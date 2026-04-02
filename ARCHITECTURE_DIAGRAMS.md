# QGRE Engine Architecture Diagrams

## 1. UML Class/Component Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          QGRE Engine Architecture                           │
└─────────────────────────────────────────────────────────────────────────────┘

                              CONFIG LAYER
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│   QGREConfig     │───▶│   SPOConfig      │    │  SkillConfig     │
│  (top-level)     │    │  • lr: 0.1       │    │  • score_key     │
└──────────────────┘    │  • aspiration_β  │    │  • mastery_thd   │
        │               │  • staleness_win │    │  • learnability_ │
        │               │  • baseline_prior│    │    threshold     │
        │               └──────────────────┘    └──────────────────┘
        │
    ┌───┴────────────────────────────┐
    │   MODEL / DATA / ALGORITHM      │
    │   [Embedded configs]            │
    └────────────────────────────────┘


                           STATE MANAGEMENT
┌──────────────────┐    ┌──────────────────┐
│   GameState      │───▶│ TutorialState    │
│  • step_count    │    │ (2D matrix)      │
│  • tier_phases   │    │ • tier → phase   │
│  • skill_tree    │    │ • mastery[tier]  │
└──────────────────┘    └──────────────────┘
        ▲                       ▲
        │ updates               │ uses
        │                       │
        └───────────────────────┘


                         ADVANTAGE COMPUTATION
┌──────────────────────────────────────────────────────────────────┐
│              QGREStepAdvantageEstimator                           │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ Per-Quality Baseline (SPO):                                 │ │
│  │   V: dict[prompt_id][quality_name] = float                 │ │
│  │   V_last_seen: dict[prompt_id][quality_name] = step_num    │ │
│  │                                                              │ │
│  │ Staleness Decay:                                            │ │
│  │   if (current_step - last_seen) > staleness_window:         │ │
│  │     V_decayed = V * decay + baseline_prior * (1 - decay)   │ │
│  │                                                              │ │
│  │ Variance-Aware LR:                                          │ │
│  │   if reward_variance < var_threshold:                       │ │
│  │     effective_lr = base_lr * min_var_ratio                 │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  Methods:                                                        │
│  • get_baseline(prompt_id, quality_name) → float                │
│  • update_baseline(prompt_id, quality_name, reward, lr)         │
│  • compute_step_advantages(...) → dict[step_num][quality]       │
└──────────────────────────────────────────────────────────────────┘
        ▲
        │ reads/updates
        │
        └─ [Uses RewardResult.scores]


                      VPRM CRITIC (Optional)
┌──────────────────────────────────────────────────────────────────┐
│               VPRMCritic (Learned Baselines)                      │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ Per-Quality MLPs:                                            │ │
│  │   heads: dict[quality_name] = QualityMLP (online)           │ │
│  │   target_heads: dict[quality_name] = QualityMLP (target)    │ │
│  │                                                              │ │
│  │ QualityMLP:                                                 │ │
│  │   hidden_dim → 128 (ReLU) → 128 (ReLU) → 1 (value)         │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  Methods:                                                        │
│  • forward(hidden_states, regions) → dict[quality_name]         │
│  • compute_advantages(hidden_states, regions, rewards)          │
│    → (advantages, critic_losses)                                │
│  • update_target_network(tau) [Polyak averaging]                │
└──────────────────────────────────────────────────────────────────┘
        ▲
        │ [Optional: VPRM enabled]
        │
        └─ [Falls back to SPO when < 2 distinct regions]


                        REWARD & SEGMENTATION
┌──────────────────┐    ┌──────────────────────┐
│  RewardResult    │    │   Segmenter Fn       │
│  • reward: float │    │ token_ids → regions  │
│  • scores: dict  │───▶│ (uniform, qwen3_xml, │
│    {quality:r}   │    │  hif_json, label)    │
│  • scored_spans  │    └──────────────────────┘
│    {quality:     │            ▲
│    [(start,end)] │            │
└──────────────────┘            │
        ▲                        │
        │                        │
        └────────────────────────┴─ Broadcast step advantages
                                    to per-token via regions


                          LOSS COMPUTATION
┌──────────────────────────────────────────────────────────────────┐
│                ClippedPGLossFn (from NeMo RL)                    │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ Inputs (per batch):                                          │ │
│  │   • curr_logprobs: [batch, seq]                             │ │
│  │   • prev_logprobs: [batch, seq]                             │ │
│  │   • advantages: [batch, seq] (per-token, per-quality)       │ │
│  │   • mask: [batch, seq] (1=valid, 0=pad)                     │ │
│  │   • reference_logprobs: [batch, seq] (optional KL)          │ │
│  │   • kl_region_weights: per-token region multipliers         │ │
│  │                                                              │ │
│  │ Processing:                                                 │ │
│  │ 1. Compute probability ratios:                              │ │
│  │      log_ratio = curr_logprobs - prev_logprobs             │ │
│  │      ratio = exp(log_ratio)                                │ │
│  │      ratio_clamped = clamp(ratio, [1-low, 1+high])         │ │
│  │                                                              │ │
│  │ 2. λ-return (GRPO-λ):                                       │ │
│  │      if lambda_return > 0:                                 │ │
│  │        advantages = apply_eligibility_traces(advantages)   │ │
│  │                                                              │ │
│  │ 3. Clipped surrogate loss:                                  │ │
│  │      clip_loss = max(-r*ratio, -r*ratio_clamped)           │ │
│  │                                                              │ │
│  │ 4. KL regularization:                                       │ │
│  │      if kl_penalty > 0 && reference_logprobs:              │ │
│  │        kl = kl_region_weights * KL(curr, reference)        │ │
│  │        kl_loss = masked_mean(kl, mask)                     │ │
│  │                                                              │ │
│  │ 5. Final loss aggregation:                                 │ │
│  │    loss = actor_loss + kl_loss                             │ │
│  │                                                              │ │
│  │ Returns:                                                    │ │
│  │   (loss, metrics_dict)                                      │ │
│  │   + [optional] per_token_loss [batch, seq] for per-quality  │ │
│  └─────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
        ▲
        │
        └─ Takes per-token advantages from estimator


                        TRAINER ORCHESTRATION
┌──────────────────────────────────────────────────────────────────┐
│                     QGRETrainer                                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ Training Loop (per step):                                    │ │
│  │  1. Generate completions                                     │ │
│  │  2. Reward function → RewardResult.scores                    │ │
│  │  3. Segmenter → regions per token                            │ │
│  │  4. QGREStepAdvantageEstimator.compute_advantages()          │ │
│  │     → per-quality per-step advantages                        │ │
│  │  5. [Optional] VPRM: compute_advantages()                    │ │
│  │  6. Broadcast step→token advantages by region               │ │
│  │  7. ClippedPGLossFn() → actor_loss + kl_loss                │ │
│  │  8. Backward + optimizer step                                │ │
│  │  9. Update SPO baselines V for next step                     │ │
│  │ 10. SkillNode.record_score() → ready_to_advance check      │ │
│  │ 11. [Tutorial] Unlock/phase-advance on frontier steps       │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  State Exports:                                                  │
│  • global_step (orchestration clock)                            │ │
│  • advantage_estimator (SPO baseline state)                     │ │
│  • vprm_critic (learned critic state)                           │ │
│  • game_state (mastery matrix state)                            │ │
│  • completion_logger (output tracking)                          │ │
└──────────────────────────────────────────────────────────────────┘
```

## 2. Energy Map (Data Flow Diagram)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│         Per-Quality Score Flow through the QGRE Engine                       │
│              (Single Quality: "q_correct", for example)                      │
└─────────────────────────────────────────────────────────────────────────────┘


STAGE 1: REWARD SIGNAL ENTRY
────────────────────────────────────────────────────────────────────────────────

    reward_fn(completion_text)
           ▼
    RewardResult(
        reward=0.85,
        scores={
            "q_correct": 0.92,  ◄─── Per-quality score
            "q_format": 0.78,
            "q_length": 0.95
        },
        scored_spans={
            "q_correct": [(0, 150), (200, 280)],  ◄─── Optional character spans
            ...
        }
    )


STAGE 2: BASELINE LOOKUP (SPO)
────────────────────────────────────────────────────────────────────────────────

    Input: prompt_id=42, quality_name="q_correct", score=0.92

    get_baseline(prompt_id=42, quality_name="q_correct")
        ▼
    Check staleness:
        last_seen_step = V_last_seen[42]["q_correct"] = 150
        current_step = 200  (from trainer.global_step)
        steps_since = 200 - 150 = 50

        if steps_since > staleness_window (50):
            decay = 0.9 ^ (50 / 50) = 0.9
            V_baseline = V[42]["q_correct"] * 0.9 + baseline_prior * 0.1
                       = 0.75 * 0.9 + 0.5 * 0.1
                       = 0.675 + 0.05 = 0.725
        else:
            V_baseline = V[42]["q_correct"] = 0.75

    Result: V_baseline = 0.725


STAGE 3: ADVANTAGE COMPUTATION
────────────────────────────────────────────────────────────────────────────────

    Raw advantage:
        A_raw = reward - V_baseline
              = 0.92 - 0.725
              = 0.195

    Target-aware aspiration gap (SPO):
        if aspiration_beta > 0:
            aspiration_target = 0.8  (from config)
            A_final = A_raw + beta * (reward - aspiration_target)
                    = 0.195 + 0.5 * (0.92 - 0.8)
                    = 0.195 + 0.5 * 0.12
                    = 0.195 + 0.06
                    = 0.255  ◄─── Final per-quality advantage
        else:
            A_final = A_raw = 0.195


STAGE 4: BASELINE UPDATE & VARIANCE TRACKING
────────────────────────────────────────────────────────────────────────────────

    Compute effective learning rate:
        reward_variance = track running variance of "q_correct" scores
        if reward_variance < var_threshold (0.01):
            effective_lr = base_lr * min_var_ratio
                        = 0.1 * 0.01 = 0.001
        else:
            effective_lr = base_lr = 0.1

    Update baseline:
        V_new = V + effective_lr * (reward - V)
              = 0.75 + 0.1 * (0.92 - 0.75)
              = 0.75 + 0.1 * 0.17
              = 0.75 + 0.017
              = 0.767

    V[42]["q_correct"] = 0.767
    V_last_seen[42]["q_correct"] = 200


STAGE 5: VPRM CRITIC (Optional Path)
────────────────────────────────────────────────────────────────────────────────

    [If VPRM enabled and region_count >= spo_fallback_min_regions]

    hidden_states: [seq_len, 4096]  (from model forward pass)
    regions: ["THINK", "STEP_1", "STEP_1", "STEP_2", "STEP_2", ...]

    Step 1: Pool hidden states by region
        STEP_1_pool = mean(hidden_states where region == "STEP_1")
                    = [4096]

    Step 2: Forward through critic head for "q_correct"
        heads["q_correct"]: nn.Sequential(
            Linear(4096, 128) → ReLU →
            Linear(128, 128) → ReLU →
            Linear(128, 1)
        )
        V_critic = heads["q_correct"](STEP_1_pool) = scalar tensor

    Step 3: Compute VPRM advantage
        A_vprm = reward - V_critic
               = 0.92 - 0.71 = 0.21
        A_vprm_clipped = clamp(A_vprm, [-clip_adv, clip_adv])
                       = clamp(0.21, [-5, 5]) = 0.21

    Step 4: Critic loss (for critic update)
        critic_loss = MSE(V_critic, reward)
                    = (0.71 - 0.92)^2
                    = 0.0441

    Use A_vprm or fallback to SPO based on sample properties


STAGE 6: PER-STEP AGGREGATION
────────────────────────────────────────────────────────────────────────────────

    step_advantages[quality_name] = aggregate from all samples

    For "q_correct" quality:
        step_advs["q_correct"] = [0.255, 0.180, -0.05, 0.120, ...]  (per sample)

    Apply frontier amplification:
        if step_num in frontier_steps and amplification > 0:
            step_advs["q_correct"] *= (1.0 + frontier_amplification)
                                     *= (1.0 + 2.0)
                                     *= 3.0


STAGE 7: BROADCAST TO TOKENS BY REGION
────────────────────────────────────────────────────────────────────────────────

    Input: step_advs = {"q_correct": [0.255, 0.180, -0.05, ...]},
           regions = ["THINK", "STEP_1", "STEP_1", "STEP_2", ...]

    For each unique region, build label_to_adv map:
        region == "STEP_1" ▶ step_num = 1 ▶ step_advs[1]["q_correct"]

        if region has virtual steps mapped to it (step_region_map):
            label_to_adv["STEP_1"] = step_advs[1]["q_correct"]
                                   + sum(step_advs[vs]["q_correct"] for vs in virtual_steps)

    Per-token assignment:
        token_advs[t] = label_to_adv[regions[t]]

    Result: token_advs = [<THINK>, 0.255, 0.255, 0.21, 0.21, ...]
                          per-token advantages broadcast from per-step


STAGE 8: PER-TOKEN LOSS & LOGPROB RATIO
────────────────────────────────────────────────────────────────────────────────

    For each token t in sequence:
        curr_logprob[t] = logprob from current policy
        prev_logprob[t] = logprob from generation policy
        mask[t] = 1 if valid, 0 if padding
        advantage[t] = 0.255 (from STAGE 7)

    Probability ratio:
        log_ratio[t] = curr_logprob[t] - prev_logprob[t]
        ratio[t] = exp(log_ratio[t])

    Clipped ratio:
        ratio_clamped[t] = clamp(ratio[t], [1 - 0.2, 1 + 0.28])

    Clipped loss per token:
        loss1[t] = -advantage[t] * ratio[t]
        loss2[t] = -advantage[t] * ratio_clamped[t]
        per_token_loss[t] = max(loss1[t], loss2[t])


STAGE 9: KL REGULARIZATION (Region-Weighted)
────────────────────────────────────────────────────────────────────────────────

    if kl_penalty > 0 and reference_logprobs provided:

        kl_region_weights[t] based on regions[t]:
            region == "THINK"  ▶ kl_think_multiplier = 0.1
            region == "FORMAT" ▶ kl_format_multiplier = 2.0
            region == "STEP"   ▶ kl_step_multiplier = 1.0

        kl[t] = kl_region_weights[t] * calculate_kl(
                    curr_logprobs[t],
                    reference_logprobs[t],
                    type="k3"  [exponential]
                )

        kl_loss = masked_mean(kl, mask)


STAGE 10: PER-QUALITY LOSS AGGREGATION
────────────────────────────────────────────────────────────────────────────────

    Total loss:
        actor_loss = sum of clipped losses, masked and normalized
        kl_loss = KL regularization term (computed above)
        loss = actor_loss + kl_loss

    Metrics logged:
        metrics = {
            "loss": loss.item(),
            "actor_loss": actor_loss.item(),
            "kl_penalty": kl_loss.item(),
            "probs_ratio_mean": mean of ratios,
            "probs_ratio_clamped_mean": mean of clipped ratios,
        }

    Per-quality aggregation (from per_token_loss):
        if return_per_token_loss=True:
            return loss, metrics, weighted_loss  [batch, seq]


STAGE 11: BACKWARD & GRADIENT FLOW
────────────────────────────────────────────────────────────────────────────────

    loss.backward()

    Gradient flows:
        dL/d(curr_logprob) ← from clipped PG loss and KL
        dL/d(model_params) ← via backprop through logit→logprob→loss

    If VPRM critic enabled:
        dL_critic/d(critic_heads["q_correct"]) ← from MSE(V_critic, reward)
        critic_optimizer.step()  [separate from policy optimizer]


STAGE 12: SKILL TREE UPDATE (Tutorial)
────────────────────────────────────────────────────────────────────────────────

    SkillNode.record_score(quality_score):
        recent_scores.append(0.92)  [deque with maxlen=mastery_window]
        _total_completions += 1

    Check mastery:
        mastery_score = mean(recent_scores) = 0.85
        learnability = p * (1 - p) = 0.85 * 0.15 = 0.1275

    Check ready_to_advance:
        if mastery_score >= mastery_threshold (0.8)
           AND learnability < learnability_threshold (0.10)
           ▶ Skill is mastered, unlock prerequisites


STAGE 13: PHASE GATING (Quality-Based)
────────────────────────────────────────────────────────────────────────────────

    if score_key = "q_correct":  (skill tracks this quality)
        skill_mastery = average of recent_scores for "q_correct"

        quality_phases = build_phase_qualities(step_qualities):
            phase 1: [q_format]
            phase 2: [q_format, q_grounding]
            phase 3: [q_format, q_grounding, q_accuracy]

    if skill_mastery > tier_advance_threshold (0.85)
       AND current_phase >= tier_advance_quality_phase (3)
       ▶ Unlock next tier
       ▶ Add new prompts to active pool


FINAL OUTPUT
────────────────────────────────────────────────────────────────────────────────

Per-quality score (0.92) →
  Baseline (0.725) →
    Advantage (0.255) →
      Broadcast to tokens (0.255 per STEP_1 token) →
        Clipped PG loss per token →
          Actor loss + KL loss →
            Gradients →
              Policy update →
                Baseline update (V[42]["q_correct"] = 0.767) →
                  Next step cycles with decayed old baselines
```

## Architectural Disconnections & Gaps

### 1. **VPRM Fallback Logic Not Logged**
- `critic.py:compute_advantages()` silently falls back to SPO when `region_count < spo_fallback_min_regions`
- **Issue**: No log/metric indicating which path (VPRM vs SPO) was used per sample
- **Impact**: Difficult to debug if critic isn't being used as expected
- **Location**: Need explicit flag or counter in trainer step metrics

### 2. **Critic Loss Aggregation Missing**
- VPRM critic returns `critic_losses: dict[quality_name] → torch.Tensor` (scalar MSE per quality)
- **Issue**: These per-quality MSE losses aren't shown being combined into a single backward signal
- **Missing**: Somewhere in trainer loop, need `critic_loss_total = sum(critic_losses.values())` before `critic_loss_total.backward()`
- **Location**: `trainer.py` training loop (not shown in excerpt)

### 3. **Aspiration Warmup Ramp Not Visible**
- `SkillConfig.aspiration_warmup_steps = 20` defined in config
- `QGRETrainer` sets `advantage_estimator._aspiration_beta = spo_cfg.aspiration_beta` once (static)
- **Issue**: The linear ramp from 0→1 over 20 steps after unlock isn't computed in shown code
- **Missing**: Compute `aspiration_warmup_factor` based on `(current_step - skill_unlock_step) / aspiration_warmup_steps` and apply to `_aspiration_beta` per-prompt
- **Location**: Likely in `compute_step_advantages()` or trainer loop

### 4. **Per-Quality Loss Logging Undefined**
- `ClippedPGLossFn` returns single scalar loss, loses per-quality structure
- **Issue**: No per-quality loss metrics (e.g., `loss_q_correct`, `loss_q_format`) in shown metrics dict
- **Missing**: If `return_per_token_loss=True`, need to aggregate per-token losses by quality and log separately
- **Location**: Trainer uses return value but doesn't unpack `per_token_loss`

### 5. **Scored Spans → Token Mask Bridge Missing**
- `RewardResult.scored_spans: dict[quality_name] → [(char_start, char_end), ...]`
- **Issue**: Character offsets provided but no visible code converts them to token-level masks
- **Missing**: Need tokenizer char-to-token mapping, then AND scored_span_mask with existing token mask for per-quality token filtering
- **Location**: Should be in `compute_step_advantages()` or loss function, before advantage broadcast

### 6. **Phase Transition Trigger Not Shown**
- `SkillNode.ready_to_advance` property checks if skill can advance
- **Issue**: Property is read-only; no explicit code calls `game_state.advance_tier()` or `skill._status = SkillStatus.MASTERED`
- **Missing**: Trainer loop should check `skill.ready_to_advance` and call state update method
- **Location**: Trainer's tutorial integration loop (not in excerpt)

### 7. **Frontier Amplification Scope Unclear**
- `apply_frontier_amplification()` multiplies `step_advs[step_num]` for each step
- **Issue**: "Frontier steps" computed where? For which quality?
- **Missing**: Definition of frontier steps (steps that are bottleneck for phase advancement)
- **Location**: Likely in `compute_step_advantages()` but not shown in advantages.py excerpt

---

## Summary

The architecture is **modular and well-separated** (SPO vs VPRM, advantages vs loss, state management). However, integration points between modules lack explicit logging and error handling. Key gaps are in per-quality metric aggregation, aspiration warmup ramp, and fallback path observability.

**Recommendation**: Add explicit wrapper around VPRM fallback decision, aggregate per-quality losses as separate MLflow metrics, and make aspiration warmup ramp visible in trainer step loop.

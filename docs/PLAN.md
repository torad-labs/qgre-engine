# QGRE-Optimized Training Engine — Plan

## Vision

A single-binary, single-GPU training engine for QGRE (Quality-Gated Reward Escalation)
that eliminates all unnecessary infrastructure overhead. No Ray, no multi-process
serialization, no HTTP servers. Just: generate → score → compute advantages → backward → update.

Built on extracted components from **NeMo RL** (Apache-2.0, NVIDIA) for the core algorithm
layer, with QGRE's phase-gated curriculum on top. NeMo RL provides production-quality
GRPO/GDPO math; QGRE provides the novel domain curriculum that NeMo RL doesn't have.

## Why

Our current stack has 6 layers of indirection on a single GPU:

```
Python script → Ray driver → Ray worker → verl ray_trainer → fsdp_workers → dp_actor → PyTorch
                  ↕ (serialization)           ↕ (IPC)              ↕ (DataProto)
              Ray GCS server            RewardLoopWorker        HF/vLLM rollout
```

Each `↕` is a serialization boundary. Ray adds ~15% overhead on single GPU (scheduling,
object store, GCS, dashboard, metrics). The DataProto ↔ TensorDict conversions happen
4+ times per step. None of this is needed for single-GPU GRPO.

The target:

```
Python loop → vLLM generate → reward function → GRPO loss → backward → optimizer step
```

One process. One GPU. Direct function calls. No serialization.

---

---

## NeMo RL Adoption Strategy

### What NeMo RL is

NVIDIA's open-source RL training library (Apache-2.0, 1400 stars, active as of March 2026).
Built for multi-node H100 clusters with Megatron Core, Ray, vLLM. Designed for scale.

### What we take (pure algorithm layer — zero infrastructure deps)

| File | Lines | What it gives us |
|------|-------|-----------------|
| `advantage_estimator.py` | 222 | **GDPO advantage** — per-component reward baselines. Prevents QGRE's format_reward from drowning grounding_reward in Phase 3+. Also: leave-one-out baseline (better than group-mean for n=4), Reinforce++ |
| `loss/loss_functions.py` | ~200 | **Clipped PG loss** — NVIDIA's tested implementation of the GRPO/PPO clip math |
| `utils.py` | 845 | Baseline computation, KL calculation, reward component extraction |
| `logits_sampling_utils.py` | 285 | Log prob computation utilities |

Total: ~1550 lines of clean, tested GRPO/GDPO math. No Ray, no Megatron, no infrastructure.

### What we DON'T take

- Ray orchestration (we're eliminating Ray)
- Megatron Core training backend (we use Unsloth + PyTorch)
- vLLM server mode (we use Unsloth's in-process fast_generate)
- NeMo Gym environment abstraction (our reward_fn is more sophisticated)
- Multi-node support (single GPU target)
- Distillation, DPO, SFT algorithms (not needed for QGRE)

### Why GDPO matters for QGRE

QGRE has 4+ reward components that activate progressively:
```
Phase 1: format_reward only
Phase 2: format + grounding
Phase 3: format + grounding + chain_coherence
Phase 4: format + grounding + chain + accuracy + archetype
Phase 5: format + grounding + chain + accuracy + archetype + node_f1
```

Standard GRPO normalizes the SUM of all active rewards per group. When format_reward
saturates (≈1.0 for all completions), its variance drops to zero and the advantage signal
comes entirely from the newest reward component. This is fine in Phase 1-2 but causes
**multi-objective crowding** in Phase 3+ — accuracy dominates because it has highest variance.

GDPO normalizes each component separately BEFORE summing:
```
advantage = normalize(format_advantages) + normalize(grounding_advantages) + ...
```

Each component maintains its own signal strength regardless of the others. Format stays
relevant even when accuracy dominates. This directly addresses QGRE Failure Mode #5
(multi-objective reward crowding) from the MASTER-PLAN.

### QGRE vs GDPO — complementary, not competing

| | GDPO (NVIDIA) | QGRE (ours) |
|---|------|------|
| **Layer** | Advantage computation math | Reward function curriculum design |
| **Problem** | Multi-reward signal collapse during normalization | Cold start on novel domains with no ground truth methodology |
| **Mechanism** | Per-component normalization of advantages | Phase-gated quality escalation with mastery tracking |
| **Assumes** | All rewards active simultaneously | Rewards unlock progressively based on demonstrated mastery |
| **Domain** | Any multi-reward RL setup | Novel domains where correct reasoning methodology is unknown |

**They compose**: GDPO handles the normalization math, QGRE handles the curriculum.
GDPO makes QGRE's Phase 3+ more stable. QGRE provides the progressive structure GDPO lacks.

### Primary Algorithm: SPO + VPRM + GDPO (unified per-step advantages)

**SPO (Single-stream Policy Optimization)** — ICLR 2026, Tencent. Replaces GRPO's group-based
normalization with persistent per-prompt value tracking. Eliminates degenerate groups, enables
adaptive curriculum, allows n=1 (more prompt diversity per step).

**The QGRE composition (novel — nobody has built this):**

```
Old (per-sequence, disconnected):
  SPO:  advantage = reward - V(prompt)              → uniform per-token (broadcast)
  VPRM: token_rewards = per-step-region rewards     → per-token (segment propagation)
  Problem: two systems, not connected

New (unified per-step):
  advantage_t = normalize_per_step(step_reward[step_of_t] - V(prompt, step_of_t))
  where step_reward = mean of phase-active qualities for that step (QGRE gating)
  and normalize_per_step = GDPO-style normalization across the batch, per step
```

Four problems solved in ONE system:
1. **Credit assignment** (VPRM) — correct step 1 gets positive advantage even when step 4 fails
2. **Degenerate groups** (SPO) — persistent value tracker, no groups needed
3. **Reward crowding** (GDPO) — per-step normalization preserves each step's signal
4. **Cold start curriculum** (QGRE) — phase gates control which qualities are active per step

**Implementation: Unified per-step advantage estimator**

```python
# Which qualities belong to which step region
STEP_QUALITIES = {
    1: ["q_format_tags", "q_tag_content", "q_node_in_prompt", "q_node_format", "q_node_length"],
    2: ["q_chain_s2_refs_s1"],
    3: ["q_chain_s3_refs_s2", "q_self_consistency"],
    4: ["q_step4_valid_json", "q_step4_has_keys", "q_existence_correct",
        "q_archetype_correct", "q_node_f1"],
}

class QGREStepAdvantageEstimator:
    """Unified: SPO + GDPO + VPRM + QGRE phase gating.

    One system replaces both QGREAdvantageEstimator AND compute_step_region_rewards.
    Per-step persistent value tracking (SPO) with per-step batch normalization (GDPO)
    and segment propagation to per-token advantages (VPRM).
    """

    def __init__(self, lr=0.1):
        self.V = defaultdict(lambda: defaultdict(float))  # V[prompt_hash][step_num]
        self.lr = lr
        self._step_seen = defaultdict(set)  # warm-start tracking per prompt

    def compute_advantages(self, batch_prompt_ids, batch_token_ids,
                           batch_reward_results, batch_active_qualities):
        """
        Args:
            batch_prompt_ids: list[int] — prompt identifier per completion
            batch_token_ids: list[list[int]] — token IDs per completion
            batch_reward_results: list[RewardResult] — from reward_fn
            batch_active_qualities: list[list[str]] — phase-active qualities per completion
        Returns:
            batch_advantages: list[Tensor[seq_len]] — per-token advantages
        """
        batch_size = len(batch_token_ids)

        # Phase 1: Segment tokens + compute per-step rewards
        all_regions = []
        all_step_rewards = []  # [{step_num: float}]
        for i in range(batch_size):
            regions = segment_completion(batch_token_ids[i])
            step_rews = {}
            for step_num, quality_keys in STEP_QUALITIES.items():
                active = [k for k in quality_keys if k in batch_active_qualities[i]]
                if active:
                    step_rews[step_num] = float(np.mean([
                        batch_reward_results[i].scores.get(k, 0.0) for k in active
                    ]))
                else:
                    step_rews[step_num] = 0.0
            all_step_rewards.append(step_rews)
            all_regions.append(regions)

        # Phase 2: Per-step SPO advantages + GDPO normalization
        step_advs = {s: torch.zeros(batch_size) for s in range(1, 5)}
        for step_num in range(1, 5):
            for i in range(batch_size):
                pid = batch_prompt_ids[i]
                r = all_step_rewards[i].get(step_num, 0.0)
                v = self.V[pid][step_num]

                # Warm-start: first observation for this step = set baseline, no advantage
                if step_num not in self._step_seen[pid] and v == 0.0 and r != 0.0:
                    v = r
                    self._step_seen[pid].add(step_num)

                step_advs[step_num][i] = r - v
                self.V[pid][step_num] = v + self.lr * (r - v)

            # GDPO-style: normalize this step's advantages across the batch
            if step_advs[step_num].std() > 1e-8:
                step_advs[step_num] = (
                    (step_advs[step_num] - step_advs[step_num].mean())
                    / (step_advs[step_num].std() + 1e-8)
                )

        # Phase 3: Broadcast per-step advantages to per-token by region
        batch_advantages = []
        for i in range(batch_size):
            token_advs = torch.zeros(len(batch_token_ids[i]))
            for t, region in enumerate(all_regions[i]):
                if region.startswith("STEP_"):
                    sn = int(region.split("_")[1])
                    token_advs[t] = step_advs[sn][i]
                # THINK, FORMAT, OTHER → 0 advantage (no gradient for structure/exploration)
            batch_advantages.append(token_advs)

        return batch_advantages

    def on_tier_advance(self, new_tier, prompt_tier_map):
        """Reset value tracker for newly-ungated prompts."""
        for pid, tier in prompt_tier_map.items():
            if tier == new_tier:
                self.V[pid] = defaultdict(float)
                self._step_seen[pid] = set()
```

**GRPO fallback mode** uses the same structure with group-mean baseline instead of SPO tracker:
```python
# GRPO mode: per-step GROUP normalization (no persistent tracker)
for step_num in range(1, 5):
    group_rewards = [all_step_rewards[i][step_num] for i in group_indices]
    mean, std = np.mean(group_rewards), np.std(group_rewards) + 1e-8
    for i in group_indices:
        step_advs[step_num][i] = (all_step_rewards[i][step_num] - mean) / std
```
Same region propagation, same GDPO normalization. Only the baseline computation differs.

**Why n=1 is better for QGRE:**

Current: 2 prompts × 4 completions = 8 samples/step
- Only 2 unique prompts per step
- 30% of groups are degenerate (zero variance) → wasted compute
- Group-mean baseline is noisy with n=4

SPO with n=1: 8 prompts × 1 completion = 8 samples/step
- 8 unique prompts per step (4x more diversity)
- Zero degenerate groups (no groups at all)
- Persistent value tracker provides stable baseline from history
- More archetypes seen per step → faster curriculum advancement
- Same compute budget, fundamentally different signal quality

**Fallback: Dr.GRPO (if SPO doesn't converge)**

SPO + per-component tracking is untested in combination. If it doesn't converge:
- Fall back to standard Dr.GRPO (our current working config)
- Add GDPO normalization on top (NeMo RL extraction)
- Keep QGRE phase gating (already working)
- This is the conservative path — proven components, just slower

**Benchmark plan:**
1. SPO + per-component tracking: 50 steps, measure reward curve slope
2. Dr.GRPO + GDPO: 50 steps, same data, measure reward curve slope
3. Dr.GRPO (current): baseline from our 500-step run
4. Compare: convergence speed, reward variance, degenerate group rate

### Integration plan (updated for SPO-first)

1. Extract `advantage_estimator.py` + `utils.py` + `loss_functions.py` from NeMo RL
2. Extract SPO core from verl PR #3503 (persistent value tracker, ~100 lines)
3. Build `QGREAdvantageEstimator` — combines SPO tracker + GDPO normalization + QGRE phase filter
4. Wire into QGRETrainer as primary advantage estimator
5. Keep Dr.GRPO + GDPO as fallback (switch via config flag)
6. Benchmark: SPO vs GDPO vs Dr.GRPO on same data for 50 steps

---

## Phase 1: Eliminate Ray (highest impact, lowest risk)

### What Ray does in our single-GPU setup

1. **Process management** — spawns TaskRunner + WorkerDict + RewardLoopWorker as Ray actors
2. **RPC dispatch** — `actor_rollout_wg.generate_sequences()` is a Ray remote call that serializes DataProto, sends to worker, deserializes, executes, serializes result, sends back
3. **Object store** — intermediate tensors go through Ray's plasma store
4. **Scheduling** — Ray's GCS decides when/where to run each stage
5. **Dashboard + metrics** — Ray dashboard on port 8265

On single GPU, ALL of this is waste. The worker is on the same GPU as the driver.
The "remote" call goes through IPC to a process 10 feet away in memory.

### What exists

- **simple_GRPO** (github.com/lsdefine/simple_GRPO, 1600 stars) — Pure PyTorch GRPO in ~500 lines.
  No Ray, no verl. Proves the concept. Missing: vLLM integration, LoRA weight sync,
  curriculum, reward sub-components.

- **tiny-grpo** (multiple repos) — Even more minimal. Single file GRPO training.

- **TRL colocated vLLM** (PR #3394, merged May 2025) — IBM's contribution. vLLM runs in-process
  on same GPU as training. No HTTP server, no separate process. Uses vLLM's `external_launcher`
  for torch-compatible execution. **3x faster GRPO.** Ray-less solution.

- **vLLM RLHF colocate example** (vllm/examples/offline_inference/rlhf_colocate.py) — Official
  example of sleep/wake + CUDA IPC for weight sharing. Shows the low-level API.

- **NeMo RL** — NVIDIA's clean GRPO implementation. More modular than verl.

### Architecture

```python
# The entire training loop — no Ray, no verl
# Loss from NeMo RL (Apache-2.0), advantages from QGREStepAdvantageEstimator
from nemo_rl_extracted.loss_functions import ClippedPGLossFn
from qgre_engine.advantages import QGREStepAdvantageEstimator
from qgre_engine.segments import segment_completion

class QGRETrainer:
    def __init__(self, model, tokenizer, reward_fn, config):
        self.model = model                    # Unsloth PeftModel
        self.vllm_engine = model.vllm_engine  # Shared-memory vLLM
        self.reward_fn = reward_fn            # _hypergraph_reward_internal — returns RewardResult
        self.optimizer = AdamW8bit(model.parameters(), lr=config.lr)
        self.game_state = GameState()         # QGRE curriculum tracking

        # Unified advantage estimator: SPO + GDPO + VPRM + QGRE phase gating
        self.advantage_estimator = QGREStepAdvantageEstimator(lr=config.spo_lr)
        self.loss_fn = ClippedPGLossFn(config.loss)

    def step(self, prompts):
        # 1. Generate (vLLM, in-process, shared GPU memory)
        completions = self.vllm_engine.generate(prompts, sampling_params)

        # 2. Score (direct call — returns RewardResult with full per-quality scores)
        reward_results = [self.reward_fn(p, c, meta) for p, c, meta in zip(...)]
        active_qualities = [PHASE_QUALITIES[rr.phase] for rr in reward_results]

        # 3. Compute per-token advantages (unified: segment → step rewards → SPO → GDPO → broadcast)
        #    Step 1 correct + step 4 wrong → step 1 tokens get positive advantage, step 4 negative
        token_advantages = self.advantage_estimator.compute_advantages(
            batch_prompt_ids=prompt_ids,
            batch_token_ids=[c.token_ids for c in completions],
            batch_reward_results=reward_results,
            batch_active_qualities=active_qualities,
        )

        # 4. Compute log probs + policy loss (NeMo RL clipped PG loss)
        log_probs = self.compute_log_probs(completions)
        loss = self.loss_fn(log_probs, old_log_probs, token_advantages, response_mask)

        # 5. Backward + update (standard PyTorch)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        # 6. Sync LoRA to vLLM (in-memory, ~12MB)
        self.model.load_lora(self.lora_path)

        # 7. Log to MLflow (direct call, no wrapper)
        mlflow.log_metrics({
            "reward/mean": mean_reward,
            "reward/step_1": mean_step_1_reward,
            "reward/step_4": mean_step_4_reward,
            "advantage/step_1": mean_step_1_adv,
            "advantage/step_4": mean_step_4_adv,
        }, step=self.step_num)
```

### What we use from NeMo RL (replacing verl's algorithm layer)

- **ClippedPGLoss** — tested clip math with proper loss aggregation
- **calculate_kl** — KL penalty computation (for kl_cov mode)
- **logits_sampling_utils** — log prob computation

NOTE: GDPOAdvantageEstimator is NOT used directly. Its per-component normalization
concept is implemented natively in QGREStepAdvantageEstimator as per-STEP normalization.
The mapping: GDPO component = QGRE step region. This is simpler and avoids the
adapter layer that was previously needed (old Step 0d).

### What we keep from our stack

- **Reward function** (`reward_fn.py`) — already standalone, QGRE curriculum
- **Curriculum gating** (`GameState`) — already in reward_fn, MLflow persistence
- **Unsloth model loading** — fast_inference + vLLM + LoRA
- **MLflow tracking** — direct calls, no wrapper needed
- **Completion logging** — direct JSONL write
- **Tokenizer** — nothink template, PAD=151654

### What we drop

| Component | Why it exists | Why we drop it |
|-----------|--------------|----------------|
| Ray | Multi-GPU orchestration | Single GPU — direct calls |
| DataProto | Serialization format for Ray IPC | No IPC needed |
| fsdp_workers.py | FSDP wrapping + worker lifecycle | No FSDP on single GPU |
| ray_trainer.py | 1700-line orchestrator | Replace with 200-line loop |
| single_controller | Ray RPC dispatch | Direct function calls |
| TensorDict conversions | DataProto ↔ TensorDict ↔ back | Native tensors throughout |
| RewardLoopWorker | Async reward in Ray actor | Direct function call |
| HFAgentLoopManager | Async rollout orchestration | Direct vLLM call |

### Estimated impact

- **Step time**: 30s → ~20s (remove serialization + Ray overhead)
- **Memory**: ~500MB less (no Ray object store, GCS, dashboard)
- **Code**: 1700 lines (ray_trainer.py) → ~200 lines
- **Startup**: ~30s faster (no Ray cluster init)
- **Debugging**: Direct stack traces instead of Ray remote error wrapping

### Implementation plan (corrected build order from TWO pressure tests)

**6 prerequisites FIRST — resolved cheapest-to-most-complex, each unblocks a bundle:**

```
Step 0a: GameState serializer (1 hour)
  UNBLOCKS: Persistence bundle (F24 → F21 checkpoint resume)
  - Problem: GameState contains deque, defaultdict(lambda: ...), nested
    quality_windows dict. These don't torch.save naively.
  - Fix: to_dict() / from_dict() methods on GameState that convert:
    deque → list, defaultdict → dict, nested windows → plain dict
  - Test: round-trip test — serialize, deserialize, compare all fields
  - Why first: cheapest fix (30 lines), unblocks crash recovery

Step 0b: NeMo RL extraction + strip Ray imports (2 hours)
  UNBLOCKS: Algorithm bundle (F5 LOO baseline, F7 clipped PG loss)
  - Extract: advantage_estimator.py, loss_functions.py, utils.py, logits_sampling_utils.py
  - Strip: Ray/Megatron imports in utils.py (replace with direct torch calls)
  - Test: import extracted modules, verify no external deps beyond torch/numpy
  - Why second: cheapest constraint to remove, unlocks 2 algorithm members
    that are currently suppressed at near-zero Net values

Step 0c: Batch reward tensor construction (1 hour)
  UNBLOCKS: GDPO input format (F4 GDPO advantage estimator)
  - Problem: reward_fn returns list[dict] per completion:
    [{"score": 0.65, "format_reward": 1.0, "grounding_reward": 0.3}, ...]
    GDPO needs dict[str, Tensor] per component:
    {"format_reward": tensor([1.0, 0.0, 1.0, ...]), "grounding_reward": tensor([0.3, 0.0, ...])}
  - Fix: ~20 lines — iterate list[dict], stack into per-component tensors
  - Test: verify tensor shapes match batch_size for each component

Step 0d: QGREStepAdvantageEstimator (3 hours)
  UNBLOCKS: Entire advantage computation — this IS the core algorithm
  - Unified system: SPO + GDPO + VPRM + QGRE phase gating (see algorithm section)
  - Replaces the old QGREAdvantageEstimator AND separate VPRM implementation
  - Includes: segment_completion(), STEP_QUALITIES mapping, per-step SPO tracker,
    per-step GDPO normalization, region→token broadcast, warm-start logic,
    on_tier_advance() reset, GRPO fallback mode
  - Uses batch reward tensor construction from 0c
  - Phase gating: reads active_qualities from reward_fn output, filters per step
  - Test: verify per-step advantages differ when step 1 correct + step 4 wrong
    (step 1 positive, step 4 negative — credit assignment works)

Step 0e: DataLoader epoch iteration (2 hours)
  UNBLOCKS: Training Loop bundle (F1 — the loop needs data to loop over)
  - Problem: QGRETrainer skeleton shows def step(self, prompts) but doesn't
    show where prompts come from. verl's RLHFDataset + StatefulDataLoader handles:
    * Load parquet, filter overlong prompts
    * apply_chat_template (list[dict] → token IDs) using nothink tokenizer
    * Left-pad to max_prompt_length
    * Shuffle per epoch with configurable seed
    * Batch into train_batch_size chunks
    * Expand each prompt × n for rollout generation
    * Track epoch/step for resume
  - Fix: Custom DataLoader class (~50 lines) OR use HF Dataset + torch DataLoader
  - Reference: verl's rl_dataset.py for the exact transformation pipeline
  - Test: verify identical token IDs as verl for same parquet + tokenizer

Step 0f: Checkpoint resume logic (1 hour)
  UNBLOCKS: Full crash recovery
  - Uses GameState serializer from 0a
  - Adds: checkpoint discovery (scan dir for latest global_step_N)
  - Adds: torch.save/load for full state:
    model_state_dict, optimizer_state_dict, scheduler_state_dict,
    game_state.to_dict(), global_step, curriculum_tier,
    torch.get_rng_state(), torch.cuda.get_rng_state()
  - Adds: resume detection on startup (find latest checkpoint, restore)
  - CRITICAL: Must recreate vLLM engine on resume (ms-swift #8233 — LoRA double-counting)
  - Test: train 10 steps, save, resume, verify step 11 produces same loss

Step 0g: LoRA verification harness (1 hour)
  UNBLOCKS: Silent training death prevention
  - Problem: LoRA sync can silently fail (unsloth #3802, ms-swift #8233, unsloth #3864)
  - Build: LoRAVerifier class with three functions:
    (a) verify_sync(): hash LoRA weight tensors before/after load, assert match
    (b) verify_active(): generate 1 token from fixed prompt, compare to base-model output
    (c) periodic_recreate(): tear down and recreate vLLM engine every N steps (default 50)
  - Integrates into QGRETrainer.step() as post-sync hook
  - Integrates into QGRETrainer.resume() as mandatory step
  - Test: mock model where load_lora is a no-op → verify_sync raises
  - Test: verify_active with real LoRA → output differs from base
```

**Total prerequisite effort: ~11 hours (added 0g). Then build in this order:**

```
Step 1: Write QGRETrainer class (~450 lines total with all wiring) — 5 hours
  NOTE: QGREStepAdvantageEstimator is already built in Step 0d.
  QGRETrainer wires it into the training loop.
  Includes:
  - Training loop (from simple_GRPO reference)
  - Gradient accumulation (micro-batch splitting)
  - Log prob computation (on-policy mode: old_lp = lp.detach())
  - Loss computation (kl_cov from verl + clipped PG from NeMo RL)
  - LLDS loss (extract from verl core_algos, ~40 lines)
  - Gradient clipping (clip_grad_norm_)
  - Logits→fp32 cast (nan prevention)
  - Response mask computation
  - Calls QGREStepAdvantageEstimator.compute_advantages() for per-token advantages
  - SPO mode: n=1, per-step persistent value tracker + low-advantage filter
  - GRPO mode: n=8, per-step group normalize + filter_groups
  - Mode switch via config flag (spo vs grpo)
Step 2: Wire Unsloth model loading + vLLM fast_generate (already working) — 1 hour
Step 3: Wire reward function (already standalone) — 30 min
  - Call _hypergraph_reward_internal directly (not compute_score) to get RewardResult
  - Pass RewardResult + active_qualities to QGREStepAdvantageEstimator
Step 4: Wire MLflow tracking (direct mlflow.log_metric calls) — 1 hour
  - Per-step reward metrics (step_1_reward, step_2_reward, etc.)
  - Per-step advantage metrics (step_1_advantage, step_2_advantage, etc.)
Step 5: Wire checkpoint save/resume (from 0a + 0f) — 1 hour
  - Include QGREStepAdvantageEstimator.V and ._step_seen in checkpoint
Step 6: Wire completion JSONL dump — 30 min
Step 7: Equivalence test: fixed completions through algorithm layer — 2 hours
  - Save 50 steps of verl completions + rewards to JSONL
  - Feed same data to QGREStepAdvantageEstimator + loss
  - Compare advantages within 1%, loss within 1%
Step 8: Credit assignment test — verify per-step advantages — 1 hour
  - Synthetic test: step 1 correct + step 4 wrong → step 1 advantage > 0, step 4 < 0
  - Synthetic test: all steps correct → all advantages similar
  - Synthetic test: phase 1 (format only) → only step 1 has non-zero advantage
```

**Total implementation: ~10 hours prerequisites + ~12 hours assembly = ~22 hours of focused work.**
(Saved 4h by merging QGREAdvantageEstimator + VPRM + QGRE adapter into unified
QGREStepAdvantageEstimator in Step 0d. No separate Step 1 or VPRM wiring needed.)

**Pre-build checklist (before Step 0a):**
- [ ] Add `quality_scores` and `active_qualities` to compute_score() return dict (10 min)
- [ ] Pin Unsloth + vLLM versions in requirements.txt
- [ ] Ensure MLflow is initialized BEFORE first compute_score() call in custom engine
      (GameState._init_tier_from_mlflow reads active_run — must exist)
- [ ] Define `RewardResult` dataclass in `qgre/types.py` — consumed by advantages, trainer, examples (DONE)
- [ ] Add `generation` section to config schema with temperature, top_p, top_k, stop_token_ids (DONE)
- [ ] Snapshot SPO implementation from paper (arxiv 2509.13232) — verl PR #3503 is closed/unmerged,
      author fork may disappear. Implement from paper spec (Algorithm 1), not extraction.
- [ ] Create stub reward_fn in examples/hypergraph/ for testing (returns RewardResult with synthetic scores)
- [ ] SPO is primary build path (DECIDED 2026-03-18). GRPO fallback second. Both share 90% of Step 0d.
- [ ] NeMo RL extraction: clone v0.5.0 tag, extract + strip deps (DECIDED 2026-03-18). Not rewrite.
- [ ] LoRA verification harness: separate Step 0g (DECIDED 2026-03-18). See build plan.

### Risks (from two pressure tests + full plan review)

| Risk | Severity | Mitigation |
|------|----------|-----------|
| GRPO loss subtleties lost in extraction | High | Equivalence test (step 8) — fixed completions, not generated |
| GDPO poisons from gated rewards | High | Adapter (step 0d) filters inactive components |
| GameState serialization round-trip | High | to_dict/from_dict with explicit round-trip test |
| DataLoader produces different tokens than verl | High | Compare token IDs for same parquet + tokenizer |
| Checkpoint resume misses state | High | Verify: train 10, save, kill, resume, step 11 same loss |
| vLLM version breaks Unsloth patches | Medium | Pin Unsloth+vLLM versions in requirements.txt |
| Unsloth monkey-patch fragility (KEYSTONE) | Medium | Abstract model loading behind interface |
| SPO value tracker spike on phase advance | Medium | Warm-start V with batch mean (see below) |
| SPO value tracker spike on tier advance | Medium | Reset V for ungated prompts (see below) |
| n=1 vs n=8 mode switch | Medium | Config flag, both paths tested |
| LoRA weight sync correctness | **High** | Post-sync verification (hash weights before/after) — see Exa findings |
| LoRA double-counting on resume | **High** | ms-swift #8233: first batch OK, subsequent batches malformed after resume. Must recreate vLLM on resume |
| VRAM leak during GRPO training | High | unsloth #3864: VRAM increases each step. Recreate vLLM engine every N steps |
| fast_generate ignoring LoRA | High | unsloth #3802: model-dependent. Verify LoRA applied by comparing outputs to base model |
| NeMo RL API changes between versions | Medium | loss_functions moved from algorithms.loss_functions to algorithms.loss.loss_functions in nightly. Pin to v0.5.0 |
| No sampling params in config | Medium | temperature=1.0 critical for diversity. Added to config schema |
| RewardResult type undefined | Medium | No file defined it — all modules referenced phantom type. Created qgre/types.py |
| SPO source fragility | Medium | verl PR #3503 closed/unmerged, author fork URL unknown. Implement from paper, not extraction |

### Hidden dependencies (from two pressure tests)

**Found in first pass:**
- **Checkpoint resume** — Without it, crash at step 200 loses hours
- **QGRE→GDPO adapter** — Without it, GDPO performs WORSE than GRPO
- **vLLM compat layer** — Unsloth patches vLLM internals; upgrades break silently

**Found in second pass:**
- **GameState serialization** — deque/defaultdict don't torch.save. Separate problem from checkpointing.
- **DataLoader epoch iteration** — PHANTOM. Plan showed `step(prompts)` but not where prompts come from.
- **Batch reward tensor construction** — PHANTOM. Bridge between reward_fn (list[dict]) and GDPO (dict[str, Tensor]).
- **NeMo RL extraction is cheapest constraint** — 2 hours of work unlocks 2 suppressed algorithm members.

**Found in third pass (full plan review, 2026-03-18):**
- **compute_score doesn't expose raw per-quality scores** — returns aggregated dict. VPRMs need rr.scores.
- **SPO + phase advance = advantage spike** — new components get V=0 → full reward as advantage.
- **SPO + tier advance = worse spike** — ungated prompts jump from 0.02 → real scores.
- **filter_groups incompatible with SPO n=1** — no groups to filter.
- **Equivalence test design wrong** — can't compare reward curves with non-deterministic generation.

---

### Issue resolutions (full plan review, 2026-03-18)

**RESOLVED: Advantage + VPRM structural gap (Eli parallax review, 2026-03-18)**
The plan had two disconnected systems: QGREAdvantageEstimator (per-sequence)
and VPRM segment propagation (per-token). Merged into QGREStepAdvantageEstimator
which does per-step value tracking (SPO) + per-step normalization (GDPO) +
segment propagation (VPRM) + phase gating (QGRE) in one unified system.
GDPO's "component" maps to VPRM's "step region". Saved 4h build time, eliminated
adapter layer (old Step 0d), eliminated separate Step 1.

**RESOLVED: compute_score interface for VPRMs**
Add `rr.scores` to the return dict of `compute_score()`:
```python
return {
    "score": result,
    "format_reward": ...,
    "grounding_reward": ...,
    "accuracy_reward": ...,
    "mastery_phase": float(rr.phase),
    "redirection_bonus": redirection,
    "trap_correct": ...,
    # NEW: raw per-quality scores for VPRM step-level rewards
    "quality_scores": dict(rr.scores),
    # NEW: active qualities for this phase (for VPRM phase gating)
    "active_qualities": list(PHASE_QUALITIES.get(rr.phase, _P1)),
}
```
Effort: 10 min. Add to reward_fn.py before engine build.

**RESOLVED: SPO value tracker spike on phase advancement**
When a new quality activates (phase advance), its V=0 produces advantage = full reward → spike.
Fix: warm-start new components with batch mean.
```python
class QGREAdvantageEstimator:
    def compute_advantage(self, ...):
        for component in active_components:
            rewards = reward_components[component]
            for i in range(len(rewards)):
                pid = prompt_ids[i].item()
                r = rewards[i].item()
                v = self.V[pid][component]
                if v == 0.0 and component not in self._seen_components:
                    # NEW component — warm-start with batch mean, not 0
                    v = rewards.mean().item()
                    self.V[pid][component] = v
                    self._seen_components.add(component)
                adv = r - v
                self.V[pid][component] = v + self.lr * (r - v)
                ...
```

**RESOLVED: SPO value tracker spike on tier advancement**
When max_active_tier advances, previously-gated prompts (reward=0.02) start
getting real scores. V(prompt) is still based on 0.02 → massive advantage.
Fix: when tier advances, reset V for all prompts in the newly-ungated tier.
```python
def on_tier_advance(self, new_tier, prompt_tier_map):
    """Reset value tracker for newly-ungated prompts."""
    for pid, tier in prompt_tier_map.items():
        if tier == new_tier:
            self.V[pid] = defaultdict(float)  # reset all components
```

**RESOLVED: filter_groups incompatible with SPO (n=1)**
SPO doesn't need filter_groups — there are no groups to have zero variance.
The equivalent function for SPO: skip prompts where |reward - V(prompt)| < epsilon
(advantage too small to provide useful gradient). This replaces zero-variance
filtering with low-advantage filtering.
```python
if spo_mode:
    # Replace filter_groups with low-advantage filter
    useful = (advantages.abs() > 0.01).any(dim=-1)  # per-sequence
    if useful.sum() < min_batch:
        pass  # use all — don't filter below minimum
    else:
        batch = batch[useful]
```

**RESOLVED: Equivalence test design**
Can't compare reward curves with non-deterministic generation. Instead:
1. Generate 50 steps with verl, save completions + rewards to JSONL
2. Feed SAME completions + rewards to custom engine's advantage + loss computation
3. Compare: advantages within 1%, loss within 1%
4. This tests the ALGORITHM layer, not generation or reward (which are already shared)

**RESOLVED: n=1 vs n=8 mode switch**
QGRETrainer supports both via config:
```yaml
algorithm:
  mode: spo          # or "grpo"
  spo:
    lr: 0.1          # EMA learning rate
    n: 1             # completions per prompt
  grpo:
    n: 8             # completions per prompt
    filter_groups: true
```
Both paths share: generation, reward, loss, backward. They differ ONLY in
advantage computation (SPO tracker vs group normalize).
- **F1 (No Ray) at near-equilibrium** — barely positive Net(+0.006). The training loop is the LAST thing to assemble, not the first.

---

### Exa Research Findings (2026-03-18)

Live search across GitHub issues, Unsloth docs, NeMo RL docs, verl PRs, and RL training guides.

**CRITICAL: LoRA sync is NOT "Low risk" — reclassified to High**

1. **ms-swift #8233 (2026-03-06):** `vllm_enable_lora` causes LoRA double-counting after
   `resume_from_checkpoint` in colocate mode. First rollout batch generates correctly, ALL
   subsequent batches produce malformed text. Root cause: LoRA weights applied twice after
   resume. Fix: recreate vLLM engine on resume, not just reload weights.
   - **Impact on QGRE:** Checkpoint resume (Step 0f) MUST recreate vLLM engine, not just
     restore model state. Without this, resume silently produces garbage.

2. **unsloth #3802 (2025-12-29):** `fast_generate()` ignores LoRA on some models (Gemma 3).
   `model.generate()` (HF/torch) uses LoRA correctly, `model.fast_generate()` (vLLM) behaves
   as base model. Was fixed for Gemma but the pattern may recur with other models.
   - **Impact on QGRE:** Must verify LoRA is active after every `save_lora`/`load_lora` cycle.
     Add a post-sync sanity check: generate a known prompt, verify output differs from base.

3. **unsloth #3864 (2026-01-07):** GRPO Training VRAM usage increases with each step → OOM.
   Confirmed on RTX Quadro 8000 (48GB). The vLLM engine leaks memory during LoRA hot-swap.
   - **Impact on QGRE:** Add periodic vLLM engine recreation (every 50-100 steps) to trainer
     loop. Budget ~5s per recreation. Include in Step 1 (QGRETrainer).

4. **ms-swift #3515 (2025-03-15):** Fix for vLLM 0.7.3 memory leak when n>1. Patched in
   ms-swift but may not be in upstream vLLM. Check vLLM version compatibility.

**SPO source status (confirmed)**

- verl PR #3503 by @dzh19990407: CLOSED (2025-10-15), never merged. 2329 additions, 6 files.
- Source branch: `feature/spo-implementation`. Author's fork: not publicly linked.
- Paper PDF available at: https://zhongwen.one/pdfs/SPO.pdf
- Notion page: https://zhongwenxu.notion.site/
- **Decision:** Implement SPO from paper spec (Algorithm 1 in paper). The core is ~50 lines:
  persistent EMA value tracker V(x) += lr * (r - V(x)), global batch normalization, KL-adaptive
  learning rate. Do NOT depend on the closed PR.

**NeMo RL extraction details (confirmed)**

- Source code for `loss_functions.py` viewable at:
  https://docs.nvidia.com/nemo/rl/0.2.1/_modules/nemo_rl/algorithms/loss_functions.html
- `ClippedPGLossFn` requires `ClippedPGLossConfig` TypedDict with fields:
  reference_policy_kl_penalty, ratio_clip_min, ratio_clip_max, ratio_clip_c,
  use_on_policy_kl_approximation, use_importance_sampling_correction, token_level_loss
- Dependencies to strip: `nemo_rl.algorithms.interfaces.LossFunction`,
  `nemo_rl.algorithms.utils.{calculate_kl_penalty_joschu2020, masked_mean}`,
  `nemo_rl.distributed.batched_data_dict.BatchedDataDict`,
  `nemo_rl.models.dtensor.parallelize.get_logprobs_from_vocab_parallel_logits`
- **API changed in nightly:** `nemo_rl.algorithms.loss_functions` → `nemo_rl.algorithms.loss.loss_functions`.
  Pin extraction to v0.5.0 tag for stability.
- NeMo RL loss design note: microbatch normalization requires global token count passed
  explicitly. Their `masked_mean` handles this. Extract `masked_mean` too.

**Sampling temperature (confirmed)**

- DeepSeek-R1 GRPO: temperature=1.0, num_generations=16, max_length=32768
- DAPO (verl best practices): temperature=1.0 standard for exploration
- TRL GRPOConfig: temperature is a first-class parameter
- vLLM SamplingParams default: temperature=1.0 (offline), 0.7 (API server — different!)
- **Decision:** temperature=1.0 in config. For SPO with n=1, temperature=1.0 is even more
  critical — there's no within-group diversity, so each completion must be independently diverse.
  Added `generation` section to config.yaml.

**Unsloth LoRA hot-swapping pattern (confirmed)**

- Unsloth docs: `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True` for dynamic LoRA loading
- Standard pattern: `model.save_lora("path")` → reload via vLLM adapter mechanism
- Known issue: vLLM LoRA loading can silently succeed even with invalid path (vllm #18372)
- **Decision:** After each `save_lora`/`load_lora`, verify by:
  (a) checking file exists and has expected size,
  (b) optionally generating 1 token from a fixed prompt and comparing to expected range

---

## Phase 1.5: Step-Level Process Rewards (HIGH impact)

**See: SPECIAL-TOKENS-SUPERPOWER.md for full details + CPA results + research validation.**

### SHIP: Step-level process rewards (VPRM-style)
Place rewards at each `</stepN>` boundary, not just at the last token.
The model gets credit for correct step 1 even when step 4 fails.
Fixes the credit assignment problem where partial success gets zero gradient.

**CRITICAL: Step tags are multi-token sequences (5-8 tokens), NOT single tokens.**
Implementation requires decoded-text regex + char→token position mapping.
`<think>`/`</think>` ARE single special tokens (151667/151668).
**Revised effort: 4 hours (was 2). Impact: HIGH.**

**SPO interaction (from λ-GRPO research, arxiv 2509.21154):**
Sullivan proves GRPO already induces implicit process rewards through within-group
token overlap. BUT: SPO uses n=1 (no groups) → implicit PRM disappears entirely.
**With SPO as our primary algorithm, explicit VPRMs are ESSENTIAL, not optional.**

### HOLD: Entropy-weighted credit assignment (GTPO-style)
GTPO validated: ICML accepted (ByteDance), SOTA on AIME + MATH 500.
GitHub: winstonsmith1897/GTPO (MIT). Confirmed working in practice.
**Blocked by:** entropy computation disabled on 16GB (entropy_coeff=0.0).
**Resolves on:** H100 80GB. Not a design problem — hardware constraint.
**Effort: 3 hours once entropy available. Impact: MEDIUM.**

### HOLD: Region-specific KL control (THR-style)
Think blocks: LOW KL (explore) — detection trivial (single special tokens).
Format/content tokens: HIGH/MEDIUM KL — detection shares multi-token complexity with VPRMs.
**Blocked by:** custom QGRE engine (keystone C1). CPA shows strong recovery (Net=+0.014) once engine exists.
**Effort: 4 hours once engine exists. Impact: MEDIUM. Defer to Phase 2.**

### KILL: Difficulty scaling (F-GRPO)
SPO's persistent value tracker handles difficulty naturally. GameState already
tracks per-archetype success rates. Building F-GRPO is redundant.
**Do not implement.**

### Paper contribution (updated framing)
**"Verifiable Process Rewards for Novel-Domain Structured Reasoning"**
— VPRMs with programmatic verifiers in a domain where no neural PRM exists.
Novel enough alone. Add GTPO as "preliminary results" once H100 data available.
Do NOT force the 4-technique monolith — it's a genuine void under pressure.

---

## Phase 2: Triton kernels for reward computation (medium impact)

### Current bottleneck

The reward function runs on CPU in Python. For each completion:
- XML tag parsing (regex)
- JSON extraction from step4
- Node matching against ground truth
- F1 computation
- GameState updates

With 8 completions per step at ~2048 tokens each, this is ~16K tokens of string processing in Python.

### What to move to GPU

- **Format tag detection** — scan for `<step1_extraction>`, `</step1_extraction>`, etc.
  This is a pattern match on token IDs, not strings. Can be a Triton kernel that scans
  the output token buffer directly.
- **JSON extraction from step4** — locate `<step4_output>` tags, extract content between them,
  validate JSON structure. Token-level operation.
- **Node matching** — check if extracted node names appear in the prompt token buffer.
  Token subsequence matching on GPU.

### What stays on CPU

- GameState updates (stateful, branching logic)
- MLflow logging
- Curriculum tier advancement
- Archetype-specific Elo tracking

### Estimated impact

- Reward computation: ~10ms → ~1ms (10x, but small fraction of total step time)
- Main benefit: enables scaling to larger batch sizes without reward becoming bottleneck

---

## Phase 3: torch.compile the training loop (medium impact)

### What torch.compile does

Traces the forward + backward computation graph and generates fused CUDA kernels.
Eliminates Python overhead between operations, fuses small operations into larger kernels,
reduces GPU kernel launch overhead.

### What can be compiled

- The forward pass through the model (already partially compiled by Unsloth)
- `logprobs_from_logits` computation
- GRPO advantage computation
- Policy loss computation

### What cannot be compiled

- vLLM generate (has its own compilation)
- Reward function (Python control flow)
- LoRA weight sync
- MLflow logging

### Estimated impact

- Training portion of step: ~15s → ~10s (30% faster)
- Combined with Phase 1: total step ~15s

---

## Phase 4: Custom CUDA kernels for critical path (low impact, high effort)

### What this means

Replace the remaining Python/PyTorch operations with hand-written CUDA/Triton kernels:
- Fused log_softmax + gather + advantage_weighted_loss (single kernel)
- Fused attention with LoRA adapters (Unsloth already does this)
- Fused KV cache management for generation

### Why probably not worth it

- Unsloth already provides optimized backward kernels
- vLLM already provides optimized generation kernels
- The remaining Python overhead after Phase 1-3 is ~2-3s/step
- Custom CUDA kernels are hard to maintain across GPU architectures

---

## Priority Order (UPDATED 2026-03-18 — Phase 4 is NOT "probably never")

| Phase | Impact | Effort | When | Status |
|-------|--------|--------|------|--------|
| 1. Eliminate Ray + SPO algorithm | 30% faster + better convergence | 2 weeks | Now | **DONE** |
| 1.5 VPRM step-level rewards | Credit assignment fix | 4 hours | Phase 1 | **DONE** (in advantage estimator) |
| 2. Triton reward | 10x reward speed | 1 week | After Phase 1 | NEXT |
| 3. torch.compile | 30% faster training | 3 days | After Phase 1 | BLOCKED (Unsloth compat TBD) |
| 4. Custom CUDA/Triton | Enables 16+ batch × 4096 tokens | 1 week | After Phase 2 | **REQUIRED** (not optional) |

### Phase 4 reassessment (2026-03-18)

The original plan said Phase 4 is "probably never worth it." This was wrong.

Evidence: 8×4096 stress test OOM'd at gpu_memory_utilization=0.6 because
lm_head output = [4096, 151936] = 1.17GB per sequence in bf16. Fixed by
lowering to 0.35, but this trades KV cache capacity for training headroom.

With Triton 3.5.1 and Liger Kernel 0.7.0 now available, Phase 4 is achievable
and would eliminate the VRAM tradeoff entirely. Four QGRE-specific Triton kernels
identified (none exist anywhere else):

1. **Fused segment_completion** — parallel token ID pattern matching on GPU
2. **Fused advantage_broadcast** — region→advantage lookup, parallel
3. **Fused LLDS gate** — three-level gate (traj+token+action) in one kernel
4. **Fused region-KL-weighted loss** — region lookup + KL scaling + masking

These are custom to QGRE — no library provides them.

## Target performance

| Metric | Current (verl+Ray+GRPO) | Phase 1 (QGRETrainer+SPO) | Phase 1+3 |
|--------|------------------------|--------------------------|-----------|
| Step time | 30s | ~20s | ~15s |
| Startup | ~45s | ~15s | ~15s |
| Memory overhead | ~500MB (Ray) | 0 | 0 |
| Code complexity | ~5000 lines | ~500 lines | ~600 lines |
| Debugging | Ray remote traces | Direct stack traces | Direct stack traces |
| Degenerate groups | ~30% wasted | 0% (no groups) | 0% |
| Prompts per step | 2 (×n=4) | 8 (×n=1) | 8 |
| Advantage estimation | Group-mean (noisy at n=4) | Persistent tracker (stable) | Persistent tracker |
| Multi-reward handling | Sum → normalize (crowding) | Per-component tracking (no crowding) | Per-component |

---

---

## NeMo RL Extraction Checklist

```
□ 1. Clone NeMo RL: git clone https://github.com/NVIDIA-NeMo/RL.git
□ 2. Extract to src/qgre-engine/nemo_extracted/:
     - algorithms/advantage_estimator.py (222 lines)
     - algorithms/loss/loss_functions.py (~200 lines)
     - algorithms/utils.py (845 lines)
     - algorithms/logits_sampling_utils.py (285 lines)
□ 3. Strip Ray/Megatron imports (utils.py has some — replace with direct torch calls)
□ 4. Write QGREAdvantageEstimator that wraps GDPOAdvantageEstimator:
     - Maps QGRE phase-active rewards to GDPO component rewards
     - Only passes currently-active phase rewards (not gated ones)
     - Passes through to GDPO for per-component normalization
□ 5. Write adapter: reward_fn dict output → GDPO component tensor format
□ 6. Benchmark: GRPO vs GDPO advantages on same data, measure crowding in Phase 3+
□ 7. Preserve Apache-2.0 headers on all extracted files
```

---

---

## Build Status (2026-03-18, session progress)

### COMPLETED (77 tests passing)

| Step | Module | Status | Tests |
|------|--------|--------|-------|
| 0a | checkpoint.py (GameState serializer) | DONE | 5 |
| 0b | nemo_extracted/*.py (loss, KL, logits) | DONE | 10 |
| 0c | advantages.py (batch reward tensors) | DONE | 3 |
| 0d | segments.py + advantages.py (CORE ALGORITHM) | DONE | 16 |
| 0e | data.py (DataLoader) | DONE | 9 |
| 0f | checkpoint.py (save/resume/discover) | DONE | 6 |
| 0g | lora_verify.py (LoRA verification) | DONE | 7 |
| 1 | config.py + trainer.py (QGRETrainer) | DONE | 7 |
| 4 | logging.py (MLflow tracking) | DONE | 2 |
| 5 | checkpoint wiring in trainer | DONE | 2 |
| 6 | logging.py (JSONL dump) | DONE | 3 |
| 7 | test_equivalence.py (algorithm consistency) | DONE | 4 |
| 8 | test_advantages.py (credit assignment) | DONE | 2 (in advantages tests) |

### REMAINING — GPU wiring + smoke tests

```
Step 2: Generation backend — Unsloth model loading + vLLM fast_generate
  - Write qgre/generation.py: UnslothBackend implementing GenerationBackend protocol
  - Clean rewrite — do NOT copy patterns from training-dojo v1
  - Smoke test model: unsloth/Qwen3-1.7B-unsloth-bnb-4bit (4-bit, ~1.5GB)
  - RTX 5080 16GB — 1.7B fits comfortably with vLLM + training
  - Temperature: 1.0 (standard for GRPO diversity)
  - Stub-only generation for smoke tests — NOT full training
  - Test: model loads, generates tokens, LoRA sync works

Step 3: Reward function adapter
  - Write adapter that imports reward_fn.py from training-dojo (symlink or copy)
  - For smoke tests: use stub reward_fn already in examples/hypergraph/
  - For real training (hypergraph-scan-v2): use real reward_fn.py from training-dojo
  - Test: stub reward_fn returns correct RewardResult type

GPU Smoke Tests (tests/test_smoke.py):
  - test_model_loads: Unsloth model loads on RTX 5080
  - test_generate_tokens: fast_generate produces non-empty output
  - test_lora_sync: save_lora → load_lora → output differs from base
  - test_three_steps_no_crash: 3 full training steps, no nan, no OOM
  - test_vram_no_growth: 10 steps, memory growth < 10%

GPU Wiring Tests (tests/test_wiring.py):
  - test_generation_backend_protocol: UnslothBackend satisfies GenerationBackend
  - test_reward_result_from_stub: stub reward_fn → RewardResult type check
  - test_full_step_with_generation: generate → score → advantages → loss → backward

Scope boundary:
  - We are NOT doing full training runs in this repo
  - We ARE verifying the engine works end-to-end on 3-5 steps
  - Real training (hypergraph-scan-v2) happens in training-dojo
  - This engine must be pip-installable and importable from training-dojo
```

---

## Reference Implementations (all Apache-2.0 or MIT)

### Primary references for our engine

| Repo | Stars | Lines | What to take | License |
|------|-------|-------|-------------|---------|
| **NeMo RL** (NVIDIA) | 1400 | ~1550 extractable | GDPO advantage, clipped PG loss, KL utils | Apache-2.0 |
| **simple_GRPO** (lsdefine) | 1600 | ~200 | Training loop structure, vLLM integration pattern | Apache-2.0 |
| **policy-gradients** (zafstojano) | 13 | ~500 | GRPO + PPO + REINFORCE in clean PyTorch, single GPU | Apache-2.0 |
| **NVlabs/GDPO** (NVIDIA) | - | - | Official GDPO implementation (HF-TRL, verl, NeMo RL) | Apache-2.0 |

### Checkpoint resume pattern (from PyTorch docs + community)
```python
# Standard pattern — works for all our needs
checkpoint = {
    'step': global_step,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'game_state': game_state.__dict__,      # QGRE-specific: archetype phases, elo, windows
    'curriculum_tier': game_state.max_active_tier,
    'rng_state': torch.get_rng_state(),
    'cuda_rng_state': torch.cuda.get_rng_state(),
}
torch.save(checkpoint, path)

# Resume
checkpoint = torch.load(path)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
# ... restore GameState, RNG, curriculum
```

### GDPO integration pattern (from paper + NVlabs code)
```python
# GDPO: normalize each reward component separately, then sum
# Standard GRPO (broken for multi-reward):
#   advantages = normalize(r1 + r2 + r3)  # collapse: (0,2) == (1,1)
#
# GDPO (correct):
#   advantages = normalize(r1) + normalize(r2) + normalize(r3)
#   Each component preserves its own signal strength
#
# QGRE adapter requirement:
#   Only pass phase-ACTIVE rewards to GDPO
#   Gated rewards (0.02) must be EXCLUDED, not normalized
```

### Key quotes from research

> "Most production-grade RL libraries are hard to debug, as they use vLLM for inference,
> FSDP for training, and Ray to handle the distributed communication. Good luck
> navigating through that." — zafstojano/policy-gradients README

> "Directly applying GRPO to normalize distinct rollout reward combinations causes them
> to collapse into identical advantage values, reducing the resolution of the training
> signal." — GDPO paper (NVIDIA, arxiv.org/abs/2601.05242)

> "Training completed in under 1 hour on 1×A800 GPUs. Both Qwen2.5-7B and Qwen2.5-3B
> exhibited an 'Aha moment' within the first 30 optimization steps." — simple_GRPO README

> "VPRMs achieve up to 20% higher F1 than state-of-the-art models and 6.5% higher than
> verifiable outcome rewards." — VPRMs paper (IBM Research, arxiv.org/abs/2601.17223)

> "We prove theoretically that the GRPO RL algorithm induces a non-trivial process
> reward model (PRM)." — "GRPO is Secretly a PRM" (Sullivan, arxiv.org/abs/2509.21154)

### Full references

- **SPO** (Tencent): arxiv.org/abs/2509.13232 (ICLR 2026, peer-reviewed)
  - verl PR #3503 (closed, code at author's fork): 2329 lines, core SPO ~100 lines
  - Key result: +3.4 pp over GRPO on Qwen3-8B across 5 math benchmarks
  - Notion page: zhongwenxu.notion.site (author's project page)
- **NeMo RL** (NVIDIA): github.com/NVIDIA-NeMo/RL (Apache-2.0, 1400 stars)
  - Extractable: `algorithms/advantage_estimator.py`, `algorithms/loss/loss_functions.py`, `algorithms/utils.py`
  - GDPO paper: arxiv.org/abs/2601.05242 (Shih-Yang Liu et al., NVIDIA, Jan 2026)
  - Official GDPO code: github.com/NVlabs/GDPO (implementations for HF-TRL, verl, NeMo RL)
- **simple_GRPO**: github.com/lsdefine/simple_GRPO (1600 stars, ~200 lines, evolved into LSRL)
- **policy-gradients**: github.com/zafstojano/policy-gradients (GRPO+PPO+REINFORCE, single GPU, educational)
- **tiny-grpo** (open-thought): github.com/open-thought/tiny-grpo (331 stars, minimal hackable)
- **simple-grpo** (davidheineman): github.com/davidheineman/simple-grpo (uses nano-vllm)
- TRL colocated vLLM: huggingface.co/blog/vllm-colocate (IBM, 3x faster, merged PR #3394)
- vLLM RLHF colocate: vllm docs, examples/offline_inference/rlhf_colocate.py
- GRPO++: cameronrwolfe.substack.com/p/grpo-tricks (comprehensive tricks guide)
- RLHFless: arxiv.org/abs/2602.22718 (serverless RLHF, identifies overhead problem)
- Hands-on GRPO from scratch: medium.com/@baicenxiao (2-GPU vLLM + PyTorch, step-by-step)
- PyTorch checkpoint docs: docs.pytorch.org/tutorials/recipes/saving_and_loading_a_general_checkpoint.html
- **VPRMs** (IBM Research): arxiv.org/abs/2601.17223 — verifiable process rewards with rule-based verifiers (Jan 2026)
- **"GRPO is Secretly a PRM"** (Sullivan): arxiv.org/abs/2509.21154 — proves GRPO induces implicit PRM (ICLR 2026 under review)
- **GTPO** (ByteDance): arxiv.org/abs/2508.04349 — entropy-weighted token rewards (ICML accepted). Code: github.com/winstonsmith1897/GTPO
- **PRPO** (Shanghai University): arxiv.org/abs/2601.07182 — process + outcome reward alignment (Feb 2026)
- **StepGRPO**: emergentmind.com/topics/step-wise-group-relative-policy-optimization-stepgrpo — per-step normalization
- **λ-GRPO (learnable)**: arxiv.org/abs/2510.06870 — learnable token preferences (+1.9% on Qwen2.5-1.5B). ICLR 2026, withdrawn.

---

## Verifiable Tests Per Deliverable

Every build step has CPU-runnable tests that verify correctness WITHOUT requiring a GPU
or a full training run. GPU-required tests are marked explicitly. Each test file maps to
one build step. All tests use `pytest` and synthetic data.

### Step 0a: GameState Serializer (`tests/test_checkpoint.py`)

```python
def test_gamestate_roundtrip():
    """Create GameState with deque, defaultdict, nested dicts.
    to_dict() → from_dict() → assert all fields equal."""

def test_gamestate_json_serializable():
    """to_dict() output passes json.dumps without error."""

def test_gamestate_preserves_deque_maxlen():
    """After round-trip, deque.maxlen matches original."""

def test_gamestate_preserves_defaultdict_behavior():
    """After from_dict(), accessing missing key returns default, not KeyError."""
```

### Step 0b: NeMo RL Extraction (`tests/test_nemo_extracted.py`)

```python
def test_import_loss_functions():
    """import qgre.nemo_extracted.loss_functions succeeds with no external deps."""

def test_import_kl():
    """import qgre.nemo_extracted.kl succeeds."""

def test_import_logits():
    """import qgre.nemo_extracted.logits succeeds."""

def test_clipped_pg_loss_nonzero():
    """ClippedPGLossFn on synthetic log_probs + advantages → non-zero, finite loss."""
    # Synthetic: 4 sequences × 32 tokens, random log_probs, random advantages
    # Assert: loss.isfinite(), loss != 0

def test_clipped_pg_loss_clip_bounds():
    """Ratio outside [1-eps_low, 1+eps_high] → loss is clipped, not raw."""
    # Create old_lp and new_lp with large ratio (>2.0)
    # Assert: loss matches clipped version, not unclipped

def test_kl_calculation_matches_manual():
    """KL divergence on small tensors matches manual computation."""
    # 2 distributions, 4 tokens, compute by hand, assert allclose

def test_masked_mean_correctness():
    """masked_mean with known mask → matches manual mean over unmasked tokens."""
```

### Step 0c: Batch Reward Tensor Construction (`tests/test_advantages.py` — partial)

```python
def test_batch_reward_tensors_shape():
    """list[dict] with 4 items → dict[str, Tensor] with shape [4] per component."""

def test_batch_reward_tensors_missing_keys():
    """Dicts with different key sets → missing keys zero-filled."""

def test_batch_reward_tensors_empty():
    """Empty list → empty dict or dict with zero-length tensors."""
```

### Step 0d: QGREStepAdvantageEstimator (`tests/test_advantages.py` — main, `tests/test_segments.py`)

**Segmentation tests:**
```python
def test_segment_known_sequence():
    """Hand-crafted token IDs with known step boundaries → correct region labels."""
    # Construct: [THINK_START, ..., THINK_END, OPEN_ANGLE, STEP_TOKEN, 16, ..., 29, content, CLOSE_SLASH, STEP_TOKEN, 16, ..., 29]
    # Assert: regions = ["THINK", ..., "THINK", "FORMAT", ..., "FORMAT", "STEP_1", ..., "FORMAT", ...]

def test_segment_no_think_block():
    """nothink template: no THINK_START/END → no THINK regions, only STEP/FORMAT/OTHER."""

def test_segment_malformed_tags():
    """Missing closing tag → region extends to end as current type (graceful, not crash)."""

def test_segment_all_four_steps():
    """Full completion with steps 1-4 → exactly 4 STEP regions + FORMAT regions for tags."""
```

**Credit assignment tests (the crown tests):**
```python
def test_credit_step1_correct_step4_wrong():
    """Step 1 scores=[1.0, 1.0], step 4 scores=[0.0, 0.0].
    Assert: step 1 advantage > 0, step 4 advantage < 0."""

def test_credit_all_steps_correct():
    """All steps score 1.0.
    Assert: all step advantages similar magnitude (within 0.1 of each other)."""

def test_credit_phase1_format_only():
    """active_qualities = phase 1 qualities only.
    Assert: only step 1 has non-zero advantage (format qualities live in step 1)."""

def test_credit_phase4_all_active():
    """active_qualities = all qualities.
    Assert: all steps have non-zero advantages."""
```

**SPO value tracker tests:**
```python
def test_spo_warmstart_no_spike():
    """First observation for a prompt: V is set to reward value, advantage ≈ 0.
    Assert: |advantage| < 0.01 on first observation."""

def test_spo_second_observation_has_advantage():
    """Second observation with different reward → advantage = reward - V(old).
    Assert: advantage ≈ new_reward - old_reward (within EMA tolerance)."""

def test_spo_on_tier_advance_resets_v():
    """Call on_tier_advance(new_tier=2, prompt_map).
    Assert: V[affected_prompt] is reset to defaultdict(float)."""

def test_spo_value_tracker_ema_convergence():
    """Feed 100 identical rewards → V converges to reward value.
    Assert: |V - reward| < 0.01 after 100 updates."""
```

**GRPO fallback tests:**
```python
def test_grpo_fallback_group_normalize():
    """4 completions per prompt, different rewards.
    Assert: advantages have mean≈0 and std≈1 within group."""

def test_grpo_fallback_degenerate_group():
    """4 completions with identical rewards → all advantages ≈ 0.
    Assert: no nan, no inf, all advantages < 0.01."""
```

**GDPO normalization tests:**
```python
def test_gdpo_per_step_normalize():
    """8 completions, step 1 has high variance, step 4 has low variance.
    After per-step normalization, both steps have std≈1.
    Assert: step_1_advs.std() ≈ 1.0, step_4_advs.std() ≈ 1.0."""

def test_gdpo_preserves_sign():
    """Completion with step 1 above batch mean → step 1 advantage > 0.
    Completion with step 1 below batch mean → step 1 advantage < 0."""
```

### Step 0e: DataLoader (`tests/test_data.py`)

```python
def test_load_parquet():
    """Load small test parquet (5 rows) → 5 prompts returned."""

def test_overlong_filter():
    """One prompt exceeds max_prompt_length after tokenization → filtered out."""

def test_prompt_expansion():
    """5 prompts × n=4 → 20 items in expanded batch."""

def test_shuffle_different_epochs():
    """Epoch 1 and epoch 2 with different seeds → different order."""

def test_batch_assembly():
    """10 prompts, batch_size=4 → 3 batches (4, 4, 2)."""

def test_epoch_tracker_counts():
    """After iterating 10 steps → epoch_tracker.total_steps == 10."""
```

### Step 0f: Checkpoint Resume (`tests/test_checkpoint.py` — continued)

```python
def test_checkpoint_save_load_roundtrip():
    """Save full state dict → load → all fields match."""
    # Includes: model_state_dict (mock), optimizer_state_dict (mock),
    # game_state, global_step, rng_state

def test_checkpoint_discovery_finds_latest():
    """Create dir with global_step_10, global_step_50, global_step_30.
    Assert: discover_latest_checkpoint() returns global_step_50."""

def test_checkpoint_rng_state_restored():
    """Save RNG → generate 5 random numbers → restore RNG → generate 5 again.
    Assert: both sequences identical."""

def test_checkpoint_includes_advantage_estimator_state():
    """V tracker and _step_seen persist through save/load.
    Assert: V[prompt_id][step_num] matches after round-trip."""
```

### Step 1: QGRETrainer (`tests/test_trainer.py`)

```python
def test_trainer_forward_finite_loss():
    """Synthetic batch through trainer._compute_loss() → loss is finite, non-zero.
    Uses mock model that returns random logits."""

def test_gradient_accumulation_equivalence():
    """1 batch of size 8 vs 2 batches of size 4 with accumulation.
    Assert: final loss within 1e-5."""

def test_response_mask_masks_padding():
    """Sequence with PAD tokens after EOS → mask is 0 for PAD positions."""

def test_on_policy_mode():
    """old_log_probs == log_probs.detach().
    Assert: torch.equal(old_lp, lp.detach())."""

def test_mode_switch_spo_vs_grpo():
    """Config mode='spo' → uses SPO estimator. mode='grpo' → uses GRPO estimator.
    Assert: different advantage values for same input."""
```

### Steps 2-3: Generation + Reward Wiring (`tests/test_wiring.py`) — GPU required

```python
@pytest.mark.gpu
def test_model_loads():
    """Unsloth model loads without error on available GPU."""

@pytest.mark.gpu
def test_fast_generate_produces_tokens():
    """fast_generate with 1 prompt → output has >0 tokens."""

@pytest.mark.gpu
def test_lora_sync_changes_output():
    """Generate with base → train 1 step → save_lora → load_lora → generate again.
    Assert: outputs differ (LoRA is actually applied)."""

def test_reward_result_type():
    """RewardResult has .scores (dict), .phase (int), .reward (float)."""

def test_reward_fn_returns_correct_type():
    """Mock reward_fn returns RewardResult with expected keys."""
```

### Step 4: MLflow Logging (`tests/test_logging.py`)

```python
def test_mlflow_metrics_logged(mock_mlflow):
    """After 1 trainer step, mlflow.log_metrics called with expected metric names.
    Assert: 'reward/mean', 'reward/step_1', 'advantage/step_1' in logged keys."""

def test_per_step_metrics_present(mock_mlflow):
    """Metrics include step_1 through step_4 for both reward and advantage."""
```

### Step 5: Checkpoint Wiring (`tests/test_checkpoint.py` — continued)

```python
def test_train_save_load_step_counter():
    """Train 3 steps → save → new trainer → load → step counter == 3."""

def test_v_tracker_persists_across_checkpoint():
    """SPO V tracker values present after save/load cycle."""
```

### Step 6: JSONL Dump (`tests/test_logging.py` — continued)

```python
def test_completion_jsonl_valid():
    """Written line parses as valid JSON."""

def test_completion_jsonl_fields():
    """JSON contains: input, output, score, reward_components, step."""
```

### Step 7: Equivalence Test (`tests/test_equivalence.py`)

```python
def test_advantages_match_reference():
    """Fixed completions + rewards from verl JSONL → advantages within 1% of reference.
    Load reference data, run through QGREStepAdvantageEstimator, compare."""

def test_loss_matches_reference():
    """Same fixed data → loss within 1% of reference.
    Tests algorithm layer only, not generation or reward."""
```

### Step 8: Credit Assignment Integration (`tests/test_advantages.py` — integration)

```python
def test_full_pipeline_credit_assignment():
    """Realistic synthetic data: 8 completions with varied per-step quality.
    Run through full pipeline: segment → step_rewards → SPO → GDPO → broadcast.
    Assert: per-token advantages reflect per-step quality, not sequence-level."""

def test_think_tokens_get_zero_advantage():
    """Tokens in THINK region → advantage == 0."""

def test_format_tokens_get_zero_advantage():
    """Tokens in FORMAT region (tag tokens) → advantage == 0."""
```

### GPU Smoke Test (`tests/test_smoke.py`) — THE MISSING TEST

```python
@pytest.mark.gpu
@pytest.mark.slow
def test_three_steps_no_crash():
    """Load model, run 3 full training steps on GPU.
    Assert: no crash, no nan loss, no OOM, final loss < initial loss * 2."""

@pytest.mark.gpu
def test_lora_sync_verification():
    """After optimizer step + save_lora + load_lora:
    Hash LoRA weight tensors → generate → verify output is not base-model-like.
    This catches the ms-swift #8233 failure mode."""

@pytest.mark.gpu
def test_vram_does_not_grow():
    """Run 10 steps, measure GPU memory at step 1, 5, 10.
    Assert: memory_step_10 < memory_step_1 * 1.1 (less than 10% growth).
    This catches the unsloth #3864 leak."""
```

### Test Infrastructure

```
tests/
  conftest.py          — fixtures: mock_model, mock_tokenizer, synthetic_batch,
                         synthetic_reward_results, known_token_sequences
  test_checkpoint.py   — Steps 0a, 0f, 5
  test_nemo_extracted.py — Step 0b
  test_segments.py     — Step 0d (segmentation only)
  test_advantages.py   — Steps 0c, 0d, 8 (advantage computation)
  test_data.py         — Step 0e
  test_trainer.py      — Step 1
  test_wiring.py       — Steps 2-3 (GPU required)
  test_logging.py      — Steps 4, 6
  test_equivalence.py  — Step 7
  test_smoke.py        — GPU smoke test (slow)
```

**Run order:**
```bash
# CPU tests — run after each step, takes seconds
pytest tests/ -k "not gpu and not slow"

# GPU tests — run after Step 2 wiring complete
pytest tests/ -m gpu

# Full suite including smoke test
pytest tests/ -m "gpu and slow"
```

**Test data fixtures (in conftest.py):**
- `known_token_ids`: Hand-crafted sequences with known step boundaries using real Qwen3 token IDs
- `synthetic_reward_results`: List of RewardResult with controlled per-quality scores
- `synthetic_batch`: Prompt IDs, token IDs, reward results, active qualities — all synthetic
- `mock_game_state`: GameState with non-trivial state (non-empty deque, nested dicts)
- `small_parquet`: 5-row parquet file written to tmp dir for DataLoader tests

# QGRE Engine

Single-GPU GRPO training engine for novel-domain structured reasoning.

No Ray. No verl. No TRL. Just: generate → score → advantages → loss → backward → update.

## What is QGRE?

**Quality-Gated Reward Escalation** is a phase-gated curriculum for training LLMs on novel domains where correct reasoning methodology is unknown. Instead of rewarding all qualities from step 1, QGRE unlocks reward components progressively: format first, then grounding, then chain coherence, then accuracy.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     QGRETrainer.step()                          │
│                                                                 │
│  ┌──────────┐   ┌──────────┐   ┌──────────────┐   ┌─────────┐ │
│  │ Generate  │──▸│  Score   │──▸│  Advantages  │──▸│  Loss   │ │
│  │ (vLLM)   │   │(reward_fn)│  │  (SPO+GDPO   │   │(NeMo RL)│ │
│  │          │   │          │   │  +VPRM+phase) │   │         │ │
│  └──────────┘   └──────────┘   └──────────────┘   └────┬────┘ │
│       ▲                                                 │      │
│       │              ┌──────────────┐                   ▼      │
│       └──────────────│ LoRA Sync    │◂── backward + optimizer  │
│                      │ (save/load)  │         step             │
│                      └──────────────┘                          │
└─────────────────────────────────────────────────────────────────┘
  One process. One GPU. Direct function calls.
```

### What each layer does

| Layer | Module | Responsibility |
|-------|--------|---------------|
| **Model + LoRA** | `generation.py` (Unsloth) | Loads QLoRA model (4-bit base + full-precision adapters). Switches between `for_training()` and `for_inference()` mode. Manages vLLM engine lifecycle. |
| **Generation** | `generation.py` (vLLM) | `fast_generate()` produces completions via in-process vLLM. Decodes prompts, returns token IDs + text. Handles SamplingParams (temperature, top_p, stop tokens). |
| **Reward** | Your `reward_fn` | Scores completions → `RewardResult(reward, scores, phase)`. You provide this. The engine consumes `.scores` per quality and `.phase` for curriculum gating. |
| **Segmentation** | `segments.py` | `segment_completion()` splits token IDs into regions: THINK, STEP_1..4, FORMAT, OTHER. Uses Qwen3 token ID patterns (not regex). Assigns each token to its step. |
| **Advantages** | `advantages.py` | `QGREStepAdvantageEstimator` computes per-token advantages. Four techniques unified: SPO baseline, GDPO normalization, VPRM segment propagation, QGRE phase gating. |
| **Loss** | `nemo_extracted/loss_functions.py` | `ClippedPGLossFn` from NeMo RL (Apache-2.0). Clipped PG loss with DAPO-style asymmetric clipping, KL regularization, importance sampling. |
| **Backward** | `trainer.py` (PyTorch) | Standard `loss.backward()` + `optimizer.step()`. NaN guard, gradient clipping, gradient accumulation. Unsloth's `for_training()` mode disables inplace ops for autograd compatibility. |
| **Persistence** | `checkpoint.py`, `logging.py` | Checkpoint save/resume (model, optimizer, GameState, SPO V-tracker, RNG). MLflow metrics. JSONL completion logs. |

## The Six Pillars

### Pillar 1: Data Pipeline (`data.py`)
Loads prompts from parquet or dicts. Tokenizes via chat template, left-pads, filters overlong, shuffles per epoch, batches, expands ×n for rollout.

### Pillar 2: Generation Engine (`generation.py`)
Unsloth `FastLanguageModel` + vLLM `fast_generate`. QLoRA (4-bit quantized base + LoRA adapters). Mode switching: `set_inference_mode()` before generate, `set_training_mode()` before backward.

### Pillar 3: Reward & Curriculum (your `reward_fn` + `types.py`)
You provide the reward function. It returns `RewardResult` with per-quality scores. `GameState` tracks curriculum phase, archetype mastery, Elo ratings, quality windows. Serializes safely (deque → list, defaultdict → dict).

### Pillar 4: Algorithm Layer (`advantages.py` + `segments.py`)
The core innovation. `QGREStepAdvantageEstimator` unifies:
- **SPO**: Persistent EMA value tracker per prompt per step. No groups needed (n=1). Warm-starts on first observation.
- **GDPO**: Per-step batch normalization. Each step's advantages are independently normalized so no step's signal drowns another.
- **VPRM**: Token-level segmentation. Tokens in STEP_1 get step 1's advantage. THINK/FORMAT tokens get zero (no gradient for structure).
- **Phase gating**: Only active qualities contribute. Phase 1 = format only. Phase 4 = all qualities.

### Pillar 5: Training Loop (`trainer.py` + `config.py`)
`QGRETrainer.step()` orchestrates: advantages → pad → forward → response mask → loss → NaN check → backward → clip → optimizer step → log. Config from YAML with validation (warns on unknown keys).

### Pillar 6: Persistence (`checkpoint.py` + `logging.py`)
Full state checkpoint: model, optimizer, scheduler, GameState, SPO V-tracker, RNG state. Auto-discovery of latest checkpoint. MLflow per-step metrics. JSONL completion dumps.

## Install

```bash
pip install -e ".[unsloth]"
```

For development:
```bash
pip install -e ".[dev]"
```

## Quick Start

```python
from qgre import RewardResult, GameState
from qgre.config import QGREConfig
from qgre.generation import UnslothBackend
from qgre.trainer import QGRETrainer
from qgre.data import QGREDataLoader

# 1. Load config
cfg = QGREConfig.from_yaml("examples/hypergraph/config.yaml")

# 2. Load model
backend = UnslothBackend(cfg.model, cfg.generation)
model, tokenizer = backend.load()

# 3. Your reward function
def my_reward_fn(prompt, completion, meta=None):
    # Score the completion, return per-quality scores
    return RewardResult(reward=0.8, scores={"q_format": 1.0, "q_accuracy": 0.6}, phase=1)

# 4. Create trainer
trainer = QGRETrainer(model=model, tokenizer=tokenizer, reward_fn=my_reward_fn,
                      config=cfg, generation_backend=backend)
trainer.setup_optimizer()

# 5. Training loop
loader = QGREDataLoader(prompts=my_prompts, tokenizer=tokenizer,
                        max_prompt_length=cfg.data.max_prompt_length,
                        train_batch_size=cfg.data.train_batch_size)

for epoch in range(10):
    for batch in loader:
        backend.set_inference_mode()
        output = backend.generate(batch.input_ids.cuda(), batch.attention_mask.cuda())

        rewards = [my_reward_fn(batch.raw_prompts[i], output.texts[i])
                   for i in range(len(output.texts))]

        backend.set_training_mode()
        metrics = trainer.step(batch, output.token_ids, rewards)
        print(f"Step {trainer.global_step}: loss={metrics['loss']:.4f}")
```

## Bring Your Own Domain

The engine is domain-agnostic. You provide three things:

1. **A reward function** returning `RewardResult` with per-quality `scores` dict
2. **A step_qualities mapping** — which qualities belong to which step (any number of steps)
3. **A segmenter** (optional) — how to split completions into step regions

The engine handles everything else: generation, advantage computation, loss, optimization, checkpointing, logging.

```python
from qgre.trainer import QGRETrainer
from qgre.segments import qwen3_xml_segmenter, uniform_segmenter

# Example 1: HIF hypergraph scanning (5 steps, XML tags)
hif_qualities = {
    1: ["q_valid_json", "q_hif_schema"],
    2: ["q_node_grounding", "q_node_verbatim"],
    3: ["q_incidence_refs_nodes", "q_internal_consistency"],
    4: ["q_existence_correct", "q_archetype_correct"],
    5: ["q_node_f1", "q_edge_f1"],
}

trainer = QGRETrainer(
    model=model, tokenizer=tokenizer, reward_fn=hif_reward_fn,
    config=cfg, step_qualities=hif_qualities, segmenter=qwen3_xml_segmenter,
)

# Example 2: Math (1 step, uniform advantages — equivalent to standard GRPO)
math_qualities = {1: ["q_correct_answer"]}

trainer = QGRETrainer(
    model=model, tokenizer=tokenizer, reward_fn=math_reward_fn,
    config=cfg, step_qualities=math_qualities, segmenter=uniform_segmenter,
)

# Example 3: Custom segmenter for JSON-structured outputs
def my_json_segmenter(token_ids: list[int]) -> list[str]:
    """Your custom logic to split token IDs into STEP_1, STEP_2, etc."""
    # Return list of region labels, same length as token_ids
    ...

trainer = QGRETrainer(
    model=model, tokenizer=tokenizer, reward_fn=my_reward_fn,
    config=cfg, step_qualities=my_qualities, segmenter=my_json_segmenter,
)
```

Step qualities can also be set in the YAML config:

```yaml
algorithm:
  step_qualities:
    1: [q_valid_json, q_schema]
    2: [q_grounding]
    3: [q_consistency]
    4: [q_accuracy]
    5: [q_f1_score]
```

**Phase gating is automatic.** Phase N includes all qualities from steps 1 through N. Phase 1 = only step 1 qualities. Phase 5 = all qualities. The reward function controls which phase is active via `RewardResult.phase`.

## Critical: Unsloth Mode Switching

Unsloth patches model internals with inplace-optimized kernels for fast inference. These break PyTorch autograd during backward. **You must switch modes:**

```python
backend.set_inference_mode()   # Before fast_generate — enables fast kernels
# ... generate completions ...

backend.set_training_mode()    # Before forward+backward — disables inplace ops
# ... trainer.step() ...
```

Without this, `loss.backward()` crashes with `RuntimeError: variable modified by inplace operation`. This is documented in [Unsloth #895](https://github.com/unslothai/unsloth/issues/895) and [#2434](https://github.com/unslothai/unsloth/issues/2434).

## Config

```yaml
model:
  path: unsloth/Qwen3-1.7B-unsloth-bnb-4bit   # QLoRA model
  lora_rank: 8
  lora_alpha: 16
  load_in_4bit: true
  fast_inference: true
  gpu_memory_utilization: 0.6

generation:
  temperature: 1.0        # 1.0 for GRPO diversity
  top_p: 1.0
  max_tokens: 2048
  stop_token_ids: [151643, 151645]

algorithm:
  mode: spo               # "spo" (n=1, persistent tracker) or "grpo" (n=8, group baseline)
  spo:
    lr: 0.1               # EMA learning rate
    n: 1
  clip_ratio_low: 0.2
  clip_ratio_high: 0.28

training:
  total_steps: 800
  lr: 5.0e-6
  save_freq: 50
```

## Known Constraints

- **16GB VRAM budget**: Qwen3-1.7B 4-bit fits comfortably. 8B OOMs with vLLM on RTX 5080.
- **LoRA engine recreation**: vLLM leaks memory during LoRA hot-swap ([Unsloth #3864](https://github.com/unslothai/unsloth/issues/3864)). `LoRAVerifier.should_recreate_engine()` signals every 50 steps.
- **`force_on_policy_ratio=True`**: Disables ratio clipping by design (ratio=1.0 always). This is correct for on-policy training where `old_logprobs = curr_logprobs.detach()`. The `clip_ratio_low/high` config values have no effect in this mode.
- **GPU smoke tests run individually**: The RTX 5080 can't load 3 separate model instances in one pytest session. Run: `pytest tests/test_smoke.py::test_three_steps_no_crash --gpu -v`

## Tests

```bash
# CPU tests (85 tests, ~2 seconds)
pytest tests/

# GPU smoke test (requires CUDA GPU)
pytest tests/test_smoke.py::test_three_steps_no_crash --gpu -v

# All GPU tests (run individually to avoid OOM)
pytest tests/test_smoke.py::test_three_steps_no_crash --gpu -v
pytest tests/test_smoke.py::test_lora_sync_verification --gpu -v
pytest tests/test_smoke.py::test_vram_does_not_grow --gpu -v
```

## Before Launching v2 — Blind Spots

These are load-bearing issues identified during the build. None block the launch, but all should be instrumented and monitored.

### 1. SPO n=1 means each completion is used once

With `mode: spo` and `n: 1`, every generated completion is scored and used for exactly one gradient update, then discarded. There's no replay buffer, no importance sampling. If your `reward_fn` is slow (graph parsing, LLM-judge calls), each completion costs real wall time and is consumed immediately. The v1 run with `n=8` amortized reward cost across 8 completions per prompt. With SPO, effective throughput per reward-fn call drops proportionally. **Make sure your reward function runs in < 100ms per completion.**

### 2. Checkpoint before LoRA engine recreation

`LoRAVerifier.should_recreate_engine()` signals every 50 steps to tear down and rebuild the vLLM engine (workaround for [Unsloth #3864](https://github.com/unslothai/unsloth/issues/3864) memory leak). If recreation fails (OOM during rebuild from VRAM fragmentation), you lose the run. **Save a checkpoint before every engine recreation** so a failure there doesn't lose progress. Not yet implemented — add to the training loop in v2.

### 3. Segmentation is Qwen3-specific

`segment_completion()` uses hardcoded Qwen3 token IDs (`THINK_START=151667`, `STEP_TOKEN=9520`, etc.). Switching to a different model architecture requires updating these IDs. This is deliberate — token-ID matching is faster and more reliable than regex on decoded text — but it's a coupling point. **If reviewers ask for a model ablation, segmentation needs updating.**

### 4. "Bring Your Own Domain" has a boundary

The training loop is genuinely domain-agnostic. The advantage computation is not — it assumes a **step-decomposable domain** where structured output has 1-4 verifiable intermediate steps. Domains without intermediate structure (e.g., single-turn Q&A, code completion) would need a different advantage estimator. **The paper should be precise: the loop is general, the step-advantage estimation requires domains with verifiable intermediate steps.**

### 5. No evaluation harness

The engine trains, checkpoints, and logs metrics to MLflow. But there is no built-in offline evaluation. How does v2 know when training is working? Options:
- Emit checkpoints that the training-dojo evaluation pipeline consumes
- Add mid-training eval hooks (every N steps, run the model on an eval set)
- Monitor MLflow per-step reward curves for plateau/regression

**For v2, wire the eval pipeline from training-dojo before starting the run.** Don't fly blind for 800 steps.

## References

- QGRE paper: (forthcoming)
- [VPRMs](https://arxiv.org/abs/2601.17223) — Verifiable Process Rewards (IBM Research, Jan 2026)
- [SPO](https://arxiv.org/abs/2509.13232) — Single-stream Policy Optimization (Tencent, ICLR 2026)
- [GDPO](https://arxiv.org/abs/2601.05242) — Group Decomposed Policy Optimization (NVIDIA, Jan 2026)
- [GTPO](https://arxiv.org/abs/2508.04349) — Entropy-weighted token rewards (ByteDance, ICML)
- [NeMo RL](https://github.com/NVIDIA-NeMo/RL) — Loss functions extracted under Apache-2.0

## License

Apache-2.0. NeMo RL extracted components retain their original Apache-2.0 headers.

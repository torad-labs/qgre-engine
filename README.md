# QGRE Engine

Single-GPU GRPO training engine for novel-domain structured reasoning.

No Ray. No verl. No TRL. Just: generate → score → advantages → loss → backward → update.

## What is QGRE?

**Quality-Gated Reward Escalation** is a phase-gated curriculum for training LLMs on novel domains where correct reasoning methodology is unknown. Instead of rewarding all qualities from step 1, QGRE unlocks reward components progressively: format first, then grounding, then chain coherence, then accuracy.

## Quick Start (CLI)

```bash
python -m qgre train \
  --config examples/math/config.yaml \
  --reward examples.math.reward_fn:math_reward_fn
```

That's it. The engine loads the model, data, creates the trainer, and runs the full training loop.

For structured output domains with XML step tags:

```bash
python -m qgre train \
  --config examples/hypergraph/config.yaml \
  --reward examples.hypergraph.reward_fn:reward_fn
```

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
| **Reward** | Your `reward_fn` | Scores completions → `RewardResult(reward, scores)`. You provide this. The engine consumes `.scores` per quality for step-level credit. |
| **Segmentation** | `segments.py` | Pluggable segmenter splits token IDs into regions: THINK, STEP_1..N, FORMAT, OTHER. Built-in: `qwen3_xml_segmenter` (1-9 steps), `uniform_segmenter`, or your own. |
| **Advantages** | `advantages.py` | `QGREStepAdvantageEstimator` computes per-token advantages. Four techniques unified: SPO baseline, GDPO normalization, VPRM segment propagation, QGRE phase gating. |
| **Loss** | `nemo_extracted/loss_functions.py` | `ClippedPGLossFn` from NeMo RL (Apache-2.0). Clipped PG loss with DAPO-style asymmetric clipping, KL regularization, importance sampling. |
| **Backward** | `trainer.py` (PyTorch) | Standard `loss.backward()` + `optimizer.step()`. NaN guard, gradient clipping, gradient accumulation. Unsloth's `for_training()` mode disables inplace ops for autograd compatibility. |
| **Persistence** | `checkpoint.py`, `logging.py` | Checkpoint save/resume (model, optimizer, GameState, SPO V-tracker, RNG). MLflow metrics. JSONL completion logs. |

## How QGRE Works — The Core Algorithm

### The Problem

Standard GRPO gives the model a single reward signal per completion. For novel domains with multi-step structured output, this creates a credit assignment problem: the model can't tell which step was good and which was bad.

### The Solution: Step-Level Advantages

QGRE breaks the reward into per-step quality scores and computes per-token advantages:

```
                    Completion Token Sequence
     ┌────────────┬────────────┬────────────┬────────────┐
     │   STEP_1   │   STEP_2   │   STEP_3   │   STEP_4   │
     │ (format)   │ (ground)   │ (chain)    │ (accuracy) │
     └──────┬─────┴──────┬─────┴──────┬─────┴──────┬─────┘
            ▼            ▼            ▼            ▼
     ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
     │ q_format │  │ q_ground │  │ q_chain  │  │ q_accuracy│
     │ = 0.95   │  │ = 0.60   │  │ = 0.30   │  │ = 0.10   │
     └──────┬───┘  └──────┬───┘  └──────┬───┘  └──────┬───┘
            ▼            ▼            ▼            ▼
     ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
     │ A(step1) │  │ A(step2) │  │ A(step3) │  │ A(step4) │
     │ = +0.85  │  │ = +0.10  │  │ = -0.20  │  │ = -0.40  │
     └──────────┘  └──────────┘  └──────────┘  └──────────┘
     ◀── Per-token advantages broadcast to all tokens in step ──▸
```

### Four Techniques Unified

1. **SPO (Single-stream Policy Optimization)**: Persistent EMA baseline per prompt per step. `V = V + lr * (r - V)`. No groups needed — works with n=1.

2. **GDPO (Group Decomposed)**: Per-step batch normalization. Each step's advantages are independently normalized so no step's gradient signal drowns another.

3. **VPRM (Verifiable Process Rewards)**: Token-level segmentation. A segmenter maps token IDs to regions (STEP_1, STEP_2, FORMAT, THINK). Each token gets its step's advantage. THINK and FORMAT tokens get zero advantage (no gradient for structure).

4. **Phase Gating**: Progressive quality unlock. Phase 1 = only step 1 qualities matter. Phase N = all qualities from steps 1..N. The engine tracks mastery per step and auto-advances when mastery threshold is met.

### Phase Advancement

```
Phase 1: Train format only        → mastery > 0.8 → advance
Phase 2: Train format + grounding → mastery > 0.8 → advance
Phase 3: Train all through chain  → mastery > 0.8 → advance
Phase 4: Train everything         → full training
```

The engine manages this automatically via `GameState`. No external curriculum logic needed.

## Bring Your Own Domain

The engine is domain-agnostic. You provide three things:

### 1. A reward function

```python
from qgre.types import RewardResult

def my_reward_fn(prompt: str, completion: str, metadata: dict | None = None) -> RewardResult:
    # Score each quality independently
    scores = {
        "q_format": check_format(completion),      # 0.0 - 1.0
        "q_grounding": check_grounding(completion), # 0.0 - 1.0
        "q_accuracy": check_accuracy(completion),    # 0.0 - 1.0
    }
    return RewardResult(
        reward=sum(scores.values()) / len(scores),
        scores=scores,
    )
```

### 2. A step_qualities mapping

```yaml
# In config.yaml
algorithm:
  step_qualities:
    1: [q_format]            # Step 1: format quality
    2: [q_grounding]         # Step 2: grounding quality
    3: [q_accuracy]          # Step 3: accuracy quality
```

Or programmatically:
```python
step_qualities = {
    1: ["q_format"],
    2: ["q_grounding"],
    3: ["q_accuracy"],
}
```

### 3. A segmenter (optional)

Built-in options:
- `uniform` (default) — all tokens get STEP_1. Use for single-step domains (math, code, Q&A).
- `qwen3_xml` — parses `<step1_...>...</step1_...>` XML tags via Qwen3 token IDs. Use for structured multi-step output.
- Custom — any `Callable[[list[int]], list[str]]` that maps token IDs to region labels.

```yaml
algorithm:
  segmenter: qwen3_xml   # or "uniform" or "my_module:my_segmenter"
```

## Full Config Reference

```yaml
model:
  path: unsloth/Qwen3-1.7B-unsloth-bnb-4bit   # Any Unsloth-supported model
  lora_rank: 8
  lora_alpha: 16
  load_in_4bit: true
  fast_inference: true
  gpu_memory_utilization: 0.35    # vLLM KV cache fraction (colocated mode)

data:
  train_files:
    - data/train.parquet
  max_prompt_length: 3200
  train_batch_size: 16
  prompt_column: prompt            # Column name in parquet
  metadata_columns: [ground_truth] # Passed to reward_fn as metadata dict

generation:
  temperature: 1.0        # 1.0 for GRPO diversity
  top_p: 1.0
  top_k: -1               # Disabled
  max_tokens: 4096
  stop_token_ids: [151643, 151645]  # Model-specific EOS tokens

algorithm:
  mode: spo               # "spo" (n=1, persistent tracker) or "grpo" (n=8, group baseline)
  segmenter: uniform      # "uniform", "qwen3_xml", or "module:function"

  spo:
    lr: 0.1               # EMA learning rate for value tracker
    n: 1                  # Completions per prompt
    # KL-adaptive lr (SPO paper Algorithm 1) — adjusts lr based on KL divergence
    kl_adaptive: false    # true to enable
    kl_threshold: 0.1
    kl_factor: 2.0
    lr_factor: 1.5
    min_lr: 0.01
    max_lr: 0.5

  grpo:
    n: 8                  # Completions per prompt (group size)
    filter_groups: true   # DAPO: zero out degenerate groups (all-identical rewards)

  clip_ratio_low: 0.2
  clip_ratio_high: 0.28
  # KL regularization is off by default — meaningless with on-policy single-epoch.
  # Enable with loss_mode: kl_cov + kl_cov_ratio: 0.0002 when multi-epoch is added.
  loss_mode: pg              # "pg" (no KL) or "kl_cov" (requires multi-epoch)
  kl_cov_ratio: 0.0          # KL penalty weight (0.0 = off)
  llds_coef: 0.05            # LLDS loss coefficient (arXiv:2512.04220)
  loss_type: grpo            # "grpo" or "dr_grpo" (unbiased, arXiv:2503.20783)

  # Region-specific KL multipliers
  kl_think_multiplier: 0.1   # Low KL for think tokens (explore)
  kl_format_multiplier: 2.0  # High KL for format tokens (exploit)
  kl_step_multiplier: 1.0    # Normal KL for step content

  # Opt-in research features (all off by default):
  lambda_return: 0.0         # GRPO-λ eligibility traces (0=off, 0.95=typical)
  length_penalty_coef: 0.0   # Dynamic length control (0=off)
  length_penalty_threshold: 0.5

  step_qualities:             # Domain-specific quality mapping
    1: [q_format]
    2: [q_grounding]
    3: [q_accuracy]

training:
  total_steps: 800
  lr: 5.0e-6
  warmup_steps: 10
  lr_scheduler: cosine       # "cosine" or "linear"
  save_freq: 50
  gradient_accumulation_steps: 1
  max_grad_norm: 1.0         # Gradient clipping max norm
  mastery_threshold: 0.8     # Quality score required to advance phase

logging:
  mlflow_experiment: my-experiment
  completion_dir: output/completions
  checkpoint_dir: output/checkpoints
```

## Programmatic API

```python
from qgre import RewardResult, GameState
from qgre.config import QGREConfig
from qgre.generation import UnslothBackend
from qgre.trainer import QGRETrainer
from qgre.data import QGREDataLoader, load_prompts_from_parquet

# 1. Load config
cfg = QGREConfig.from_yaml("config.yaml")

# 2. Load model
backend = UnslothBackend(cfg.model, cfg.generation)
model, tokenizer = backend.load()

# 3. Load data
prompts = load_prompts_from_parquet("data/train.parquet")
loader = QGREDataLoader(
    prompts=prompts,
    tokenizer=tokenizer,
    max_prompt_length=cfg.data.max_prompt_length,
    train_batch_size=cfg.data.train_batch_size,
    n_completions=cfg.algorithm.spo.n,
    metadata_columns=cfg.data.metadata_columns,
)

# 4. Create trainer with your reward function
trainer = QGRETrainer(
    model=model,
    tokenizer=tokenizer,
    reward_fn=my_reward_fn,
    config=cfg,
    generation_backend=backend,
)

# 5. Train (handles everything: generate, score, step, checkpoint, log)
trainer.train(loader, backend)
```

## Optimizations Built In

All opt-in via config. Zero breaking changes when disabled.

| Feature | Config | Paper |
|---------|--------|-------|
| LLDS loss | `llds_coef: 0.05` | arXiv:2512.04220 |
| AdamW 8-bit | Automatic (bitsandbytes) | — |
| Low-advantage filter | Auto (SPO mode) | SPO paper |
| seq-mean-token-sum-norm | Always on | verl core_algos.py |
| Region-specific KL | `kl_think_multiplier: 0.1` | THR-style |
| selective_log_softmax | Always on | TRL PR #2799 |
| Dr.GRPO unbiased mode | `loss_type: dr_grpo` | arXiv:2503.20783 |
| DAPO Dynamic Sampling | `filter_groups: true` | DAPO paper |
| KL-adaptive SPO lr | `spo.kl_adaptive: true` | SPO Algorithm 1 |
| Prioritized sampling | Auto (SPO mode) | SPO Section 3.2 |
| GRPO-λ eligibility traces | `lambda_return: 0.95` | ICLR 2026 |
| Dynamic length control | `length_penalty_coef: 0.01` | Huawei |
| neg_logprob_mean metric | Always on (metric only) | Policy collapse monitor |
| Triton fused logprobs | Auto (if Triton available) | Custom kernel |

## File Structure

```
qgre/
  __init__.py          — Package root, exports RewardResult, GameState, segmenters
  __main__.py          — CLI: python -m qgre train --config --reward --segmenter
  types.py             — RewardResult dataclass, GameState
  config.py            — All config dataclasses, YAML loader
  segments.py          — Segmenters: qwen3_xml, uniform, custom
  advantages.py        — QGREStepAdvantageEstimator (SPO+GDPO+VPRM+phase)
  data.py              — DataLoader: parquet → tokenize → pad → batch
  checkpoint.py        — Save/resume full training state
  logging.py           — MLflow tracking + JSONL completion dump
  trainer.py           — QGRETrainer: the training loop
  generation.py        — UnslothBackend: vLLM colocated generation
  lora_verify.py       — LoRA weight sync verification
  fused_logprobs.py    — Chunked logprobs (no full logits materialization)
  triton_logprobs.py   — Triton fused lm_head→logprobs kernel
  nemo_extracted/      — ClippedPGLossFn, KL, logits (Apache-2.0 from NeMo RL)
examples/
  hypergraph/          — Multi-step XML structured output (SPO mode)
  math/                — Single-step math (GRPO mode)
tests/                 — 132 tests (123 CPU + 9 GPU)
```

## Tests

```bash
# All CPU tests (123 tests, ~25 seconds)
python -m pytest tests/ -q

# Specific module
python -m pytest tests/test_advantages.py -v

# GPU smoke test (requires CUDA GPU + Qwen3-1.7B)
python -m pytest tests/test_smoke.py::test_three_steps_no_crash --gpu -v
```

## Checkpoint & Resume

Full state is saved and restored automatically:
- Model weights (LoRA)
- Optimizer state (AdamW8bit)
- LR scheduler state
- GameState (phase, mastery, step counts)
- SPO value tracker (V per prompt per step)
- PyTorch + CUDA RNG state (exact reproducibility)

Resume is automatic — `trainer.train()` checks for the latest checkpoint and picks up where it left off.

## Known Constraints

- **16GB VRAM budget**: Qwen3-1.7B 4-bit fits comfortably. 8B OOMs with vLLM on RTX 5080.
- **Segmentation is model-specific**: `qwen3_xml_segmenter` uses Qwen3 token IDs. Other models need a custom segmenter or `uniform`.
- **Unsloth mode switching required**: Must call `set_training_mode()` before backward, `set_inference_mode()` before generate.
- **`force_on_policy_ratio=True`**: The engine uses on-policy training — ratio clipping config has no effect (ratio is always 1.0 by design).

## References

- QGRE paper: (forthcoming)
- [VPRMs](https://arxiv.org/abs/2601.17223) — Verifiable Process Rewards (IBM Research, Jan 2026)
- [SPO](https://arxiv.org/abs/2509.13232) — Single-stream Policy Optimization (Tencent, ICLR 2026)
- [GDPO](https://arxiv.org/abs/2601.05242) — Group Decomposed Policy Optimization (NVIDIA, Jan 2026)
- [Dr.GRPO](https://arxiv.org/abs/2503.20783) — Unbiased GRPO (Mar 2025)
- [LLDS](https://arxiv.org/abs/2512.04220) — Lazy Likelihood Displacement (Dec 2025)
- [NeMo RL](https://github.com/NVIDIA-NeMo/RL) — Loss functions extracted under Apache-2.0

## License

Apache-2.0. NeMo RL extracted components retain their original Apache-2.0 headers.

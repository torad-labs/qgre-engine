# QGRE Engine

**Quality-Gated Reward Escalation.** A single-GPU training engine that grounds LLM reasoning in novel domains — without supervised fine-tuning, without a critic network, without a cluster.

The core claim: SFT teaches models to *match outputs*. QGRE teaches models to *match inputs* — to ground their reasoning in the structure of the problem. A 1.7B model trained with QGRE on Hamiltonian mechanics writes correct physics derivations in 50 steps. Not because it memorized them. Because it learned the protocol.

No Ray. No verl. No TRL. Just: generate → score → advantages → loss → backward → update.

## Results

**Hamiltonian mechanics** (Qwen3-1.7B, 4-bit quantized, single RTX 5080 16GB):

| Step | Avg Reward | Min | Max | What the model produces |
|------|-----------|-----|-----|------------------------|
| 0 | 0.61 | 0.40 | 0.96 | Guessing — some correct structure by chance |
| 3 | 0.93 | 0.85 | 0.98 | Identifies kinetic + potential energy, derives Hamilton's equations |
| 18 | 0.98 | 0.96 | 1.00 | Near-perfect derivations with `<think>` reasoning traces |
| 47 | 0.94 | 0.94 | 0.95 | Converged — consistent quality across all prompts |

50 steps. ~35 seconds per step at 4096 tokens. No SFT warm-up. No hand-crafted examples. The curriculum *is* the warm-up.

```
SCORE: 0.98
<think>
Okay, so I need to derive the Hamiltonian H(x, p) from first principles
for a block attached to a spring on a frictionless surface. The block has
mass 3 kg, the spring constant is 6 N/m...
```

## Quick Start

```bash
pip install -e .

# Hamiltonian mechanics (SPO mode, verifiable via sympy)
python -m qgre train \
  --config examples/hamiltonian/config.yaml \
  --reward examples.hamiltonian.reward_fn:hamiltonian_reward

# Multi-step XML structured output
python -m qgre train \
  --config examples/hypergraph/config.yaml \
  --reward examples.hypergraph.reward_fn:reward_fn

# Single-step math
python -m qgre train \
  --config examples/math/config.yaml \
  --reward examples.math.reward_fn:math_reward_fn
```

That's it. The engine loads the model, data, creates the trainer, and runs the full training loop.

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
| **Model + LoRA** | `generation.py` (Unsloth) | Loads QLoRA model (4-bit base + full-precision adapters). Switches between `for_training()` and `for_inference()` mode. Manages vLLM engine lifecycle including periodic recreation to prevent VRAM leak. |
| **Generation** | `generation.py` (vLLM) | `fast_generate()` produces completions via in-process vLLM. Decodes prompts, returns token IDs + text. Handles SamplingParams (temperature, top_p, stop tokens). |
| **Reward** | Your `reward_fn` | Scores completions → `RewardResult(reward, scores)`. You provide this. The engine consumes `.scores` per quality for step-level credit. Every score must give partial credit — binary 0/1 kills gradient signal. |
| **Segmentation** | `segments.py` | Pluggable segmenter splits token IDs into regions: THINK, STEP_1..N, FORMAT, OTHER. Built-in: `qwen3_xml` (XML step tags), `hif_json` (HIF JSON sections), `uniform` (all STEP_1), or your own. |
| **Advantages** | `advantages.py` | `QGREStepAdvantageEstimator` computes per-token advantages. Four techniques unified: SPO baseline, GDPO normalization, VPRM segment propagation, QGRE phase gating. NaN-guarded (ms-swift #8123). |
| **Loss** | `nemo_extracted/loss_functions.py` | `ClippedPGLossFn` from NeMo RL (Apache-2.0). Clipped PG loss with DAPO-style asymmetric clipping, configurable KL regularization (k1/k2/k3), importance sampling, region-specific KL weights. |
| **Backward** | `trainer.py` (PyTorch) | Standard `loss.backward()` + `optimizer.step()`. NaN guard, gradient clipping, gradient accumulation. Unsloth's `for_training()` mode called before each micro-batch forward pass for autograd compatibility. |
| **Monitoring** | `trainer.py` | Completion length tracking (mean/max/min per step), stagnation detection (timeout + plateau), neg_logprob_mean for policy collapse monitoring. All logged to MLflow. |
| **Persistence** | `checkpoint.py`, `logging.py` | Checkpoint save/resume (model, optimizer, scheduler, GameState with stagnation counters, SPO V-tracker, RNG). MLflow metrics. JSONL completion logs. |

## How QGRE Works — The Core Algorithm

### The Problem

Standard RL training gives the model a single reward signal per completion. For novel domains with multi-step structured output, this creates a credit assignment problem: the model can't tell which step was good and which was bad. A completion that formats perfectly but grounds poorly gets the same gradient as one that grounds perfectly but formats poorly.

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

1. **SPO (Single-stream Policy Optimization)**: Persistent EMA baseline per prompt per step. `V = V + lr * (r - V)`. No groups needed — works with n=1. One completion per prompt, every completion teaches. The baseline remembers what this prompt usually scores.

2. **GDPO (Group Decomposed)**: Per-step batch normalization. Each step's advantages are independently normalized so no step's gradient signal drowns another. When format saturates at 0.95, grounding at 0.30 still gets full gradient bandwidth.

3. **VPRM (Verifiable Process Rewards)**: Token-level segmentation. A segmenter maps token IDs to regions (STEP_1, STEP_2, FORMAT, THINK). Each token gets its step's advantage. THINK and FORMAT tokens get zero advantage — reasoning is free, structure is free.

4. **Phase Gating**: Progressive quality unlock. Phase 1 = only step 1 qualities matter. Phase N = all qualities from steps 1..N. The engine tracks mastery per step and auto-advances when mastery threshold is met.

### Phase Advancement

```
Phase 1: Train format only        → mastery > 0.8 → advance
Phase 2: Train format + grounding → mastery > 0.8 → advance
Phase 3: Train all through chain  → mastery > 0.8 → advance
Phase 4: Train everything         → full training
```

The engine manages this automatically via `GameState`. No external curriculum logic needed.

### Stagnation Detection

Training can stall. The engine monitors two signals:

- **Plateau**: mastery improvement < 0.02 over the last 50 steps. The model stopped learning but hasn't hit the threshold.
- **Timeout**: a phase exceeds 200 steps without advancement. The curriculum is stuck.

Both are logged to MLflow as the `stagnation` metric (0=normal, 1=stagnating, 2=stuck). Detection only — the engine does not intervene. Your training run decides what to do with the signal.

### The n=1 Economics

Standard GRPO generates 8 completions per prompt and compares them. SPO generates 1 and compares it to a persistent memory of what this prompt usually scores.

For a 4096-token completion at ~35s generation time:
- GRPO (n=8): 8 × 35s = 280s generation per training step
- SPO (n=1): 1 × 35s = 35s generation per training step

8× faster per step. The trade-off: no within-group comparison. SPO compensates with VPRMs (per-region credit) and persistent baselines (cross-step credit). The signal-per-step is slightly noisier but the steps-per-hour is 8× higher.

## Bring Your Own Domain

The engine is domain-agnostic. You provide three things:

### 1. A reward function

```python
from qgre.types import RewardResult

def my_reward_fn(prompt: str, completion: str, metadata: dict | None = None) -> RewardResult:
    # Score each quality independently — ALWAYS partial credit, NEVER binary 0/1
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

| Segmenter | Config value | Use case |
|-----------|-------------|----------|
| `uniform` | `segmenter: uniform` | Single-step domains (math, code, Q&A). All tokens → STEP_1. |
| `qwen3_xml` | `segmenter: qwen3_xml` | Multi-step XML tags (`<step1>...<step2>..`). Token ID pattern matching — no decode needed. |
| `hif_json` | `segmenter: hif_json` | HIF JSON output. Decode-and-regex on `nodes`, `edges`, `incidences`, `scan-results`. Binds tokenizer at registration. |
| Custom | `segmenter: my_module:my_fn` | Any `Callable[[list[int]], list[str]]` that maps token IDs to region labels. |

```yaml
algorithm:
  segmenter: qwen3_xml   # or "uniform" or "hif_json" or "my_module:my_segmenter"
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
  temperature: 1.0        # 1.0 for RL diversity; 0.6 for Qwen3 recommended
  top_p: 1.0
  top_k: -1               # Disabled
  max_tokens: 4096
  stop_token_ids: [151643, 151645]  # Model-specific EOS tokens (Qwen3)

algorithm:
  mode: spo               # "spo" (n=1, persistent tracker) or "grpo" (n=8, group baseline)
  segmenter: uniform      # "uniform", "qwen3_xml", "hif_json", or "module:function"
  reference_policy_kl_type: k3  # "k1" (unbiased), "k2" (squared), "k3" (exponential)

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
  # KL regularization is off by default — meaningless with on-policy single-epoch
  # where reference_logprobs == curr_logprobs. Enable with loss_mode: kl_cov when
  # stored generation-time logprobs are implemented.
  loss_mode: pg              # "pg" (no KL) or "kl_cov" (KL regularized)
  kl_cov_ratio: 0.0          # KL penalty weight (0.0 = off)
  llds_coef: 0.05            # LLDS collapse prevention (arXiv:2512.04220)
  loss_type: grpo            # "grpo" or "dr_grpo" (unbiased, arXiv:2503.20783)

  # Region-specific KL multipliers (validated by Archer, ICLR 2026)
  kl_think_multiplier: 0.1   # Low KL for think tokens (explore freely)
  kl_format_multiplier: 2.0  # High KL for format tokens (lock structure)
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
  stagnation_timeout: 200    # Steps in a phase before STUCK signal
  plateau_window: 50         # Steps to measure plateau slope
  plateau_threshold: 0.02    # Minimum improvement to avoid STAGNATING signal

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

## What's Built In

Research-backed features. All opt-in via config. Zero breaking changes when disabled.

| Feature | Config | Source |
|---------|--------|--------|
| SPO persistent baseline | `mode: spo` | SPO (Tencent, ICLR 2026) |
| GDPO per-step normalization | Always on | GDPO (NVIDIA, Jan 2026) |
| VPRM segment propagation | Via segmenter | VPRMs (IBM Research, Jan 2026) |
| Phase-gated curriculum | Via step_qualities | QGRE (this work) |
| Stagnation detection | `stagnation_timeout`, `plateau_window` | Scaf-GRPO informed |
| Completion length tracking | Always on | Verbosity drift detection |
| GDPO NaN guard | Always on | ms-swift #8123 |
| LLDS collapse prevention | `llds_coef: 0.05` | arXiv:2512.04220 |
| AdamW 8-bit | Automatic (bitsandbytes) | — |
| Low-advantage filter | Auto (SPO mode) | SPO paper |
| seq-mean-token-sum-norm | Always on | verl core_algos.py |
| KL estimator selection | `reference_policy_kl_type: k1` | Comedy of Estimators (ICLR 2026) |
| Region-specific KL | `kl_think_multiplier: 0.1` | Archer (ICLR 2026) |
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
  __init__.py          — Package root, exports RewardResult, GameState, StagnationStatus
  __main__.py          — CLI: python -m qgre train --config --reward --segmenter
  types.py             — RewardResult, GameState, StagnationStatus
  config.py            — All config dataclasses, YAML loader
  segments.py          — Segmenters: qwen3_xml, hif_json, uniform, custom
  advantages.py        — QGREStepAdvantageEstimator (SPO+GDPO+VPRM+phase)
  data.py              — DataLoader: parquet → tokenize → left-pad → batch → prioritized sampling
  checkpoint.py        — Save/resume full training state (including stagnation counters)
  logging.py           — MLflow tracking + JSONL completion dump (context manager)
  trainer.py           — QGRETrainer: the training loop
  generation.py        — UnslothBackend: vLLM colocated generation
  lora_verify.py       — LoRA weight sync verification
  fused_logprobs.py    — Chunked logprobs (no full logits materialization)
  triton_logprobs.py   — Triton fused lm_head→logprobs kernel
  nemo_extracted/      — ClippedPGLossFn, KL, logits, LLDS (Apache-2.0 from NeMo RL)
examples/
  hamiltonian/         — Physics derivation (SPO mode, verifiable via sympy)
  hypergraph/          — Multi-step XML structured output (SPO mode)
  math/                — Single-step math
tests/                 — 130 CPU tests + 9 GPU tests
```

## Tests

```bash
# All CPU tests (130 tests, ~27 seconds)
python -m pytest tests/ -q

# Segmentation tests (XML + HIF JSON)
python -m pytest tests/test_segments.py -v

# Advantage computation
python -m pytest tests/test_advantages.py -v

# Specific module
python -m pytest tests/test_checkpoint.py -v

# GPU smoke test (requires CUDA GPU + Qwen3-1.7B)
python -m pytest tests/test_smoke.py::test_three_steps_no_crash --gpu -v
```

## Checkpoint & Resume

Full state is saved and restored automatically:
- Model weights (LoRA adapters)
- Optimizer state (AdamW8bit)
- LR scheduler state
- GameState (phase, mastery windows, stagnation counters, phase history)
- SPO value tracker (V per prompt per step)
- PyTorch + CUDA RNG state (exact reproducibility)

Resume is automatic — `trainer.train()` checks for the latest checkpoint and picks up where it left off. Stagnation detection state survives checkpoint/resume with backward-compatible defaults.

## Known Constraints

- **16GB VRAM budget**: Qwen3-1.7B 4-bit at `gpu_memory_utilization=0.35` peaks at 6.2GB. 8B requires higher utilization and tighter micro-batching.
- **On-policy only**: `force_on_policy_ratio=True` means ratio clipping config has no effect (ratio is always 1.0 by design). KL regularization requires stored generation-time logprobs (not yet implemented) — until then, `loss_mode: pg` is the correct default.
- **Segmenters are model-specific**: `qwen3_xml` uses Qwen3 token IDs. `hif_json` uses decoded text with regex. Other models need a custom segmenter or `uniform`.
- **Unsloth mode switching required**: Must call `set_training_mode()` before backward, `set_inference_mode()` before generate. The engine handles this automatically.
- **vLLM recreation**: Engine recreates the vLLM backend every 50 steps to prevent VRAM leak (Unsloth #3864). Failures are logged with warnings, not silently swallowed.

## References

- QGRE paper: (forthcoming)
- [SPO](https://arxiv.org/abs/2509.13232) — Single-stream Policy Optimization (Tencent, ICLR 2026)
- [GDPO](https://arxiv.org/abs/2601.05242) — Group Decomposed Policy Optimization (NVIDIA, Jan 2026)
- [VPRMs](https://arxiv.org/abs/2601.17223) — Verifiable Process Rewards (IBM Research, Jan 2026)
- [Dr.GRPO](https://arxiv.org/abs/2503.20783) — Unbiased GRPO (Mar 2025)
- [LLDS](https://arxiv.org/abs/2512.04220) — Lazy Likelihood Displacement (Dec 2025)
- [Comedy of Estimators](https://arxiv.org/abs/2512.21852) — KL estimator analysis (Bengio et al., Dec 2025)
- [Archer](https://openreview.net/forum?id=ee326398473daf76d49b49cda4dea9d699fbf61b) — Dual-token KL constraints (ICLR 2026)
- [Scaf-GRPO](https://arxiv.org/abs/2510.19807) — Scaffolded progressive training (Feb 2026)
- [NeMo RL](https://github.com/NVIDIA-NeMo/RL) — Loss functions extracted under Apache-2.0

## License

Apache-2.0. NeMo RL extracted components retain their original Apache-2.0 headers.

---

Built by [Torad Labs](https://torad.ai). The engine behind the QGRE paper.

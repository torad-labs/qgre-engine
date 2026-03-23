from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    path: str = ""  # Required — set in YAML or constructor
    lora_rank: int = 8
    lora_alpha: int = 16
    load_in_4bit: bool = True
    fast_inference: bool = True
    gpu_memory_utilization: float = 0.35  # Colocate: leaves ~65% VRAM for training (Leeroopedia heuristic)


@dataclass
class DataConfig:
    train_files: list[str] = field(default_factory=list)
    max_prompt_length: int = 3200
    train_batch_size: int = 16
    prompt_column: str = "prompt"
    metadata_columns: list[str] = field(default_factory=lambda: ["ground_truth", "extra_info"])
    # Difficulty-gated curriculum: maps GameState phase → allowed difficulty values.
    # Prompts with difficulty not in the current phase's set get zero sampling weight.
    # The difficulty value comes from the metadata column named by difficulty_column.
    difficulty_column: str | None = None  # e.g. "difficulty"
    difficulty_schedule: dict | None = None  # e.g. {1: ["tier1","edge"], 2: ["tier1","edge","tier2"], ...}
    system_prompt_column: str | None = None  # e.g. "system_prompt" — separate system message in chat template
    # 2D mastery matrix curriculum (takes precedence over difficulty_schedule when set)
    tier_order: list[str] | None = None           # e.g. ["tier1", "edge", "tier2", "tier3"]
    tier_advance_quality_phase: int = 3           # Quality phase required on current tier to unlock next
    tier_advance_threshold: float = 0.85          # Mastery threshold for tier advancement
    initial_tiers: list[str] | None = None        # Starting tiers, e.g. ["tier1", "edge"]


@dataclass
class GenerationConfig:
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    max_tokens: int = 4096
    stop_token_ids: list[int] = field(default_factory=lambda: [151643, 151645])  # Qwen3: <|endoftext|> + <|im_end|>


@dataclass
class SPOConfig:
    lr: float = 0.1
    n: int = 1
    # KL-adaptive lr (SPO paper Algorithm 1): adjust EMA lr based on KL divergence
    kl_adaptive: bool = False
    kl_threshold: float = 0.1
    kl_factor: float = 2.0
    lr_factor: float = 1.5
    min_lr: float = 0.01
    max_lr: float = 0.5


@dataclass
class GRPOConfig:
    n: int = 8
    filter_groups: bool = True


@dataclass
class LabelPatternConfig:
    pattern: str = ""
    region: str = "STEP_1"


@dataclass
class LabelSegmenterConfig:
    patterns: list[LabelPatternConfig] = field(default_factory=list)
    default_region: str = "STEP_1"
    ignore_case: bool = False


@dataclass
class AlgorithmConfig:
    mode: str = "spo"  # "spo" or "grpo"
    spo: SPOConfig = field(default_factory=SPOConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    clip_ratio_low: float = 0.2
    clip_ratio_high: float = 0.28
    # KL regularization requires multi-epoch training (ppo_epochs > 1) to be meaningful.
    # With on-policy (generate then immediately train, 1 epoch), KL between current and
    # generation is zero by definition — no optimizer step occurs between them.
    loss_mode: str = "pg"  # "pg" (no KL) or "kl_cov" (requires multi-epoch)
    kl_cov_ratio: float = 0.0
    reference_policy_kl_type: str = "k3"  # "k1" (linear/unbiased), "k2" (squared), "k3" (exponential)
    llds_coef: float = 0.05
    # entropy_coeff removed: -mean(logprob) has wrong gradient direction for entropy bonus.
    # neg_logprob_mean is logged as a metric only (no backprop). See Fix 3 notes.
    step_qualities: dict | None = None  # {step_num: [quality_names]} — domain-specific
    segmenter: str = "uniform"  # "uniform", "qwen3_xml", "label", or "module.path:function_name"
    label_segmenter: LabelSegmenterConfig | None = None  # Config for segmenter="label"
    # Region-specific KL multipliers (THR-style, PLAN.md lines 798-802)
    # THINK=explore freely, FORMAT=lock structure, STEP=focus on quality
    kl_think_multiplier: float = 0.1   # Low KL for think tokens (explore)
    kl_format_multiplier: float = 2.0  # High KL for format tokens (exploit)
    kl_step_multiplier: float = 1.0    # Normal KL for step content
    # Dr.GRPO: remove length and std normalization biases (arXiv:2503.20783)
    # "grpo": standard GRPO (divides by horizon length + normalizes by std)
    # "dr_grpo": removes both normalizations (unbiased gradients)
    loss_type: str = "grpo"  # "grpo" or "dr_grpo"
    # GRPO-λ eligibility traces (ICLR 2026): per-token credit via λ-return approximation
    lambda_return: float = 0.0  # 0=off, 0.95=typical. Composes with VPRM step-level rewards.
    # Dynamic length control (Huawei): penalize length only when group accuracy is high
    length_penalty_coef: float = 0.0  # 0=off
    length_penalty_threshold: float = 0.5  # correctness ratio above which length penalty applies


@dataclass
class TrainingConfig:
    total_steps: int = 800
    lr: float = 5e-6
    warmup_steps: int = 10
    lr_scheduler: str = "cosine"
    save_freq: int = 50
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    mastery_threshold: float = 0.8
    stagnation_timeout: int = 200
    plateau_window: int = 50
    plateau_threshold: float = 0.02


@dataclass
class LoggingConfig:
    mlflow_experiment: str = "qgre-training"
    completion_dir: str = "output/completions"
    checkpoint_dir: str = "output/checkpoints"


@dataclass
class QGREConfig:
    """Top-level config for the QGRE training engine."""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @staticmethod
    def from_yaml(path: str | Path) -> QGREConfig:
        """Load config from YAML file."""
        with open(path) as f:
            raw = yaml.safe_load(f)
        return QGREConfig._from_dict(raw)

    @staticmethod
    def _from_dict(d: dict) -> QGREConfig:

        def _pick(cls, raw: dict, section: str = "") -> dict:
            """Filter raw dict to only keys recognized by the dataclass. Warns on unknown keys."""
            known = set(cls.__dataclass_fields__)
            unknown = set(raw) - known
            if unknown:
                import warnings
                warnings.warn(f"Unknown {section} config keys (ignored): {unknown}")
            return {k: v for k, v in raw.items() if k in known}

        cfg = QGREConfig()
        if "model" in d:
            cfg.model = ModelConfig(**_pick(ModelConfig, d["model"], "model"))
        if "data" in d:
            data_fields = _pick(DataConfig, d["data"], "data")
            # Ensure difficulty_schedule keys are ints
            if "difficulty_schedule" in data_fields and data_fields["difficulty_schedule"] is not None:
                data_fields["difficulty_schedule"] = {
                    int(k): list(v) for k, v in data_fields["difficulty_schedule"].items()
                }
            cfg.data = DataConfig(**data_fields)
        if "generation" in d:
            cfg.generation = GenerationConfig(**_pick(GenerationConfig, d["generation"], "generation"))
        if "algorithm" in d:
            alg = d["algorithm"]
            spo = SPOConfig(**_pick(SPOConfig, alg.get("spo", {}), "algorithm.spo")) if "spo" in alg else SPOConfig()
            grpo = GRPOConfig(**_pick(GRPOConfig, alg.get("grpo", {}), "algorithm.grpo")) if "grpo" in alg else GRPOConfig()
            algo_top = {k: v for k, v in alg.items() if k not in ("spo", "grpo", "label_segmenter")}
            algo_fields = _pick(AlgorithmConfig, algo_top, "algorithm")
            # Parse label_segmenter config
            if "label_segmenter" in alg and alg["label_segmenter"] is not None:
                ls_raw = alg["label_segmenter"]
                patterns = [LabelPatternConfig(**p) for p in ls_raw.get("patterns", [])]
                algo_fields["label_segmenter"] = LabelSegmenterConfig(
                    patterns=patterns,
                    default_region=ls_raw.get("default_region", "STEP_1"),
                    ignore_case=ls_raw.get("ignore_case", False),
                )
            # Ensure step_qualities keys are ints (YAML may parse them as ints or strings)
            if "step_qualities" in algo_fields and algo_fields["step_qualities"] is not None:
                algo_fields["step_qualities"] = {
                    int(k): list(v) for k, v in algo_fields["step_qualities"].items()
                }
            cfg.algorithm = AlgorithmConfig(spo=spo, grpo=grpo, **algo_fields)
        if "training" in d:
            cfg.training = TrainingConfig(**_pick(TrainingConfig, d["training"], "training"))
        if "logging" in d:
            cfg.logging = LoggingConfig(**_pick(LoggingConfig, d["logging"], "logging"))
        return cfg

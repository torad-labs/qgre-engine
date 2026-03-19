from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    path: str = "unsloth/Qwen3-1.7B-unsloth-bnb-4bit"
    tokenizer_path: str | None = None
    lora_rank: int = 8
    lora_alpha: int = 16
    load_in_4bit: bool = True
    fast_inference: bool = True
    gpu_memory_utilization: float = 0.6


@dataclass
class DataConfig:
    train_files: list[str] = field(default_factory=list)
    max_prompt_length: int = 3200
    max_response_length: int = 2048
    train_batch_size: int = 16
    prompt_column: str = "prompt"
    metadata_columns: list[str] = field(default_factory=lambda: ["ground_truth", "extra_info"])


@dataclass
class GenerationConfig:
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    max_tokens: int = 4096
    stop_token_ids: list[int] = field(default_factory=lambda: [151643, 151645])


@dataclass
class SPOConfig:
    lr: float = 0.1
    n: int = 1


@dataclass
class GRPOConfig:
    n: int = 8
    filter_groups: bool = True


@dataclass
class AlgorithmConfig:
    mode: str = "spo"  # "spo" or "grpo"
    spo: SPOConfig = field(default_factory=SPOConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    clip_ratio_low: float = 0.2
    clip_ratio_high: float = 0.28
    loss_mode: str = "kl_cov"
    kl_cov_ratio: float = 0.0002
    llds_coef: float = 0.05
    entropy_coeff: float = 0.001
    step_qualities: dict | None = None  # {step_num: [quality_names]} — domain-specific


@dataclass
class TrainingConfig:
    total_steps: int = 800
    lr: float = 5e-6
    warmup_steps: int = 10
    lr_scheduler: str = "cosine"
    save_freq: int = 50
    gradient_accumulation_steps: int = 1


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
            cfg.data = DataConfig(**_pick(DataConfig, d["data"], "data"))
        if "generation" in d:
            cfg.generation = GenerationConfig(**_pick(GenerationConfig, d["generation"], "generation"))
        if "algorithm" in d:
            alg = d["algorithm"]
            spo = SPOConfig(**_pick(SPOConfig, alg.get("spo", {}), "algorithm.spo")) if "spo" in alg else SPOConfig()
            grpo = GRPOConfig(**_pick(GRPOConfig, alg.get("grpo", {}), "algorithm.grpo")) if "grpo" in alg else GRPOConfig()
            algo_top = {k: v for k, v in alg.items() if k not in ("spo", "grpo")}
            algo_fields = _pick(AlgorithmConfig, algo_top, "algorithm")
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

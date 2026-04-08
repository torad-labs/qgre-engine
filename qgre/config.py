from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import yaml


if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class ModelConfig:
    path: str = ""  # Required — set in YAML or constructor
    lora_rank: int = 8
    lora_alpha: int = 16
    load_in_4bit: bool = True
    fast_inference: bool = True
    gpu_memory_utilization: float = (
        0.35  # Colocate: leaves ~65% VRAM for training (Leeroopedia heuristic)
    )
    max_lora_rank: int = (
        0  # 0 = auto (max(64, lora_rank * 2)); vLLM rejects adapters with rank > this
    )
    weight_sync_strategy: str = (
        "direct_copy"  # "direct_copy" (4-bit) or "merge" (full-precision deployment)
    )
    pad_token: str = (
        ""  # Required — set per-model in YAML. Must NOT be EOS, stop token, or vision-reserved.
    )
    pad_token_id: int = -1  # Required — set per-model in YAML.
    lora_target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )  # LoRA target modules — model-architecture-specific. Qwen3/LLaMA default shown.
    modules_to_save: list[str] = field(
        default_factory=lambda: ["lm_head"]
    )  # embed_tokens removed — fim_pad is pre-trained


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
    difficulty_schedule: dict | None = None  # DEPRECATED: use tier_order + initial_tiers instead

    def __post_init__(self):
        """Validate config after initialization."""
        if self.difficulty_schedule is not None:
            import warnings

            warnings.warn(
                "difficulty_schedule is deprecated. Use tier_order + initial_tiers for 2D curriculum. "
                "difficulty_schedule will be removed in a future version.",
                stacklevel=2,
            )
        # R3-CSM-009: Validate initial_tiers is list or None
        if self.initial_tiers is not None and not isinstance(self.initial_tiers, list):
            raise TypeError(
                f"R3-CSM-009: initial_tiers must be a list or None, got {type(self.initial_tiers).__name__}. "
                f"Use [{self.initial_tiers!r}] if you meant a single-item list.",
            )
        # R3-CSM-002: Validate initial_tiers not empty when tier_order is non-empty
        if self.tier_order and not self.initial_tiers:
            raise ValueError(
                f"R3-CSM-002: tier_order is non-empty {self.tier_order} but initial_tiers is empty. "
                "Curriculum requires starting tiers. Set initial_tiers or leave tier_order empty.",
            )
        # R3-CSM-003: Validate tier_order not empty when initial_tiers is non-empty
        if self.initial_tiers and not self.tier_order:
            raise ValueError(
                f"R3-CSM-003: initial_tiers is non-empty {self.initial_tiers} but tier_order is empty. "
                "Curriculum requires progression order. Set tier_order or leave initial_tiers empty.",
            )
        # R3-CSM-004: Validate no duplicate tiers in tier_order
        if self.tier_order and len(self.tier_order) != len(set(self.tier_order)):
            duplicates = [t for t in self.tier_order if self.tier_order.count(t) > 1]
            raise ValueError(
                f"R3-CSM-004: tier_order contains duplicate tiers: {duplicates}. "
                f"Each tier should appear only once in progression.",
            )
        # R3-CSM-010: Validate no duplicate tiers in initial_tiers
        if self.initial_tiers and len(self.initial_tiers) != len(set(self.initial_tiers)):
            duplicates = [t for t in self.initial_tiers if self.initial_tiers.count(t) > 1]
            raise ValueError(
                f"R3-CSM-010: initial_tiers contains duplicate tiers: {duplicates}. "
                f"Each tier should appear only once in starting set.",
            )
        # DP3-009: Validate initial_tiers subset of tier_order
        if self.tier_order and self.initial_tiers:
            tier_order_set = set(self.tier_order)
            initial_set = set(self.initial_tiers)
            if not initial_set.issubset(tier_order_set):
                raise ValueError(
                    f"DP3-009: initial_tiers {self.initial_tiers} contains tiers not in tier_order {self.tier_order}. "
                    f"Invalid tiers: {initial_set - tier_order_set}",
                )

    system_prompt_column: str | None = (
        None  # e.g. "system_prompt" — separate system message in chat template
    )
    # 2D mastery matrix curriculum (takes precedence over difficulty_schedule when set)
    tier_order: list[str] | None = None  # e.g. ["tier1", "edge", "tier2", "tier3"]
    tier_advance_quality_phase: int = 3  # Quality phase required on current tier to unlock next
    tier_advance_threshold: float = 0.85  # Mastery threshold for tier advancement
    initial_tiers: list[str] | None = None  # Starting tiers, e.g. ["tier1", "edge"]


@dataclass
class GenerationConfig:
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    min_p: float = 0.1
    max_tokens: int = 4096
    repetition_penalty: float = (
        1.0  # 1.0 = disabled, >1.0 penalizes repeated tokens in vLLM sampling
    )
    stop_token_ids: list[int] = field(
        default_factory=list
    )  # Required per-model. Qwen3: [151643, 151645]
    max_logprobs: int = 5  # vLLM max_logprobs for LLDS logprob extraction
    # LoRA dropout during generation: partially revert to base model for exploration
    lora_dropout_rate: float = 0.0  # 0.0 = disabled, 0.15 = recommended
    lora_dropout_anneal_steps: int = 500  # Linear anneal to 0 over this many steps


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
    # Target-aware aspiration gap: A += beta * (reward - target). Preserves shaped reward gradients
    # even when baseline matches the constant partial credit. beta=0 disables.
    aspiration_beta: float = 0.5
    aspiration_target: float = 0.8  # Usually mastery_threshold
    # Variance-aware baseline: slow baseline lr when reward variance drops (prevents dead gradient)
    var_aware: bool = True
    var_threshold: float = 0.01  # Variance below this triggers slowdown
    var_lr: float = 0.05  # EMA rate for variance tracking
    min_var_ratio: float = 0.01  # Floor: baseline lr never drops below lr * min_var_ratio
    # Per-quality baseline staleness: decay toward prior when quality hasn't been seen in N steps
    staleness_window: int = 50  # Steps before baseline starts decaying toward prior
    baseline_prior: float = 0.5  # Prior value for unseen/stale qualities (middle of [0,1])


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
class SkillConfig:
    """Configuration for a single skill node in the tutorial DAG.

    Prompts can be specified directly (prompts: [id1, id2]) or matched
    from training data metadata (match_metadata: {system: freefall}).
    When match_metadata is set, prompts are resolved at init time from
    the dataloader's items based on metadata column values.
    """

    prompts: list[str] = field(default_factory=list)
    match_metadata: dict[str, str] | None = (
        None  # e.g. {system: freefall} — matches prompts by metadata column values
    )
    prerequisites: list[str] = field(default_factory=list)
    mastery_threshold: float = 0.8
    regression_threshold: float = 0.6
    mastery_window: int = 20
    review_probability: float = 0.15
    score_key: str | None = (
        None  # Quality key from RewardResult.scores to track mastery (e.g. "q_V_correct"). None = overall reward.
    )
    aspiration_warmup_steps: int = (
        20  # After unlock, ramp aspiration beta from 0 → full over N steps
    )
    # Learnability-based advancement: advance only when variance collapses (skill is stable)
    # learnability = p(1-p) where p = mastery_score. At p=0.5: 0.25 (max), p=0.9: 0.09 (ready)
    learnability_threshold: float = 0.10  # Advance when p(1-p) < this (p > 0.89 or p < 0.11)


_VALID_POST_MASTERY_BEHAVIORS = {"review_only", "pause", "continue_all"}


@dataclass
class TutorialConfig:
    """Skill-tree tutorial system — prerequisite-gated prompt filtering."""

    enabled: bool = False
    skill_tree: dict[str, SkillConfig] = field(default_factory=dict)
    post_mastery_behavior: str = "review_only"  # review_only | pause | continue_all
    untracked_always_active: bool = True
    sequential_mastery: bool = False  # Focus on one skill at a time instead of all active skills


@dataclass
class VPRMConfig:
    enabled: bool = False
    intermediate_dim: int = 128  # MLP hidden layer size
    lr: float = 1e-4  # Critic learning rate (separate from policy lr)
    clip_advantage: float = 5.0  # Per-quality advantage clipping bound
    spo_fallback_min_regions: int = 2  # Min distinct regions to use critic (else SPO fallback)
    polyak_tau: float = 0.01  # Polyak averaging rate for target network
    use_target_network: bool = True  # Stable baselines via slow-moving target
    target_warmup_steps: int = 500  # Hard-sync target every 100 steps during warmup


@dataclass
class EGRSConfig:
    """Entropy-Gated Reinforcement System — 2x2 matrix for token-level gradient control.

    Classifies tokens into 4 quadrants based on (correct/wrong) x (confident/uncertain):
    - Q1 (uncertain+correct): Reinforce (learning)
    - Q2 (confident+correct): Do nothing (already learned)
    - Q3 (confident+wrong): Entropy boost (shake confidence)
    - Q4 (uncertain+wrong): Flag for hint injection (provide direction)

    See docs/entropy-gated-reinforcement.md for full design.
    """

    enabled: bool = False  # Master switch for EGRS
    reward_threshold: float = 0.5  # Span reward above this = "correct"
    entropy_threshold: float = 0.5  # Normalized entropy below this = "confident"
    gate_temperature: float = 0.1  # Sigmoid temperature for soft gating (lower = sharper)
    exploration_weight: float = 0.1  # Lambda for entropy bonus in Q3 (confident+wrong)
    hint_enabled: bool = True  # Enable hint injection for Q4 (uncertain+wrong)
    hint_token_count: int = 3  # Max tokens to inject as hint
    mastery_threshold: float = 0.8  # Mastery score at which hints stop
    # Hint extractor: "hamiltonian", "generic", or "none"
    # - hamiltonian: Uses T_expr, V_expr, H_expr from metadata
    # - generic: Uses hint_extractor_mapping to map span_id → metadata field
    # - none: Generic hints like "Focus on STEP_N"
    hint_extractor: str = "none"
    hint_extractor_mapping: dict = field(default_factory=dict)  # For generic extractor

    def __post_init__(self):
        """Validate EGRS config after initialization."""
        if self.gate_temperature <= 0:
            raise ValueError(
                f"gate_temperature must be > 0, got {self.gate_temperature}. "
                "Zero temperature causes division by zero in confidence_gate.",
            )
        if self.reward_threshold >= 1.0:
            import warnings

            warnings.warn(
                f"reward_threshold={self.reward_threshold} >= 1.0 means no spans are ever 'correct'. "
                "This makes all EGRS tokens Q3 or Q4. Consider lowering threshold.",
                stacklevel=2,
            )
        if self.hint_enabled and self.hint_token_count < 1:
            raise ValueError(
                f"hint_enabled=True but hint_token_count={self.hint_token_count} < 1. "
                "Must have at least 1 token for hints.",
            )


@dataclass
class LoRAProConfig:
    """LoRA-Pro gradient adjustment for better approximation of full fine-tuning.

    Paper: "LoRA-Pro: Are Low-Rank Adapters Properly Optimized?" (ICLR 2025)
    https://arxiv.org/abs/2407.18242

    When enabled, adjusts LoRA A/B gradients after backward() so that the
    equivalent low-rank gradient better approximates what full fine-tuning
    would produce. Adds ~4GB memory overhead, no additional training time.
    """

    enabled: bool = False  # Enable LoRA-Pro gradient adjustment
    beta1: float = 0.9  # Adam beta1 for equivalent gradient momentum
    beta2: float = 0.999  # Adam beta2 for equivalent gradient momentum
    eps: float = 1e-8  # Epsilon for numerical stability
    delta: float = 1e-8  # Regularization for matrix pseudo-inverse
    use_rslora: bool = True  # RSLoRA scaling (alpha/sqrt(r)) vs standard (alpha/r)
    grad_scale: float = 1.0  # Post-adjustment gradient multiplier (counteracts 1/s² attenuation for RL)
    grad_floor: float = 1e-7  # Minimum gradient norm (prevents numerical collapse)


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
    reference_policy_kl_type: str = (
        "k3"  # "k1" (linear/unbiased), "k2" (squared), "k3" (exponential)
    )
    llds_coef: float = 0.05
    # entropy_coeff removed: -mean(logprob) has wrong gradient direction for entropy bonus.
    # neg_logprob_mean is logged as a metric only (no backprop). See Fix 3 notes.
    step_qualities: dict | None = None  # {step_num: [quality_names]} — domain-specific
    step_region_map: dict | None = (
        None  # {virtual_step: region_step} — maps steps without segments to a region
    )
    segmenter: str = "uniform"  # "uniform", "qwen3_xml", "label", or "module.path:function_name"
    label_segmenter: LabelSegmenterConfig | None = None  # Config for segmenter="label"
    # Fused logprobs: chunked lm_head projection saves ~2GB VRAM by not materializing full logit tensor
    use_fused_logprobs: bool = True
    fused_logprob_chunk_size: int = (
        256  # Tokens per chunk (lower = less memory, more kernel launches)
    )
    kl_input_clamp: float = (
        20.0  # Clamp KL input before computing penalty (prevents gradient explosion)
    )
    kl_output_clamp: float = 10.0  # Clamp KL output after computing penalty
    spo_filter_threshold: float = (
        0.001  # Skip sequences with max|advantage| below this (near-zero signal)
    )
    # Region-specific KL multipliers (THR-style, PLAN.md lines 798-802)
    # THINK=explore freely, FORMAT=lock structure, STEP=focus on quality
    kl_think_multiplier: float = 0.1  # Low KL for think tokens (explore)
    kl_format_multiplier: float = 2.0  # High KL for format tokens (exploit)
    kl_step_multiplier: float = 1.0  # Normal KL for step content
    # Dr.GRPO: remove length and std normalization biases (arXiv:2503.20783)
    # "grpo": standard GRPO (divides by horizon length + normalizes by std)
    # "dr_grpo": removes both normalizations (unbiased gradients)
    loss_type: str = "grpo"  # "grpo" or "dr_grpo"
    # GRPO-λ eligibility traces (ICLR 2026): per-token credit via λ-return approximation
    lambda_return: float = 0.0  # 0=off, 0.95=typical. Composes with VPRM step-level rewards.
    # Dynamic length control (Huawei): penalize length only when group accuracy is high
    length_penalty_coef: float = 0.0  # 0=off
    length_penalty_threshold: float = 0.5  # correctness ratio above which length penalty applies
    # Phase-aware frontier amplification: multiply advantages for steps blocking advancement.
    # Mastered steps get weight 1.0, frontier steps get (1 + frontier_amplification).
    # 0=off, 2.0=triple gradient on bottleneck steps (recommended for multi-step curriculum).
    frontier_amplification: float = 2.0
    # Scale factor for final token advantages before loss computation.
    # Compresses advantage signal to fit small model logit resolution.
    # 1.0 = no scaling, 0.1 = 10x compression (recommended for 1-3B models).
    advantage_scale: float = 1.0
    # Entropy-Regulated Importance Constraint (ERIC): dampen gradients on committed anchor tokens
    # Based on ERPO insight: low-entropy tokens are "committed" and shouldn't be pushed hard.
    # Combined with position-based causal weighting for proper anchor identification.
    # CRITICAL: Only dampens POSITIVE advantages (protecting correct anchors).
    # Negative advantages (confident mistakes) get FULL gradient for correction.
    attention_constrained_advantage: bool = False  # Enable entropy-regulated importance constraint
    attention_constraint_strength: float = (
        1.0  # Dampening multiplier (1.0 = standard, 2.0 = aggressive)
    )
    attention_constraint_mode: str = (
        "entropy_position"  # "entropy", "position", or "entropy_position" (recommended)
    )
    attention_position_decay: float | None = (
        None  # Position weight decay. None = auto from seq_len. Manual: 0.5=sqrt, 1.0=linear
    )
    attention_sample_layer: int = (
        -1
    )  # DEPRECATED: entropy-based constraint doesn't need attention layers

    def __post_init__(self):
        """Validate AlgorithmConfig after initialization."""
        # R3-CSM-005: Validate step_qualities values are lists
        if self.step_qualities is not None:
            for step, qualities in self.step_qualities.items():
                if not isinstance(qualities, list):
                    raise TypeError(
                        f"R3-CSM-005: step_qualities values must be lists. "
                        f"Step {step} has value {qualities!r} of type {type(qualities).__name__}. "
                        f"Use [{qualities!r}] if you meant a single-item list.",
                    )
                # R3-CSM-008: Validate no empty quality lists
                if not qualities:
                    raise ValueError(
                        f"R3-CSM-008: step_qualities step {step} has empty quality list. "
                        f"Empty quality list produces zero gradient. Remove the step or add qualities.",
                    )


@dataclass
class TrainingConfig:
    total_steps: int = 800
    lr: float = 5e-6
    warmup_steps: int = 10
    lr_scheduler: str = "cosine"
    save_freq: int = 50
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    empty_cache_between_microbatches: bool = True  # torch.cuda.empty_cache() after each micro-batch
    mastery_threshold: float = 0.8
    stagnation_timeout: int = 200
    plateau_window: int = 50
    plateau_threshold: float = 0.02
    embedding_lr_ratio: float = 0.1  # Embedding/lm_head LR = base_lr * this (lower prevents drift)
    micro_batch_seq_threshold: int = 2048  # Sequences >= this use micro_batch_size=1 (VRAM safety)
    kv_cache_flush_freq: int = 50  # Flush vLLM KV cache every N steps (0=never)
    quality_window_size: int = 20  # Rolling window for mastery score tracking
    seed: int = -1  # Random seed for training. -1 = time-based (non-reproducible), 0+ = fixed seed.
    gradient_probe_steps: int = (
        0  # If > 0, capture actual logit changes on physics tokens for first N steps
    )
    gradient_probe_prompt: str = (
        ""  # Fixed prompt for logit measurement (if empty, uses first batch prompt)
    )
    log_attention_patterns: bool = False  # Log attention entropy and collapse patterns at each step
    attention_log_freq: int = (
        10  # Log attention stats every N steps (if log_attention_patterns enabled)
    )


@dataclass
class LoggingConfig:
    mlflow_experiment: str = "qgre-training"
    completion_dir: str = "output/completions"
    checkpoint_dir: str = "output/checkpoints"
    log_freq: int = 5  # Print progress table every N steps
    grad_log_freq: int = 10  # Log per-quality gradient flow proxy every N steps


@dataclass
class QGREConfig:
    """Top-level config for the QGRE training engine."""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    vprm: VPRMConfig = field(default_factory=VPRMConfig)
    tutorial: TutorialConfig = field(default_factory=TutorialConfig)
    egrs: EGRSConfig = field(default_factory=EGRSConfig)
    lora_pro: LoRAProConfig = field(default_factory=LoRAProConfig)

    def validate(self) -> None:
        """Validate config for common misconfigurations. Called after from_yaml()."""
        import warnings

        if not self.model.pad_token or self.model.pad_token_id < 0:
            raise ValueError(
                "model.pad_token and model.pad_token_id are required.\n"
                "Set per-model in YAML. Qwen3 example: pad_token='<|fim_pad|>', pad_token_id=151662",
            )
        if not self.model.lora_target_modules:
            raise ValueError(
                "model.lora_target_modules must not be empty.\n"
                "Set per-model architecture. Qwen3/LLaMA: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]",
            )
        if not self.generation.stop_token_ids:
            warnings.warn(
                "generation.stop_token_ids is empty — generation may not terminate.\n"
                "Set per-model. Qwen3: [151643, 151645]",
                stacklevel=2,
            )
        # CFG-R2-1: Validate train_files is non-empty
        if not self.data.train_files:
            raise ValueError(
                "data.train_files is empty — no training data configured.\n"
                "Add at least one parquet file path to data.train_files in YAML.",
            )
        # CFG-R3-1: metadata_columns must be list[str], not string
        if not isinstance(self.data.metadata_columns, list):
            raise ValueError(
                f"data.metadata_columns must be list[str], got {type(self.data.metadata_columns).__name__}.\n"
                "String iterates as characters. Use YAML list syntax: [col1, col2]",
            )
        # CFG-R3-3: tier_order must be list[str], not string
        if self.data.tier_order is not None and not isinstance(self.data.tier_order, list):
            raise ValueError(
                f"data.tier_order must be list[str], got {type(self.data.tier_order).__name__}.\n"
                "String iterates as characters. Use YAML list syntax: [tier1, tier2, tier3]",
            )
        # CFG-R3-2: initial_tiers must be list[str], not string
        if self.data.initial_tiers is not None and not isinstance(self.data.initial_tiers, list):
            raise ValueError(
                f"data.initial_tiers must be list[str], got {type(self.data.initial_tiers).__name__}.\n"
                "String iterates as characters. Use YAML list syntax: [tier1, tier2]",
            )

    @staticmethod
    def from_yaml(path: str | Path) -> QGREConfig:
        """Load config from YAML file."""
        with open(path) as f:
            raw = yaml.safe_load(f)
        cfg = QGREConfig._from_dict(raw)
        cfg.validate()
        return cfg

    @staticmethod
    def _from_dict(d: dict) -> QGREConfig:
        def _pick(cls: type, raw: dict, section: str = "") -> dict:
            """Filter raw dict to only keys recognized by the dataclass. Warns on unknown keys."""
            known = set(cls.__dataclass_fields__)
            unknown = set(raw) - known
            if unknown:
                import warnings

                warnings.warn(f"Unknown {section} config keys (ignored): {unknown}", stacklevel=2)
            return {k: v for k, v in raw.items() if k in known}

        cfg = QGREConfig()
        if "model" in d:
            cfg.model = ModelConfig(**_pick(ModelConfig, d["model"], "model"))
        if "data" in d:
            data_fields = _pick(DataConfig, d["data"], "data")
            # CFG-R1-1: Ensure difficulty_schedule keys are ints with clear error message
            if (
                "difficulty_schedule" in data_fields
                and data_fields["difficulty_schedule"] is not None
            ):
                try:
                    data_fields["difficulty_schedule"] = {
                        int(k): list(v) for k, v in data_fields["difficulty_schedule"].items()
                    }
                except (ValueError, TypeError) as e:
                    raise ValueError(
                        f"data.difficulty_schedule keys must be integers (phase numbers). "
                        f"Got non-integer key: {e}",
                    ) from e
            cfg.data = DataConfig(**data_fields)
        if "generation" in d:
            cfg.generation = GenerationConfig(
                **_pick(GenerationConfig, d["generation"], "generation")
            )
        if "algorithm" in d:
            alg = d["algorithm"]
            spo = (
                SPOConfig(**_pick(SPOConfig, alg.get("spo", {}), "algorithm.spo"))
                if "spo" in alg
                else SPOConfig()
            )
            grpo = (
                GRPOConfig(**_pick(GRPOConfig, alg.get("grpo", {}), "algorithm.grpo"))
                if "grpo" in alg
                else GRPOConfig()
            )
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
            # CFG-R1-7: Ensure step_qualities keys are ints with clear error message
            if "step_qualities" in algo_fields and algo_fields["step_qualities"] is not None:
                try:
                    # R3-CSM-001: Check if value is string before converting to list
                    converted = {}
                    for k, v in algo_fields["step_qualities"].items():
                        if isinstance(v, str):
                            raise TypeError(
                                f"R3-CSM-001: step_qualities values must be lists, not strings. "
                                f"Got string '{v}' for step {k}. Use [{v!r}] instead of {v!r}.",
                            )
                        converted[int(k)] = list(v)
                    algo_fields["step_qualities"] = converted
                except (ValueError, TypeError) as e:
                    if "R3-CSM-001" in str(e):
                        raise
                    raise ValueError(
                        f"algorithm.step_qualities keys must be integers (step numbers). "
                        f"Got non-integer key: {e}",
                    ) from e
            # Ensure step_region_map keys/values are ints
            if "step_region_map" in algo_fields and algo_fields["step_region_map"] is not None:
                algo_fields["step_region_map"] = {
                    int(k): int(v) for k, v in algo_fields["step_region_map"].items()
                }
            cfg.algorithm = AlgorithmConfig(spo=spo, grpo=grpo, **algo_fields)
        if "training" in d:
            cfg.training = TrainingConfig(**_pick(TrainingConfig, d["training"], "training"))
        if "logging" in d:
            cfg.logging = LoggingConfig(**_pick(LoggingConfig, d["logging"], "logging"))
        if "vprm" in d:
            cfg.vprm = VPRMConfig(**_pick(VPRMConfig, d["vprm"], "vprm"))
        if "tutorial" in d:
            tut_raw = d["tutorial"]
            skill_tree = {}
            for key, skill_raw in tut_raw.get("skill_tree", {}).items():
                if isinstance(skill_raw, dict):
                    skill_tree[key] = SkillConfig(
                        **_pick(SkillConfig, skill_raw, f"tutorial.skill_tree.{key}")
                    )
                else:
                    raise ValueError(
                        f"tutorial.skill_tree.{key}: expected dict, got {type(skill_raw).__name__}. "
                        f"Check YAML formatting — skill entries must be mappings.",
                    )
            tut_top = {k: v for k, v in tut_raw.items() if k != "skill_tree"}
            tut_fields = _pick(TutorialConfig, tut_top, "tutorial")
            pmb = tut_fields.get("post_mastery_behavior", "review_only")
            if pmb not in _VALID_POST_MASTERY_BEHAVIORS:
                raise ValueError(
                    f"tutorial.post_mastery_behavior='{pmb}' is invalid. "
                    f"Must be one of: {sorted(_VALID_POST_MASTERY_BEHAVIORS)}",
                )
            cfg.tutorial = TutorialConfig(skill_tree=skill_tree, **tut_fields)
        if "egrs" in d:
            egrs_fields = _pick(EGRSConfig, d["egrs"], "egrs")
            # CSM-005: Validate hint_extractor_mapping is a dict
            if (
                "hint_extractor_mapping" in egrs_fields
                and egrs_fields["hint_extractor_mapping"] is not None
            ):
                mapping = egrs_fields["hint_extractor_mapping"]
                if not isinstance(mapping, dict):
                    raise TypeError(
                        f"CSM-005: egrs.hint_extractor_mapping expected dict, got {type(mapping).__name__}. "
                        "Check YAML formatting — must be a mapping, not a string.",
                    )
                # CR-003: Validate mapping values are strings (metadata field names)
                for span_id, field_name in mapping.items():
                    if not isinstance(field_name, str):
                        raise TypeError(
                            f"CR-003: egrs.hint_extractor_mapping['{span_id}'] must be str, "
                            f"got {type(field_name).__name__}. Expected metadata field name.",
                        )
            cfg.egrs = EGRSConfig(**egrs_fields)
        return cfg

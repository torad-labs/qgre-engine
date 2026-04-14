from __future__ import annotations

import contextlib
import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import torch
from torch import nn

from qgre.advantages import (
    QGREStepAdvantageEstimator,
    apply_egrs_matrix,
    build_phase_qualities,
    compute_span_correctness,
)
from qgre.attention_bonds import (
    compute_bond_strength,
    compute_entropy_importance,
    compute_entropy_importance_from_hidden,
    compute_normalized_entropy,
    select_attention_layer,
)
from qgre.checkpoint import (
    discover_latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from qgre.logging import log_step_metrics, log_training_params
from qgre.nemo_extracted.kl import masked_mean
from qgre.nemo_extracted.llds import compute_llds_loss
from qgre.nemo_extracted.logits import logprobs_from_logits
from qgre.nemo_extracted.loss_functions import ClippedPGLossFn
from qgre.segments import Segmenter, uniform_segmenter
from qgre.sync_state import SyncState
from qgre.types import (
    AlignedLossFrame,
    GameState,
    RewardResult,
    SampleData,
    TrainerState,
    TrainingContext,
    TrainingStep,
    WeightLoaderState,
)


if TYPE_CHECKING:
    from collections.abc import Callable

    from qgre.config import QGREConfig
    from qgre.data import PromptBatch, QGREDataLoader


class GenerationBackend(Protocol):
    """Abstract generation interface — shields trainer from Unsloth internals."""

    # Core generation
    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs) -> Any: ...

    # Mode switching for training/inference
    def set_training_mode(self) -> None: ...
    def set_inference_mode(self) -> None: ...

    # Weight synchronization components
    @property
    def weight_exporter(self) -> Any: ...
    @property
    def weight_loader(self) -> Any: ...

    # Internal model access (for fused logprobs, etc.)
    @property
    def model(self) -> Any: ...
    @property
    def tokenizer(self) -> Any: ...

    # Unsloth-specific (optional - may not exist on all backends)
    @property
    def _FastLanguageModel(self) -> Any: ...  # noqa: N802 (matches Unsloth library naming)


class QGRETrainer:
    """Single-GPU QGRE training engine.

    Orchestrates: generate → score → advantages → loss → backward → update.
    No Ray, no verl. Direct function calls.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        reward_fn: Callable[..., RewardResult],
        config: QGREConfig,
        generation_backend: GenerationBackend | None = None,
        game_state: GameState | None = None,
        step_qualities: dict[int, list[str]] | None = None,
        segmenter: Segmenter | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.config = config
        self.generation_backend = generation_backend
        # Single source of truth for all sync-related state. Injected into WeightBus,
        # WeightLoader, and apply_lora_dropout so they share lifecycle, dropout, and
        # initialization tracking.
        self.sync_state = SyncState()
        self.game_state = game_state or GameState(
            mastery_threshold=config.training.mastery_threshold,
            stagnation_timeout=config.training.stagnation_timeout,
            plateau_window=config.training.plateau_window,
            plateau_threshold=config.training.plateau_threshold,
            quality_window_size=config.training.quality_window_size,
        )
        self._dataloader = None  # Set in train()

        # 2D curriculum setup
        data_cfg = config.data
        self._tier_order = data_cfg.tier_order
        self._difficulty_column = data_cfg.difficulty_column
        self._tier_advance_phase = data_cfg.tier_advance_quality_phase
        self._tier_advance_threshold = data_cfg.tier_advance_threshold
        if self._tier_order:
            # Default initial_tiers to first tier if not set
            if data_cfg.initial_tiers is None:
                data_cfg.initial_tiers = [self._tier_order[0]]
            # Type check: ensure lists, not strings
            if not isinstance(data_cfg.initial_tiers, list):
                raise ValueError(
                    f"initial_tiers must be list[str], got {type(data_cfg.initial_tiers).__name__}"
                )
            if not isinstance(self._tier_order, list):
                raise ValueError(
                    f"tier_order must be list[str], got {type(self._tier_order).__name__}"
                )
            self.game_state.active_tiers = list(data_cfg.initial_tiers)
            for t in data_cfg.initial_tiers:
                self.game_state.tier_phases.setdefault(t, 1)
                self.game_state.tier_steps_at_phase_start.setdefault(t, 0)

        # Step qualities and phase mapping — configurable per domain
        # step_qualities: from constructor arg, config YAML, or error
        sq = step_qualities or config.algorithm.step_qualities
        if sq is None:
            raise ValueError(
                "step_qualities is required. Pass to QGRETrainer constructor or set in config YAML:\n"
                "  algorithm:\n"
                "    step_qualities:\n"
                "      1: [q_format]\n"
                "      2: [q_accuracy]",
            )
        self.step_qualities = sq
        self.phase_qualities = build_phase_qualities(sq)
        self._sq_validated = False  # Validate step_qualities keys against first reward result

        # Algorithm setup
        alg = config.algorithm
        mode = alg.mode
        spo_lr = alg.spo.lr if mode == "spo" else 0.1

        # Resolve segmenter: explicit arg > config > uniform default
        if segmenter is None and alg.segmenter != "uniform":
            if alg.segmenter == "qwen3_xml":
                from qgre.segments import qwen3_xml_segmenter

                segmenter = qwen3_xml_segmenter
            elif alg.segmenter == "hif_json":
                from qgre.segments import make_hif_json_segmenter

                segmenter = make_hif_json_segmenter(tokenizer)
            elif alg.segmenter == "hamiltonian":
                from qgre.segments import make_hamiltonian_segmenter

                segmenter = make_hamiltonian_segmenter(tokenizer)
            elif alg.segmenter == "label":
                from qgre.segments import make_label_segmenter

                if alg.label_segmenter is None or not alg.label_segmenter.patterns:
                    raise ValueError(
                        "segmenter='label' requires algorithm.label_segmenter.patterns in config"
                    )
                segmenter = make_label_segmenter(tokenizer, alg.label_segmenter)
            elif ":" in alg.segmenter:
                import importlib

                mod_path, fn_name = alg.segmenter.rsplit(":", 1)
                segmenter = getattr(importlib.import_module(mod_path), fn_name)

        spo_cfg = alg.spo
        # Target-aware aspiration gap uses mastery threshold as default target
        aspiration_target = (
            spo_cfg.aspiration_target
            if spo_cfg.aspiration_target > 0
            else config.training.mastery_threshold
        )
        self.advantage_estimator = QGREStepAdvantageEstimator(
            lr=spo_lr,
            mode=mode,
            step_qualities=sq,
            segmenter=segmenter or uniform_segmenter,
            normalize_advantages=alg.loss_type != "dr_grpo",
            filter_groups=alg.grpo.filter_groups,
            step_region_map=alg.step_region_map,
            frontier_amplification=alg.frontier_amplification,
            var_aware=spo_cfg.var_aware,
            var_threshold=spo_cfg.var_threshold,
            var_lr=spo_cfg.var_lr,
            min_var_ratio=spo_cfg.min_var_ratio,
        )
        self.advantage_estimator._aspiration_beta = spo_cfg.aspiration_beta
        self.advantage_estimator._aspiration_target = aspiration_target
        self.advantage_estimator._advantage_scale = config.algorithm.advantage_scale

        # VPRM critic — per-region per-dimension learned baseline
        self.vprm_critic = None
        self.vprm_optimizer = None
        self._vprm_initialized = False
        self._vprm_config = config.vprm
        self._vprm_sq = sq

        # EGRS hint registry — tracks (prompt, span) pairs needing hints
        self.hint_registry = None
        self.hint_extractor = None
        if config.egrs.enabled and config.egrs.hint_enabled:
            from qgre.hints import (
                HintRegistry,
                make_generic_hint_extractor,
                make_hamiltonian_hint_extractor,
            )

            self.hint_registry = HintRegistry(
                mastery_threshold=config.egrs.mastery_threshold,
            )
            # Create hint extractor based on config
            if config.egrs.hint_extractor == "hamiltonian":
                self.hint_extractor = make_hamiltonian_hint_extractor()
                logging.getLogger(__name__).info(
                    "EGRS: Hint registry initialized with Hamiltonian extractor. "
                    "Q4 hints will use T_expr, V_expr, H_expr from metadata.",
                )
            elif config.egrs.hint_extractor == "generic" and config.egrs.hint_extractor_mapping:
                self.hint_extractor = make_generic_hint_extractor(
                    config.egrs.hint_extractor_mapping
                )
                logging.getLogger(__name__).info(
                    f"EGRS: Hint registry initialized with generic extractor. "
                    f"Mapping: {config.egrs.hint_extractor_mapping}",
                )
            elif config.egrs.hint_extractor == "generic" and not config.egrs.hint_extractor_mapping:
                # R3-CSM-007: Warn if generic extractor requested but mapping is empty
                logging.getLogger(__name__).warning(
                    "R3-CSM-007: hint_extractor='generic' but hint_extractor_mapping is empty/None. "
                    "Generic extractor requires mapping. Falling back to no extractor (generic hints only).",
                )
                self.hint_extractor = None
            else:
                logging.getLogger(__name__).info(
                    "EGRS: Hint registry initialized without extractor. "
                    "Q4 hints will be generic ('Focus on STEP_N').",
                )

        # Loss function (NeMo RL extracted)
        self.loss_fn = ClippedPGLossFn(
            {
                "reference_policy_kl_penalty": alg.kl_cov_ratio
                if alg.loss_mode == "kl_cov"
                else 0.0,
                "reference_policy_kl_type": alg.reference_policy_kl_type,
                "kl_input_clamp_value": alg.kl_input_clamp,
                "kl_output_clamp_value": alg.kl_output_clamp,
                "ratio_clip_min": alg.clip_ratio_low,
                "ratio_clip_max": alg.clip_ratio_high,
                "ratio_clip_c": None,
                "use_on_policy_kl_approximation": True,
                "use_importance_sampling_correction": False,
                "truncated_importance_sampling_ratio": None,
                "token_level_loss": True,  # nosec B105 - not a password
                "force_on_policy_ratio": False,
                "remove_length_normalization": alg.loss_type == "dr_grpo",
                "lambda_return": alg.lambda_return,
            }
        )

        # LLDS requires stored generation-time logprobs to be meaningful.
        # Without them, old_logprob == curr_logprob and all LLDS gates return zero.
        # This flag is set dynamically per step based on whether generation_logprobs is provided.
        self._has_stored_logprobs = False

        # Fused logprobs: chunked lm_head projection saves ~2GB by not materializing
        # full [seq, vocab] logit tensor. Uses torch.checkpoint per chunk to prevent
        # autograd from storing all chunks for backward.
        self._use_fused_logprobs = config.algorithm.use_fused_logprobs
        self._fused_chunk_size = config.algorithm.fused_logprob_chunk_size
        self._fused_validated = False  # One-time validation on first use

        # Triton fused logprobs: single GPU launch replaces chunked path.
        # Full Triton forward + backward — zero [seq, vocab] allocation, self-
        # consistent gradients.
        #
        # Triton requires CUDA. Disable on CPU-only environments (used for
        # tests and offline smoke checks) without raising, since there's no
        # GPU for the kernels to run on. On CUDA-capable machines, missing
        # triton install is a hard error: don't silently fall back to cuBLAS,
        # because that changes numerics mid-run and hides configuration bugs.
        self._use_triton_logprobs = config.algorithm.use_triton_logprobs
        if self._use_triton_logprobs:
            from qgre.triton_logprobs import HAS_TRITON as _HAS_TRITON

            if not torch.cuda.is_available():
                logging.getLogger(__name__).info(
                    "CUDA unavailable — disabling Triton logprobs and using "
                    "the chunked cuBLAS path (CPU-only environment).",
                )
                self._use_triton_logprobs = False
            elif not _HAS_TRITON:
                raise RuntimeError(
                    "algorithm.use_triton_logprobs=true but triton is not "
                    "installed. Install triton (pip install triton) or set "
                    "algorithm.use_triton_logprobs=false to use the chunked "
                    "cuBLAS path. No silent fallback — the numerics differ.",
                )
        self._triton_validated = False

        # Advantage observability — populated each batch at the SPO filter
        # point, then merged into the metrics dict on both the normal and
        # filtered-early-return branches. Zero-valued defaults so the metrics
        # schema is consistent on every step (dashboards/CSV exports expect
        # the same columns every row).
        self._last_advantage_stats: dict[str, float] = {
            "adv/abs_max": 0.0,
            "adv/abs_mean": 0.0,
            "adv/min": 0.0,
            "adv/max": 0.0,
            "adv/nonzero_count": 0.0,
            "adv/nonzero_fraction": 0.0,
        }

        # Completion logger (lazily initialized on first log to handle checkpoint resume)
        self._completion_logger_path = config.logging.completion_dir
        self._completion_logger = None

        # Training state
        self.global_step = 0
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: Any = None
        self._accumulated_loss = 0.0
        self._accumulation_count = 0  # CP2-004: Track actual accumulation steps
        self._accumulated_samples = 0  # TL-R2-04: Track samples for accurate loss averaging

        # SPO filter monitoring — track drop rate to detect data starvation
        self._spo_filter_stats = {"total": 0, "passed": 0, "dropped": 0, "warned": False}
        self._spo_filter_idx: list[int] | None = None

        # Checkpoint resume state — initialized here to avoid fragile hasattr checks
        self._resumed_mid_accumulation = False
        self._needs_weight_sync = False

        # Gradient probe — measure actual logit changes on physics tokens for advantage_scale calibration
        self._gradient_probe_steps = config.training.gradient_probe_steps
        self._gradient_probe_log = []
        self._gradient_probe_prompt_ids = None  # Fixed prompt for measurement
        self._gradient_probe_physics_tokens = None  # Physics token IDs to track

        # Attention pattern monitoring — detect laminar→turbulent transitions
        self._log_attention = config.training.log_attention_patterns
        self._attention_log_freq = config.training.attention_log_freq
        self._attention_log = []  # Per-step attention statistics

        # Training context — device, dtype, step counter (created once, reused across training)
        # Device inference: use model's actual device, fallback to cuda only if CUDA available
        model_devices = [p.device for p in model.parameters()]
        if model_devices:
            # Use first non-CPU device if available, otherwise use the model's actual device
            _device = next((d for d in model_devices if d.type != "cpu"), model_devices[0])
        else:
            _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ctx = TrainingContext.from_config(config, device=str(_device))

        # LoRA-Pro gradient adjustment — initialized lazily after optimizer setup
        self._lora_pro_config = config.lora_pro
        self._lora_pro_adjuster = None

    @property
    def completion_logger(self):
        """Lazy completion logger — creates on first access, handles resume."""
        if self._completion_logger is None:
            from qgre.logging import CompletionLogger

            self._completion_logger = CompletionLogger(self._completion_logger_path)
        return self._completion_logger

    def _init_vprm_critic(self, hidden_dim: int, device: str | torch.device):
        """Lazily initialize VPRM critic once hidden_dim is known from first forward pass."""
        if self._vprm_initialized:
            return
        from qgre.critic import VPRMCritic

        self.vprm_critic = VPRMCritic(
            hidden_dim=hidden_dim,
            step_qualities=self._vprm_sq,
            intermediate_dim=self._vprm_config.intermediate_dim,
            clip_advantage=self._vprm_config.clip_advantage,
        ).to(device)
        self.vprm_optimizer = torch.optim.Adam(
            [p for p in self.vprm_critic.parameters() if p.requires_grad],
            lr=self._vprm_config.lr,
        )
        self._vprm_initialized = True

    def setup_optimizer(self):
        """Create optimizer and LR scheduler from config.

        Uses AdamW8bit from bitsandbytes for ~4x memory savings on optimizer states.
        Falls back to regular AdamW if bitsandbytes not available.
        """
        # Split params: embedding/lm_head modules_to_save get reduced LR
        # (Unsloth recommendation for modules_to_save to prevent drift)
        modules_to_save_names = tuple(self.config.model.modules_to_save)
        embedding_params = []
        other_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if any(mod in name for mod in modules_to_save_names):
                embedding_params.append(param)
            else:
                other_params.append(param)

        base_lr = self.config.training.lr
        embed_lr = base_lr * self.config.training.embedding_lr_ratio
        param_groups = [
            {"params": other_params, "lr": base_lr},
            {"params": embedding_params, "lr": embed_lr},
        ]

        if embedding_params:
            print(
                f"Optimizer: {len(other_params)} params at lr={base_lr}, "
                f"{len(embedding_params)} embedding params at lr={embed_lr}"
            )
        else:
            import warnings

            warnings.warn(
                "No embedding/lm_head params with requires_grad=True found. "
                "If using modules_to_save=['lm_head'], this means PEFT adapter init failed.",
                stacklevel=2,
            )
            # Remove empty param group — optimizer doesn't like empty groups
            param_groups = [{"params": other_params, "lr": base_lr}]

        # Validate model has trainable parameters before setting up optimizer
        trainable_count = sum(1 for p in self.model.parameters() if p.requires_grad)
        if trainable_count == 0:
            raise RuntimeError(
                "Model has no trainable parameters. Cannot start training. "
                "Check PEFT adapter configuration and requires_grad settings.",
            )

        # AdamW8bit saves ~4x memory on optimizer states (PLAN.md line 323)
        use_8bit = False
        try:
            device = next(
                (p.device for p in self.model.parameters() if p.device.type != "cpu"),
                next(self.model.parameters()).device,
            )
            if device.type == "cuda":
                from bitsandbytes.optim import AdamW8bit

                self.optimizer = AdamW8bit(param_groups)
                use_8bit = True
        except ImportError:
            pass  # bitsandbytes not installed

        if not use_8bit:
            self.optimizer = torch.optim.AdamW(param_groups)

        cfg = self.config.training
        main_scheduler = None
        if cfg.lr_scheduler == "cosine":
            # eta_min = base_lr * 0.1. This means embedding group (base_lr/10) has
            # flat LR — intentional, embeddings need stable learning rate.
            main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,  # type: ignore[arg-type]
                T_max=cfg.total_steps,
                eta_min=cfg.lr * 0.1,
            )
        elif cfg.lr_scheduler == "linear":
            main_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,  # type: ignore[arg-type]
                start_factor=1.0,
                end_factor=0.1,
                total_iters=cfg.total_steps,
            )

        if main_scheduler is not None and cfg.warmup_steps > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,  # type: ignore[arg-type]
                start_factor=0.01,
                end_factor=1.0,
                total_iters=cfg.warmup_steps,
            )
            self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                self.optimizer,  # type: ignore[arg-type]
                schedulers=[warmup, main_scheduler],
                milestones=[cfg.warmup_steps],
            )
        else:
            self.scheduler = main_scheduler

        # Initialize LoRA-Pro gradient adjuster if enabled
        if self._lora_pro_config.enabled:
            from qgre.lora_pro import LoRAProAdjuster, LoRAProConfig

            lora_pro_cfg = LoRAProConfig(
                enabled=True,
                beta1=self._lora_pro_config.beta1,
                beta2=self._lora_pro_config.beta2,
                eps=self._lora_pro_config.eps,
                delta=self._lora_pro_config.delta,
                use_rslora=self._lora_pro_config.use_rslora,
            )
            self._lora_pro_adjuster = LoRAProAdjuster(
                model=self.model,
                lora_rank=self.config.model.lora_rank,
                lora_alpha=self.config.model.lora_alpha,
                config=lora_pro_cfg,
            )
            logging.getLogger(__name__).info(
                f"LoRA-Pro: Enabled with rank={self.config.model.lora_rank}, "
                f"alpha={self.config.model.lora_alpha}, "
                f"rslora={self._lora_pro_config.use_rslora}"
            )

    def compute_response_mask(
        self,
        input_ids: torch.Tensor,
        prompt_lengths: list[int],
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """Compute response mask: 1 for response tokens, 0 for prompt + padding after EOS."""
        batch_size, seq_len = input_ids.shape
        mask = torch.zeros(batch_size, seq_len, dtype=torch.float32, device=input_ids.device)

        for i in range(batch_size):
            start = prompt_lengths[i]
            if eos_token_id is not None:
                eos_positions = (input_ids[i, start:] == eos_token_id).nonzero(as_tuple=True)[0]
                end = start + eos_positions[0].item() + 1 if len(eos_positions) > 0 else seq_len
            else:
                end = seq_len
            mask[i, start:end] = 1.0

        return mask

    def step(
        self,
        batch: PromptBatch,
        completions: list[list[int]],
        reward_results: list[RewardResult],
        generation_logprobs: list[list[float]] | None = None,
        completion_texts: list[str] | None = None,  # Original texts for span validation
    ) -> dict[str, float]:
        """Execute one training step given pre-generated completions and rewards.

        Phase is engine-managed via GameState. The engine:
        1. Uses game_state.phase to determine active qualities
        2. Computes per-step advantages via segmenter + step_qualities
        3. Records per-step mastery scores to GameState
        4. Checks for phase advancement after each step

        Returns metrics dict.
        """
        assert self.optimizer is not None, "Call setup_optimizer() first"

        # Reset SPO filter index mapping at start of each step
        self._spo_filter_idx = None

        if not self._sq_validated and reward_results:
            if not self.phase_qualities:
                raise ValueError(
                    "step_qualities must not be empty. "
                    "Provide at least one phase with quality keys.",
                )
            # Check for phases with empty quality lists — they contribute no keys and
            # produce zero gradients for any sample in that phase.
            empty_phases = [phase for phase, keys in self.step_qualities.items() if not keys]
            if empty_phases:
                raise ValueError(
                    f"step_qualities phases {sorted(empty_phases)} have empty quality lists. "
                    "Each phase must declare at least one quality key. "
                    "An empty list produces zero gradients for all samples in that phase.",
                )
            # Validate against ALL reward_results — use intersection of score keys so that
            # any sample missing a required key is caught (not just sample[0]).
            available_keys: set[str] = set(reward_results[0].scores.keys())
            for idx, rr in enumerate(reward_results[1:], start=1):
                available_keys &= set(rr.scores.keys())
            # Validate ALL phase keys — a future phase referencing an absent key
            # causes silent zero learning signal, not just the starting phase.
            all_sq_keys: set[str] = set()
            for keys in self.phase_qualities.values():
                all_sq_keys.update(keys)
            missing = all_sq_keys - available_keys
            if missing:
                raise ValueError(
                    f"step_qualities keys {sorted(missing)} not found in reward_fn output "
                    f"for all samples (checked {len(reward_results)} samples). "
                    f"Keys present in ALL samples: {sorted(available_keys)}. "
                    "This would cause silent zero learning signal.",
                )
            self._sq_validated = True

        # LLDS activates when real generation-time logprobs are available
        self._has_stored_logprobs = generation_logprobs is not None

        # Per-prompt active qualities based on each prompt's tier and that tier's quality phase
        active_qualities = []
        for i in range(len(reward_results)):
            meta = batch.metadata[i] if i < len(batch.metadata) else {}
            tier = self._get_prompt_tier(meta)
            tier_phase = self.game_state.tier_phases.get(tier, 1)
            fallback = (
                max(p for p in self.phase_qualities if p <= tier_phase)
                if any(p <= tier_phase for p in self.phase_qualities)
                else min(self.phase_qualities)
            )
            active_qualities.append(
                self.phase_qualities.get(tier_phase, self.phase_qualities[fallback])
            )

        # Compute frontier steps — steps blocking phase advancement (below mastery threshold).
        # These get amplified advantages to focus gradient on the bottleneck.
        frontier_steps = set()
        max_phase = max(self.step_qualities.keys())
        for tier in self.game_state.active_tiers:
            current_phase = self.game_state.tier_phases.get(tier, 1)
            if current_phase <= max_phase:
                mastery = self.game_state.get_tier_step_mastery(tier, current_phase)
                if mastery < self.config.training.mastery_threshold:
                    frontier_steps.add(current_phase)

        # Build prompt contexts once — used by advantages, VPRM, tutorial recording
        batch_contexts = self.game_state.build_prompt_contexts(
            prompt_ids=batch.prompt_ids,
            metadata=batch.metadata,
            difficulty_column=self._difficulty_column,
            active_tiers=set(self.game_state.active_tiers),
        )

        # Compute per-token advantages — span-based (if scored_spans populated) or region-based (legacy)
        use_spans = any(rr.scored_spans for rr in reward_results)
        batch_token_masks: list[dict[str, torch.Tensor]] = []  # Available for per-quality loss

        # Debug: log when scored_spans are missing entirely
        if not use_spans:
            logging.getLogger(__name__).warning(
                f"Step {self.global_step}: No scored_spans in batch. "
                f"scored_spans counts: {[len(rr.scored_spans) if rr.scored_spans else 0 for rr in reward_results]}",
            )
        if use_spans:
            from qgre.spans import build_char_to_token_map, scored_spans_to_token_masks

            # Build per-sample token masks from scored_spans
            batch_token_masks: list[dict[str, torch.Tensor]] = []
            for i, rr in enumerate(reward_results):
                if rr.scored_spans:
                    # SPAN VALIDATION: Compare decoded text with reward_fn text
                    if completion_texts is not None and i < len(completion_texts):
                        our_decode = self.tokenizer.decode(completions[i], skip_special_tokens=True)
                        reward_fn_text = completion_texts[i]
                        if our_decode != reward_fn_text:
                            _log = logging.getLogger(__name__)
                            _log.error(
                                f"SPAN VALIDATION FAILED sample {i}: decoded text != reward_fn text. "
                                f"Decoded len={len(our_decode)}, reward_fn len={len(reward_fn_text)}. "
                                f"Character positions will be WRONG. Falling back to segmenter.",
                            )
                            # Find first difference for debugging
                            for j, (a, b) in enumerate(
                                zip(our_decode, reward_fn_text, strict=False)
                            ):
                                if a != b:
                                    _log.error(
                                        f"  First diff at char {j}: decoded='{a}' vs reward_fn='{b}'"
                                    )
                                    break
                            batch_token_masks.append({})  # Skip spans for this sample
                            continue

                    # Build char→token map directly from original token_ids (no re-encoding)
                    char_map = build_char_to_token_map(completions[i], self.tokenizer)
                    if char_map is not None:
                        masks = scored_spans_to_token_masks(
                            rr.scored_spans, char_map, len(completions[i]), self.ctx
                        )

                        # SPAN CONTENT LOGGING: Write to separate file for verification
                        # Check output/hamiltonian/spans.log to verify spans are correct
                        if self.global_step % self.config.logging.log_freq == 0 and i == 0:
                            text = (
                                completion_texts[i]
                                if completion_texts and i < len(completion_texts)
                                else None
                            )
                            if text:
                                span_log_path = (
                                    Path(self.config.logging.checkpoint_dir).parent / "spans.log"
                                )
                                with open(span_log_path, "a") as f:
                                    f.write(f"\n=== Step {self.global_step} Sample {i} ===\n")
                                    for q_name, spans in rr.scored_spans.items():
                                        for span_idx, (cs, ce) in enumerate(spans):
                                            span_text = (
                                                text[cs:ce] if cs < len(text) else "<OUT_OF_BOUNDS>"
                                            )
                                            tok_indices = sorted(
                                                {
                                                    char_map[c]
                                                    for c in range(
                                                        max(0, cs), min(ce, len(char_map))
                                                    )
                                                    if char_map[c] >= 0
                                                }
                                            )
                                            tok_range = (
                                                f"{tok_indices[0]}-{tok_indices[-1]}"
                                                if tok_indices
                                                else "NONE"
                                            )
                                            display_text = span_text[:80].replace("\n", "\\n")
                                            f.write(
                                                f"  {q_name}[{span_idx}]: chars({cs},{ce}) → toks({tok_range}) '{display_text}'\n"
                                            )
                    else:
                        logging.getLogger(__name__).error(
                            f"char_to_token_map returned None for sample {i} — cannot map scored_spans to tokens. "
                            "Falling back to uniform advantage (no span signal). Check tokenizer decode consistency.",
                        )
                        masks = {}  # Mapping failed — fall back to segmenter
                else:
                    masks = {}
                batch_token_masks.append(masks)

            # A4: Validate all samples have same mask status (all empty or all populated)
            # Mixed batches: some samples have span masks, some don't. Keep
            # the working ones — destroying 3 good span mappings because 1
            # sample failed is worse than having heterogeneous advantage modes.
            # Samples without spans get zero advantage for this step (empty mask).
            non_empty_masks = sum(1 for m in batch_token_masks if m)
            if non_empty_masks == 0:
                logging.getLogger(__name__).warning(
                    f"All span mappings failed: {len(batch_token_masks)} samples, "
                    f"scored_spans present: {[bool(rr.scored_spans) for rr in reward_results]}",
                )
            if any(m for m in batch_token_masks):
                token_advantages, _quality_metrics = (
                    self.advantage_estimator.compute_advantages_with_spans(
                        batch_prompt_ids=batch.prompt_ids,
                        batch_token_ids=completions,
                        batch_reward_results=reward_results,
                        batch_active_qualities=active_qualities,
                        batch_token_masks=batch_token_masks,
                        group_size=self.config.algorithm.grpo.n
                        if self.config.algorithm.mode == "grpo"
                        else None,
                        frontier_steps=frontier_steps,
                        batch_contexts=batch_contexts,
                        ctx=self.ctx,
                    )
                )
                # Still run segmenter for KL region weights (THINK/FORMAT multipliers).
                # VPRM critic now uses spans directly via compute_advantages_from_spans.
                batch_regions = [self.advantage_estimator.segmenter(c) for c in completions]
            else:
                use_spans = False  # All mappings failed, fall back to segmenter

        if not use_spans:
            token_advantages, batch_regions = self.advantage_estimator.compute_advantages(
                batch_prompt_ids=batch.prompt_ids,
                batch_token_ids=completions,
                batch_reward_results=reward_results,
                batch_active_qualities=active_qualities,
                group_size=self.config.algorithm.grpo.n
                if self.config.algorithm.mode == "grpo"
                else None,
                frontier_steps=frontier_steps,
                batch_contexts=batch_contexts,
                ctx=self.ctx,
            )

        # Build full sequences on model device.
        # modules_to_save offloads original_module to CPU — skip CPU params for device detection.
        device = next(
            (p.device for p in self.model.parameters() if p.device.type != "cpu"),
            next(self.model.parameters()).device,
        )
        max_comp_len = max(len(adv) for adv in token_advantages)  # type: ignore[possibly-undefined]

        padded_advs = torch.zeros(len(completions), max_comp_len, device=device)
        for i, adv in enumerate(token_advantages):  # type: ignore[possibly-undefined]
            padded_advs[i, : len(adv)] = adv.to(device)

        # Build per-sample KL region weights from segmenter regions (THR-style, PLAN.md lines 798-802)
        # These go into SampleData for clean SPO filter reindexing.
        # Gate: skip entirely when KL is disabled (default config).
        alg = self.config.algorithm
        per_sample_kl_weights: list[torch.Tensor | None] = []
        if alg.kl_cov_ratio > 0 and alg.loss_mode == "kl_cov":
            region_map = {"THINK": alg.kl_think_multiplier, "FORMAT": alg.kl_format_multiplier}
            for i, regions in enumerate(batch_regions):  # type: ignore[possibly-undefined]
                kl_weights = torch.ones(len(regions), device=device)
                for t, region in enumerate(regions):
                    if region in region_map:
                        kl_weights[t] = region_map[region]
                    elif region.startswith("STEP_"):
                        kl_weights[t] = alg.kl_step_multiplier
                per_sample_kl_weights.append(kl_weights)
        else:
            per_sample_kl_weights = [None] * len(completions)

        # Validate generation-time logprobs for LLDS (when available from backend)
        # Store per-sample tensors in SampleData; padding happens after SPO filter.
        per_sample_gen_logprobs: list[torch.Tensor | None] = []
        if generation_logprobs is not None:
            valid = True
            for i, (lps, comp) in enumerate(zip(generation_logprobs, completions, strict=False)):
                if len(lps) != len(comp):
                    import warnings

                    warnings.warn(
                        f"T3: Logprobs length ({len(lps)}) != completion length ({len(comp)}) "
                        f"for sample {i}. LLDS is disabled for this step due to length mismatch. "
                        "Check generation backend configuration.",
                        stacklevel=2,
                    )
                    valid = False
                    break
            if valid:
                for lps in generation_logprobs:
                    per_sample_gen_logprobs.append(
                        torch.tensor(lps, dtype=torch.float32, device=device)
                    )
            else:
                self._has_stored_logprobs = False
                per_sample_gen_logprobs = [None] * len(completions)
        else:
            per_sample_gen_logprobs = [None] * len(completions)

        # Build samples list — single source of truth for per-sample data.
        # SPO filter reindexing becomes one line instead of 6 separate list operations.
        # All downstream code accesses samples[i].field instead of mapping indices.
        samples: list[SampleData] = [
            SampleData(
                completion=completions[i],
                reward_result=reward_results[i],
                context=batch_contexts[i],
                active_qualities=active_qualities[i],
                regions=batch_regions[i] if batch_regions else None,  # type: ignore[reportPossiblyUnboundVariable]
                token_masks=batch_token_masks[i]
                if batch_token_masks and i < len(batch_token_masks)
                else None,
                kl_region_weights=per_sample_kl_weights[i],
                gen_logprobs=per_sample_gen_logprobs[i],
            )
            for i in range(len(completions))
        ]

        # Build padded batch tensors for model input
        comp_tensor = torch.zeros(len(completions), max_comp_len, dtype=torch.long, device=device)
        for i, c in enumerate(completions):
            comp_tensor[i, : len(c)] = torch.tensor(c, dtype=torch.long, device=device)

        # Attention mask: 1 for real tokens, 0 for padding beyond completion length.
        # Uses actual completion lengths, NOT token values — token ID 0 is a valid
        # token in many vocabularies (e.g. Qwen3). Using (comp_tensor != 0) would
        # silently mask real tokens, corrupting hidden states and gradients.
        comp_attention_mask = torch.zeros_like(comp_tensor, dtype=torch.long)
        for i, c in enumerate(completions):
            comp_attention_mask[i, : len(c)] = 1

        # Bundle all step-level state into TrainingStep — eliminates reindexing bugs.
        # SPO filter becomes step.filter(idx) which atomically reindexes ALL fields.
        step = TrainingStep(
            samples=samples,
            reward_results=reward_results,
            active_qualities=active_qualities,
            batch_regions=batch_regions,  # type: ignore[reportPossiblyUnboundVariable, arg-type]
            batch_contexts=batch_contexts,
            completions=completions,
            padded_advs=padded_advs,
            comp_tensor=comp_tensor,
            comp_attention_mask=comp_attention_mask,
        )

        # Advantage observability — diagnoses "training stuck" by distinguishing
        # "reward is flat" from "filter too aggressive" from "aspiration zeroing".
        adv_abs = step.padded_advs.abs()
        adv_nonzero_mask = adv_abs > self.config.algorithm.spo_filter_threshold
        has_elements = adv_abs.numel() > 0
        nonzero_count = int(adv_nonzero_mask.sum().item())
        if has_elements:
            advantage_stats: dict[str, float] = {
                "adv/abs_max": float(adv_abs.max().item()),
                "adv/abs_mean": float(adv_abs.mean().item()),
                "adv/min": float(step.padded_advs.min().item()),
                "adv/max": float(step.padded_advs.max().item()),
                "adv/nonzero_count": float(nonzero_count),
                "adv/nonzero_fraction": nonzero_count / adv_abs.numel(),
            }
        else:
            advantage_stats = dict(self._last_advantage_stats)  # keep schema consistent
        self._last_advantage_stats = advantage_stats

        # SPO low-advantage filter: skip sequences with near-zero signal (PLAN.md lines 658-671)
        if self.config.algorithm.mode == "spo":
            useful = adv_nonzero_mask.any(dim=-1)
            # Track filter stats for data starvation detection
            batch_total = len(step)
            batch_passed = useful.sum().item()
            self._spo_filter_stats["total"] += batch_total
            self._spo_filter_stats["passed"] += batch_passed
            self._spo_filter_stats["dropped"] += batch_total - batch_passed
            # Warn if drop rate exceeds 50% after 100+ samples
            if not self._spo_filter_stats["warned"] and self._spo_filter_stats["total"] >= 100:
                drop_rate = self._spo_filter_stats["dropped"] / self._spo_filter_stats["total"]
                if drop_rate > 0.5:
                    import warnings

                    warnings.warn(
                        f"SPO filter drop rate {drop_rate:.1%} exceeds 50% "
                        f"(dropped {self._spo_filter_stats['dropped']}/{self._spo_filter_stats['total']}). "
                        "Model may be stuck — reward function returns near-zero signal for most samples. "
                        "Check reward function thresholds and phase gating.",
                        stacklevel=2,
                    )
                    self._spo_filter_stats["warned"] = True
            if useful.sum() == 0:
                # All advantages near-zero — skip backward pass but still record mastery + log completions
                metrics = {
                    "loss": 0.0,
                    "reward/mean": sum(rr.reward for rr in step.reward_results) / len(step),
                    "global_step": self.global_step,
                    "phase": self.game_state.phase,
                    "skipped": True,
                    **advantage_stats,  # Preserve advantage observability on filtered steps
                }
                self._record_mastery_and_advance(
                    step.reward_results,
                    step.active_qualities,
                    batch,
                    metrics,
                    batch_contexts=step.batch_contexts,
                )
                # Log completions even on skipped steps
                for i, sample in enumerate(step.samples):
                    if self.tokenizer is not None:
                        comp_text = self.tokenizer.decode(
                            sample.completion, skip_special_tokens=True
                        )
                    else:
                        comp_text = str(sample.completion)
                    self.completion_logger.log_completion(
                        step=self.global_step,
                        prompt=batch.raw_prompts[i] if i < len(batch.raw_prompts) else "",
                        completion=comp_text,
                        reward=sample.reward_result.reward,
                        reward_components=sample.reward_result.scores,
                        phase=self.game_state.phase,
                    )
                # FIX 15: Don't increment _accumulation_count on early return without backward
                # The count tracks actual backward() calls, not skipped steps
                # FIX R3-T2: Don't increment _accumulated_samples either — no backward means
                # this step doesn't contribute to the loss average. Incrementing the denominator
                # would dilute the average incorrectly.
                self._skip_microbatch("SPO filtered all samples")
                self.global_step += 1
                self.ctx.step = self.global_step
                # T3: Scheduler step removed from early-return path - normal path handles it
                return metrics
            if useful.sum() >= 2 and useful.sum() < len(step):
                idx = useful.nonzero(as_tuple=True)[0]
                # ATOMIC FILTER: TrainingStep.filter() reindexes ALL fields in one call.
                # This prevents the bug class where a field is forgotten during manual reindexing.
                step = step.filter(idx)
                # Track that filtering was applied (for VPRM/EGRS index mapping)
                self._spo_filter_idx = step.filter_idx
                # TP1/TP5: Reindex batch_contexts and active_qualities to match filtered step
                batch_contexts = [batch_contexts[i] for i in idx.tolist()]
                active_qualities = [active_qualities[i] for i in idx.tolist()]

        # Rebuild batch tensors from samples (after SPO filter if applied)
        max_comp_len = step.comp_tensor.shape[1]
        device = step.comp_tensor.device

        # Rebuild gen_logprobs_padded from samples
        has_gen_lp = any(s.gen_logprobs is not None for s in step.samples)
        if has_gen_lp:
            gen_logprobs_padded = torch.zeros(
                len(step), max_comp_len, device=device, dtype=torch.float32
            )
            for i, s in enumerate(step.samples):
                if s.gen_logprobs is not None:
                    lp_len = min(len(s.gen_logprobs), max_comp_len)
                    gen_logprobs_padded[i, :lp_len] = s.gen_logprobs[:lp_len].to(device)
        else:
            gen_logprobs_padded = None

        # Rebuild kl_region_weights from samples
        has_kl_weights = any(s.kl_region_weights is not None for s in step.samples)
        if has_kl_weights:
            kl_region_weights = torch.ones(len(step), max_comp_len, device=device)
            for i, s in enumerate(step.samples):
                if s.kl_region_weights is not None:
                    kl_len = min(len(s.kl_region_weights), max_comp_len)
                    kl_region_weights[i, :kl_len] = s.kl_region_weights[:kl_len]
        else:
            kl_region_weights = None

        # AlignedLossFrame: centralize ALL Level 1 shifts in one place.
        # L1 shift ([:, 1:]) aligns from token space to next-token-prediction space.
        # Shape validation happens inside build() — catches misalignment at construction.
        prompt_lengths = [0] * step.comp_tensor.shape[0]
        raw_response_mask = self.compute_response_mask(step.comp_tensor, prompt_lengths)
        aligned_frame = AlignedLossFrame.build(
            response_mask=raw_response_mask,
            padded_advs=step.padded_advs,
            gen_logprobs_padded=gen_logprobs_padded,
            kl_region_weights=kl_region_weights,
        )

        if aligned_frame.response_mask.sum() == 0:
            raise RuntimeError(
                f"Step {self.global_step}: no response tokens in any completion — cannot compute loss.",
            )

        # Micro-batched forward + backward — avoids OOM on logits tensor
        # Full logits = batch × seq × vocab ≈ 8 × 4096 × 151K × 4B = 18.6GB (impossible on 16GB)
        # Micro-batch size adapts to sequence length to avoid OOM on long completions.
        # At 4096 tokens, Unsloth MLP activation = 2 × 4096 × 8960 × 2B = 140MB per seq.
        # micro_batch_size=1 for seq ≥ 2048, micro_batch_size=2 for shorter.
        actual_batch = step.comp_tensor.shape[0]  # May differ after SPO filter
        micro_batch_size = (
            1
            if max_comp_len >= self.config.training.micro_batch_seq_threshold
            else max(1, min(2, actual_batch))
        )
        n_micro = (actual_batch + micro_batch_size - 1) // micro_batch_size
        total_loss = 0.0
        all_metrics = {}

        for mb_start in range(0, actual_batch, micro_batch_size):
            mb_end = min(mb_start + micro_batch_size, actual_batch)
            mb_ids = step.comp_tensor[mb_start:mb_end]
            mb_attn_mask = step.comp_attention_mask[mb_start:mb_end]
            # L1-level view for EGRS mutation (before L2 shift in slice_for_microbatch)
            mb_advs = aligned_frame.advantages[mb_start:mb_end]
            mb_hidden_states = None  # Set by fused/non-fused path when VPRM enabled

            # Ensure Unsloth training mode before EACH forward pass (not just at init).
            # Unsloth's inplace attention kernels require this transition before backward.
            # Source: Unsloth #895, #2434 — "modified by inplace operation" fix.
            if hasattr(self, "_FastLanguageModel"):
                self._FastLanguageModel.for_training(self.model)  # type: ignore[attr-defined]
            elif self.generation_backend and hasattr(self.generation_backend, "_FastLanguageModel"):
                self.generation_backend._FastLanguageModel.for_training(self.model)
            else:
                try:
                    from unsloth import FastLanguageModel as FLM

                    FLM.for_training(self.model)
                except ImportError:
                    import warnings

                    warnings.warn(
                        "Could not call FastLanguageModel.for_training() — Unsloth not importable. "
                        "If using Unsloth models, inplace attention kernels may cause backward errors.",
                        stacklevel=2,
                    )

            # Forward pass: fused (chunked lm_head + checkpoint) or non-fused (full lm_head)
            # Both paths start from hidden states (UNSLOTH_RETURN_HIDDEN_STATES=1 is global).
            # Fused saves ~2GB VRAM via chunking. Non-fused materializes full logit tensor.
            if self._use_fused_logprobs:
                from qgre.fused_logprobs import (
                    chunked_logprobs_from_hidden,
                    get_hidden_states_and_lm_head,
                )

                # Optionally extract attentions for bond strength computation
                # NOTE: Unsloth's fast forward asserts output_attentions=False (inplace attention kernels
                # don't produce attention matrices). Heuristic detection is unreliable (Unsloth patches
                # methods without changing class names), so we try the call and catch the AssertionError.
                # IMPORTANT: Cache the decision so we don't retry every iteration.
                if not hasattr(self, "_attention_output_attentions"):
                    self._attention_output_attentions = (
                        self.config.algorithm.attention_constrained_advantage
                    )
                output_attentions = self._attention_output_attentions
                if output_attentions:
                    try:
                        hidden_states, lm_head, attentions = get_hidden_states_and_lm_head(  # type: ignore[misc]
                            self.model,
                            mb_ids,
                            output_attentions=True,
                            attention_mask=mb_attn_mask,
                        )
                    except (AssertionError, NotImplementedError) as _attn_err:
                        # Unsloth's fast forward asserts output_attentions=False.
                        # Fall back to hidden state + entropy proxy for importance.
                        logging.getLogger(__name__).warning(
                            f"Attention extraction failed ({type(_attn_err).__name__}), "
                            f"using hidden state + entropy proxy for importance constraint",
                        )
                        self._attention_output_attentions = False
                        self._attention_disabled_reason = "unsloth_incompatible"
                        self._use_importance_proxy = True  # Use proxy instead
                        # Retry without attention extraction
                        hidden_states, lm_head = get_hidden_states_and_lm_head(  # type: ignore[misc]
                            self.model,
                            mb_ids,
                            attention_mask=mb_attn_mask,
                        )
                        attentions = None
                else:
                    hidden_states, lm_head = get_hidden_states_and_lm_head(  # type: ignore[misc]
                        self.model,
                        mb_ids,
                        attention_mask=mb_attn_mask,
                    )
                    attentions = None
                if hidden_states is not None and lm_head is not None:
                    # Fused path: compute logprobs without materializing full [seq, vocab] tensor.
                    # Triton kernel availability is decided at init (fail-fast)
                    # — no runtime try/except. A kernel exception here means a
                    # real bug (bad indexing, CUDA error, label out of vocab),
                    # not a degraded-mode condition. Silent fallback would
                    # change numerics mid-run and hide the bug.
                    _use_triton = self._use_triton_logprobs
                    if _use_triton:
                        from qgre.triton_logprobs import triton_logprobs_with_grad

                        if not self._triton_validated:
                            logging.getLogger(__name__).info(
                                "Triton fused logprobs enabled — single GPU launch, "
                                "zero vocab-tensor allocation in forward and backward",
                            )
                        mb_lp = triton_logprobs_with_grad(
                            hidden_states[:, :-1, :],
                            lm_head,
                            mb_ids[:, 1:],
                        )
                    else:
                        mb_lp = chunked_logprobs_from_hidden(
                            hidden_states[:, :-1, :],
                            lm_head,
                            mb_ids[:, 1:],
                            chunk_size=self._fused_chunk_size,
                        )

                    # One-time validation: verify autograd graph exists.
                    # Chunked path cross-checks against cuBLAS reference (same
                    # bf16 accumulator → tight atol=1e-3).
                    # Triton path does NOT cross-check against cuBLAS — it uses
                    # full fp32 element-wise reduction while cuBLAS uses bf16
                    # accumulators. At hidden_dim=2048 the expected divergence is
                    # O(sqrt(2048) * eps_bf16) ≈ 0.18, which is a *precision
                    # advantage* of Triton, not a bug. Self-consistency (same
                    # element-wise path in forward and backward) IS the
                    # correctness guarantee. Test suite validates at small scale
                    # where fp32 and bf16 agree (atol=1e-3).
                    if not self._fused_validated and not _use_triton and mb_lp.shape[1] > 0:
                        self._validate_logprob_path(
                            mb_lp,
                            hidden_states,
                            lm_head,
                            mb_ids,
                            atol=1e-3,
                            path_name="fused",
                        )
                        self._fused_validated = True
                    elif _use_triton and not self._triton_validated and mb_lp.shape[1] > 0:
                        if mb_lp.grad_fn is None:
                            raise RuntimeError(
                                "Triton logprobs has no grad_fn — autograd graph is broken.",
                            )
                        if not mb_lp.isfinite().all():
                            raise RuntimeError(
                                "Triton logprobs contain NaN/Inf on first step. "
                                "Check model stability (hidden state explosion).",
                            )
                        logging.getLogger(__name__).info(
                            "Triton logprobs validated (grad_fn present, "
                            "output finite, self-consistent fp32 path)",
                        )
                        self._triton_validated = True
                    # Compute importance for advantage constraint
                    # Priority: attention bond strength > hidden state proxy > disabled
                    if attentions is not None:
                        # VRAM fix: sample ONE layer only, not all layers
                        layer_idx = self.config.algorithm.attention_sample_layer
                        attention_single = select_attention_layer(attentions, layer_idx)
                        if self.global_step == 0 and not getattr(self, "_attn_logged", False):
                            shape = attention_single.shape
                            vram_mb = shape[0] * shape[1] * shape[2] * shape[3] * 4 / 1024 / 1024
                            logging.getLogger(__name__).info(
                                f"Attention constraint enabled (attention mode): layer={layer_idx}, "
                                f"shape={list(shape)}, estimated_vram={vram_mb:.1f}MB",
                            )
                            self._attn_logged = True
                        bond_strength = compute_bond_strength(
                            attention_single,
                            seq_len=mb_ids.shape[1],
                            mode=self.config.algorithm.attention_constraint_mode,
                            batch_size=mb_ids.shape[0],
                            device=mb_ids.device,
                        )
                        mb_token_entropy = None  # EGRS not supported with attention mode
                        del attentions
                    elif (
                        getattr(self, "_use_importance_proxy", False)
                        and self.config.algorithm.attention_constrained_advantage
                    ):
                        # Attention unavailable but constraint enabled — use ERIC (entropy-regulated importance).
                        # Chunks the lm_head projection + softmax through seq in slices so we never
                        # materialize the full [batch, seq, vocab] fp32 tensor. Peak per chunk ~450 MB
                        # vs ~14 GB for the naive approach that crashed on 16 GB.
                        if self.global_step == 0 and not getattr(self, "_proxy_logged", False):
                            logging.getLogger(__name__).info(
                                f"ERIC importance (chunked lm_head): mode={self.config.algorithm.eric_mode} "
                                f"decay={self.config.algorithm.attention_position_decay}",
                            )
                            self._proxy_logged = True
                        with torch.no_grad():
                            if self.config.egrs.enabled:
                                bond_strength, mb_token_entropy = (
                                    compute_entropy_importance_from_hidden(
                                        hidden_states[:, :-1, :],
                                        lm_head,
                                        seq_len=mb_ids.shape[1],
                                        mode=self.config.algorithm.eric_mode,
                                        position_decay=self.config.algorithm.attention_position_decay,
                                        return_normalized_entropy=True,
                                    )
                                )
                            else:
                                bond_strength = compute_entropy_importance_from_hidden(
                                    hidden_states[:, :-1, :],
                                    lm_head,
                                    seq_len=mb_ids.shape[1],
                                    mode=self.config.algorithm.eric_mode,
                                    position_decay=self.config.algorithm.attention_position_decay,
                                )
                                mb_token_entropy = None
                    else:
                        bond_strength = None
                        mb_token_entropy = None
                    # Keep hidden_states for VPRM critic if enabled; else free memory
                    mb_hidden_states = hidden_states.detach() if self.config.vprm.enabled else None
                    del hidden_states
                else:
                    # Hidden states mode didn't take effect — crash with diagnostics
                    raise RuntimeError(
                        f"Step {self.global_step}: fused logprobs unavailable — "
                        f"UNSLOTH_RETURN_HIDDEN_STATES did not take effect. "
                        f"Delete unsloth_compiled_cache/ and restart. "
                        f"To disable fused logprobs entirely, set algorithm.use_fused_logprobs=false.",
                    )
            else:
                # Non-fused path: full lm_head projection without chunking.
                # Costs ~2GB more VRAM than fused (materializes full [seq, vocab] tensor).
                # This is the degraded-but-correct escape hatch.
                from qgre.fused_logprobs import get_hidden_states_and_lm_head

                # Optionally extract attentions for bond strength computation
                # NOTE: Unsloth's fast forward asserts output_attentions=False. Uses same cached
                # decision as fused path (try call, catch AssertionError, disable and retry).
                if not hasattr(self, "_attention_output_attentions"):
                    self._attention_output_attentions = (
                        self.config.algorithm.attention_constrained_advantage
                    )
                output_attentions = self._attention_output_attentions
                if output_attentions:
                    try:
                        hs, lm_head_nf, attentions = get_hidden_states_and_lm_head(  # type: ignore[misc]
                            self.model,
                            mb_ids,
                            output_attentions=True,
                            attention_mask=mb_attn_mask,
                        )
                    except (AssertionError, NotImplementedError) as _attn_err:
                        # Same as fused path: fall back to proxy
                        logging.getLogger(__name__).warning(
                            f"Attention extraction failed ({type(_attn_err).__name__}), "
                            f"using hidden state + entropy proxy for importance constraint",
                        )
                        self._attention_output_attentions = False
                        self._attention_disabled_reason = "unsloth_incompatible"
                        self._use_importance_proxy = True
                        hs, lm_head_nf = get_hidden_states_and_lm_head(  # type: ignore[misc]
                            self.model,
                            mb_ids,
                            attention_mask=mb_attn_mask,
                        )
                        attentions = None
                else:
                    hs, lm_head_nf = get_hidden_states_and_lm_head(  # type: ignore[misc]
                        self.model,
                        mb_ids,
                        attention_mask=mb_attn_mask,
                    )
                    attentions = None
                if hs is None or lm_head_nf is None:
                    raise RuntimeError(
                        f"Step {self.global_step}: model did not return hidden states. "
                        f"UNSLOTH_RETURN_HIDDEN_STATES did not take effect. "
                        f"Delete unsloth_compiled_cache/ and restart.",
                    )
                mb_logits = lm_head_nf(hs[:, :-1, :]).float()
                # Compute token entropy for EGRS
                if self.config.egrs.enabled:
                    vocab_size = mb_logits.shape[-1]
                    mb_token_entropy = compute_normalized_entropy(mb_logits, vocab_size)
                else:
                    mb_token_entropy = None
                # Compute importance for advantage constraint
                if attentions is not None:
                    layer_idx = self.config.algorithm.attention_sample_layer
                    attention_single = select_attention_layer(attentions, layer_idx)
                    if self.global_step == 0 and not getattr(self, "_attn_logged", False):
                        shape = attention_single.shape
                        vram_mb = shape[0] * shape[1] * shape[2] * shape[3] * 4 / 1024 / 1024
                        logging.getLogger(__name__).info(
                            f"Attention constraint enabled (attention mode): layer={layer_idx}, "
                            f"shape={list(shape)}, estimated_vram={vram_mb:.1f}MB",
                        )
                        self._attn_logged = True
                    bond_strength = compute_bond_strength(
                        attention_single,
                        seq_len=mb_ids.shape[1],
                        mode=self.config.algorithm.attention_constraint_mode,
                        batch_size=mb_ids.shape[0],
                        device=mb_ids.device,
                    )
                    del attentions
                elif (
                    getattr(self, "_use_importance_proxy", False)
                    and self.config.algorithm.attention_constrained_advantage
                ):
                    # Attention unavailable but constraint enabled — use ERIC (entropy-regulated importance).
                    # Non-fused path: mb_logits already materialized for logprobs; reuse it.
                    # compute_entropy_importance chunks softmax internally so the 3× peak doesn't OOM.
                    if self.global_step == 0 and not getattr(self, "_proxy_logged", False):
                        logging.getLogger(__name__).info(
                            f"ERIC importance (non-fused): mode={self.config.algorithm.eric_mode} "
                            f"decay={self.config.algorithm.attention_position_decay}",
                        )
                        self._proxy_logged = True
                    # mb_logits is [batch, seq-1, vocab]; pass mb_ids.shape[1] so the output
                    # pads to full seq_len and matches downstream token_advs shape.
                    bond_strength = compute_entropy_importance(
                        mb_logits,
                        seq_len=mb_ids.shape[1],
                        mode=self.config.algorithm.eric_mode,
                        position_decay=self.config.algorithm.attention_position_decay,
                    )
                else:
                    bond_strength = None
                mb_hidden_states = hs.detach() if self.config.vprm.enabled else None
                del hs
                mb_lp = logprobs_from_logits(mb_logits, mb_ids[:, 1:])
                del mb_logits

            # old_logprobs: generation-time logprobs for LLDS, or detached current logprobs.
            # gen_logprobs is L1-shifted in aligned_frame (same coordinate as mb_lp).
            if aligned_frame.gen_logprobs is not None:
                mb_gen_lp = aligned_frame.gen_logprobs[mb_start:mb_end]
                min_lp_len = min(mb_lp.shape[1], mb_gen_lp.shape[1])
                mb_old_lp = mb_gen_lp[:, :min_lp_len]
                # L5: Use -100 instead of -1e9 to prevent exp(inf) in KL computation
                if mb_lp.shape[1] > min_lp_len:
                    pad = torch.full(
                        (mb_lp.shape[0], mb_lp.shape[1] - min_lp_len), -100.0, device=device
                    )
                    mb_old_lp = torch.cat([mb_old_lp, pad], dim=1)
            else:
                mb_old_lp = mb_lp.detach()

            # VPRM critic: replace SPO advantages with critic-based advantages
            mb_critic_loss = torch.tensor(0.0, device=device, requires_grad=True)
            mb_critic_count = 0
            if self.config.vprm.enabled and mb_hidden_states is not None:
                from qgre.advantages import compute_advantages_vprm

                # Lazy init critic on first forward pass (now we know hidden_dim)
                if not self._vprm_initialized:
                    self._init_vprm_critic(
                        hidden_dim=mb_hidden_states.shape[-1],
                        device=device,
                    )

                # TL-R3-07: Validate hidden_dim matches checkpoint (only on first forward after resume)
                # Don't delete _vprm_checkpoint_hidden_dim to allow check after resume
                if hasattr(self, "_vprm_checkpoint_hidden_dim") and not getattr(
                    self, "_vprm_checkpoint_validated", False
                ):
                    actual_hidden_dim = mb_hidden_states.shape[-1]
                    if self._vprm_checkpoint_hidden_dim != actual_hidden_dim:
                        raise RuntimeError(
                            f"VPRM critic hidden_dim mismatch: checkpoint has {self._vprm_checkpoint_hidden_dim} "
                            f"but model produces {actual_hidden_dim}. Model architecture changed between checkpoints.",
                        )
                    self._vprm_checkpoint_validated = True  # Only check once per resume

                # TL-5: Wrap VPRM computation in try-finally for guaranteed cleanup
                try:
                    # Compute VPRM advantages per-sample in this micro-batch
                    for mb_i in range(mb_end - mb_start):
                        filtered_i = mb_start + mb_i
                        # With TrainingStep, all fields are already reindexed after filter.
                        # Use filtered_i directly — no orig_i mapping needed for step data.
                        # orig_i is only needed for external state (mastery, etc.)
                        orig_i = step.get_original_idx(filtered_i)

                        # Get sample data directly from step (already reindexed)
                        sample_regions = step.batch_regions[filtered_i]
                        sample_rr = step.reward_results[filtered_i]
                        sample_aq = step.active_qualities[filtered_i]
                        # Get hidden states for this sample
                        sample_hs = mb_hidden_states[mb_i]  # [seq_len, hidden_dim]
                        # Trim to completion length with bounds validation
                        comp_len = len(step.completions[filtered_i])
                        if comp_len > sample_hs.shape[0]:
                            import warnings

                            warnings.warn(
                                f"VPRM: completion length ({comp_len}) > hidden states length ({sample_hs.shape[0]}) "
                                f"for sample orig_i={orig_i}. Clamping to hidden states length.",
                                stacklevel=2,
                            )
                            comp_len = sample_hs.shape[0]
                        sample_hs_trimmed = sample_hs[:comp_len]

                        # Get span token masks from step.samples (already filtered by TrainingStep)
                        sample_token_masks = step.samples[filtered_i].token_masks

                        # Extract bond strength for this sample if available
                        sample_bond_strength = None
                        if bond_strength is not None:
                            # bond_strength shape: [micro_batch_size, seq_len]
                            # Extract for this sample and trim to completion length
                            sample_bond_strength = bond_strength[mb_i, :comp_len].to(device)
                            # Log bond strength stats periodically
                            if self.global_step % 10 == 0 and mb_i == 0 and filtered_i == 0:
                                bs = sample_bond_strength
                                logging.getLogger(__name__).warning(
                                    f"Step {self.global_step} bond_strength: "
                                    f"min={bs.min().item():.4f}, max={bs.max().item():.4f}, "
                                    f"mean={bs.mean().item():.4f}, nonzero={(bs > 0).sum().item()}/{len(bs)}",
                                )

                        vprm_advs, vprm_loss, used_critic = compute_advantages_vprm(
                            critic=self.vprm_critic,
                            hidden_states=sample_hs_trimmed,
                            regions=sample_regions[:comp_len],  # type: ignore[index]
                            reward_result=sample_rr,
                            step_qualities=self.step_qualities,
                            active_qualities=sample_aq,
                            step_region_map=self.config.algorithm.step_region_map,
                            frontier_steps=frontier_steps,
                            frontier_amplification=self.config.algorithm.frontier_amplification,
                            min_regions=self.config.vprm.spo_fallback_min_regions,
                            # Use step.batch_contexts (already reindexed after filter)
                            aspiration_beta=self.advantage_estimator._aspiration_beta
                            * step.batch_contexts[filtered_i].aspiration_warmup,
                            aspiration_target=step.batch_contexts[filtered_i].aspiration_target,
                            ctx=self.ctx,
                            token_masks=sample_token_masks,  # Pass span masks for span-aware critic
                            bond_strength=sample_bond_strength,  # Pass attention bond strength
                            constraint_strength=self.config.algorithm.attention_constraint_strength,
                        )

                        if used_critic:
                            # When spans are active, DON'T overwrite span advantages —
                            # spans provide better token targeting. Only collect critic loss
                            # (critic still learns the baseline for future use).
                            # When spans are NOT active, replace SPO advantages with VPRM.
                            if not use_spans:
                                # VPRM returns raw-space advantages: vprm_advs[t] = advantage for token t.
                                # mb_advs is in logprob space: mb_advs[t] = advantage for token t+1.
                                # Offset by 1: vprm_advs[1:] maps raw token t+1 to logprob position t.
                                if vprm_advs.shape[0] > 1:
                                    adv_len = min(vprm_advs.shape[0] - 1, mb_advs.shape[1])
                                    mb_advs[mb_i, :adv_len] = vprm_advs[1 : adv_len + 1]
                            mb_critic_loss = mb_critic_loss + vprm_loss
                            mb_critic_count += 1
                finally:
                    # TL-5: Guaranteed cleanup of mb_hidden_states
                    if "mb_hidden_states" in locals():
                        del mb_hidden_states

            # EGRS 2x2 matrix: modify advantages based on (correct/wrong) x (confident/uncertain)
            # This replaces the simpler ERIC dampening when EGRS is enabled
            mb_entropy_adjustments = None
            if self.config.egrs.enabled and mb_token_entropy is not None:
                egrs_cfg = self.config.egrs
                mb_entropy_adjustments = torch.zeros_like(mb_advs)
                for mb_i in range(mb_advs.shape[0]):
                    filtered_i = mb_start + mb_i
                    # With TrainingStep, use filtered_i for step data, orig_i for external state
                    orig_i = step.get_original_idx(filtered_i)

                    # Get sample data directly from step (already reindexed)
                    sample_regions = step.batch_regions[filtered_i]
                    sample_rr = step.reward_results[filtered_i]
                    if sample_rr is None:
                        logging.getLogger(__name__).warning(
                            f"EGRS: reward_results[{orig_i}] is None — sample skipped. "
                            "Check if reward_fn returned None instead of RewardResult.",
                        )
                        continue
                    # Compute span correctness from reward result
                    span_correct = compute_span_correctness(
                        sample_rr,
                        self.step_qualities,
                        egrs_cfg.reward_threshold,
                    )
                    # Entropy is computed on shifted logprobs (tokens [1:]), so
                    # mb_token_entropy.shape[1] = seq_len - 1. Use the minimum
                    # of regions length and entropy length.
                    comp_len = min(len(sample_regions), mb_token_entropy.shape[1])  # type: ignore[arg-type]
                    sample_entropy = mb_token_entropy[mb_i, :comp_len]
                    sample_advs = mb_advs[mb_i, :comp_len]
                    # Get importance for ERIC dampening (if available)
                    sample_importance = None
                    if bond_strength is not None and mb_i < bond_strength.shape[0]:
                        sample_importance = bond_strength[mb_i, :comp_len]
                    # Apply EGRS matrix with ERIC dampening for Q1
                    modified_advs, entropy_adj, hint_flags = apply_egrs_matrix(
                        sample_advs,
                        sample_regions[:comp_len],  # type: ignore[index]
                        sample_entropy,
                        span_correct,
                        entropy_threshold=egrs_cfg.entropy_threshold,
                        gate_temperature=egrs_cfg.gate_temperature,
                        exploration_weight=egrs_cfg.exploration_weight,
                        importance=sample_importance,
                        eric_strength=self.config.algorithm.attention_constraint_strength,
                    )
                    # Update advantages in place
                    mb_advs[mb_i, :comp_len] = modified_advs
                    mb_entropy_adjustments[mb_i, :comp_len] = entropy_adj
                    # Flag hints for registry (if enabled)
                    # Extract hint tokens using hint_extractor if available
                    if self.hint_registry is not None and hint_flags:
                        prompt_id = (
                            batch.prompt_ids[orig_i] if orig_i < len(batch.prompt_ids) else -1
                        )
                        sample_meta = batch.metadata[orig_i] if orig_i < len(batch.metadata) else {}
                        for step_num, t in hint_flags:
                            span_id = f"STEP_{step_num}" if step_num > 0 else "THINK"
                            # Extract hint tokens using extractor if available
                            hint_tokens = []
                            if self.hint_extractor is not None:
                                hint_text = self.hint_extractor(span_id, sample_meta)
                                if hint_text:
                                    # Tokenize hint text (limit to hint_token_count)
                                    hint_tokens = self.tokenizer.encode(
                                        hint_text,
                                        add_special_tokens=False,
                                    )[
                                        : self.config.egrs.hint_token_count * 10
                                    ]  # Allow more tokens for math
                            # Skip flagging if hint_tokens is empty
                            if hint_tokens:
                                self.hint_registry.flag_for_hint(
                                    prompt_id,
                                    span_id,
                                    hint_tokens,
                                    current_mastery=0.0,
                                    current_step=self.global_step,
                                )

            # AlignedLossFrame L2 shift: advantages[t+1] pairs with logprob[t].
            # EGRS mutation of mb_advs happened above; slice_for_microbatch reads the mutated values.
            mb_frame = aligned_frame.slice_for_microbatch(mb_start, mb_end)
            loss_len = mb_frame.loss_len

            # Request per-token loss when computing per-quality metrics.
            # Coerce to bool — `use_spans and batch_token_masks` returns
            # the list itself on the truthy path, which is a latent type bug
            # the loss_fn signature doesn't accept.
            need_per_token = bool(use_spans and batch_token_masks)
            loss_result = self.loss_fn(
                curr_logprobs=mb_lp[:, :loss_len],
                prev_logprobs=mb_old_lp[:, :loss_len],
                advantages=mb_frame.advantages,
                mask=mb_frame.mask.float(),
                reference_logprobs=mb_old_lp[:, :loss_len],
                kl_region_weights=mb_frame.kl_weights,
                return_per_token_loss=need_per_token,
            )
            if need_per_token:
                mb_loss, mb_metrics, mb_per_token_loss = loss_result  # type: ignore[misc]
            else:
                mb_loss, mb_metrics = loss_result  # type: ignore[misc]
                mb_per_token_loss = None

            # neg_logprob_mean: monitor for policy collapse (metric only, not a loss term).
            # -mean(log p(token)) is NOT a valid entropy loss — its gradient pushes the wrong
            # direction (increases prob of sampled tokens instead of spreading mass).
            # See NeMo RL docs: "not recommended for direct backpropagation."
            with torch.no_grad():
                neg_logprob_mean = -masked_mean(mb_lp[:, :loss_len], mb_frame.mask.float())
                mb_metrics["neg_logprob_mean"] = neg_logprob_mean.item()

            # Per-quality loss computation (when using span-based advantages)
            # Uses actual per-token loss from ClippedPGLossFn (ratio * advantage),
            # NOT the wrong formula (-logprob * advantage) used before.
            #
            # To match total loss normalization (seq-mean-token-sum-norm), we:
            # 1. Sum per-token loss within each quality span
            # 2. Track span counts to compute proper mean across samples
            if mb_per_token_loss is not None and batch_token_masks:
                with torch.no_grad():
                    for mb_i_inner in range(mb_per_token_loss.shape[0]):
                        # Use filtered_i to access step.samples (already filtered by TrainingStep)
                        filtered_i = mb_start + mb_i_inner
                        sample_masks = step.samples[filtered_i].token_masks
                        if sample_masks:
                            for q_name, q_mask in sample_masks.items():
                                # Shift mask to align with loss positions (logprobs[t] predicts token[t+1])
                                # q_mask is in token space [0..completion_len-1]
                                # Loss is computed for positions predicting tokens [1..seq_len-1]
                                # So q_mask[i] contributes to loss position i-1; we need q_mask[1:]
                                q_mask_shifted = q_mask[1:]  # Always shift by 1
                                # Align to loss_len: truncate if longer (batch has longer samples),
                                # pad with zeros if shorter (this sample is shorter than the batch max).
                                # Padding positions contribute zero loss because they're masked out.
                                if q_mask_shifted.shape[0] >= loss_len:
                                    q_mask_shifted = q_mask_shifted[:loss_len]
                                else:
                                    pad_len = loss_len - q_mask_shifted.shape[0]
                                    q_mask_shifted = torch.cat(
                                        [
                                            q_mask_shifted,
                                            torch.zeros(
                                                pad_len, device=q_mask.device, dtype=q_mask.dtype
                                            ),
                                        ]
                                    )
                                q_mask_shifted = q_mask_shifted.to(mb_per_token_loss.device)
                                # Sum actual per-token loss contribution for this quality's span
                                q_loss = (
                                    mb_per_token_loss[mb_i_inner] * q_mask_shifted[:loss_len]
                                ).sum()
                                q_count = q_mask_shifted[:loss_len].sum()
                                if q_count > 0:
                                    # Accumulate sum and count separately for proper averaging later
                                    loss_key = f"loss/{q_name}"
                                    count_key = f"_loss_count/{q_name}"
                                    mb_metrics[loss_key] = (
                                        mb_metrics.get(loss_key, 0.0) + q_loss.item()
                                    )
                                    mb_metrics[count_key] = (
                                        mb_metrics.get(count_key, 0.0) + q_count.item()
                                    )

            # Dynamic length control (Huawei): penalize length when group accuracy is high
            lp_coef = self.config.algorithm.length_penalty_coef
            if lp_coef > 0:
                lp_thresh = self.config.algorithm.length_penalty_threshold
                # Compute group_correctness using step.reward_results (already reindexed after filter)
                mb_reward_results = step.reward_results[mb_start:mb_end]
                group_correctness = sum(rr.reward for rr in mb_reward_results) / len(
                    mb_reward_results
                )
                if group_correctness > lp_thresh:
                    # High correctness → add length penalty to encourage efficiency
                    seq_lengths = mb_frame.mask.sum(dim=-1)
                    mean_len = seq_lengths.mean()
                    length_penalty = lp_coef * (seq_lengths / max(mean_len, 1.0)).mean()
                    mb_loss = mb_loss + length_penalty
                    mb_metrics["length_penalty"] = length_penalty.item()

            # LLDS auxiliary loss — prevents Lazy Likelihood Displacement death spiral
            # (arXiv:2512.04220). Fires when generation-time logprobs are available
            # (old_logprob ≠ curr_logprob). Three-level gate: trajectory + token + action.
            llds_coef = self.config.algorithm.llds_coef
            if llds_coef > 0 and self._has_stored_logprobs:
                llds_loss, llds_mask = compute_llds_loss(
                    log_prob=mb_lp[:, :loss_len],
                    old_log_prob=mb_old_lp[:, :loss_len],
                    advantages=mb_frame.advantages,
                    response_mask=mb_frame.mask.float(),
                )
                mb_loss = mb_loss + llds_coef * llds_loss
                mb_metrics["llds_loss"] = llds_loss.item()
                mb_metrics["llds_mask_ratio"] = llds_mask.sum().item() / max(
                    mb_frame.mask.sum().item(), 1
                )

            # VPRM critic loss: weight by sample count and add to policy loss
            # T2-5: Guard div by zero to prevent NaN
            if mb_critic_count > 0:
                mb_critic_loss = mb_critic_loss / mb_critic_count
                critic_weight = mb_critic_count / len(mb_advs)
                if mb_critic_loss.requires_grad:
                    mb_loss = mb_loss + critic_weight * mb_critic_loss
                    mb_metrics["critic_loss"] = mb_critic_loss.item()

            # EGRS entropy loss: maximize entropy for Q3 (confident+wrong) tokens
            # Formula: loss += -sum(adjustment * entropy) where adjustment > 0 for Q3
            # Minimizing negative entropy = maximizing entropy (shakes confidence)
            if (
                self.config.egrs.enabled
                and mb_entropy_adjustments is not None
                and mb_token_entropy is not None
            ):
                # Both tensors are already in logprob space (computed from L1-shifted mb_advs
                # and logits[:, :-1] respectively). No additional shift needed.
                egrs_len = min(mb_entropy_adjustments.shape[1], mb_token_entropy.shape[1], loss_len)
                egrs_mask = mb_frame.mask[:, :egrs_len].float()
                egrs_adj = mb_entropy_adjustments[:, :egrs_len]
                egrs_ent = mb_token_entropy[:, :egrs_len]
                # Sum over tokens with positive adjustment (Q3 only), normalized by token count
                egrs_token_count = egrs_mask.sum().clamp(min=1.0)
                egrs_loss = -(egrs_adj * egrs_ent * egrs_mask).sum() / egrs_token_count
                if egrs_loss.abs() > 0:
                    mb_loss = mb_loss + egrs_loss
                    mb_metrics["egrs_entropy_loss"] = egrs_loss.item()
                # Log quadrant distribution for debugging
                # Q1: uncertain+correct (scaled advantage), Q2: confident+correct (zero),
                # Q3: confident+wrong (entropy boost), Q4: uncertain+wrong (hint flag)
                with torch.no_grad():
                    q3_count = (egrs_adj > 0).sum().item()
                    mb_metrics["egrs/q3_tokens"] = q3_count
                    # Note: Q1/Q2/Q4 counts require per-sample tracking done in apply_egrs_matrix loop
                    # The hint_registry tracks Q4 flags if enabled

            # Check for NaN BEFORE backward
            if torch.isnan(mb_loss).any():
                mb_idx = mb_start // micro_batch_size + 1
                raise RuntimeError(
                    f"Step {self.global_step} micro-batch {mb_idx}/{n_micro}: "
                    f"loss is NaN — aborting before backward() to prevent gradient corruption.",
                )
            # TRN-R1-2: Check loss for NaN/inf BEFORE backward() to prevent optimizer corruption
            if not torch.isfinite(mb_loss):
                mb_idx = mb_start // micro_batch_size + 1
                raise RuntimeError(
                    f"Step {self.global_step} micro-batch {mb_idx}/{n_micro}: "
                    f"loss is {mb_loss.item()} — aborting before backward() to prevent gradient corruption.",
                )

            try:
                (mb_loss / n_micro).backward()
            except Exception as bwd_exc:
                # TL-2: Zero gradients on backward failure to prevent corrupted state
                logging.getLogger(__name__).exception(
                    f"TL-2: backward() raised exception at step {self.global_step}: {bwd_exc}. "
                    "Zeroing gradients to prevent gradient corruption.",
                )
                self.optimizer.zero_grad()
                # Clear accumulated loss from successful micro-batches to prevent corrupt update
                self._reset_accumulators()
                # Clear CUDA cache to free memory before re-raising
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise
            # TL-3: Only add to total_loss AFTER successful backward
            total_loss += mb_loss.item() / n_micro

            del mb_lp

            # OPT-2: Release CUDA cached blocks between micro-batches.
            # Prevents false OOM from allocator fragmentation on tight memory budgets.
            if self.config.training.empty_cache_between_microbatches and torch.cuda.is_available():
                torch.cuda.empty_cache()

            # TRN-R1-3: Track metric types for correct aggregation
            # T2-1: Initialize _per_token_metrics unconditionally to prevent AttributeError on early return
            self._per_token_metrics = {"kl", "llds_loss", "llds_mask_ratio", "critic_loss"}
            # TL-R2-05: Accumulate per_token_metrics across micro-batches
            if not all_metrics:
                all_metrics = dict(mb_metrics.items())
            else:
                for k, v in mb_metrics.items():
                    all_metrics[k] = all_metrics.get(k, 0) + v

        # TRN-R1-3: Aggregate per-token metrics correctly (average), not scalar sum
        metrics = {}
        for k, v in all_metrics.items():
            if k in self._per_token_metrics:
                metrics[k] = v / n_micro  # Per-token: average across micro-batches
            elif k.startswith("_loss_count/"):
                continue  # Skip count keys, used below for per-quality loss computation
            else:
                metrics[k] = v  # Scalar: sum is correct (e.g., policy_loss is already normalized)
        if n_micro > 1 and "loss" not in self._per_token_metrics:
            # Total loss is sum of micro-batch losses (each already divided by n_micro)
            metrics["loss"] = total_loss
        else:
            metrics["loss"] = total_loss

        # Per-quality loss: compute mean from accumulated sum and count
        # This gives mean per-token loss for each quality span across all samples
        for k in list(all_metrics.keys()):
            if k.startswith("loss/") and not k.startswith("_loss_count/"):
                q_name = k[5:]  # Strip "loss/" prefix
                count_key = f"_loss_count/{q_name}"
                if count_key in all_metrics and all_metrics[count_key] > 0:
                    metrics[k] = all_metrics[k] / all_metrics[count_key]
                else:
                    metrics[k] = 0.0

        # Track accumulated loss across gradient accumulation steps
        self._record_microbatch(len(step), total_loss)

        # Optimizer step (backward already done in micro-batches above)
        if (self.global_step + 1) % self.config.training.gradient_accumulation_steps == 0:
            # TL-8: Guard gradient probe initialization with batch size check
            if (
                self._gradient_probe_steps > 0
                and self.global_step < self._gradient_probe_steps
                and self._gradient_probe_prompt_ids is None
            ):
                if batch.input_ids.shape[0] == 0 or batch.input_ids.shape[1] == 0:
                    logging.getLogger(__name__).warning(
                        "TL-8: Cannot initialize gradient probe with empty batch"
                    )
                else:
                    # Get model device
                    model_device = next(self.model.parameters()).device

                    # Use first batch prompt as fixed measurement prompt
                    self._gradient_probe_prompt_ids = (
                        batch.input_ids[0:1].clone().to(model_device)
                    )  # [1, seq_len]

                    # Physics tokens: p, V, T, H, x, m
                    physics_strs = ["p", "V", "T", "H", "x", "m"]
                    self._gradient_probe_physics_tokens = []
                    for tok_str in physics_strs:
                        tok_ids = self.tokenizer.encode(tok_str, add_special_tokens=False)
                        if len(tok_ids) == 1:
                            self._gradient_probe_physics_tokens.append(tok_ids[0])

                    if not self._gradient_probe_physics_tokens:
                        # Fallback: use top-50 most common tokens if physics tokens aren't single-token
                        self._gradient_probe_physics_tokens = list(range(50, 100))

            # Gradient probe: capture logits BEFORE optimizer step
            logits_before = None
            if (
                self._gradient_probe_steps > 0
                and self.global_step < self._gradient_probe_steps
                and self._gradient_probe_prompt_ids is not None
            ):
                with torch.no_grad():
                    self.model.eval()
                    # TL-11: Move probe tensor to model device each time
                    model_device = next(self.model.parameters()).device
                    probe_ids = self._gradient_probe_prompt_ids.to(model_device)
                    probe_outputs = self.model(probe_ids)
                    # Get logits for last position, physics tokens only
                    physics_tokens_tensor = torch.tensor(
                        self._gradient_probe_physics_tokens, device=probe_outputs.logits.device
                    )
                    logits_before = probe_outputs.logits[0, -1, physics_tokens_tensor].clone()
                    self.model.train()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.config.training.max_grad_norm
            )

            # Gradient coherence monitoring — detect laminar→turbulent transitions
            # Measures "eddies forming": layers disagreeing, energy dissipating into noise
            # MUST happen AFTER backward + clipping, BEFORE optimizer.step() (which zeros grads)
            # Note: global_step increments at END of step(), so check (global_step + 1) to log at correct step
            if self._log_attention and (self.global_step + 1) % self._attention_log_freq == 0:
                from qgre.gradient_coherence import compute_gradient_coherence

                coherence_stats = compute_gradient_coherence(self.model)

                # Diagnostic: check if near-zero cosine is real or artifact
                if self.global_step < 100 and self.global_step % 30 == 0:
                    print(f"[GRADIENT DEBUG step {self.global_step}]")
                    print(
                        f"  Layers: {coherence_stats['n_layers']}, Comparisons: {coherence_stats['n_comparisons']}, Nonzero norms: {coherence_stats['nonzero_norms']}"
                    )
                    print(
                        f"  Cosine range: [{coherence_stats['min_cosine']:.6f}, {coherence_stats['max_cosine']:.6f}], mean={coherence_stats['mean_cosine']:.6f}"
                    )
                    print(f"  First 5 layer norms: {coherence_stats['per_layer_norms'][:5]}")
                    print(f"  First 5 cosines: {coherence_stats['per_layer_cosines'][:5]}")

                coherence_stats["step"] = self.global_step
                coherence_stats["phase"] = self.game_state.phase
                coherence_stats["reward"] = metrics.get("reward/mean", 0.0)
                coherence_stats["loss"] = metrics.get("loss", 0.0)

                self._attention_log.append(coherence_stats)

                # Add key metrics to step metrics for MLflow logging
                metrics["grad/temporal_cosine"] = coherence_stats["temporal_cosine"]
                metrics["grad/spatial_cosine"] = coherence_stats["spatial_cosine"]
                metrics["grad/mean_cosine"] = coherence_stats["mean_cosine"]  # backward compat
                metrics["grad/min_cosine"] = coherence_stats["min_cosine"]
                metrics["grad/norm_ratio"] = coherence_stats["norm_ratio"]
                metrics["grad/mean_norm"] = coherence_stats["mean_grad_norm"]
                metrics["grad/lora_weight_norm"] = coherence_stats["lora_weight_norm"]

                # Turbulence detection (if detector initialized)
                if not hasattr(self, "_turbulence_detector"):
                    from qgre.gradient_coherence import TurbulenceDetector

                    self._turbulence_detector = TurbulenceDetector(
                        calibration_steps=30,
                        cosine_threshold_low=0.02,
                        cosine_threshold_high=0.10,
                        transition_window=5,
                    )

                state = self._turbulence_detector.update(self.global_step, coherence_stats)
                metrics["grad/turbulence_state"] = {
                    "CALIBRATING": 0,
                    "LAMINAR": 1,
                    "TRANSITIONAL": 2,
                    "TURBULENT": 3,
                }.get(state, 0)

            # LoRA-Pro: Adjust gradients before optimizer step
            if self._lora_pro_adjuster is not None:
                lora_pro_metrics = self._lora_pro_adjuster.adjust_gradients(self.global_step)
                metrics.update(lora_pro_metrics)

            self.optimizer.step()

            # Gradient probe: capture logits AFTER optimizer step and compute delta
            if (
                self._gradient_probe_steps > 0
                and self.global_step < self._gradient_probe_steps
                and logits_before is not None
            ):
                with torch.no_grad():
                    self.model.eval()
                    probe_outputs = self.model(self._gradient_probe_prompt_ids)
                    physics_tokens_tensor = torch.tensor(
                        self._gradient_probe_physics_tokens, device=probe_outputs.logits.device
                    )
                    logits_after = probe_outputs.logits[0, -1, physics_tokens_tensor].clone()
                    self.model.train()

                    # Compute logit delta (absolute change, averaged across physics tokens)
                    logit_delta = (logits_after - logits_before).abs()
                    mean_delta = logit_delta.mean().item()
                    max_delta = logit_delta.max().item()

                    self._gradient_probe_log.append(
                        {
                            "step": self.global_step,
                            "loss": total_loss,
                            "advantage_scale": self.config.algorithm.advantage_scale,
                            "mean_logit_delta": mean_delta,
                            "max_logit_delta": max_delta,
                            "per_token_deltas": logit_delta.cpu().tolist(),
                        }
                    )
            # Clear optimizer momentum state to prevent double-stepping on resume
            if hasattr(self, "_resumed_mid_accumulation") and self._resumed_mid_accumulation:
                for group in self.optimizer.param_groups:
                    for p in group["params"]:
                        state = self.optimizer.state[p]
                        if "exp_avg" in state:
                            state["exp_avg"].zero_()
                        if "exp_avg_sq" in state:
                            state["exp_avg_sq"].zero_()
            self.optimizer.zero_grad()
            # Clear flag AFTER momentum clearing
            if hasattr(self, "_resumed_mid_accumulation"):
                self._resumed_mid_accumulation = False
            # VPRM critic optimizer steps at same cadence as policy optimizer
            if self.vprm_critic is not None and self.vprm_optimizer is not None:
                has_grad = any(
                    p.grad is not None and p.grad.abs().sum() > 0
                    for p in self.vprm_critic.parameters()
                )
                if has_grad:
                    torch.nn.utils.clip_grad_norm_(
                        self.vprm_critic.parameters(), max_norm=self.config.training.max_grad_norm
                    )
                    self.vprm_optimizer.step()
                    # Polyak-average target network (or hard-sync during warmup)
                    if self._vprm_config.use_target_network:
                        if (
                            self.global_step < self._vprm_config.target_warmup_steps
                            and self.global_step % 100 == 0
                        ):
                            self.vprm_critic.sync_target_to_online()
                        elif self.global_step >= self._vprm_config.target_warmup_steps:
                            self.vprm_critic.update_target_network(tau=self._vprm_config.polyak_tau)
                # Always zero VPRM optimizer regardless of has_grad
                self.vprm_optimizer.zero_grad()
                # Divergence monitoring — independent of has_grad (reads .data, no grad needed)
                if self.global_step % 50 == 0:
                    with torch.no_grad():
                        # Single .item() call to avoid 54 GPU syncs
                        divergence = sum(
                            (op.data - tp.data).pow(2).mean()
                            for q in self.vprm_critic.quality_names
                            for op, tp in zip(
                                self.vprm_critic.heads[q].parameters(),
                                self.vprm_critic.target_heads[q].parameters(),
                                strict=False,
                            )
                        ).item()  # type: ignore[union-attr]
                        if not math.isfinite(divergence):
                            import warnings

                            warnings.warn(
                                f"Step {self.global_step}: target_divergence={divergence} — critic may contain NaN",
                                stacklevel=2,
                            )
                        metrics["target_divergence"] = divergence
            if self.scheduler is not None:
                self.scheduler.step()
            # TL-R2-04: Divide by accumulated samples, not accumulation count
            actual_samples = self._accumulated_samples if self._accumulated_samples > 0 else 1
            metrics["accumulated_loss"] = self._accumulated_loss / actual_samples
            self._reset_accumulators()
        elif self._accumulated_samples > 0:
            actual_samples = self._accumulated_samples
            metrics["accumulated_loss"] = self._accumulated_loss / actual_samples

        # KL-adaptive SPO learning rate (SPO paper Algorithm 1)
        spo_cfg = self.config.algorithm.spo
        if spo_cfg.kl_adaptive and self.config.algorithm.mode == "spo":
            kl_val = metrics.get("kl_penalty", 0.0)
            self.advantage_estimator.adapt_lr(
                kl=kl_val,
                kl_threshold=spo_cfg.kl_threshold,
                kl_factor=spo_cfg.kl_factor,
                lr_factor=spo_cfg.lr_factor,
                min_lr=spo_cfg.min_lr,
                max_lr=spo_cfg.max_lr,
            )

        # Log metrics (use step.X for all step-level data)
        reward_mean = sum(rr.reward for rr in step.reward_results) / len(step)
        metrics["reward/mean"] = reward_mean
        metrics["global_step"] = self.global_step
        # Advantage observability — computed at filter time, see SPO filter above.
        metrics.update(self._last_advantage_stats)

        # Track completion lengths for verbosity drift detection (uses filtered step data)
        comp_lengths = [len(c) for c in step.completions]
        if comp_lengths:
            metrics["completion_length/mean"] = float(sum(comp_lengths) / len(comp_lengths))
            metrics["completion_length/max"] = float(max(comp_lengths))
            metrics["completion_length/min"] = float(min(comp_lengths))
        else:
            metrics["completion_length/mean"] = 0.0
            metrics["completion_length/max"] = 0.0
            metrics["completion_length/min"] = 0.0

        self._record_mastery_and_advance(
            step.reward_results,
            step.active_qualities,
            batch,
            metrics,
            batch_contexts=step.batch_contexts,
        )

        # Log completions — use get_original_idx to map filtered→original for batch.raw_prompts
        completions_text = []
        for i, rr in enumerate(step.reward_results):
            orig_i = step.get_original_idx(i)  # Map filtered index to original batch index
            # Decode token IDs to text for readable logs
            comp_tokens = step.completions[i]
            if self.tokenizer is not None:
                comp_text = self.tokenizer.decode(comp_tokens, skip_special_tokens=True)
            else:
                comp_text = str(comp_tokens)
            completions_text.append(comp_text)
            self.completion_logger.log_completion(
                step=self.global_step,
                prompt=batch.raw_prompts[orig_i] if orig_i < len(batch.raw_prompts) else "",
                completion=comp_text,
                reward=rr.reward,
                reward_components=rr.scores,
                phase=self.game_state.phase,
            )
        self.global_step += 1
        self.ctx.step = self.global_step
        return metrics

    def save(self, path: str | Path | None = None):
        """Save checkpoint."""
        if path is None:
            path = Path(self.config.logging.checkpoint_dir) / f"global_step_{self.global_step}.pt"

        # Build TrainerState with all mutable state
        trainer_state = TrainerState(
            global_step=self.global_step,
            accumulated_loss=self._accumulated_loss,
            accumulation_count=self._accumulation_count,
            accumulated_samples=self._accumulated_samples,
            resumed_mid_accumulation=getattr(self, "_resumed_mid_accumulation", False),
            fused_validated=getattr(self, "_fused_validated", False),
            triton_validated=getattr(self, "_triton_validated", False),
            needs_weight_sync=getattr(self, "_needs_weight_sync", False),
            rng_state=torch.get_rng_state(),
            cuda_rng_state=torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            mlflow_run_id=getattr(self, "_mlflow_run_id", None),
        )

        # Persist SyncState through the legacy WeightLoaderState container so
        # restore_failed and the lifecycle survive a restart. SyncState.state_dict()
        # uses uppercase enum names ("READY"); WeightLoaderLifecycle is lowercase
        # ("ready") — convert at the seam.
        sync_dict = self.sync_state.state_dict()
        weight_loader_state = WeightLoaderState(
            initialized=sync_dict["initialized"],
            restore_failed=sync_dict["restore_failed"],
            lifecycle=sync_dict["lifecycle"].lower(),
        )

        save_checkpoint(
            path=path,
            global_step=self.global_step,
            model_state_dict=self.model.state_dict(),
            optimizer_state_dict=self.optimizer.state_dict() if self.optimizer else None,
            scheduler_state_dict=self.scheduler.state_dict() if self.scheduler else None,
            game_state=self.game_state,
            advantage_estimator_state=self.advantage_estimator.state_dict(),
            vprm_critic_state=self.vprm_critic.state_dict_with_meta() if self.vprm_critic else None,
            vprm_optimizer_state=self.vprm_optimizer.state_dict() if self.vprm_optimizer else None,
            dataloader_state=self._dataloader.state_dict() if self._dataloader else None,
            training_context=self.ctx.to_dict(),
            hint_registry_state=self.hint_registry.to_dict() if self.hint_registry else None,
            lora_pro_state=self._lora_pro_adjuster.state_dict()
            if self._lora_pro_adjuster
            else None,
            trainer_state=trainer_state,  # Use StateSpec instead of individual fields
            weight_loader_state=weight_loader_state,
        )

    def resume(self, checkpoint_dir: str | Path) -> bool:
        """Try to resume from latest checkpoint. Returns True if resumed."""
        latest = discover_latest_checkpoint(checkpoint_dir)
        if latest is None:
            return False

        # Clear stale gradient cache — temporal cosine must not compare against
        # gradients from a different model state (pre-checkpoint).
        from qgre.gradient_coherence import reset_gradient_cache

        reset_gradient_cache()

        checkpoint = load_checkpoint(latest)
        # CheckpointState: access trainer fields via checkpoint.trainer
        self.global_step = checkpoint.trainer.global_step

        if not checkpoint.model_state_dict:
            raise RuntimeError(
                f"Checkpoint {latest} missing model_state_dict — cannot resume with random weights",
            )
        # Filter out bitsandbytes quantization metadata keys that don't exist in
        # a freshly initialized model (absmax, quant_map, quant_state, etc.)
        # These are saved by model.state_dict() but cause errors on load_state_dict()
        # because bnb creates them lazily during quantization, not at init time.
        state_dict = checkpoint.model_state_dict
        model_keys = set(self.model.state_dict().keys())
        filtered = {k: v for k, v in state_dict.items() if k in model_keys}
        skipped = len(state_dict) - len(filtered)
        if skipped > 0:
            import warnings

            warnings.warn(
                f"Checkpoint resume: skipped {skipped} keys not in model (bnb quant metadata)",
                stacklevel=2,
            )
        self.model.load_state_dict(filtered, strict=False)

        # Validate optimizer compatibility before loading
        optimizer_loaded = False
        if checkpoint.optimizer_state_dict and self.optimizer:
            ckpt_opt = checkpoint.optimizer_state_dict
            # R2-CSM-002: Validate optimizer state dict structure before load
            if (
                not isinstance(ckpt_opt, dict)
                or "state" not in ckpt_opt
                or "param_groups" not in ckpt_opt
            ):
                import warnings

                warnings.warn(
                    "R2-CSM-002: Malformed optimizer_state_dict (missing 'state' or 'param_groups'). "
                    "Skipping optimizer restore, momentum lost.",
                    stacklevel=2,
                )
            else:
                ckpt_groups = len(ckpt_opt.get("param_groups", []))
                model_groups = len(self.optimizer.param_groups)

                if ckpt_groups != model_groups:
                    print(f"┌{'─' * 60}┐")
                    print(f"│{'⚠️  CHECKPOINT CONFIG MISMATCH':^60}│")
                    print(f"├{'─' * 60}┤")
                    print(
                        f"│  Checkpoint optimizer: {ckpt_groups} param groups{' ' * (35 - len(str(ckpt_groups)))}│"
                    )
                    print(
                        f"│  Current optimizer:    {model_groups} param groups{' ' * (35 - len(str(model_groups)))}│"
                    )
                    print(f"│  (modules_to_save likely changed){' ' * 24}│")
                    print(f"├{'─' * 60}┤")
                    print(f"│  Model weights: ✓ LOADED{' ' * 34}│")
                    print(f"│  Optimizer state: ✗ RESET (momentum lost){' ' * 16}│")
                    print(f"│  Game state: ✓ LOADED{' ' * 36}│")
                    print(f"│  Step counter: ✓ LOADED{' ' * 34}│")
                    print(f"└{'─' * 60}┘")
                else:
                    has_nan = False
                    for state in ckpt_opt.get("state", {}).values():
                        if any(
                            torch.isnan(v).any() if isinstance(v, torch.Tensor) else False
                            for v in state.values()
                        ):
                            has_nan = True
                            break
                    if has_nan:
                        import warnings

                        warnings.warn(
                            "Optimizer state contains NaN values. Skipping load to prevent corruption.",
                            stacklevel=2,
                        )
                    else:
                        self.optimizer.load_state_dict(ckpt_opt)
                        optimizer_loaded = True

        if checkpoint.scheduler_state_dict and self.scheduler:
            # R2-CSM-003: Validate scheduler state dict is non-empty before load
            if (
                not isinstance(checkpoint.scheduler_state_dict, dict)
                or not checkpoint.scheduler_state_dict
            ):
                import warnings

                warnings.warn(
                    "R2-CSM-003: Empty or invalid scheduler_state_dict. "
                    "Skipping scheduler restore, learning rate schedule will restart.",
                    stacklevel=2,
                )
            else:
                # Load scheduler state even when optimizer is skipped (mid-accumulation NaN)
                import warnings  # Import here for use in all branches below

                if not optimizer_loaded:
                    warnings.warn(
                        "Optimizer skipped due to NaN but scheduler will still load. "
                        "Learning rate schedule continues from checkpoint.",
                        stacklevel=2,
                    )
                # Check if T_max changed (indicating new schedule)
                if hasattr(self.scheduler, "T_max"):
                    ckpt_T_max = checkpoint.scheduler_state_dict.get("T_max")
                    current_T_max = self.scheduler.T_max
                    if ckpt_T_max is not None and ckpt_T_max != current_T_max:
                        warnings.warn(
                            f"Scheduler T_max changed: checkpoint={ckpt_T_max}, config={current_T_max}. "
                            "Skipping scheduler state restore — learning rate schedule will restart. "
                            "This is expected if you changed total_steps in config.",
                            stacklevel=2,
                        )
                    else:
                        try:
                            self.scheduler.load_state_dict(checkpoint.scheduler_state_dict)
                        except (ValueError, KeyError) as e:
                            warnings.warn(
                                f"Scheduler state incompatible with current config, resetting: {e}. "
                                "Learning rate schedule will restart from step 0.",
                                stacklevel=2,
                            )
                else:
                    try:
                        self.scheduler.load_state_dict(checkpoint.scheduler_state_dict)
                    except (ValueError, KeyError) as e:
                        warnings.warn(
                            f"Scheduler state incompatible with current config, resetting: {e}. "
                            "Learning rate schedule will restart from step 0.",
                            stacklevel=2,
                        )
        if checkpoint.game_state:
            self.game_state = checkpoint.game_state
        # CheckpointState: advantage_estimator is AdvantageEstimatorState which wraps state_dict
        if checkpoint.advantage_estimator and checkpoint.advantage_estimator.state_dict:
            self.advantage_estimator.load_state_dict(checkpoint.advantage_estimator.state_dict)
        # CP3-003: Warn if checkpoint has critic but config disabled
        if checkpoint.vprm_critic_state and not self.config.vprm.enabled:
            import warnings

            warnings.warn(
                "CP3-003: Checkpoint contains VPRM critic state but config.vprm.enabled=False. "
                "Critic will not be restored. Set vprm.enabled=True to restore critic.",
                stacklevel=2,
            )
        # Restore VPRM critic + optimizer (if saved)
        if checkpoint.vprm_critic_state and self.config.vprm.enabled:
            from qgre.critic import VPRMCritic

            device = next(
                (p.device for p in self.model.parameters() if p.device.type != "cpu"),
                next(self.model.parameters()).device,
            )

            # R2-CSM-001: Validate required keys exist before access
            required_keys = {"hidden_dim", "step_qualities", "model_state"}
            missing_keys = required_keys - set(checkpoint.vprm_critic_state.keys())
            if missing_keys:
                raise KeyError(
                    f"R2-CSM-001: VPRM critic state missing required keys: {missing_keys}. "
                    f"Available keys: {list(checkpoint.vprm_critic_state.keys())}. "
                    "Checkpoint may be corrupted or from incompatible version.",
                )

            # CSM-007: Validate step_qualities match between checkpoint and config (raise, not warn)
            ckpt_step_qualities = checkpoint.vprm_critic_state["step_qualities"]
            config_step_qualities = self.step_qualities
            if ckpt_step_qualities != config_step_qualities:
                raise ValueError(
                    f"CSM-007: step_qualities mismatch between checkpoint and config. "
                    f"Checkpoint: {ckpt_step_qualities}, Config: {config_step_qualities}. "
                    "Cannot restore VPRM critic with different step_qualities (shape mismatch). "
                    "Update config to match checkpoint, or start fresh training.",
                )

            # C14: Restore with intermediate_dim from current config, not checkpoint
            # (allows config changes without breaking restore)
            self.vprm_critic = VPRMCritic(
                hidden_dim=checkpoint.vprm_critic_state["hidden_dim"],
                step_qualities=config_step_qualities,  # Use current config
                intermediate_dim=self._vprm_config.intermediate_dim,  # C14: from config
                step_region_map=checkpoint.vprm_critic_state.get("step_region_map"),
            )
            # Load weights with strict=False to handle architecture mismatches
            self.vprm_critic.load_state_dict(
                checkpoint.vprm_critic_state["model_state"], strict=False
            )
            self.vprm_critic.to(device)

            # Store checkpoint hidden_dim for validation on first forward pass
            self._vprm_checkpoint_hidden_dim = checkpoint.vprm_critic_state["hidden_dim"]
            self.vprm_optimizer = torch.optim.Adam(
                [p for p in self.vprm_critic.parameters() if p.requires_grad],
                lr=self._vprm_config.lr,
            )
            if checkpoint.vprm_optimizer_state:
                # R2-CSM-006: Validate vprm_optimizer_state has required structure
                if (
                    not isinstance(checkpoint.vprm_optimizer_state, dict)
                    or "state" not in checkpoint.vprm_optimizer_state
                    or "param_groups" not in checkpoint.vprm_optimizer_state
                ):
                    import warnings

                    warnings.warn(
                        "R2-CSM-006: Malformed vprm_optimizer_state (missing 'state' or 'param_groups'). "
                        "Skipping VPRM optimizer restore.",
                        stacklevel=2,
                    )
                else:
                    self.vprm_optimizer.load_state_dict(checkpoint.vprm_optimizer_state)
            self._vprm_initialized = True
        # Restore EGRS hint registry (if saved)
        if checkpoint.hint_registry_state and self.config.egrs.enabled:
            from qgre.hints import HintRegistry

            self.hint_registry = HintRegistry.from_dict(checkpoint.hint_registry_state)
            logging.getLogger(__name__).info(
                f"EGRS: Restored hint registry with {len(self.hint_registry)} entries",
            )
        # CSM-003: Warn if hint_registry_state exists but EGRS disabled
        elif checkpoint.hint_registry_state and not self.config.egrs.enabled:
            # H-5: Check isinstance before .get() to avoid AttributeError on non-dict
            if isinstance(checkpoint.hint_registry_state, dict):
                hint_count = len(checkpoint.hint_registry_state.get("hints", []))
            else:
                hint_count = 0
            import warnings

            warnings.warn(
                f"CT-1: Checkpoint contains {hint_count} hint entries but EGRS is disabled in config. "
                f"All {hint_count} flagged hints will be LOST. "
                "Set egrs.enabled=True to restore hint registry.",
                stacklevel=2,
            )
        # Restore LoRA-Pro momentum state (if saved and enabled)
        if checkpoint.lora_pro_state and self._lora_pro_adjuster is not None:
            self._lora_pro_adjuster.load_state_dict(checkpoint.lora_pro_state)
            logging.getLogger(__name__).info(
                f"LoRA-Pro: Restored momentum state for {len(checkpoint.lora_pro_state)} layers",
            )
        elif checkpoint.lora_pro_state and self._lora_pro_adjuster is None:
            import warnings

            warnings.warn(
                "Checkpoint contains LoRA-Pro state but lora_pro.enabled=False in config. "
                "LoRA-Pro momentum will not be restored.",
                stacklevel=2,
            )
        # CheckpointState: RNG state is in trainer
        if checkpoint.trainer.rng_state is not None:
            import warnings

            warnings.warn(
                "Restoring RNG state from checkpoint. Note: this overrides config.training.seed. "
                "If you changed the seed in config after this checkpoint, it will be ignored.",
                stacklevel=2,
            )
            torch.set_rng_state(checkpoint.trainer.rng_state)  # type: ignore[arg-type]
        if checkpoint.trainer.cuda_rng_state is not None and torch.cuda.is_available():
            # CP3-002: Save device index in checkpoint, restore to same device
            # Note: checkpoint format already includes device index in cuda_rng_state
            # This is correct — no fix needed, but documenting expected behavior
            torch.cuda.set_rng_state(checkpoint.trainer.cuda_rng_state)  # type: ignore[arg-type]

        # TRN-R2-1: Restore accumulated loss from TrainerState
        # Reset partial accumulation to avoid mixing batch sizes
        if (
            checkpoint.trainer.accumulation_count > 0
            and checkpoint.trainer.accumulation_count
            < self.config.training.gradient_accumulation_steps
        ):
            logging.getLogger(__name__).warning(
                f"Checkpoint saved mid-accumulation (count={checkpoint.trainer.accumulation_count}). "
                "Resetting accumulated loss and scheduler state to avoid batch size inconsistency.",
            )
            self._reset_accumulators()
            # Reset scheduler last_epoch to match global_step
            if self.scheduler is not None and hasattr(self.scheduler, "last_epoch"):
                # Scheduler last_epoch should match optimizer steps, not global_step
                # Since we reset accumulation, the next optimizer step will be at the correct boundary
                pass  # Document that last_epoch is not reset - it tracks optimizer steps, not accumulation
        else:
            self._accumulated_loss = checkpoint.trainer.accumulated_loss
            self._accumulated_samples = checkpoint.trainer.accumulated_samples
            self._accumulation_count = checkpoint.trainer.accumulation_count
            if not torch.isfinite(torch.tensor(self._accumulated_loss)):
                logging.getLogger(__name__).warning(
                    f"Accumulated loss is not finite ({self._accumulated_loss}). Resetting accumulators."
                )
                self._reset_accumulators()
        # DI1: Store dataloader state for restore in train() when dataloader is available
        # CheckpointState: dataloader is now DataLoaderState dataclass
        from dataclasses import asdict

        self._pending_dataloader_state = (
            asdict(checkpoint.dataloader) if checkpoint.dataloader else None
        )

        # Restore training context (device, dtype, step)
        if checkpoint.training_context:
            self.ctx = TrainingContext.from_dict(checkpoint.training_context)
        else:
            # Fallback: reconstruct from global_step if not in checkpoint (backward compat)
            model_devices = [p.device for p in self.model.parameters()]
            if model_devices:
                _device = next((d for d in model_devices if d.type != "cpu"), model_devices[0])
            else:
                _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.ctx = TrainingContext.from_config(self.config, device=str(_device))
            self.ctx.step = self.global_step

        # TRN-R2-5: Compute _resumed_mid_accumulation at resume time
        # This flag means "we JUST resumed mid-accumulation, clear momentum on next step"
        # Must be determined by current resume state, not restored from checkpoint
        n_accum = self.config.training.gradient_accumulation_steps
        self._resumed_mid_accumulation = self.global_step % n_accum != 0

        # Restore validation flags from checkpoint — same weights, same validation status.
        # Prevents wasted 1.2 GB cross-validation allocation on first step after resume.
        self._fused_validated = checkpoint.trainer.fused_validated
        self._triton_validated = getattr(checkpoint.trainer, "triton_validated", False)

        # R3-T1: Restore MLflow run_id for cross-resume continuity
        self._mlflow_run_id = checkpoint.trainer.mlflow_run_id

        # W11: Restore WeightLoaderState to generation_backend.weight_loader
        # FIX 7: Always restore restore_failed, even if weight_loader path is skipped
        if checkpoint.weight_loader:
            wl_state = checkpoint.weight_loader
            # Always restore restore_failed to sync_state (it may be on a different object)
            self.sync_state.restore_failed = wl_state.restore_failed

        if (
            checkpoint.weight_loader
            and hasattr(self, "generation_backend")
            and self.generation_backend is not None
            and hasattr(self.generation_backend, "weight_loader")
            and self.generation_backend.weight_loader is not None
        ):
            wl_state = checkpoint.weight_loader
            # Drive lifecycle through SyncState rather than the read-only legacy
            # _direct_ready / _load_lora_called properties (which now derive from
            # state.lifecycle and have no setters). By the time resume() runs in
            # train(), backend.weight_loader._state has already been swapped to
            # self.sync_state, so mutations here land on the authoritative state.
            loader_state = self.generation_backend.weight_loader._state
            from qgre.sync_state import SyncLifecycle

            # R3-C3: Restore lifecycle directly from checkpoint to preserve ERROR states
            lifecycle_str = (wl_state.lifecycle or "uninitialized").upper()
            try:
                loader_state.lifecycle = SyncLifecycle[lifecycle_str]
            except KeyError:
                # Unknown lifecycle value — fall back to UNINITIALIZED to force a clean re-sync
                import warnings

                warnings.warn(
                    f"Unknown lifecycle '{wl_state.lifecycle}' in checkpoint, falling back to UNINITIALIZED",
                    stacklevel=2,
                )
                loader_state.lifecycle = SyncLifecycle.UNINITIALIZED

            loader_state.initialized = wl_state.initialized
            # Note: restore_failed already restored above (outside this block)
            # W1: Restore lora_request_id if present (backward compat: may be None in old checkpoints)
            if hasattr(wl_state, "lora_request_id") and wl_state.lora_request_id is not None:
                # Note: _lora_request is the actual LoRARequest object, not restored from checkpoint
                # This field is for tracking re-registration; actual object rebuilt on first sync
                pass  # Currently no restoration needed — document that re-registration is intentional
            # Don't restore cleaned_up=True (would prevent future use)
            if not wl_state.cleaned_up:
                self.generation_backend.weight_loader._cleaned_up = False

        # Zero gradients AFTER resume to avoid clearing loaded optimizer state
        if self.optimizer is not None:
            self.optimizer.zero_grad()
        else:
            raise RuntimeError(
                "Optimizer is None after checkpoint resume. "
                "Check that optimizer was saved and restored correctly.",
            )
        if self.vprm_optimizer is not None:
            self.vprm_optimizer.zero_grad()

        # LoRA verification on resume (PLAN.md line 487-488: mandatory step)
        try:
            from qgre.lora_verify import LoRAVerifier
        except ImportError as e:
            # Only silence if Unsloth is missing — warn for unexpected import errors
            if "unsloth" in str(e).lower():
                pass  # Expected — Unsloth not installed
            else:
                import warnings

                warnings.warn(f"LoRA verification import failed unexpectedly: {e}", stacklevel=2)
        else:
            try:
                is_active = LoRAVerifier.verify_active(
                    self.model, self.tokenizer, state=self.sync_state
                )
                if not is_active:
                    import warnings

                    warnings.warn(
                        "LoRA verification returned False after resume. "
                        "LoRA adapters may not be active. Check model state.",
                        stacklevel=2,
                    )
            except (RuntimeError, ValueError, AttributeError) as e:
                import warnings

                warnings.warn(
                    f"LoRA verification failed after resume: {e}. "
                    f"LoRA adapters may not be active. Check model state.",
                    stacklevel=2,
                )

        # WS2: Flag that vLLM needs weight sync (model weights restored but vLLM has stale weights)
        self._needs_weight_sync = True

        return True

    def _get_prompt_tier(self, metadata: dict) -> str:
        """Get the difficulty tier for a prompt from its metadata."""
        if self._difficulty_column:
            return metadata.get(self._difficulty_column, "default")
        return "default"

    def _record_mastery_and_advance(
        self, reward_results, active_qualities, batch, metrics, batch_contexts
    ):
        """Record per-tier mastery scores, check per-tier phase advancement, check tier unlock."""
        from collections import defaultdict

        import numpy as np

        max_phase = max(self.step_qualities.keys())

        # Group reward results by tier
        # Note: reward_results and batch_contexts are FILTERED (post-SPO), batch.metadata is UNFILTERED
        # Use batch_contexts[i].tier which is already computed and aligned with filtered indices
        tier_groups = defaultdict(list)
        for i, rr in enumerate(reward_results):
            tier = batch_contexts[i].tier
            tier_groups[tier].append((rr, active_qualities[i]))

        # Record mastery per tier, check per-tier quality phase advance
        for tier, items in tier_groups.items():
            tier_active_qs = items[0][1]  # All items in same tier share active qualities
            for step_num, quality_keys in self.step_qualities.items():
                active_keys = [k for k in quality_keys if k in tier_active_qs]
                if active_keys:
                    scores = [
                        float(np.mean([rr.scores.get(k, 0.0) for k in active_keys]))
                        for rr, _ in items
                    ]
                    mean_score = float(np.mean(scores))
                    self.game_state.record_tier_step_score(tier, step_num, mean_score)
                    metrics[f"mastery/{tier}/step_{step_num}"] = mean_score

            if self.game_state.check_tier_phase_advance(tier, max_phase):
                new_phase = self.game_state.tier_phases[tier]
                metrics[f"tier_phase_advanced/{tier}"] = new_phase
                # Reset SPO baselines for ALL prompts in this tier (not just current batch)
                col = self._difficulty_column
                if self._dataloader and col:
                    tier_pids = [
                        item["prompt_id"]
                        for item in self._dataloader.items
                        if item.get("metadata", {}).get(col, "default") == tier
                    ]
                    import warnings

                    warnings.warn(
                        f"[RESET TRIGGER] tier={tier}, phase→{new_phase}, dataloader path, found {len(tier_pids)} prompts",
                        stacklevel=2,
                    )
                else:
                    tier_pids = [
                        batch_contexts[i].prompt_id
                        for i in range(len(reward_results))
                        if batch_contexts[i].tier == tier
                    ]
                    import warnings

                    warnings.warn(
                        f"[RESET TRIGGER] tier={tier}, phase→{new_phase}, batch path, found {len(tier_pids)} prompts",
                        stacklevel=2,
                    )
                self.advantage_estimator.on_tier_advance(
                    new_tier=new_phase,
                    prompt_tier_map=dict.fromkeys(tier_pids, new_phase),  # type: ignore[arg-type]
                )

        # Check tier unlock — tutorial gates tier advancement
        if self._tier_order:
            # Pre-check: which tier WOULD be next?
            active_set = set(self.game_state.active_tiers)
            candidate_tier = None
            for t in self._tier_order:
                if t not in active_set:
                    candidate_tier = t
                    break

            # Only attempt unlock if tutorial allows it
            if candidate_tier is None or self.game_state.can_tier_unlock(candidate_tier):
                new_tier = self.game_state.check_tier_unlock(
                    self._tier_order,
                    self._tier_advance_phase,
                    self._tier_advance_threshold,
                )
                if new_tier:
                    metrics["tier_unlocked"] = new_tier
                    print(f"\n┌{'─' * 60}┐")
                    print(f"│{'🔓 TIER UNLOCKED':^60}│")
                    print(f"├{'─' * 60}┤")
                    print(f"│  Step: {self.global_step:<51}│")
                    print(f"│  Tier: {new_tier:<51}│")
                    print(f"│  Active: {', '.join(self.game_state.active_tiers):<49}│")
                    print(f"└{'─' * 60}┘")
                    self._apply_difficulty_gate()
                    # Reset baselines for ALL prompts in ALL active tiers on tier unlock
                    # New tier means new prompt distribution — stale baselines must go
                    col = self._difficulty_column
                    if self._dataloader and col:
                        active_set = set(self.game_state.active_tiers)
                        all_active_pids = [
                            item["prompt_id"]
                            for item in self._dataloader.items
                            if item.get("metadata", {}).get(col, "default") in active_set
                        ]
                    else:
                        all_active_pids = [
                            batch_contexts[i].prompt_id for i in range(len(reward_results))
                        ]
                    self.advantage_estimator.on_tier_advance(
                        new_tier=0,  # Not used anymore — full reset for all affected pids
                        prompt_tier_map=dict.fromkeys(all_active_pids, 0),
                    )

        # Tutorial skill tree: record per-skill mastery score
        if self.game_state.tutorial_enabled:
            cache_snapshot = self.game_state.snapshot_pool_version()
            for i, rr in enumerate(reward_results):
                ctx = batch_contexts[i]
                score = self.game_state.resolve_mastery_score(ctx.prompt_id_str, rr)
                self.game_state.record_completion(ctx.prompt_id_str, score)
            # Re-apply difficulty gate if tutorial state changed (skill mastered/unlocked/relocked)
            if self.game_state.did_prompt_pool_change(cache_snapshot):
                self._apply_difficulty_gate()
            metrics.update(self.game_state.get_tutorial_metrics())

        self.game_state.step_count = self.global_step
        metrics["phase"] = self.game_state.phase

        # Per-tier stagnation
        for tier in self.game_state.active_tiers:
            stag = self.game_state.check_tier_stagnation(tier)
            metrics[f"stagnation/{tier}"] = {"normal": 0, "stagnating": 1, "stuck": 2}[stag.value]

    def _apply_difficulty_gate(self):
        """Apply difficulty gate using active_tiers from GameState.

        Sets dataloader to only sample prompts from active tiers, with equal
        weight per tier (prevents large tiers drowning small ones).

        When the tutorial system is enabled, tutorial-tracked prompts in active
        skills BYPASS the tier gate (tutorial is the authority for its prompts).
        Untracked prompts still respect the tier gate. Locked skill prompts are
        always zeroed out.
        """
        if self._dataloader is None or not hasattr(self._dataloader, "set_difficulty_gate"):
            return
        col = self._difficulty_column
        if not col:
            # Clear difficulty gate when tier_order is None
            if hasattr(self._dataloader, "_difficulty_gate"):
                self._dataloader._difficulty_gate = None
            return  # No difficulty column → no gating (default tier, all prompts)

        allowed = set(self.game_state.active_tiers)
        self._dataloader.set_difficulty_gate(allowed, col)

        # Tutorial: tracked prompts in active skills bypass tier gate
        tutorial_active_pids = None
        tutorial_tracked_pids = None
        if self.game_state.tutorial_enabled:
            tutorial_active_pids = set(self.game_state.get_active_prompts())
            tutorial_tracked_pids = set(self.game_state._prompt_to_skill.keys())

        from collections import Counter

        tier_counts = Counter(
            item["metadata"].get(col, "default") for item in self._dataloader.items
        )
        tier_weights = {}
        for item in self._dataloader.items:
            tier = item["metadata"].get(col, "default")
            pid = item["prompt_id"]
            pid_str = str(pid)

            if tutorial_tracked_pids is not None and pid_str in tutorial_tracked_pids:
                # Tutorial-tracked prompt: skill gate is the authority, tier gate bypassed
                if pid_str in tutorial_active_pids:  # type: ignore[operator]
                    tier_weights[pid] = 1.0 / max(tier_counts.get(tier, 1), 1)
                else:
                    tier_weights[pid] = 0.0  # Locked skill
            elif tier in allowed:
                tier_weights[pid] = 1.0 / max(tier_counts[tier], 1)
                # else: difficulty gate already zeros it out

        if tier_weights:
            self._dataloader.set_priorities(tier_weights)

        active_count = sum(1 for w in tier_weights.values() if w > 0) if tier_weights else 0
        tut_count = len(tutorial_active_pids) if tutorial_active_pids else "N/A"
        print(f"\n┌{'─' * 60}┐")
        print(f"│{'⚙ DIFFICULTY GATE':^60}│")
        print(f"├{'─' * 60}┤")
        print(f"│  Tiers: {', '.join(sorted(allowed)):<50}│")
        print(f"│  Active prompts: {active_count:<41}│")
        print(f"│  Tutorial active: {tut_count!s:<40}│")
        if active_count == 0:
            print(f"│  ⚠ ZERO PROMPTS — falling back to uniform{' ' * 17}│")
        print(f"└{'─' * 60}┘")

    def _record_microbatch(self, sample_count: int, loss_value: float) -> None:
        """Record one micro-batch contribution to gradient accumulation.

        The ONLY place outside __init__ and resume() where the accumulators
        advance. Keeps _accumulation_count, _accumulated_samples, and
        _accumulated_loss in lockstep.
        """
        self._accumulation_count += 1
        self._accumulated_samples += sample_count
        self._accumulated_loss += loss_value

    def _validate_logprob_path(
        self,
        mb_lp: torch.Tensor,
        hidden_states: torch.Tensor,
        lm_head: torch.nn.Module,
        mb_ids: torch.Tensor,
        *,
        atol: float,
        path_name: str,
    ) -> None:
        """One-time validation: verify grad_fn exists and output matches cuBLAS reference."""
        if mb_lp.grad_fn is None:
            raise RuntimeError(
                f"{path_name} logprobs has no grad_fn — autograd graph is broken.",
            )
        with torch.no_grad():
            manual_logits = lm_head(hidden_states[:, :-1, :]).float()
            std_lp = logprobs_from_logits(manual_logits, mb_ids[:, 1:])
            del manual_logits
            min_len = min(mb_lp.shape[1], std_lp.shape[1])
            max_diff = (mb_lp[:, :min_len].detach() - std_lp[:, :min_len]).abs().max().item()
            del std_lp
        if max_diff > atol:
            raise RuntimeError(
                f"{path_name} logprobs diverge from cuBLAS reference "
                f"(max diff: {max_diff:.6f} > {atol}). "
                f"Set algorithm.use_{path_name}_logprobs=false to fall back.",
            )
        logging.getLogger(__name__).info(
            f"{path_name} logprobs validated (max diff vs cuBLAS: "
            f"{max_diff:.2e}, within {atol} tolerance)",
        )

    def _skip_microbatch(self, reason: str) -> None:
        """Mark a micro-batch as skipped (e.g., SPO filtered all samples).

        Does NOT advance the accumulators — a skipped batch contributes
        nothing to gradient accumulation. Logs the reason for diagnostics.
        """
        logging.getLogger(__name__).debug(
            f"Skipped microbatch at global_step={self.global_step}: {reason}"
        )

    def _reset_accumulators(self) -> None:
        """Reset accumulators after an optimizer step. The third (and only
        other) place these fields mutate."""
        self._accumulation_count = 0
        self._accumulated_samples = 0
        self._accumulated_loss = 0.0

    def _sync_weights_guarded(
        self,
        weight_bus: Any,
        backend: GenerationBackend,
        *,
        reason: str,
        restore_lora: Callable[[], None] | None = None,
    ) -> None:
        """Run weight_bus.sync with the standard guard contract.

        Owns the entire sync invariant in one place:
          1. The pre-condition gate (delegated to weight_bus.sync, which calls
             self.sync_state.check_sync_allowed at the top).
          2. The try/except that runs cleanup on failure.
          3. The restore_lora callback if provided (dropout path).
          4. Translating any exception into a clear message tagged with `reason`.

        Args:
            weight_bus: the WeightBus instance for this train() invocation.
            backend: the generation backend (must have weight_exporter and weight_loader).
            reason: short tag for diagnostics ("pre-dropout-generate",
                "pre-generate-fresh", "post-optimizer-step").
            restore_lora: optional cleanup callable; called on exception to
                restore LoRA weights and clear dropout_active before re-raising.
                Pass None for sync paths that don't follow apply_lora_dropout.
        """
        try:
            weight_bus.sync(
                backend.weight_exporter,
                backend.weight_loader,
                self.model,
                ctx=self.ctx,
                modules_to_save=self.config.model.modules_to_save,
            )
        except Exception:
            if restore_lora is not None:
                with contextlib.suppress(Exception):
                    # restore_lora records its own failure in state.restore_failed
                    restore_lora()
            raise

    def train(
        self,
        dataloader: QGREDataLoader,
        generation_backend: GenerationBackend | None = None,
    ):
        """Full end-to-end training loop: generate → score → advantages → loss → backward.

        This is the main entry point for training. It:
        1. Iterates over batches from the dataloader
        2. Generates completions via generation_backend
        3. Scores completions via self.reward_fn
        4. Calls self.step() for algorithm + backward
        5. Records mastery, checks phase advancement
        6. Saves checkpoints every save_freq steps
        7. Logs metrics to MLflow
        """
        backend = generation_backend or self.generation_backend
        if backend is None:
            raise RuntimeError(
                "No generation backend provided. Pass to train() or QGRETrainer constructor.",
            )

        # Replace the backend's placeholder SyncState with trainer's authoritative instance.
        # The backend creates a default SyncState in its __init__ because it can't know about
        # the trainer at construction time. We swap it here so all components share one state.
        if hasattr(backend, "weight_loader") and backend.weight_loader is not None:
            backend.weight_loader._state = self.sync_state

        # Weight Sync Bus — coordinates weight transfer between training and vLLM
        from qgre.weight_bus import SyncStrategy, WeightBus

        weight_bus = WeightBus(
            state=self.sync_state,
            strategy=SyncStrategy(self.config.model.weight_sync_strategy),
        )

        self._dataloader = dataloader  # Store ref for difficulty gate updates

        # Initialize tutorial system if configured
        if self.config.tutorial.enabled:
            all_prompt_ids = [str(item["prompt_id"]) for item in dataloader.items]
            self.game_state.default_aspiration_target = self.advantage_estimator._aspiration_target
            self.game_state.init_tutorial(
                self.config.tutorial,
                all_prompt_ids,
                dataloader_items=dataloader.items,
                difficulty_column=self._difficulty_column,
            )

        self.setup_optimizer()
        # zero_grad() moved after resume() to avoid clearing loaded optimizer state
        cfg = self.config.training

        # Try to resume from checkpoint BEFORE MLflow setup (so we can reuse run_id)
        self.resume(self.config.logging.checkpoint_dir)

        # MLflow experiment setup (PILLARS.md line 128)
        # R3-T1: Moved after resume so we can reuse the MLflow run_id if resuming
        try:
            import mlflow

            mlflow.set_experiment(self.config.logging.mlflow_experiment)
            # Check if we're resuming an existing run
            if hasattr(self, "_mlflow_run_id") and self._mlflow_run_id is not None:
                try:
                    mlflow.start_run(run_id=self._mlflow_run_id)
                    import warnings

                    warnings.warn(
                        f"Resumed MLflow run {self._mlflow_run_id} from checkpoint. "
                        f"Continuing from step {self.global_step}.",
                        stacklevel=2,
                    )
                except (RuntimeError, ValueError, OSError) as e:
                    import warnings

                    warnings.warn(
                        f"Failed to resume MLflow run {self._mlflow_run_id}: {e}. "
                        "Starting new run. Your training history is fragmented.",
                        stacklevel=2,
                    )
                    mlflow.start_run(run_name=f"qgre-step-{self.global_step}")
                    active = mlflow.active_run()
                    self._mlflow_run_id = active.info.run_id if active else None
            else:
                mlflow.start_run(run_name=f"qgre-step-{self.global_step}")
                active = mlflow.active_run()
                self._mlflow_run_id = active.info.run_id if active else None
            log_training_params(
                {
                    "model": {
                        "path": self.config.model.path,
                        "lora_rank": self.config.model.lora_rank,
                    },
                    "algorithm": {
                        "mode": self.config.algorithm.mode,
                        "loss_type": self.config.algorithm.loss_type,
                    },
                    "training": {"lr": cfg.lr, "total_steps": cfg.total_steps},
                }
            )
        except ImportError:
            pass  # MLflow not installed
        except (RuntimeError, ValueError, OSError) as e:
            import warnings

            warnings.warn(f"MLflow setup failed: {e}. Metrics will not be tracked.", stacklevel=2)

        # DI1: Restore dataloader state after resume (dataloader not available inside resume())
        if getattr(self, "_pending_dataloader_state", None):
            if dataloader:
                dataloader.load_state_dict(self._pending_dataloader_state)  # type: ignore[arg-type]
            else:
                import warnings

                warnings.warn(
                    "T3-5: Pending dataloader state lost - dataloader is None on resume",
                    stacklevel=2,
                )
            self._pending_dataloader_state = None

        # Difficulty-gated curriculum: set initial gate based on current phase
        self._apply_difficulty_gate()

        for _epoch in range(10000):  # Outer epoch loop — stops when total_steps reached
            for batch in dataloader:
                if self.global_step >= cfg.total_steps:
                    break

                # Batch validation — guard against empty batches from dynamic sizing or data issues
                if batch.input_ids.shape[0] == 0:
                    raise RuntimeError(
                        f"Step {self.global_step}: empty batch received. "
                        "This indicates data pipeline issues (exhausted dataset, filter dropped all samples, "
                        "or dynamic batch sizing failure). Check data loading and filtering logic.",
                    )

                # 1. Generate (inference mode) with optional LoRA dropout for exploration
                if hasattr(backend, "set_inference_mode"):
                    backend.set_inference_mode()

                # LoRA dropout: partially revert to base model during generation
                gen_cfg = self.config.generation
                restore_lora = None
                if gen_cfg.lora_dropout_rate > 0:
                    from qgre.lora_dropout import apply_lora_dropout, compute_dropout_rate

                    current_rate = compute_dropout_rate(
                        gen_cfg.lora_dropout_rate,
                        gen_cfg.lora_dropout_anneal_steps,
                        self.global_step,
                    )
                    if current_rate > 0:
                        restore_lora = apply_lora_dropout(self.model, current_rate, self.sync_state)
                        # Sync noisy weights to vLLM via Weight Sync Bus
                        try:
                            self._sync_weights_guarded(
                                weight_bus,
                                backend,
                                reason="pre-dropout-generate",
                                restore_lora=restore_lora,
                            )
                        except Exception:
                            # Sync failed — restore_lora was already called inside _sync_weights_guarded.
                            # Clear it so the finally block doesn't call it again.
                            restore_lora = None
                            raise

                # Ensure LoRA weights are synced before first generate (fixes first-batch base-model bug)
                # When lora_dropout_rate=0, the dropout path above is skipped, but we still need
                # to sync LoRA weights on step 0 so vLLM uses trained weights, not base model.
                if (
                    (self.global_step == 0 or getattr(self, "_needs_weight_sync", False))
                    and backend.weight_loader is not None
                    and gen_cfg.lora_dropout_rate == 0
                ):
                    # R3-W1: Sync can raise from check_sync_allowed. If it raises,
                    # _needs_weight_sync stays True so next step retries. Only clear on success.
                    self._sync_weights_guarded(weight_bus, backend, reason="pre-generate-fresh")
                    self._needs_weight_sync = False

                generation_succeeded = False
                output = None

                # EGRS Phase 5: Extract hints from registry for this batch
                prompt_hints = None
                if self.hint_registry is not None and self.config.egrs.hint_enabled:
                    prompt_hints = {}
                    # Track generic hint usage for warning
                    generic_hint_count = 0
                    real_hint_count = 0

                    for i, pid in enumerate(batch.prompt_ids):
                        # Get tier for this prompt (from metadata or default)
                        meta = batch.metadata[i] if i < len(batch.metadata) else {}
                        tier = meta.get("tier", "default")

                        # Create mastery lookup function for this prompt's tier
                        # Maps span_id (e.g., "STEP_1") -> mastery from tier_mastery[tier][step_num]
                        # Track malformed span_ids to warn once per pattern
                        if not hasattr(self, "_egrs_malformed_span_ids"):
                            self._egrs_malformed_span_ids = set()

                        def make_mastery_fn(t: str, malformed_set: set):
                            def mastery_fn(span_id: str, tier=t) -> float:
                                if span_id.startswith("STEP_"):
                                    try:
                                        step_num = int(span_id.split("_")[1])
                                        # Read current state each time, don't capture tier_mastery values
                                        mastery = self.game_state.get_tier_step_mastery(
                                            tier, step_num
                                        )
                                        # H-1: If step not in tier_mastery (new tier/phase), return high mastery to force hint decay
                                        if (
                                            mastery == 0.0
                                            and step_num
                                            not in self.game_state.tier_mastery.get(tier, {})
                                        ):
                                            return 1.0
                                        return mastery
                                    except (IndexError, ValueError):
                                        if span_id not in malformed_set:
                                            malformed_set.add(span_id)
                                            logging.getLogger(__name__).warning(
                                                f"EGRS: Malformed span_id '{span_id}' in mastery_fn. "
                                                "Cannot parse step number. Using mastery=0.0 (100% hint probability).",
                                            )
                                        return 0.0
                                elif span_id == "THINK":
                                    # Read current state each time, don't capture tier_mastery values
                                    return self.game_state.get_tier_step_mastery(tier, 0)
                                return 1.0  # H-1: Unknown spans get high mastery (no hints)

                            return mastery_fn

                        # Query hints with per-span mastery lookup
                        sample_hints = self.hint_registry.get_hints_for_prompt(
                            pid,
                            mastery_fn=make_mastery_fn(tier, self._egrs_malformed_span_ids),
                        )
                        if sample_hints:
                            # Convert token hints to text hints
                            text_hints = {}
                            for span_id, tokens in sample_hints.items():
                                if tokens:
                                    # Decode hint tokens to text
                                    hint_text = self.tokenizer.decode(
                                        tokens, skip_special_tokens=True
                                    )
                                    # MIO-007: Validate hint text length
                                    if len(hint_text) > 512:
                                        logging.getLogger(__name__).warning(
                                            f"MIO-007: Hint text for {span_id} exceeds 512 chars ({len(hint_text)}). "
                                            "May exceed max_prompt_length. Truncating.",
                                        )
                                        hint_text = hint_text[:512]
                                    text_hints[span_id] = f"Hint for {span_id}: {hint_text}"
                                    real_hint_count += 1
                                else:
                                    # No tokens - use generic hint (less useful)
                                    text_hints[span_id] = f"Focus on getting {span_id} correct"
                                    generic_hint_count += 1
                            if text_hints:
                                prompt_hints[i] = text_hints
                    # Warn if all hints are generic (indicates extractor not configured or metadata missing)
                    if (
                        generic_hint_count > 0
                        and real_hint_count == 0
                        and self.global_step % 100 == 0
                    ):
                        extractor_status = "configured" if self.hint_extractor else "not configured"
                        logging.getLogger(__name__).warning(
                            f"EGRS: All {generic_hint_count} hints are generic (empty tokens). "
                            f"Hint extractor: {extractor_status}. "
                            "Check egrs.hint_extractor config and metadata columns.",
                        )

                try:
                    _dev = next(
                        (p.device for p in self.model.parameters() if p.device.type != "cpu"),
                        next(self.model.parameters()).device,
                    )
                    output = backend.generate(
                        batch.input_ids.to(_dev),
                        batch.attention_mask.to(_dev),
                        prompt_hints=prompt_hints,
                    )
                    generation_succeeded = True
                except Exception:
                    # TL-1: Clear CUDA cache on generate failure to prevent tensor leaks
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    raise
                finally:
                    # Always restore clean weights — even if generate crashes.
                    #
                    # ORDER IS CRITICAL:
                    #   1. weight_bus.restore_for_training(model)  — unmerge LoRA FIRST
                    #   2. restore_lora()                          — restore dropout mask SECOND
                    #
                    # Why: with MERGE strategy, weight_bus.sync() merged the (possibly
                    # dropped-out) LoRA into base weights before generate. To reverse,
                    # restore_for_training must unmerge using the EXACT SAME LoRA state
                    # that was merged (the dropped state). If we restored the dropout
                    # mask first, restore_lora would replace the dropped LoRA with the
                    # original, and the subsequent unmerge would use the wrong values —
                    # base weights would end up at `original - drift` instead of `original`.
                    #
                    # For DIRECT_COPY strategy, restore_for_training is a no-op, so the
                    # order doesn't matter in that path.
                    try:
                        weight_bus.restore_for_training(backend.weight_exporter, self.model)
                    except Exception:
                        logging.getLogger(__name__).exception(
                            "weight_bus.restore_for_training failed in finally block. "
                            "Model may be in merged state — halting to prevent LoRA corruption."
                        )
                        raise
                    # MERGE strategy creates temporary fp32 tensors during merge/unmerge
                    # (dequantize → add LoRA → requantize via Params4bit). The CUDA allocator
                    # holds the freed blocks as reserved memory, causing a slow VRAM leak
                    # (~7 MiB/min). Flushing after each cycle prevents accumulation.
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    # Don't sync to vLLM here — step-end sync (after training) handles it.
                    # If restore itself raises on the happy path (e.g. CUDA OOM mid-copy),
                    # log it but don't mask the successful generation: apply_lora_dropout's
                    # restore closure has already set state.restore_failed=True via
                    # exit_dropout(success=False), so the next weight_bus.sync() will be
                    # blocked by check_sync_allowed() with full context. On the failure
                    # path (generation already raising), let the restore exception
                    # propagate so it chains with the original.
                    if restore_lora is not None:
                        try:
                            restore_lora()
                        except Exception:
                            logging.getLogger(__name__).exception(
                                "LoRA dropout restore failed in finally block. "
                                "state.restore_failed is set; next sync will halt with diagnostics.",
                            )
                            if not generation_succeeded:
                                raise

                # 2. Score via reward_fn
                if output is None:
                    raise RuntimeError("Generation failed, output is None")
                reward_results = []
                min_tokens = self.config.algorithm.min_completion_tokens
                for i in range(len(output.texts)):
                    prompt = batch.raw_prompts[i] if i < len(batch.raw_prompts) else ""
                    meta = batch.metadata[i] if i < len(batch.metadata) else {}
                    rr = self.reward_fn(prompt, output.texts[i], meta)
                    # Empty-output floor: completions below min_completion_tokens get
                    # a penalized copy with negative reward and zeroed quality scores.
                    # RewardResult is frozen, so with_floor() creates a new instance.
                    if min_tokens > 0 and len(output.token_ids[i]) < min_tokens:
                        rr = rr.with_floor(-0.1)
                    reward_results.append(rr)

                # Validate: reward_results must match completions length after batch expansion
                assert len(reward_results) == len(output.token_ids), (
                    f"reward_results length ({len(reward_results)}) != completions length ({len(output.token_ids)})"
                )

                # EGRS Phase 5: Track hint success/failure for registry clearing
                # With n_completions > 1, same prompt_id appears multiple times.
                # Aggregate results per (prompt_id, span_id) before updating registry:
                # - ANY failure → record_failure (model hasn't learned)
                # - ALL succeeded WITHOUT hint → can graduate
                # - ALL succeeded but some WITH hint → don't graduate yet
                if self.hint_registry is not None and self.config.egrs.hint_enabled:
                    # Validate hints_used length matches batch size
                    if output.hints_used is not None and len(output.hints_used) != len(
                        batch.prompt_ids
                    ):
                        logging.getLogger(__name__).warning(
                            f"hints_used length mismatch: got {len(output.hints_used)}, expected {len(batch.prompt_ids)}. "
                            "Skipping hint success/failure tracking for this batch.",
                        )
                    # Validate batch.prompt_ids and reward_results have same length
                    if len(batch.prompt_ids) != len(reward_results):
                        logging.getLogger(__name__).warning(
                            f"Length mismatch: batch.prompt_ids={len(batch.prompt_ids)}, reward_results={len(reward_results)}. "
                            "Truncating to shorter length for hint tracking.",
                        )
                    # Aggregate: (prompt_id, span_id) -> {successes_without_hint, successes_with_hint, failures}
                    span_outcomes: dict[tuple[int, str], dict[str, int]] = {}
                    for i, (pid, rr) in enumerate(
                        zip(batch.prompt_ids, reward_results, strict=False)
                    ):
                        # MIO-001: hints_used is now dict[str, bool] mapping span_id → was_injected
                        hints_dict = {}
                        if (
                            output.hints_used is not None
                            and len(output.hints_used) == len(batch.prompt_ids)
                            and i < len(output.hints_used)
                        ):
                            hints_dict = output.hints_used[i]
                        span_correct = compute_span_correctness(
                            rr,
                            self.step_qualities,
                            self.config.egrs.reward_threshold,
                        )
                        for step_num, is_correct in span_correct.items():
                            span_id = f"STEP_{step_num}" if step_num > 0 else "THINK"
                            hint_was_used = hints_dict.get(span_id, False)
                            key = (pid, span_id)
                            if key not in span_outcomes:
                                span_outcomes[key] = {
                                    "success_no_hint": 0,
                                    "success_with_hint": 0,
                                    "failure": 0,
                                }
                            if is_correct:
                                if hint_was_used:
                                    span_outcomes[key]["success_with_hint"] += 1
                                else:
                                    span_outcomes[key]["success_no_hint"] += 1
                            else:
                                span_outcomes[key]["failure"] += 1

                    # MIO-005: Majority voting for graduation
                    # Track graduated hints to avoid race condition in multi-completion batches
                    graduated_this_batch: set[tuple] = set()
                    for (pid, span_id), counts in span_outcomes.items():
                        total = (
                            counts["success_no_hint"]
                            + counts["success_with_hint"]
                            + counts["failure"]
                        )
                        if total == 0:
                            continue
                        # Skip if hint doesn't exist or already graduated this batch
                        if (pid, span_id) not in self.hint_registry or (
                            pid,
                            span_id,
                        ) in graduated_this_batch:
                            continue
                        # Majority failure: reset streak
                        if counts["failure"] > total / 2:
                            self.hint_registry.record_failure(pid, span_id)
                        # Majority success without hint: can graduate
                        elif counts["success_no_hint"] > total / 2:
                            graduated = self.hint_registry.record_success(
                                pid, span_id, hint_was_used=False
                            )
                            if graduated:
                                graduated_this_batch.add((pid, span_id))
                                if self.global_step % 50 == 0:
                                    logging.getLogger(__name__).info(
                                        f"EGRS: Prompt {pid} {span_id} graduated (no longer needs hints)",
                                    )
                        # Otherwise (mixed, or majority success with hint): record success with hint
                        else:
                            self.hint_registry.record_success(pid, span_id, hint_was_used=True)

                # 3. Train step (training mode)
                if hasattr(backend, "set_training_mode"):
                    backend.set_training_mode()
                try:
                    metrics = self.step(
                        batch,
                        output.token_ids,
                        reward_results,
                        generation_logprobs=output.logprobs,
                        completion_texts=output.texts,  # Pass for span validation
                    )
                except Exception as e:
                    logging.getLogger(__name__).exception(
                        f"TL-6: step() raised exception: {e}. "
                        "Advancing global_step to maintain dataloader consistency.",
                    )
                    self.global_step += 1
                    raise

                # 3b. Update prioritized sampling weights (SPO paper Section 3.2)
                if hasattr(dataloader, "set_priorities"):
                    priorities = self.advantage_estimator.get_prompt_priorities()
                    if priorities:
                        dataloader.set_priorities(priorities)

                # 4. Log progress + MLflow
                step_rewards = {
                    int(k.split("_")[-1]): v
                    for k, v in metrics.items()
                    if k.startswith("mastery/step_")
                }
                # Log mastery and phase to stdout for debugging curriculum
                if self.global_step % self.config.logging.log_freq == 0:
                    tiers_str = "/".join(self.game_state.active_tiers)
                    reward_mean = metrics.get("reward/mean", 0.0)
                    metrics.get("critic_loss", 0)
                    loss_val = metrics.get("loss", 0.0)

                    # Build quality score rows. Use the highest-scoring sample in the
                    # batch for display (not necessarily the last) to avoid showing all-zeros
                    # when the last sample was floored by min_completion_tokens.
                    rows = []
                    if reward_results:
                        best_rr = max(reward_results, key=lambda r: r.reward)
                        last_scores = best_rr.scores
                        groups = [
                            (
                                "Energy",
                                [
                                    ("kinetic T", "q_kinetic"),
                                    ("potential V", "q_potential"),
                                    ("hamiltonian H", "q_hamiltonian"),
                                ],
                            ),
                            (
                                "Equations",
                                [
                                    ("dq/dt", "q_dqdt"),
                                    ("dp/dt", "q_dpdt"),
                                    ("consistency", "q_consistency"),
                                ],
                            ),
                        ]
                        for group_name, fields in groups:
                            vals = " │ ".join(
                                f"{name:>12s} {last_scores.get(key, 0):.1f}" for name, key in fields
                            )
                            rows.append((group_name, vals))

                    # Per-quality loss row (from span-based computation)
                    quality_losses = {
                        k.replace("loss/", ""): v
                        for k, v in metrics.items()
                        if k.startswith("loss/")
                    }
                    if quality_losses:
                        loss_items = sorted(quality_losses.items())[:6]  # Top 6 for space
                        loss_vals = " │ ".join(f"{q[:10]:>10s} {v:.4f}" for q, v in loss_items)
                        rows.append(("SpanLoss", loss_vals))

                    # Learning coefficients (learnability from advantage estimator)
                    if (
                        hasattr(self.advantage_estimator, "_reward_var")
                        and self.advantage_estimator._reward_var
                    ):
                        # Get learnability for last prompt's qualities
                        last_pid = batch.prompt_ids[-1] if batch.prompt_ids else None
                        if (
                            last_pid is not None
                            and last_pid in self.advantage_estimator._reward_var
                        ):
                            vars_dict = self.advantage_estimator._reward_var[last_pid]
                            # Learnability = p(1-p), approximated from variance
                            # Filter to string keys (quality names), not int keys (step numbers)
                            learn_items = [
                                (q, min(v, 0.25))
                                for q, v in vars_dict.items()
                                if v > 0 and isinstance(q, str)
                            ][:5]
                            if learn_items:
                                learn_vals = " │ ".join(
                                    f"{q[:8]:>8s} {v:.3f}" for q, v in learn_items
                                )
                                rows.append(("Learning", learn_vals))

                    if self.game_state.tutorial_enabled:
                        tut = self.game_state.get_tutorial_metrics()
                        tut_vals = (
                            f"{'active':>12s} {tut.get('tutorial/active_skills', 0)}   │ "
                            f"{'mastered':>12s} {tut.get('tutorial/mastered_skills', 0)}   │ "
                            f"{'locked':>12s} {tut.get('tutorial/locked_skills', 0)}   │ "
                            f"{'pool':>12s} {tut.get('tutorial/active_prompt_pool_size', 0)}"
                        )
                        rows.append(("Tutorial", tut_vals))

                    # Gradient coherence metrics (laminar→turbulent detection)
                    if "grad/temporal_cosine" in metrics:
                        t_cos = metrics["grad/temporal_cosine"]
                        s_cos = metrics.get("grad/spatial_cosine", 0.0)
                        w_norm = metrics.get("grad/lora_weight_norm", 0.0)
                        turb_state = metrics.get("grad/turbulence_state", 0)
                        state_names = {0: "CALIB", 1: "LAMINAR", 2: "TRANS", 3: "TURB"}
                        grad_vals = (
                            f"{'t_cos':>8s} {t_cos:+.4f} │ "
                            f"{'s_cos':>8s} {s_cos:+.4f} │ "
                            f"{'wt_norm':>8s} {w_norm:.1f} │ "
                            f"{'state':>8s} {state_names[turb_state]}"  # type: ignore[index]
                        )
                        rows.append(("Gradients", grad_vals))

                    # Compute column widths, capped to avoid terminal wrapping.
                    # Use terminal size if interactive tty, otherwise 150.
                    import shutil as _sh

                    max_w = _sh.get_terminal_size(fallback=(150, 24)).columns - 2
                    label_w = max((len(r[0]) for r in rows), default=10) + 2
                    val_w = max((len(r[1]) for r in rows), default=40) + 2
                    natural_w = label_w + val_w + 3  # 3 for " │ "

                    # Header row (show spans status based on per-quality loss presence)
                    spans_active = any(k.startswith("loss/") for k in metrics)
                    spans_str = "spans" if spans_active else "segmenter"
                    header = f" Step {self.global_step}/{cfg.total_steps} │ Phase {self.game_state.phase} │ Tiers: {tiers_str} │ [{spans_str}] │ Reward: {reward_mean:.3f} │ Loss: {loss_val:.6f}"
                    header_w = min(max_w, max(natural_w, len(header) + 2))
                    val_col_w = header_w - label_w - 3

                    def _fit(s: str, w: int) -> str:
                        if w <= 0:
                            return ""
                        return f"{s:<{w}}" if len(s) <= w else s[: w - 1] + "…"

                    print(f"\n┌{'─' * header_w}┐")
                    print(f"│{_fit(header, header_w)}│")
                    print(f"├{'─' * label_w}┬{'─' * (header_w - label_w - 1)}┤")
                    for i, (label, vals) in enumerate(rows):
                        print(f"│{label:>{label_w}} │ {_fit(vals, val_col_w)}│")
                        if i < len(rows) - 1:
                            print(f"├{'─' * label_w}┼{'─' * (header_w - label_w - 1)}┤")
                    print(f"└{'─' * label_w}┴{'─' * (header_w - label_w - 1)}┘")

                    # Full completion — same width, lines padded or truncated
                    if output.texts:
                        comp = output.texts[-1]
                        inner_w = header_w - 2
                        print(f"┌{'─' * header_w}┐")
                        print(f"│{'COMPLETION':^{header_w}}│")
                        print(f"├{'─' * header_w}┤")
                        for line in comp.split("\n"):
                            print(f"│ {_fit(line, inner_w)} │")
                        print(f"└{'─' * header_w}┘")
                try:
                    log_step_metrics(
                        step=self.global_step - 1,
                        reward_mean=metrics.get("reward/mean", 0.0),
                        loss=metrics.get("loss", 0.0),
                        step_rewards=step_rewards or None,
                        extra={
                            k: v
                            for k, v in metrics.items()
                            if k == "phase" or k.startswith("grad/")
                        },
                    )
                except ImportError:
                    pass  # MLflow not installed
                except (RuntimeError, ValueError, OSError) as e:
                    # Warn once per unique error, don't spam logs
                    if not hasattr(self, "_mlflow_warned"):
                        import warnings

                        warnings.warn(
                            f"MLflow logging failed: {e}. Further errors will be silent.",
                            stacklevel=2,
                        )
                        self._mlflow_warned = True

                # 5. Save checkpoint
                if cfg.save_freq > 0 and self.global_step % cfg.save_freq == 0:
                    self.save()

                # 6. Weight sync via Weight Sync Bus (LoRA + modules_to_save → vLLM)
                # Skip if generation failed — stale weights shouldn't be synced
                if backend.weight_loader is not None and generation_succeeded:
                    self._sync_weights_guarded(weight_bus, backend, reason="post-optimizer-step")
                    # Flush CUDA cache after MERGE sync to prevent fragmentation buildup
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                # 7. Periodic vLLM KV cache flush to prevent VRAM leak (unsloth #3864)
                flush_freq = self.config.training.kv_cache_flush_freq
                if flush_freq > 0 and self.global_step > 0 and self.global_step % flush_freq == 0:
                    if backend.weight_loader is not None:
                        try:
                            backend.weight_loader.flush_kv_cache()
                        except (RuntimeError, AttributeError) as e:
                            import warnings

                            warnings.warn(
                                f"Step {self.global_step}: KV cache flush failed: {e}. "
                                f"VRAM leak may accumulate. Monitor GPU memory.",
                                stacklevel=2,
                            )

            if self.global_step >= cfg.total_steps:
                break

        # Save gradient probe results
        if self._gradient_probe_log:
            import json
            from pathlib import Path

            output_dir = Path("output/hamiltonian/study/gradient_probe")
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_dir / "probe_results.json", "w") as f:
                json.dump(
                    {
                        "config": {
                            "advantage_scale": self.config.algorithm.advantage_scale,
                            "lr": self.config.training.lr,
                            "lora_rank": self.config.model.lora_rank,
                            "lora_alpha": self.config.model.lora_alpha,
                            "n_steps": len(self._gradient_probe_log),
                        },
                        "measurements": self._gradient_probe_log,
                    },
                    f,
                    indent=2,
                )
            print(f"\n{'=' * 70}")
            print(f"Gradient probe results saved to {output_dir / 'probe_results.json'}")
            print(f"{'=' * 70}")

        # Save gradient coherence log
        if self._attention_log:
            import json
            from pathlib import Path

            output_dir = Path(self.config.logging.checkpoint_dir).parent / "gradient_coherence"
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_dir / "coherence_log.json", "w") as f:
                json.dump(
                    {
                        "config": {
                            "log_freq": self._attention_log_freq,
                            "n_measurements": len(self._attention_log),
                        },
                        "measurements": self._attention_log,
                    },
                    f,
                    indent=2,
                )
            print(f"\n{'=' * 70}")
            print(f"Gradient coherence log saved to {output_dir / 'coherence_log.json'}")
            print(f"{'=' * 70}")

        # TL-10: Ensure completion_logger is closed even on exception
        try:
            # Final checkpoint
            self.save()
        finally:
            if hasattr(self, "completion_logger"):
                self.completion_logger.close()

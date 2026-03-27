from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Protocol

import torch
import torch.nn as nn

from qgre.advantages import QGREStepAdvantageEstimator, build_phase_qualities
from qgre.checkpoint import (
    discover_latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from qgre.config import QGREConfig
from qgre.data import PromptBatch, QGREDataLoader
from qgre.logging import CompletionLogger, log_step_metrics, log_training_params
from qgre.nemo_extracted.kl import masked_mean
from qgre.nemo_extracted.llds import compute_llds_loss
from qgre.nemo_extracted.logits import logprobs_from_logits
from qgre.nemo_extracted.loss_functions import ClippedPGLossFn
from qgre.segments import Segmenter, uniform_segmenter
from qgre.types import GameState, RewardResult


class GenerationBackend(Protocol):
    """Abstract generation interface — shields trainer from Unsloth internals."""

    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs) -> Any:
        ...

    def save_weights(self, path: str | Path) -> None:
        ...

    def load_weights(self, path: str | Path) -> None:
        ...


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
        self.game_state = game_state or GameState(
            mastery_threshold=config.training.mastery_threshold,
            stagnation_timeout=config.training.stagnation_timeout,
            plateau_window=config.training.plateau_window,
            plateau_threshold=config.training.plateau_threshold,
        )
        self._dataloader = None  # Set in train()

        # 2D curriculum setup
        data_cfg = config.data
        self._tier_order = data_cfg.tier_order
        self._difficulty_column = data_cfg.difficulty_column
        self._tier_advance_phase = data_cfg.tier_advance_quality_phase
        self._tier_advance_threshold = data_cfg.tier_advance_threshold
        if self._tier_order and data_cfg.initial_tiers:
            self.game_state.active_tiers = list(data_cfg.initial_tiers)
            for t in data_cfg.initial_tiers:
                self.game_state.tier_phases.setdefault(t, 1)

        # Step qualities and phase mapping — configurable per domain
        # step_qualities: from constructor arg, config YAML, or error
        sq = step_qualities or config.algorithm.step_qualities
        if sq is None:
            raise ValueError(
                "step_qualities is required. Pass to QGRETrainer constructor or set in config YAML:\n"
                "  algorithm:\n"
                "    step_qualities:\n"
                "      1: [q_format]\n"
                "      2: [q_accuracy]"
            )
        self.step_qualities = sq
        self.phase_qualities = build_phase_qualities(sq)

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
                    raise ValueError("segmenter='label' requires algorithm.label_segmenter.patterns in config")
                segmenter = make_label_segmenter(tokenizer, alg.label_segmenter)
            elif ":" in alg.segmenter:
                import importlib
                mod_path, fn_name = alg.segmenter.rsplit(":", 1)
                segmenter = getattr(importlib.import_module(mod_path), fn_name)

        spo_cfg = alg.spo
        # Target-aware aspiration gap uses mastery threshold as default target
        aspiration_target = spo_cfg.aspiration_target if spo_cfg.aspiration_target > 0 else config.training.mastery_threshold
        self.advantage_estimator = QGREStepAdvantageEstimator(
            lr=spo_lr, mode=mode,
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

        # VPRM critic — per-region per-dimension learned baseline
        self.vprm_critic = None
        self.vprm_optimizer = None
        self._vprm_initialized = False
        self._vprm_config = config.vprm
        self._vprm_sq = sq

        # Loss function (NeMo RL extracted)
        self.loss_fn = ClippedPGLossFn({
            "reference_policy_kl_penalty": alg.kl_cov_ratio if alg.loss_mode == "kl_cov" else 0.0,
            "reference_policy_kl_type": alg.reference_policy_kl_type,
            "kl_input_clamp_value": 20.0,
            "kl_output_clamp_value": 10.0,
            "ratio_clip_min": alg.clip_ratio_low,
            "ratio_clip_max": alg.clip_ratio_high,
            "ratio_clip_c": None,
            "use_on_policy_kl_approximation": True,
            "use_importance_sampling_correction": False,
            "truncated_importance_sampling_ratio": None,
            "token_level_loss": True,
            "force_on_policy_ratio": True,
            "remove_length_normalization": alg.loss_type == "dr_grpo",
            "lambda_return": alg.lambda_return,
        })

        # LLDS requires stored generation-time logprobs to be meaningful.
        # Without them, old_logprob == curr_logprob and all LLDS gates return zero.
        # This flag is set to True when generation-time logprobs are wired (future work).
        self._has_stored_logprobs = False

        # Fused logprobs: chunked lm_head projection saves ~2GB by not materializing
        # full [seq, vocab] logit tensor. Uses torch.checkpoint per chunk to prevent
        # autograd from storing all chunks for backward.
        self._use_fused_logprobs = config.algorithm.use_fused_logprobs
        self._fused_chunk_size = config.algorithm.fused_logprob_chunk_size
        self._fused_validated = False  # One-time validation on first use

        # Completion logger
        self.completion_logger = CompletionLogger(config.logging.completion_dir)

        # Training state
        self.global_step = 0
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: Any = None
        self._accumulated_loss = 0.0

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
        # AdamW8bit saves ~4x memory on optimizer states (PLAN.md line 323)
        # Requires GPU tensors — falls back to regular AdamW on CPU
        use_8bit = False
        try:
            device = next(self.model.parameters()).device
            if device.type == "cuda":
                from bitsandbytes.optim import AdamW8bit
                self.optimizer = AdamW8bit(
                    self.model.parameters(),
                    lr=self.config.training.lr,
                )
                use_8bit = True
        except (ImportError, StopIteration):
            pass

        if not use_8bit:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.training.lr,
            )

        cfg = self.config.training
        main_scheduler = None
        if cfg.lr_scheduler == "cosine":
            main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=cfg.total_steps, eta_min=cfg.lr * 0.1,
            )
        elif cfg.lr_scheduler == "linear":
            main_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=1.0, end_factor=0.1, total_iters=cfg.total_steps,
            )

        if main_scheduler is not None and cfg.warmup_steps > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=0.01, end_factor=1.0, total_iters=cfg.warmup_steps,
            )
            self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                self.optimizer, schedulers=[warmup, main_scheduler], milestones=[cfg.warmup_steps],
            )
        else:
            self.scheduler = main_scheduler

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

        # Per-prompt active qualities based on each prompt's tier and that tier's quality phase
        active_qualities = []
        for i in range(len(reward_results)):
            meta = batch.metadata[i] if i < len(batch.metadata) else {}
            tier = self._get_prompt_tier(meta)
            tier_phase = self.game_state.tier_phases.get(tier, 1)
            fallback = max(p for p in self.phase_qualities if p <= tier_phase) if any(p <= tier_phase for p in self.phase_qualities) else min(self.phase_qualities)
            active_qualities.append(self.phase_qualities.get(tier_phase, self.phase_qualities[fallback]))

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

        # Compute per-token advantages — span-based (if scored_spans populated) or region-based (legacy)
        use_spans = any(rr.scored_spans for rr in reward_results)
        if use_spans:
            from qgre.spans import build_char_to_token_map, scored_spans_to_token_masks
            # Build per-sample token masks from scored_spans
            batch_token_masks: list[dict[str, torch.Tensor]] = []
            for i, rr in enumerate(reward_results):
                if rr.scored_spans:
                    # Decode completion for offset_mapping (authoritative char→token)
                    comp_text = self.tokenizer.decode(completions[i], skip_special_tokens=False) if self.tokenizer else None
                    char_map = build_char_to_token_map(completions[i], self.tokenizer, completion_text=comp_text)
                    if char_map is not None:
                        masks = scored_spans_to_token_masks(rr.scored_spans, char_map, len(completions[i]))
                    else:
                        masks = {}  # Fallback: empty masks → zero advantages (segmenter fallback below)
                else:
                    masks = {}
                batch_token_masks.append(masks)

            # Check if span mapping succeeded for any sample
            if any(m for m in batch_token_masks):
                token_advantages = self.advantage_estimator.compute_advantages_with_spans(
                    batch_prompt_ids=batch.prompt_ids,
                    batch_token_ids=completions,
                    batch_reward_results=reward_results,
                    batch_active_qualities=active_qualities,
                    batch_token_masks=batch_token_masks,
                    group_size=self.config.algorithm.grpo.n if self.config.algorithm.mode == "grpo" else None,
                    frontier_steps=frontier_steps,
                )
                # Still run segmenter for VPRM critic (needs STEP_N regions for hidden-state pooling)
                # and KL region weights
                batch_regions = [self.advantage_estimator.segmenter(c) for c in completions]
            else:
                use_spans = False  # All mappings failed, fall back to segmenter

        if not use_spans:
            token_advantages, batch_regions = self.advantage_estimator.compute_advantages(
                batch_prompt_ids=batch.prompt_ids,
                batch_token_ids=completions,
                batch_reward_results=reward_results,
                batch_active_qualities=active_qualities,
                group_size=self.config.algorithm.grpo.n if self.config.algorithm.mode == "grpo" else None,
                frontier_steps=frontier_steps,
            )

        # Build full sequences on model device
        device = next(self.model.parameters()).device
        max_comp_len = max(len(c) for c in completions)

        padded_advs = torch.zeros(len(completions), max_comp_len, device=device)
        for i, adv in enumerate(token_advantages):
            padded_advs[i, :len(adv)] = adv.to(device)

        comp_tensor = torch.zeros(len(completions), max_comp_len, dtype=torch.long, device=device)
        for i, c in enumerate(completions):
            comp_tensor[i, :len(c)] = torch.tensor(c, dtype=torch.long, device=device)

        # Attention mask: 1 for real tokens, 0 for padding beyond completion length.
        # Uses actual completion lengths, NOT token values — token ID 0 is a valid
        # token in many vocabularies (e.g. Qwen3). Using (comp_tensor != 0) would
        # silently mask real tokens, corrupting hidden states and gradients.
        comp_attention_mask = torch.zeros_like(comp_tensor, dtype=torch.long)
        for i, c in enumerate(completions):
            comp_attention_mask[i, :len(c)] = 1

        # Build KL region weights from segmenter regions (THR-style, PLAN.md lines 798-802)
        # Gate: skip entirely when KL is disabled (default config). Saves the tensor
        # allocation + nested Python loop over every token every step.
        alg = self.config.algorithm
        if alg.kl_cov_ratio > 0 and alg.loss_mode == "kl_cov":
            kl_region_weights = torch.ones(len(completions), max_comp_len, device=device)
            region_map = {"THINK": alg.kl_think_multiplier, "FORMAT": alg.kl_format_multiplier}
            for i, regions in enumerate(batch_regions):
                for t, region in enumerate(regions):
                    if t < max_comp_len:
                        if region in region_map:
                            kl_region_weights[i, t] = region_map[region]
                        elif region.startswith("STEP_"):
                            kl_region_weights[i, t] = alg.kl_step_multiplier
        else:
            kl_region_weights = None

        if not self.config.vprm.enabled:
            del batch_regions  # No longer needed — regions already mapped to KL weights or skipped

        # SPO low-advantage filter: skip sequences with near-zero signal (PLAN.md lines 658-671)
        _spo_filter_idx = None  # Maps filtered indices → original batch indices
        if self.config.algorithm.mode == "spo":
            useful = (padded_advs.abs() > 0.001).any(dim=-1)
            if useful.sum() == 0:
                # All advantages near-zero — skip backward pass but still record mastery + log completions
                metrics = {"loss": 0.0, "reward/mean": sum(rr.reward for rr in reward_results) / len(reward_results),
                           "global_step": self.global_step, "phase": self.game_state.phase, "skipped": True}
                self._record_mastery_and_advance(reward_results, active_qualities, batch, metrics)
                # Log completions even on skipped steps
                for i, rr in enumerate(reward_results):
                    comp_tokens = completions[i]
                    if self.tokenizer is not None:
                        comp_text = self.tokenizer.decode(comp_tokens, skip_special_tokens=True)
                    else:
                        comp_text = str(comp_tokens)
                    self.completion_logger.log_completion(
                        step=self.global_step,
                        prompt=batch.raw_prompts[i] if i < len(batch.raw_prompts) else "",
                        completion=comp_text,
                        reward=rr.reward,
                        reward_components=rr.scores,
                        phase=self.game_state.phase,
                    )
                self.global_step += 1
                return metrics
            if useful.sum() >= 2 and useful.sum() < len(completions):
                idx = useful.nonzero(as_tuple=True)[0]
                padded_advs = padded_advs[idx]
                comp_tensor = comp_tensor[idx]
                comp_attention_mask = comp_attention_mask[idx]
                if kl_region_weights is not None:
                    kl_region_weights = kl_region_weights[idx]
                # Track original indices for VPRM region/reward lookup
                _spo_filter_idx = idx.tolist()
            else:
                _spo_filter_idx = None

        # Response mask
        prompt_lengths = [0] * comp_tensor.shape[0]
        response_mask = self.compute_response_mask(comp_tensor, prompt_lengths)[:, 1:]

        if response_mask.sum() == 0:
            raise RuntimeError(
                f"Step {self.global_step}: no response tokens in any completion — cannot compute loss."
            )

        # Micro-batched forward + backward — avoids OOM on logits tensor
        # Full logits = batch × seq × vocab ≈ 8 × 4096 × 151K × 4B = 18.6GB (impossible on 16GB)
        # Micro-batch size adapts to sequence length to avoid OOM on long completions.
        # At 4096 tokens, Unsloth MLP activation = 2 × 4096 × 8960 × 2B = 140MB per seq.
        # micro_batch_size=1 for seq ≥ 2048, micro_batch_size=2 for shorter.
        actual_batch = comp_tensor.shape[0]  # May differ from len(completions) after SPO filter
        micro_batch_size = 1 if max_comp_len >= 2048 else max(1, min(2, actual_batch))
        n_micro = (actual_batch + micro_batch_size - 1) // micro_batch_size
        total_loss = 0.0
        all_metrics = {}

        for mb_start in range(0, actual_batch, micro_batch_size):
            mb_end = min(mb_start + micro_batch_size, actual_batch)
            mb_ids = comp_tensor[mb_start:mb_end]
            mb_attn_mask = comp_attention_mask[mb_start:mb_end]
            mb_advs = padded_advs[mb_start:mb_end]
            mb_mask = response_mask[mb_start:mb_end]
            mb_hidden_states = None  # Set by fused/non-fused path when VPRM enabled

            # Ensure Unsloth training mode before EACH forward pass (not just at init).
            # Unsloth's inplace attention kernels require this transition before backward.
            # Source: Unsloth #895, #2434 — "modified by inplace operation" fix.
            if hasattr(self, '_FastLanguageModel'):
                self._FastLanguageModel.for_training(self.model)
            elif self.generation_backend and hasattr(self.generation_backend, '_FastLanguageModel'):
                self.generation_backend._FastLanguageModel.for_training(self.model)
            else:
                try:
                    from unsloth import FastLanguageModel as FLM
                    FLM.for_training(self.model)
                except ImportError:
                    import warnings
                    warnings.warn(
                        "Could not call FastLanguageModel.for_training() — Unsloth not importable. "
                        "If using Unsloth models, inplace attention kernels may cause backward errors."
                    )

            # Forward pass: fused (chunked lm_head + checkpoint) or non-fused (full lm_head)
            # Both paths start from hidden states (UNSLOTH_RETURN_HIDDEN_STATES=1 is global).
            # Fused saves ~2GB VRAM via chunking. Non-fused materializes full logit tensor.
            if self._use_fused_logprobs:
                from qgre.fused_logprobs import get_hidden_states_and_lm_head, chunked_logprobs_from_hidden
                hidden_states, lm_head = get_hidden_states_and_lm_head(self.model, mb_ids, attention_mask=mb_attn_mask)
                if hidden_states is not None and lm_head is not None:
                    # Fused path: chunked lm_head + checkpoint avoids full [seq, vocab] logit tensor.
                    mb_lp = chunked_logprobs_from_hidden(
                        hidden_states[:, :-1, :], lm_head, mb_ids[:, 1:],
                        chunk_size=self._fused_chunk_size,
                    )
                    # One-time validation: chunking correctness only.
                    # Compares chunked vs full lm_head projection from same hidden states.
                    # Does NOT validate model-level correctness (caught by training loop).
                    if not self._fused_validated and mb_lp.shape[1] > 0:
                        if mb_lp.grad_fn is None:
                            raise RuntimeError(
                                "Fused logprobs has no grad_fn — autograd graph is broken. "
                                "Set algorithm.use_fused_logprobs=false to fall back."
                            )
                        with torch.no_grad():
                            manual_logits = lm_head(hidden_states[:, :-1, :]).float()
                            std_lp = logprobs_from_logits(manual_logits, mb_ids[:, 1:])
                            del manual_logits
                            min_len_v = min(mb_lp.shape[1], std_lp.shape[1])
                            if not torch.allclose(mb_lp[:, :min_len_v].detach(), std_lp[:, :min_len_v], atol=1e-3):
                                max_diff = (mb_lp[:, :min_len_v].detach() - std_lp[:, :min_len_v]).abs().max().item()
                                raise RuntimeError(
                                    f"Fused logprobs diverge from full projection (max diff: {max_diff:.6f}). "
                                    f"Set algorithm.use_fused_logprobs=false to fall back."
                                )
                            del std_lp
                        self._fused_validated = True
                    # Keep hidden_states for VPRM critic if enabled; else free memory
                    mb_hidden_states = hidden_states.detach() if self.config.vprm.enabled else None
                    del hidden_states
                else:
                    # Hidden states mode didn't take effect — crash with diagnostics
                    raise RuntimeError(
                        f"Step {self.global_step}: fused logprobs unavailable — "
                        f"UNSLOTH_RETURN_HIDDEN_STATES did not take effect. "
                        f"Delete unsloth_compiled_cache/ and restart. "
                        f"To disable fused logprobs entirely, set algorithm.use_fused_logprobs=false."
                    )
            else:
                # Non-fused path: full lm_head projection without chunking.
                # Costs ~2GB more VRAM than fused (materializes full [seq, vocab] tensor).
                # This is the degraded-but-correct escape hatch.
                from qgre.fused_logprobs import get_hidden_states_and_lm_head
                hs, lm_head_nf = get_hidden_states_and_lm_head(self.model, mb_ids, attention_mask=mb_attn_mask)
                if hs is None or lm_head_nf is None:
                    raise RuntimeError(
                        f"Step {self.global_step}: model did not return hidden states. "
                        f"UNSLOTH_RETURN_HIDDEN_STATES did not take effect. "
                        f"Delete unsloth_compiled_cache/ and restart."
                    )
                mb_logits = lm_head_nf(hs[:, :-1, :]).float()
                mb_hidden_states = hs.detach() if self.config.vprm.enabled else None
                del hs
                mb_lp = logprobs_from_logits(mb_logits, mb_ids[:, 1:])
                del mb_logits

            mb_old_lp = mb_lp.detach()

            # VPRM critic: replace SPO advantages with critic-based advantages
            mb_critic_loss = torch.tensor(0.0, device=device)
            mb_critic_count = 0
            if self.config.vprm.enabled and mb_hidden_states is not None:
                from qgre.advantages import compute_advantages_vprm
                # Lazy init critic on first forward pass (now we know hidden_dim)
                if not self._vprm_initialized:
                    self._init_vprm_critic(
                        hidden_dim=mb_hidden_states.shape[-1],
                        device=device,
                    )

                # Compute VPRM advantages per-sample in this micro-batch
                for mb_i in range(mb_end - mb_start):
                    filtered_i = mb_start + mb_i
                    # Map filtered index → original batch index
                    orig_i = _spo_filter_idx[filtered_i] if _spo_filter_idx is not None else filtered_i
                    # Get regions for this sample (original batch index)
                    if orig_i >= len(batch_regions):
                        import warnings
                        warnings.warn(
                            f"VPRM: orig_i={orig_i} >= len(batch_regions)={len(batch_regions)} "
                            f"— skipping sample (SPO filter mapping error?)"
                        )
                        continue
                    sample_regions = batch_regions[orig_i]
                    # Get reward result and active qualities
                    if orig_i >= len(reward_results):
                        import warnings
                        warnings.warn(
                            f"VPRM: orig_i={orig_i} >= len(reward_results)={len(reward_results)} "
                            f"— skipping sample"
                        )
                        continue
                    sample_rr = reward_results[orig_i]
                    sample_aq = active_qualities[orig_i] if orig_i < len(active_qualities) else []
                    # Get hidden states for this sample
                    sample_hs = mb_hidden_states[mb_i]  # [seq_len, hidden_dim]
                    # Trim to completion length
                    comp_len = len(completions[orig_i]) if orig_i < len(completions) else sample_hs.shape[0]
                    sample_hs_trimmed = sample_hs[:comp_len]

                    vprm_advs, vprm_loss, used_critic = compute_advantages_vprm(
                        critic=self.vprm_critic,
                        hidden_states=sample_hs_trimmed,
                        regions=sample_regions[:comp_len],
                        reward_result=sample_rr,
                        step_qualities=self.step_qualities,
                        active_qualities=sample_aq,
                        step_region_map=self.config.algorithm.step_region_map,
                        frontier_steps=frontier_steps,
                        frontier_amplification=self.config.algorithm.frontier_amplification,
                        min_regions=self.config.vprm.spo_fallback_min_regions,
                        aspiration_beta=self.advantage_estimator._aspiration_beta,
                        aspiration_target=self.advantage_estimator._aspiration_target,
                    )

                    if used_critic:
                        # When spans are active, DON'T overwrite span advantages —
                        # spans provide better token targeting. Only collect critic loss
                        # (critic still learns the baseline for future use).
                        # When spans are NOT active, replace SPO advantages with VPRM.
                        if not use_spans:
                            adv_len = min(vprm_advs.shape[0], mb_advs.shape[1])
                            mb_advs[mb_i, :adv_len] = vprm_advs[:adv_len]
                        mb_critic_loss = mb_critic_loss + vprm_loss
                        mb_critic_count += 1

                del mb_hidden_states

            # Align advantages + KL weights with logprob positions:
            # mb_lp[t] = log P(token t+1 | tokens 0..t), so it needs advantage[t+1]
            # Shift advantages and KL weights by 1 to match logprob indexing
            mb_advs_shifted = mb_advs[:, 1:]  # advantage for token being predicted
            mb_kl_shifted = kl_region_weights[mb_start:mb_end, 1:] if kl_region_weights is not None else None
            min_len = min(mb_lp.shape[1], mb_advs_shifted.shape[1], mb_mask.shape[1])
            mb_kl_weights = mb_kl_shifted[:, :min_len] if mb_kl_shifted is not None else None
            mb_loss, mb_metrics = self.loss_fn(
                curr_logprobs=mb_lp[:, :min_len],
                prev_logprobs=mb_old_lp[:, :min_len],
                advantages=mb_advs_shifted[:, :min_len],
                mask=mb_mask[:, :min_len].float(),
                reference_logprobs=mb_old_lp[:, :min_len],
                kl_region_weights=mb_kl_weights,
            )

            # neg_logprob_mean: monitor for policy collapse (metric only, not a loss term).
            # -mean(log p(token)) is NOT a valid entropy loss — its gradient pushes the wrong
            # direction (increases prob of sampled tokens instead of spreading mass).
            # See NeMo RL docs: "not recommended for direct backpropagation."
            with torch.no_grad():
                neg_logprob_mean = -masked_mean(mb_lp[:, :min_len], mb_mask[:, :min_len].float())
                mb_metrics["neg_logprob_mean"] = neg_logprob_mean.item()

            # Dynamic length control (Huawei): penalize length when group accuracy is high
            lp_coef = self.config.algorithm.length_penalty_coef
            if lp_coef > 0:
                lp_thresh = self.config.algorithm.length_penalty_threshold
                group_correctness = sum(rr.reward for rr in reward_results) / len(reward_results)
                if group_correctness > lp_thresh:
                    # High correctness → add length penalty to encourage efficiency
                    seq_lengths = mb_mask[:, :min_len].sum(dim=-1)
                    mean_len = seq_lengths.mean()
                    length_penalty = lp_coef * (seq_lengths / max(mean_len, 1.0)).mean()
                    mb_loss = mb_loss + length_penalty
                    mb_metrics["length_penalty"] = length_penalty.item()

            # LLDS auxiliary loss — prevents Lazy Likelihood Displacement death spiral
            # (arXiv:2512.04220). Only meaningful when old_logprob != curr_logprob,
            # which requires stored generation-time logprobs (not yet implemented).
            llds_coef = self.config.algorithm.llds_coef
            if llds_coef > 0 and self._has_stored_logprobs:
                llds_loss, llds_mask = compute_llds_loss(
                    log_prob=mb_lp[:, :min_len],
                    old_log_prob=mb_old_lp[:, :min_len],
                    advantages=mb_advs_shifted[:, :min_len],
                    response_mask=mb_mask[:, :min_len].float(),
                )
                mb_loss = mb_loss + llds_coef * llds_loss
                mb_metrics["llds_loss"] = llds_loss.item()
                mb_metrics["llds_mask_ratio"] = llds_mask.sum().item() / max(mb_mask[:, :min_len].sum().item(), 1)

            # VPRM critic loss: normalize by sample count and add to policy loss
            if mb_critic_count > 0:
                mb_critic_loss = mb_critic_loss / mb_critic_count
            if mb_critic_loss.requires_grad:
                mb_loss = mb_loss + mb_critic_loss
                mb_metrics["critic_loss"] = mb_critic_loss.item()

            (mb_loss / n_micro).backward()
            total_loss += mb_loss.item() / n_micro

            del mb_lp

            # OPT-2: Release CUDA cached blocks between micro-batches.
            # Prevents false OOM from allocator fragmentation on tight memory budgets.
            if self.config.training.empty_cache_between_microbatches and torch.cuda.is_available():
                torch.cuda.empty_cache()

            if not all_metrics:
                all_metrics = {k: v for k, v in mb_metrics.items()}
            else:
                for k, v in mb_metrics.items():
                    all_metrics[k] = all_metrics.get(k, 0) + v

        metrics = {k: v / n_micro for k, v in all_metrics.items()} if n_micro > 1 else all_metrics
        metrics["loss"] = total_loss
        loss_val = total_loss  # For NaN check

        if not (torch.isfinite(torch.tensor(loss_val))):
            raise RuntimeError(
                f"Step {self.global_step}: loss is {loss_val} — aborting to prevent weight corruption."
            )

        # Track accumulated loss across gradient accumulation steps
        self._accumulated_loss += total_loss

        # Optimizer step (backward already done in micro-batches above)
        if (self.global_step + 1) % self.config.training.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.training.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
            # VPRM critic optimizer steps at same cadence as policy optimizer
            # Only step when critic actually received gradients (avoids diluting Adam moments)
            if self.vprm_critic is not None and self.vprm_optimizer is not None:
                has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in self.vprm_critic.parameters())
                if has_grad:
                    torch.nn.utils.clip_grad_norm_(self.vprm_critic.parameters(), max_norm=self.config.training.max_grad_norm)
                    self.vprm_optimizer.step()
                    # Polyak-average target network (or hard-sync during warmup)
                    if self._vprm_config.use_target_network:
                        if self.global_step < self._vprm_config.target_warmup_steps and self.global_step % 100 == 0:
                            self.vprm_critic.sync_target_to_online()
                        elif self.global_step >= self._vprm_config.target_warmup_steps:
                            self.vprm_critic.update_target_network(tau=self._vprm_config.polyak_tau)
                self.vprm_optimizer.zero_grad()
                # Divergence monitoring — independent of has_grad (reads .data, no grad needed)
                if self.global_step % 50 == 0:
                    with torch.no_grad():
                        # Single .item() call to avoid 54 GPU syncs
                        divergence = sum(
                            (op.data - tp.data).pow(2).mean()
                            for q in self.vprm_critic.quality_names
                            for op, tp in zip(self.vprm_critic.heads[q].parameters(), self.vprm_critic.target_heads[q].parameters())
                        ).item()
                        if not math.isfinite(divergence):
                            import warnings
                            warnings.warn(f"Step {self.global_step}: target_divergence={divergence} — critic may contain NaN")
                        metrics["target_divergence"] = divergence
            if self.scheduler is not None:
                self.scheduler.step()
            # Report accumulated loss across gradient accumulation steps
            metrics["accumulated_loss"] = self._accumulated_loss
            self._accumulated_loss = 0.0

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

        # Log metrics
        reward_mean = sum(rr.reward for rr in reward_results) / len(reward_results)
        metrics["reward/mean"] = reward_mean
        metrics["global_step"] = self.global_step

        # Track completion lengths for verbosity drift detection
        comp_lengths = [len(c) for c in completions]
        metrics["completion_length/mean"] = float(sum(comp_lengths) / len(comp_lengths))
        metrics["completion_length/max"] = float(max(comp_lengths))
        metrics["completion_length/min"] = float(min(comp_lengths))

        self._record_mastery_and_advance(reward_results, active_qualities, batch, metrics)

        # Log completions
        completions_text = []
        for i, rr in enumerate(reward_results):
            # Decode token IDs to text for readable logs
            comp_tokens = completions[i]
            if self.tokenizer is not None:
                comp_text = self.tokenizer.decode(comp_tokens, skip_special_tokens=True)
            else:
                comp_text = str(comp_tokens)
            completions_text.append(comp_text)
            self.completion_logger.log_completion(
                step=self.global_step,
                prompt=batch.raw_prompts[i] if i < len(batch.raw_prompts) else "",
                completion=comp_text,
                reward=rr.reward,
                reward_components=rr.scores,
                phase=self.game_state.phase,
            )

        self.global_step += 1
        return metrics

    def save(self, path: str | Path | None = None):
        """Save checkpoint."""
        if path is None:
            path = Path(self.config.logging.checkpoint_dir) / f"global_step_{self.global_step}.pt"

        save_checkpoint(
            path=path,
            global_step=self.global_step,
            model_state_dict=self.model.state_dict(),
            optimizer_state_dict=self.optimizer.state_dict() if self.optimizer else None,
            scheduler_state_dict=self.scheduler.state_dict() if self.scheduler else None,
            game_state=self.game_state,
            advantage_estimator_state=self.advantage_estimator.state_dict(),
            cuda_rng_state=torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            vprm_critic_state=self.vprm_critic.state_dict_with_meta() if self.vprm_critic else None,
            vprm_optimizer_state=self.vprm_optimizer.state_dict() if self.vprm_optimizer else None,
        )

    def resume(self, checkpoint_dir: str | Path) -> bool:
        """Try to resume from latest checkpoint. Returns True if resumed."""
        latest = discover_latest_checkpoint(checkpoint_dir)
        if latest is None:
            return False

        checkpoint = load_checkpoint(latest)
        self.global_step = checkpoint["global_step"]

        if not checkpoint.get("model_state_dict"):
            raise RuntimeError(
                f"Checkpoint {latest} missing model_state_dict — cannot resume with random weights"
            )
        # Filter out bitsandbytes quantization metadata keys that don't exist in
        # a freshly initialized model (absmax, quant_map, quant_state, etc.)
        # These are saved by model.state_dict() but cause errors on load_state_dict()
        # because bnb creates them lazily during quantization, not at init time.
        state_dict = checkpoint["model_state_dict"]
        model_keys = set(self.model.state_dict().keys())
        filtered = {k: v for k, v in state_dict.items() if k in model_keys}
        skipped = len(state_dict) - len(filtered)
        if skipped > 0:
            import warnings
            warnings.warn(f"Checkpoint resume: skipped {skipped} keys not in model (bnb quant metadata)")
        self.model.load_state_dict(filtered, strict=False)

        if checkpoint.get("optimizer_state_dict") and self.optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint.get("scheduler_state_dict") and self.scheduler:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if checkpoint.get("game_state"):
            self.game_state = checkpoint["game_state"]
        if checkpoint.get("advantage_estimator_state"):
            self.advantage_estimator.load_state_dict(checkpoint["advantage_estimator_state"])
        # Restore VPRM critic + optimizer (if saved)
        if checkpoint.get("vprm_critic_state") and self.config.vprm.enabled:
            from qgre.critic import VPRMCritic
            device = str(next(self.model.parameters()).device)
            self.vprm_critic = VPRMCritic.from_checkpoint(checkpoint["vprm_critic_state"], device=device)
            self.vprm_optimizer = torch.optim.Adam(
                [p for p in self.vprm_critic.parameters() if p.requires_grad],
                lr=self._vprm_config.lr,
            )
            if checkpoint.get("vprm_optimizer_state"):
                self.vprm_optimizer.load_state_dict(checkpoint["vprm_optimizer_state"])
            self._vprm_initialized = True
        if checkpoint.get("rng_state") is not None:
            torch.set_rng_state(checkpoint["rng_state"])
        if checkpoint.get("cuda_rng_state") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(checkpoint["cuda_rng_state"])

        # Re-validate fused logprobs after resume — model weights changed
        self._fused_validated = False

        # LoRA verification on resume (PLAN.md line 487-488: mandatory step)
        try:
            from qgre.lora_verify import LoRAVerifier
        except ImportError:
            pass  # LoRA verification unavailable — Unsloth not installed
        else:
            try:
                LoRAVerifier.verify_active(self.model, self.tokenizer)
            except AttributeError as e:
                import warnings
                warnings.warn(
                    f"LoRA verification failed after resume: {e}. "
                    f"LoRA adapters may not be active. Check model state."
                )

        return True

    def _get_prompt_tier(self, metadata: dict) -> str:
        """Get the difficulty tier for a prompt from its metadata."""
        if self._difficulty_column:
            return metadata.get(self._difficulty_column, "default")
        return "default"

    def _record_mastery_and_advance(self, reward_results, active_qualities, batch, metrics):
        """Record per-tier mastery scores, check per-tier phase advancement, check tier unlock."""
        import numpy as np
        from collections import defaultdict

        max_phase = max(self.step_qualities.keys())

        # Group reward results by tier
        tier_groups = defaultdict(list)
        for i, rr in enumerate(reward_results):
            meta = batch.metadata[i] if i < len(batch.metadata) else {}
            tier = self._get_prompt_tier(meta)
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
                # Reset SPO baselines for prompts in this tier at the new phase
                tier_pids = [batch.prompt_ids[i] for i in range(len(reward_results))
                             if self._get_prompt_tier(batch.metadata[i] if i < len(batch.metadata) else {}) == tier]
                self.advantage_estimator.on_tier_advance(
                    new_tier=new_phase,
                    prompt_tier_map={pid: new_phase for pid in tier_pids},
                )

        # Check tier unlock
        if self._tier_order:
            new_tier = self.game_state.check_tier_unlock(
                self._tier_order, self._tier_advance_phase, self._tier_advance_threshold,
            )
            if new_tier:
                metrics["tier_unlocked"] = new_tier
                import warnings
                warnings.warn(f"Step {self.global_step}: tier '{new_tier}' unlocked")
                self._apply_difficulty_gate()

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
        """
        if self._dataloader is None or not hasattr(self._dataloader, "set_difficulty_gate"):
            return
        col = self._difficulty_column
        if not col:
            return  # No difficulty column → no gating (default tier, all prompts)

        allowed = set(self.game_state.active_tiers)
        self._dataloader.set_difficulty_gate(allowed, col)

        # Equal weight per tier: tier with fewer prompts gets higher per-prompt weight
        from collections import Counter
        tier_counts = Counter(
            item["metadata"].get(col, "default") for item in self._dataloader.items
        )
        tier_weights = {}
        for item in self._dataloader.items:
            tier = item["metadata"].get(col, "default")
            if tier in allowed:
                tier_weights[item["prompt_id"]] = 1.0 / max(tier_counts[tier], 1)
        if tier_weights:
            self._dataloader.set_priorities(tier_weights)

        import warnings
        warnings.warn(f"Difficulty gate → {sorted(allowed)} ({len(allowed)} tiers active)")

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
                "No generation backend provided. Pass to train() or QGRETrainer constructor."
            )

        self._dataloader = dataloader  # Store ref for difficulty gate updates
        self.setup_optimizer()
        self.optimizer.zero_grad()  # Clean slate — no stale gradients from init or resume
        cfg = self.config.training

        # MLflow experiment setup (PILLARS.md line 128)
        try:
            import mlflow
            mlflow.set_experiment(self.config.logging.mlflow_experiment)
            mlflow.start_run(run_name=f"qgre-step-{self.global_step}")
            log_training_params({
                "model": {"path": self.config.model.path, "lora_rank": self.config.model.lora_rank},
                "algorithm": {"mode": self.config.algorithm.mode, "loss_type": self.config.algorithm.loss_type},
                "training": {"lr": cfg.lr, "total_steps": cfg.total_steps},
            })
        except Exception:
            pass  # MLflow may not be configured

        # Try to resume from checkpoint
        self.resume(self.config.logging.checkpoint_dir)

        # Difficulty-gated curriculum: set initial gate based on current phase
        self._apply_difficulty_gate()

        for epoch in range(10000):  # Outer epoch loop — stops when total_steps reached
            for batch in dataloader:
                if self.global_step >= cfg.total_steps:
                    break

                # 1. Generate (inference mode) with optional LoRA dropout for exploration
                if hasattr(backend, "set_inference_mode"):
                    backend.set_inference_mode()

                # LoRA dropout: partially revert to base model during generation
                gen_cfg = self.config.generation
                restore_lora = None
                if gen_cfg.lora_dropout_rate > 0:
                    from qgre.lora_dropout import apply_lora_dropout, compute_dropout_rate
                    current_rate = compute_dropout_rate(
                        gen_cfg.lora_dropout_rate, gen_cfg.lora_dropout_anneal_steps, self.global_step,
                    )
                    if current_rate > 0:
                        restore_lora = apply_lora_dropout(self.model, current_rate)
                        # Sync noisy weights to vLLM (save AND load — save alone doesn't push to engine)
                        lora_path = str(Path(self.config.logging.checkpoint_dir) / "lora_latest")
                        if hasattr(backend, "save_weights") and hasattr(backend, "load_weights"):
                            backend.save_weights(lora_path)
                            backend.load_weights(lora_path)

                try:
                    output = backend.generate(
                        batch.input_ids.to(next(self.model.parameters()).device),
                        batch.attention_mask.to(next(self.model.parameters()).device),
                    )
                finally:
                    # Always restore clean weights — even if generate crashes
                    # Don't sync to vLLM here — step-end sync (after training) handles it
                    if restore_lora is not None:
                        restore_lora()

                # 2. Score via reward_fn
                reward_results = []
                for i in range(len(output.texts)):
                    prompt = batch.raw_prompts[i] if i < len(batch.raw_prompts) else ""
                    meta = batch.metadata[i] if i < len(batch.metadata) else {}
                    rr = self.reward_fn(prompt, output.texts[i], meta)
                    reward_results.append(rr)

                # 3. Train step (training mode)
                if hasattr(backend, "set_training_mode"):
                    backend.set_training_mode()
                metrics = self.step(batch, output.token_ids, reward_results)

                # 3b. Update prioritized sampling weights (SPO paper Section 3.2)
                if hasattr(dataloader, "set_priorities"):
                    priorities = self.advantage_estimator.get_prompt_priorities()
                    if priorities:
                        dataloader.set_priorities(priorities)

                # 4. Log progress + MLflow
                step_rewards = {int(k.split("_")[-1]): v for k, v in metrics.items()
                               if k.startswith("mastery/step_")}
                # Log mastery and phase to stdout for debugging curriculum
                if self.global_step % 5 == 0:
                    tiers_str = "/".join(self.game_state.active_tiers)
                    reward_mean = metrics.get("reward/mean", 0.0)
                    # Show ALL quality scores from last batch
                    score_parts = []
                    if reward_results:
                        last_scores = reward_results[-1].scores
                        for qk in sorted(last_scores.keys()):
                            score_parts.append(f"{qk.replace('q_','')}={last_scores[qk]:.1f}")
                    scores_str = " ".join(score_parts)
                    # Show first 150 chars of last completion
                    comp_preview = ""
                    if output.texts:
                        comp_preview = output.texts[-1].replace("\n", " ")
                    critic_str = f" critic={metrics.get('critic_loss', 0):.3f}" if metrics.get('critic_loss', 0) > 0 else ""
                    print(f"[{self.global_step}/{cfg.total_steps}] phase={self.game_state.phase} tiers={tiers_str} reward={reward_mean:.2f}{critic_str} {scores_str}")
                    if comp_preview:
                        print(f"  completion: {comp_preview}...")
                try:
                    log_step_metrics(
                        step=self.global_step - 1,
                        reward_mean=metrics.get("reward/mean", 0.0),
                        loss=metrics.get("loss", 0.0),
                        step_rewards=step_rewards if step_rewards else None,
                        extra={k: v for k, v in metrics.items() if k == "phase"},
                    )
                except Exception:
                    pass  # MLflow may not be configured

                # 5. Save checkpoint
                if self.global_step % cfg.save_freq == 0:
                    self.save()

                # 6. LoRA sync (for vLLM weight update)
                if hasattr(backend, "save_weights") and hasattr(backend, "load_weights"):
                    lora_path = Path(self.config.logging.checkpoint_dir) / "lora_latest"
                    backend.save_weights(lora_path)
                    backend.load_weights(lora_path)

                    # LoRA verification after sync (PLAN.md lines 484-487, step 0g)
                    try:
                        from qgre.lora_verify import LoRAVerifier
                        verifier = LoRAVerifier()
                        verifier.verify_sync(lora_path)
                        verifier.verify_active(self.model, self.tokenizer)
                    except ImportError:
                        pass  # LoRA verifier not installed
                    except Exception as e:
                        import warnings
                        warnings.warn(
                            f"Step {self.global_step}: LoRA verification failed: {e}. "
                            f"Training continues but weights may be desynchronized."
                        )

                # 7. Periodic vLLM recreation to prevent VRAM leak (PLAN.md line 719, unsloth #3864)
                if self.global_step > 0 and self.global_step % 50 == 0:
                    if hasattr(backend, "recreate_engine"):
                        try:
                            backend.recreate_engine()
                        except Exception as e:
                            import warnings
                            warnings.warn(
                                f"Step {self.global_step}: vLLM engine recreation failed: {e}. "
                                f"VRAM leak may accumulate. Monitor GPU memory."
                            )

            if self.global_step >= cfg.total_steps:
                break

        # Final checkpoint
        self.save()
        self.completion_logger.close()

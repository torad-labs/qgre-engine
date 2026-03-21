from __future__ import annotations

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
            elif ":" in alg.segmenter:
                import importlib
                mod_path, fn_name = alg.segmenter.rsplit(":", 1)
                segmenter = getattr(importlib.import_module(mod_path), fn_name)

        self.advantage_estimator = QGREStepAdvantageEstimator(
            lr=spo_lr, mode=mode,
            step_qualities=sq,
            segmenter=segmenter or uniform_segmenter,
            normalize_advantages=alg.loss_type != "dr_grpo",
            filter_groups=alg.grpo.filter_groups,
        )

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

        # Completion logger
        self.completion_logger = CompletionLogger(config.logging.completion_dir)

        # Training state
        self.global_step = 0
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: Any = None
        self._accumulated_loss = 0.0

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

        # Phase is engine-managed — use GameState, not RewardResult
        current_phase = self.game_state.phase
        # Fallback to highest available phase if current_phase not in mapping
        fallback_phase = max(p for p in self.phase_qualities if p <= current_phase) if any(p <= current_phase for p in self.phase_qualities) else min(self.phase_qualities)
        active_qualities = [
            self.phase_qualities.get(current_phase, self.phase_qualities[fallback_phase])
        ] * len(reward_results)

        # Compute per-token advantages (segment → step rewards → SPO/GRPO → GDPO → broadcast)
        token_advantages, batch_regions = self.advantage_estimator.compute_advantages(
            batch_prompt_ids=batch.prompt_ids,
            batch_token_ids=completions,
            batch_reward_results=reward_results,
            batch_active_qualities=active_qualities,
            group_size=self.config.algorithm.grpo.n if self.config.algorithm.mode == "grpo" else None,
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

        # Build KL region weights from segmenter regions (THR-style, PLAN.md lines 798-802)
        alg = self.config.algorithm
        kl_region_weights = torch.ones(len(completions), max_comp_len, device=device)
        region_map = {"THINK": alg.kl_think_multiplier, "FORMAT": alg.kl_format_multiplier}
        for i, regions in enumerate(batch_regions):
            for t, region in enumerate(regions):
                if t < max_comp_len:
                    if region in region_map:
                        kl_region_weights[i, t] = region_map[region]
                    elif region.startswith("STEP_"):
                        kl_region_weights[i, t] = alg.kl_step_multiplier

        # SPO low-advantage filter: skip sequences with near-zero signal (PLAN.md lines 658-671)
        if self.config.algorithm.mode == "spo":
            useful = (padded_advs.abs() > 0.01).any(dim=-1)
            if useful.sum() == 0:
                # All advantages near-zero — skip backward pass but still record mastery + check phase
                self.global_step += 1
                metrics = {"loss": 0.0, "reward/mean": sum(rr.reward for rr in reward_results) / len(reward_results),
                           "global_step": self.global_step - 1, "phase": self.game_state.phase, "skipped": True}
                self._record_mastery_and_advance(reward_results, active_qualities, batch, metrics)
                return metrics
            if useful.sum() >= 2 and useful.sum() < len(completions):
                idx = useful.nonzero(as_tuple=True)[0]
                padded_advs = padded_advs[idx]
                comp_tensor = comp_tensor[idx]
                kl_region_weights = kl_region_weights[idx]

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
            mb_advs = padded_advs[mb_start:mb_end]
            mb_mask = response_mask[mb_start:mb_end]

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
                except (ImportError, Exception):
                    pass

            # Forward through full model (preserves Unsloth gradient checkpointing).
            # selective_log_softmax avoids materializing [seq, vocab] log-prob tensor.
            mb_output = self.model(mb_ids)
            mb_logits = mb_output.logits if hasattr(mb_output, "logits") else mb_output
            del mb_output
            mb_lp = logprobs_from_logits(mb_logits[:, :-1, :], mb_ids[:, 1:])
            del mb_logits

            mb_old_lp = mb_lp.detach()

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

            (mb_loss / n_micro).backward()
            total_loss += mb_loss.item() / n_micro

            del mb_lp

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
        for i, rr in enumerate(reward_results):
            # Decode token IDs to text for readable logs
            comp_tokens = completions[i][:50]
            if self.tokenizer is not None:
                comp_text = self.tokenizer.decode(comp_tokens, skip_special_tokens=True)[:200]
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
        if checkpoint.get("rng_state") is not None:
            torch.set_rng_state(checkpoint["rng_state"])
        if checkpoint.get("cuda_rng_state") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(checkpoint["cuda_rng_state"])

        # LoRA verification on resume (PLAN.md line 487-488: mandatory step)
        try:
            from qgre.lora_verify import LoRAVerifier
            LoRAVerifier.verify_active(self.model, self.tokenizer)
        except (ImportError, Exception):
            pass  # Non-fatal if tokenizer not set or Unsloth not loaded

        return True

    def _record_mastery_and_advance(self, reward_results, active_qualities, batch, metrics):
        """Record per-step mastery scores, check phase advancement, update difficulty gate."""
        import numpy as np
        for step_num, quality_keys in self.step_qualities.items():
            active_keys = [k for k in quality_keys if k in active_qualities[0]]
            if active_keys:
                step_scores = [
                    float(np.mean([rr.scores.get(k, 0.0) for k in active_keys]))
                    for rr in reward_results
                ]
                mean_step_score = float(np.mean(step_scores))
                self.game_state.record_step_score(step_num, mean_step_score)
                metrics[f"mastery/step_{step_num}"] = mean_step_score

        max_phase = max(self.step_qualities.keys())
        old_phase = self.game_state.phase
        if self.game_state.check_phase_advance(max_phase):
            new_phase = self.game_state.phase
            metrics["phase_advanced"] = new_phase
            self.advantage_estimator.on_tier_advance(
                new_tier=new_phase,
                prompt_tier_map={pid: new_phase for pid in batch.prompt_ids},
            )
            self._apply_difficulty_gate(self._dataloader)

        self.game_state.step_count = self.global_step
        metrics["phase"] = self.game_state.phase

        stagnation = self.game_state.check_stagnation()
        metrics["stagnation"] = {"normal": 0, "stagnating": 1, "stuck": 2}[stagnation.value]

    def _apply_difficulty_gate(self, dataloader):
        """Apply difficulty-gated curriculum: only sample prompts at or below current phase.

        Uses config.data.difficulty_schedule to map phase → allowed difficulty values.
        Calls dataloader.set_difficulty_gate() to zero out prompts above the gate.
        Logs the gate change to MLflow.
        """
        schedule = self.config.data.difficulty_schedule
        col = self.config.data.difficulty_column
        if not schedule or not col or not hasattr(dataloader, "set_difficulty_gate"):
            return

        phase = self.game_state.phase
        # Find the highest phase in schedule that's <= current phase
        valid_phases = [p for p in schedule if p <= phase]
        if not valid_phases:
            return
        active_phase = max(valid_phases)
        allowed = set(schedule[active_phase])
        dataloader.set_difficulty_gate(allowed, col)

        # Log to MLflow
        try:
            from qgre.mlflow_logger import log_step_metrics
            log_step_metrics(
                step=self.global_step,
                reward_mean=0.0,
                loss=0.0,
                extra={"difficulty_gate_phase": active_phase, "difficulty_gate_tiers": len(allowed)},
            )
        except Exception:
            pass
        import warnings
        warnings.warn(f"Phase {phase}: difficulty gate → {sorted(allowed)}")

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
        self._apply_difficulty_gate(dataloader)

        for epoch in range(100):  # Outer epoch loop — stops when total_steps reached
            for batch in dataloader:
                if self.global_step >= cfg.total_steps:
                    break

                # 1. Generate (inference mode)
                if hasattr(backend, "set_inference_mode"):
                    backend.set_inference_mode()
                output = backend.generate(
                    batch.input_ids.to(next(self.model.parameters()).device),
                    batch.attention_mask.to(next(self.model.parameters()).device),
                )

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

                # 4. Log to MLflow (PLAN.md lines 517-518: per-step reward + advantage metrics)
                try:
                    step_rewards = {int(k.split("_")[-1]): v for k, v in metrics.items()
                                   if k.startswith("mastery/step_")}
                    # Log mastery and phase to stdout for debugging curriculum
                    if self.global_step % 10 == 0:
                        mastery_str = " ".join(f"s{k}={v:.2f}" for k, v in sorted(step_rewards.items()))
                        print(f"[step {self.global_step}] phase={self.game_state.phase} mastery: {mastery_str}")
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

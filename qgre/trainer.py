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
from qgre.logging import CompletionLogger, log_step_metrics
from qgre.nemo_extracted.kl import masked_mean
from qgre.nemo_extracted.logits import logprobs_from_logits
from qgre.nemo_extracted.loss_functions import ClippedPGLossFn
from qgre.segments import Segmenter, uniform_segmenter
from qgre.types import GameState, RewardResult


class GenerationBackend(Protocol):
    """Abstract generation interface — shields trainer from Unsloth internals."""

    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs) -> list[list[int]]:
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
        self.game_state = game_state or GameState()

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
        self.advantage_estimator = QGREStepAdvantageEstimator(
            lr=spo_lr, mode=mode,
            step_qualities=sq,
            segmenter=segmenter or uniform_segmenter,
        )

        # Loss function (NeMo RL extracted)
        self.loss_fn = ClippedPGLossFn({
            "reference_policy_kl_penalty": alg.kl_cov_ratio if alg.loss_mode == "kl_cov" else 0.0,
            "reference_policy_kl_type": "k3",
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
        })

        # Completion logger
        self.completion_logger = CompletionLogger(config.logging.completion_dir)

        # Training state
        self.global_step = 0
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: Any = None

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
        if cfg.lr_scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=cfg.total_steps, eta_min=cfg.lr * 0.1,
            )
        elif cfg.lr_scheduler == "linear":
            self.scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=1.0, end_factor=0.1, total_iters=cfg.total_steps,
            )
        else:
            self.scheduler = None

    def compute_loss(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        advantages: torch.Tensor,
        response_mask: torch.Tensor,
        old_logprobs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """Compute clipped PG loss from model output.

        Args:
            logits: [batch, seq, vocab] model output
            input_ids: [batch, seq] full sequence
            advantages: [batch, seq-1] per-token advantages
            response_mask: [batch, seq-1] mask (1 = response token)
            old_logprobs: [batch, seq-1] log probs from generation (on-policy: detached)

        Returns:
            (loss, metrics_dict)
        """
        curr_logprobs = logprobs_from_logits(logits[:, :-1, :], input_ids[:, 1:])

        if old_logprobs is None:
            old_logprobs = curr_logprobs.detach()

        # Align lengths
        min_len = min(curr_logprobs.shape[1], advantages.shape[1], response_mask.shape[1])
        curr_lp = curr_logprobs[:, :min_len]
        old_lp = old_logprobs[:, :min_len]
        advs = advantages[:, :min_len]
        mask = response_mask[:, :min_len].float()

        loss, metrics = self.loss_fn(
            curr_logprobs=curr_lp,
            prev_logprobs=old_lp,
            advantages=advs,
            mask=mask,
        )

        return loss, metrics

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
        active_qualities = [
            self.phase_qualities.get(current_phase, self.phase_qualities[min(self.phase_qualities)])
        ] * len(reward_results)

        # Compute per-token advantages (segment → step rewards → SPO/GRPO → GDPO → broadcast)
        token_advantages = self.advantage_estimator.compute_advantages(
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

        # Response mask
        prompt_lengths = [0] * len(completions)
        response_mask = self.compute_response_mask(comp_tensor, prompt_lengths)[:, 1:]

        if response_mask.sum() == 0:
            raise RuntimeError(
                f"Step {self.global_step}: no response tokens in any completion — cannot compute loss."
            )

        # Micro-batched forward + backward — avoids OOM on logits tensor
        # Full logits = batch × seq × vocab ≈ 8 × 4096 × 151K × 4B = 18.6GB (impossible on 16GB)
        # Micro-batch: 1-2 seqs at a time, each does forward → loss → backward
        # Gradients accumulate across micro-batches. (ref: TRL PR #2669)
        micro_batch_size = max(1, min(2, len(completions)))
        n_micro = (len(completions) + micro_batch_size - 1) // micro_batch_size
        total_loss = 0.0
        all_metrics = {}

        for mb_start in range(0, len(completions), micro_batch_size):
            mb_end = min(mb_start + micro_batch_size, len(completions))
            mb_ids = comp_tensor[mb_start:mb_end]
            mb_advs = padded_advs[mb_start:mb_end]
            mb_mask = response_mask[mb_start:mb_end]

            mb_output = self.model(mb_ids)
            mb_logits = mb_output.logits if hasattr(mb_output, "logits") else mb_output

            mb_lp = logprobs_from_logits(mb_logits[:, :-1, :], mb_ids[:, 1:])
            mb_old_lp = mb_lp.detach()  # On-policy

            min_len = min(mb_lp.shape[1], mb_advs.shape[1] - 1, mb_mask.shape[1])
            mb_loss, mb_metrics = self.loss_fn(
                curr_logprobs=mb_lp[:, :min_len],
                prev_logprobs=mb_old_lp[:, :min_len],
                advantages=mb_advs[:, :min_len],
                mask=mb_mask[:, :min_len].float(),
            )

            # Scale loss by micro-batch fraction
            (mb_loss / n_micro).backward()
            total_loss += mb_loss.item() / n_micro

            del mb_logits, mb_output, mb_lp  # Free VRAM immediately

            if not all_metrics:
                all_metrics = mb_metrics
            else:
                for k, v in mb_metrics.items():
                    all_metrics[k] = (all_metrics.get(k, 0) + v) / 2

        metrics = all_metrics
        metrics["loss"] = total_loss
        loss_val = total_loss  # For NaN check

        if not (torch.isfinite(torch.tensor(loss_val))):
            raise RuntimeError(
                f"Step {self.global_step}: loss is {loss_val} — aborting to prevent weight corruption."
            )

        # Optimizer step (backward already done in micro-batches above)
        if (self.global_step + 1) % self.config.training.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()
            if self.scheduler is not None:
                self.scheduler.step()

        # Log metrics
        reward_mean = sum(rr.reward for rr in reward_results) / len(reward_results)
        metrics["reward/mean"] = reward_mean
        metrics["global_step"] = self.global_step

        # Record per-step mastery scores to GameState
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

        # Check phase advancement
        max_phase = max(self.step_qualities.keys())
        old_phase = self.game_state.phase
        if self.game_state.check_phase_advance(max_phase):
            new_phase = self.game_state.phase
            metrics["phase_advanced"] = new_phase
            self.advantage_estimator.on_tier_advance(
                new_tier=new_phase,
                prompt_tier_map={pid: new_phase for pid in batch.prompt_ids},
            )

        self.game_state.step_count = self.global_step
        metrics["phase"] = self.game_state.phase

        # Log completions
        for i, rr in enumerate(reward_results):
            self.completion_logger.log_completion(
                step=self.global_step,
                prompt=batch.raw_prompts[i] if i < len(batch.raw_prompts) else "",
                completion=str(completions[i][:50]),
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
            game_state=self.game_state,
            advantage_estimator_state=self.advantage_estimator.state_dict(),
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
        self.model.load_state_dict(checkpoint["model_state_dict"])

        if checkpoint.get("optimizer_state_dict") and self.optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint.get("game_state"):
            self.game_state = checkpoint["game_state"]
        if checkpoint.get("advantage_estimator_state"):
            self.advantage_estimator.load_state_dict(checkpoint["advantage_estimator_state"])
        if checkpoint.get("rng_state") is not None:
            torch.set_rng_state(checkpoint["rng_state"])

        return True

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

        self.setup_optimizer()
        cfg = self.config.training

        # Try to resume from checkpoint
        self.resume(self.config.logging.checkpoint_dir)

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

                # 4. Log to MLflow
                try:
                    log_step_metrics(
                        step=self.global_step - 1,
                        reward_mean=metrics.get("reward/mean", 0.0),
                        loss=metrics.get("loss", 0.0),
                        extra={k: v for k, v in metrics.items()
                               if k.startswith("mastery/") or k == "phase"},
                    )
                except Exception:
                    pass  # MLflow may not be configured

                # 5. Save checkpoint
                if self.global_step % cfg.save_freq == 0:
                    self.save()

                # 6. LoRA sync (for vLLM weight update)
                if hasattr(backend, "save_weights") and hasattr(backend, "load_weights"):
                    from pathlib import Path
                    lora_path = Path(self.config.logging.checkpoint_dir) / "lora_latest"
                    backend.save_weights(lora_path)
                    backend.load_weights(lora_path)

            if self.global_step >= cfg.total_steps:
                break

        # Final checkpoint
        self.save()
        self.completion_logger.close()

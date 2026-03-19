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
from qgre.segments import HYPERGRAPH_V1_STEP_QUALITIES, Segmenter, qwen3_xml_segmenter
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
        sq = step_qualities or HYPERGRAPH_V1_STEP_QUALITIES
        self.step_qualities = sq
        self.phase_qualities = build_phase_qualities(sq)

        # Algorithm setup
        alg = config.algorithm
        mode = alg.mode
        spo_lr = alg.spo.lr if mode == "spo" else 0.1
        self.advantage_estimator = QGREStepAdvantageEstimator(
            lr=spo_lr, mode=mode,
            step_qualities=sq,
            segmenter=segmenter or qwen3_xml_segmenter,
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
        """Create optimizer and scheduler. Call after model is on device."""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.training.lr,
        )

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
        eos_token_id: int = 151643,
    ) -> torch.Tensor:
        """Compute response mask: 1 for response tokens, 0 for prompt + padding after EOS."""
        batch_size, seq_len = input_ids.shape
        mask = torch.zeros(batch_size, seq_len, dtype=torch.float32, device=input_ids.device)

        for i in range(batch_size):
            start = prompt_lengths[i]
            # Find first EOS after prompt
            eos_positions = (input_ids[i, start:] == eos_token_id).nonzero(as_tuple=True)[0]
            if len(eos_positions) > 0:
                end = start + eos_positions[0].item() + 1  # Include EOS
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

        This is the algorithm-only path for testing. The full training loop
        (generate → score → step) is in train().

        Returns metrics dict.
        """
        assert self.optimizer is not None, "Call setup_optimizer() first"

        # Determine active qualities per completion
        active_qualities = [
            self.phase_qualities.get(rr.phase, self.phase_qualities[min(self.phase_qualities)])
            for rr in reward_results
        ]

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

        # Forward pass — single call, extract logits
        output = self.model(comp_tensor)
        logits = output.logits if hasattr(output, "logits") else output

        # Response mask: completions from vLLM are response-only (no prompt tokens).
        # prompt_lengths=0 because comp_tensor contains ONLY completion tokens.
        # (ref: silent-failure-hunter round 2, finding #1)
        prompt_lengths = [0] * len(completions)
        response_mask = self.compute_response_mask(comp_tensor, prompt_lengths)[:, 1:]

        if response_mask.sum() == 0:
            raise RuntimeError(
                f"Step {self.global_step}: no response tokens in any completion — cannot compute loss."
            )

        # Compute loss — advantages truncated to seq_len-1 (NOT shifted by 1)
        # logprobs_from_logits already does the next-token shift internally
        # (ref: silent-failure-hunter round 2, finding #2)
        loss, metrics = self.compute_loss(
            logits=logits,
            input_ids=comp_tensor,
            advantages=padded_advs[:, :-1],
            response_mask=response_mask,
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Step {self.global_step}: loss is {loss.item()} — aborting to prevent weight corruption."
            )

        # Backward + update
        loss = loss / self.config.training.gradient_accumulation_steps
        loss.backward()

        if (self.global_step + 1) % self.config.training.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()

        # Log metrics
        reward_mean = sum(rr.reward for rr in reward_results) / len(reward_results)
        metrics["reward/mean"] = reward_mean
        metrics["global_step"] = self.global_step

        # Log completions
        for i, rr in enumerate(reward_results):
            self.completion_logger.log_completion(
                step=self.global_step,
                prompt=batch.raw_prompts[i] if i < len(batch.raw_prompts) else "",
                completion=str(completions[i][:50]),  # Truncated for logging
                reward=rr.reward,
                reward_components=rr.scores,
                phase=rr.phase,
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

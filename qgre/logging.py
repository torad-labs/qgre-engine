from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlflow


def log_step_metrics(
    step: int,
    reward_mean: float,
    loss: float,
    step_rewards: dict[int, float] | None = None,
    step_advantages: dict[int, float] | None = None,
    extra: dict[str, float] | None = None,
):
    """Log per-step metrics to MLflow."""

    metrics = {
        "reward/mean": reward_mean,
        "loss/total": loss,
    }

    if step_rewards:
        for sn, val in step_rewards.items():
            metrics[f"reward/step_{sn}"] = val

    if step_advantages:
        for sn, val in step_advantages.items():
            metrics[f"advantage/step_{sn}"] = val

    if extra:
        metrics.update(extra)

    mlflow.log_metrics(metrics, step=step)


def log_training_params(config_dict: dict[str, Any]):
    """Log training config params to MLflow at start."""

    flat = {}
    for section, values in config_dict.items():
        if isinstance(values, dict):
            for k, v in values.items():
                flat[f"{section}.{k}"] = str(v)
        else:
            flat[section] = str(values)

    mlflow.log_params(flat)


class CompletionLogger:
    """Write completions as JSONL for analysis."""

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._file = None
        self._step = 0

    def log_completion(
        self,
        step: int,
        prompt: str,
        completion: str,
        reward: float,
        reward_components: dict[str, float] | None = None,
        phase: int = 1,
    ):
        """Write a single completion as a JSONL line."""
        # CP3-008: Wrap open in try-except, restore _file state on failure
        if self._file is None or self._step != step:
            if self._file is not None:
                self._file.close()
            path = self.output_dir / f"step_{step:06d}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            prev_file = self._file
            prev_step = self._step
            try:
                self._file = open(path, "a")
                self._step = step
            except OSError as e:
                # CP3-008: Restore state on failure
                self._file = prev_file
                self._step = prev_step
                import warnings

                warnings.warn(
                    f"CP3-008: Failed to open log file {path}: {e}. Log not written.", stacklevel=2
                )
                return

        record = {
            "step": step,
            "input": prompt,
            "output": completion,
            "score": reward,
            "reward_components": reward_components or {},
            "phase": phase,
        }
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

"""Tests for MLflow logging + JSONL dump (Steps 4, 6)."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from qgre.logging import CompletionLogger, log_step_metrics


# --- Step 4: MLflow logging ---


def test_mlflow_metrics_logged():
    """log_step_metrics calls mlflow.log_metrics with expected keys."""
    with patch("qgre.logging.mlflow", create=True) as mock_mlflow:
        log_step_metrics(
            step=10,
            reward_mean=0.75,
            loss=0.32,
            step_rewards={1: 0.9, 2: 0.7, 3: 0.6, 4: 0.5},
            step_advantages={1: 0.3, 2: 0.1, 3: -0.1, 4: -0.2},
        )

        mock_mlflow.log_metrics.assert_called_once()
        call_args = mock_mlflow.log_metrics.call_args
        metrics = call_args[0][0]

        assert "reward/mean" in metrics
        assert "loss/total" in metrics
        assert "reward/step_1" in metrics
        assert "advantage/step_4" in metrics
        assert call_args[1]["step"] == 10


def test_per_step_metrics_present():
    """Metrics include step_1 through step_4 for both reward and advantage."""
    with patch("qgre.logging.mlflow", create=True) as mock_mlflow:
        log_step_metrics(
            step=5,
            reward_mean=0.5,
            loss=0.1,
            step_rewards={1: 0.8, 2: 0.6, 3: 0.4, 4: 0.2},
            step_advantages={1: 0.5, 2: 0.3, 3: 0.1, 4: -0.1},
        )

        metrics = mock_mlflow.log_metrics.call_args[0][0]
        for sn in range(1, 5):
            assert f"reward/step_{sn}" in metrics
            assert f"advantage/step_{sn}" in metrics


# --- Step 6: JSONL dump ---


def test_completion_jsonl_valid():
    """Written line parses as valid JSON."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = CompletionLogger(tmpdir)
        logger.log_completion(
            step=1,
            prompt="What is 2+2?",
            completion="The answer is 4.",
            reward=1.0,
            reward_components={"q_format": 1.0, "q_accuracy": 1.0},
            phase=2,
        )
        logger.close()

        path = Path(tmpdir) / "step_000001.jsonl"
        assert path.exists()

        with open(path) as f:
            line = f.readline()
            record = json.loads(line)
            assert isinstance(record, dict)


def test_completion_jsonl_fields():
    """JSON contains: input, output, score, reward_components, step."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = CompletionLogger(tmpdir)
        logger.log_completion(
            step=5,
            prompt="Test prompt",
            completion="Test completion",
            reward=0.75,
            reward_components={"q_a": 0.8, "q_b": 0.7},
            phase=3,
        )
        logger.close()

        path = Path(tmpdir) / "step_000005.jsonl"
        with open(path) as f:
            record = json.loads(f.readline())

        assert record["input"] == "Test prompt"
        assert record["output"] == "Test completion"
        assert record["score"] == 0.75
        assert record["reward_components"]["q_a"] == 0.8
        assert record["step"] == 5
        assert record["phase"] == 3


def test_completion_logger_multiple_steps():
    """Multiple steps → separate files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = CompletionLogger(tmpdir)
        logger.log_completion(step=1, prompt="p1", completion="c1", reward=0.5)
        logger.log_completion(step=2, prompt="p2", completion="c2", reward=0.6)
        logger.close()

        assert (Path(tmpdir) / "step_000001.jsonl").exists()
        assert (Path(tmpdir) / "step_000002.jsonl").exists()

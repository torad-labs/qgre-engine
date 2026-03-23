"""CLI entry point: python -m qgre train --config config.yaml --reward module:function"""

from __future__ import annotations

# Unsloth Standby: offload vLLM weights during training, reclaim for generation.
# MUST be set before any Unsloth import. Without this, vLLM reserves
# gpu_memory_utilization * VRAM (e.g., 0.35 * 16GB = 5.6GB) permanently,
# making 14B training on 16GB impossible.
import os
os.environ.setdefault("UNSLOTH_VLLM_STANDBY", "1")
# Append expandable_segments to existing CUDA config (don't override user's settings)
_cuda_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
if "expandable_segments" not in _cuda_conf:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = f"{_cuda_conf},expandable_segments:True".lstrip(",")

import argparse
import importlib
import sys
from pathlib import Path

from qgre.config import QGREConfig


def import_reward_fn(spec: str):
    """Import a reward function from 'module.path:function_name' spec."""
    if ":" not in spec:
        raise ValueError(
            f"Reward function spec must be 'module:function', got '{spec}'.\n"
            f"Example: examples.math.reward_fn:math_reward_fn"
        )
    module_path, fn_name = spec.rsplit(":", 1)

    # Add CWD to sys.path so relative imports work
    cwd = str(Path.cwd())
    if cwd not in sys.path:
        sys.path.insert(0, cwd)

    module = importlib.import_module(module_path)
    fn = getattr(module, fn_name, None)
    if fn is None:
        available = [a for a in dir(module) if not a.startswith("_")]
        raise ValueError(
            f"Function '{fn_name}' not found in '{module_path}'.\n"
            f"Available: {available}"
        )
    return fn


def import_segmenter(spec: str | None):
    """Import a segmenter or return None for default."""
    if spec is None:
        return None
    if spec == "uniform":
        from qgre.segments import uniform_segmenter
        return uniform_segmenter
    if spec == "qwen3_xml":
        from qgre.segments import qwen3_xml_segmenter
        return qwen3_xml_segmenter
    if spec in ("hamiltonian", "hif_json", "label"):
        return None  # Needs tokenizer — resolved later in trainer.__init__
    return import_reward_fn(spec)  # Same module:function pattern


def cmd_train(args):
    """Run QGRE training from config YAML + reward function."""
    config = QGREConfig.from_yaml(args.config)
    reward_fn = import_reward_fn(args.reward)
    # CLI --segmenter overrides config; otherwise use config value
    seg_spec = args.segmenter or config.algorithm.segmenter
    segmenter = import_segmenter(seg_spec)

    # Load model and tokenizer via Unsloth backend
    from qgre.generation import UnslothBackend
    backend = UnslothBackend(config.model, config.generation, max_prompt_length=config.data.max_prompt_length)
    model, tokenizer = backend.load()

    # Load training data
    from qgre.data import QGREDataLoader, load_prompts_from_parquet
    all_prompts = []
    for f in config.data.train_files:
        all_prompts.extend(load_prompts_from_parquet(f))

    n_completions = (
        config.algorithm.spo.n if config.algorithm.mode == "spo"
        else config.algorithm.grpo.n
    )
    dataloader = QGREDataLoader(
        prompts=all_prompts,
        tokenizer=tokenizer,
        max_prompt_length=config.data.max_prompt_length,
        train_batch_size=config.data.train_batch_size,
        n_completions=n_completions,
        prompt_column=config.data.prompt_column,
        metadata_columns=config.data.metadata_columns,
        system_prompt_column=config.data.system_prompt_column,
    )

    # Create trainer
    from qgre.trainer import QGRETrainer
    trainer = QGRETrainer(
        model=model,
        tokenizer=tokenizer,
        reward_fn=reward_fn,
        config=config,
        generation_backend=backend,
        segmenter=segmenter,
    )

    print(f"QGRE Engine v{__import__('qgre').__version__}")
    print(f"  Model: {config.model.path}")
    print(f"  Mode: {config.algorithm.mode}")
    print(f"  Steps: {config.training.total_steps}")
    print(f"  Data: {len(all_prompts)} prompts from {len(config.data.train_files)} files")
    print(f"  Reward: {args.reward}")
    print()

    trainer.train(dataloader, backend)
    print("Training complete.")


def main():
    parser = argparse.ArgumentParser(
        prog="qgre",
        description="QGRE Engine — Quality-Gated Reward Escalation training",
    )
    sub = parser.add_subparsers(dest="command")

    train_p = sub.add_parser("train", help="Run QGRE training")
    train_p.add_argument("--config", required=True, help="Path to YAML config file")
    train_p.add_argument(
        "--reward", required=True,
        help="Reward function as module:function (e.g., examples.math.reward_fn:math_reward_fn)",
    )
    train_p.add_argument(
        "--segmenter", default=None,
        help="Segmenter: 'uniform', 'qwen3_xml', or module:function (default: from config)",
    )

    args = parser.parse_args()
    if args.command == "train":
        cmd_train(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

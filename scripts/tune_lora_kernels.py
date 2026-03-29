"""Generate tuned LoRA kernel configs for your GPU.

Runs Triton autotuning for vLLM's LoRA shrink/expand kernels and saves
optimal configs as JSON files that vLLM loads via VLLM_TUNED_CONFIG_FOLDER.

Usage:
    python scripts/tune_lora_kernels.py [--output-dir output/vllm_kernel_configs]

The QGRE engine automatically sets VLLM_TUNED_CONFIG_FOLDER to point at
output/vllm_kernel_configs/ if it exists. Run this script once per GPU model.
"""

from __future__ import annotations

import argparse
import json
import time
from itertools import product
from pathlib import Path

import torch
import torch.utils.benchmark as TBenchmark


def get_gpu_name() -> str:
    """Get GPU name formatted as vLLM expects for config filenames."""
    name = torch.cuda.get_device_name()
    return name.replace(" ", "_").replace("-", "_")


def bench_lora_shrink(
    M: int, K: int, N: int, num_loras: int, num_slices: int,
    dtype: torch.dtype, config: dict,
) -> float:
    """Benchmark a single LoRA shrink config. Returns time in ms."""
    from vllm.lora.ops.triton_ops import lora_shrink, LoRAKernelMeta

    # Create test tensors
    x = torch.randn(M, K, dtype=dtype, device="cuda")
    # lora_a shape: [num_loras, num_slices, rank, hidden_size]
    lora_a = torch.randn(num_loras, num_slices, N, K, dtype=dtype, device="cuda")
    output = torch.zeros(M, num_slices * N, dtype=dtype, device="cuda")
    indices = torch.zeros(M, dtype=torch.long, device="cuda")

    meta = LoRAKernelMeta.from_input(lora_a, indices)

    try:
        timer = TBenchmark.Timer(
            stmt="lora_shrink(x, lora_a, output, num_loras, meta, config)",
            globals={
                "lora_shrink": lora_shrink,
                "x": x, "lora_a": lora_a, "output": output,
                "num_loras": num_loras, "meta": meta, "config": config,
            },
        )
        result = timer.blocked_autorange(min_run_time=0.5)
        return result.median * 1e3  # ms
    except Exception:
        return float("inf")


def bench_lora_expand(
    M: int, K: int, N: int, num_loras: int, num_slices: int,
    add_inputs: bool, dtype: torch.dtype, config: dict,
) -> float:
    """Benchmark a single LoRA expand config. Returns time in ms."""
    from vllm.lora.ops.triton_ops import lora_expand, LoRAKernelMeta

    # lora_b shape: [num_loras, num_slices, hidden_size, rank]
    lora_b = torch.randn(num_loras, num_slices, N, K, dtype=dtype, device="cuda")
    x = torch.randn(M, num_slices * K, dtype=dtype, device="cuda")
    output = torch.randn(M, N, dtype=dtype, device="cuda") if add_inputs else torch.zeros(M, N, dtype=dtype, device="cuda")
    indices = torch.zeros(M, dtype=torch.long, device="cuda")
    output_slices = tuple(N // num_slices for _ in range(num_slices))

    meta = LoRAKernelMeta.from_input(lora_b, indices)

    try:
        timer = TBenchmark.Timer(
            stmt="lora_expand(x, lora_b, output, num_loras, output_slices, add_inputs, meta, config)",
            globals={
                "lora_expand": lora_expand,
                "x": x, "lora_b": lora_b, "output": output,
                "num_loras": num_loras, "output_slices": output_slices,
                "add_inputs": add_inputs, "meta": meta, "config": config,
            },
        )
        result = timer.blocked_autorange(min_run_time=0.5)
        return result.median * 1e3
    except Exception:
        return float("inf")


# Search space for Triton kernel parameters
BLOCK_M = [16, 32, 64, 128]
BLOCK_N = [16, 32, 64, 128]
BLOCK_K = [32, 64, 128, 256]
NUM_WARPS = [4, 8]
NUM_STAGES = [2, 3, 4]
SPLIT_K_SHRINK = [4, 8, 16, 32, 64]

# Qwen3-1.7B shapes (LoRA rank 8-32, hidden_size=2048, intermediate=5632)
# These cover the actual shapes seen during QGRE training
HIDDEN_SIZES = [2048]
LORA_RANKS = [8, 16, 32]
BATCH_SIZES = [1, 4, 16, 32, 64, 128, 256]
NUM_SLICES_LIST = [1, 3]  # 1 for most modules, 3 for gate_up_proj
MAX_LORAS = 1  # Single LoRA in QGRE


def tune_shrink(hidden_sizes, lora_ranks, batch_sizes, dtype) -> dict:
    """Find optimal shrink kernel configs for all shape combinations."""
    configs = {}
    total = len(batch_sizes) * len(hidden_sizes) * len(lora_ranks) * len(NUM_SLICES_LIST)
    done = 0

    for m, k, n, num_slices in product(batch_sizes, hidden_sizes, lora_ranks, NUM_SLICES_LIST):
        done += 1
        best_time = float("inf")
        best_config = None

        # Reduced search: test key combinations
        for bm, bn, bk, nw, ns, sk in product(
            BLOCK_M, BLOCK_N[:3], BLOCK_K[:3], NUM_WARPS, NUM_STAGES[:2], SPLIT_K_SHRINK[:3]
        ):
            config = {
                "BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk,
                "num_warps": nw, "num_stages": ns, "split_k": sk,
            }
            t = bench_lora_shrink(m, k, n, MAX_LORAS, num_slices, dtype, config)
            if t < best_time:
                best_time = t
                best_config = config

        key = f"{MAX_LORAS},{num_slices},{m},{k},{n}"
        configs[key] = best_config
        print(f"  [{done}/{total}] shrink M={m} K={k} N={n} slices={num_slices}: {best_time:.3f}ms {best_config}")

    return configs


def tune_expand(hidden_sizes, lora_ranks, batch_sizes, add_inputs: bool, dtype) -> dict:
    """Find optimal expand kernel configs for all shape combinations."""
    configs = {}
    total = len(batch_sizes) * len(hidden_sizes) * len(lora_ranks) * len(NUM_SLICES_LIST)
    done = 0

    for m, k, n, num_slices in product(batch_sizes, lora_ranks, hidden_sizes, NUM_SLICES_LIST):
        done += 1
        best_time = float("inf")
        best_config = None

        for bm, bn, bk, nw, ns in product(
            BLOCK_M, BLOCK_N, BLOCK_K[:3], NUM_WARPS, NUM_STAGES[:2]
        ):
            config = {
                "BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk,
                "num_warps": nw, "num_stages": ns,
            }
            t = bench_lora_expand(m, k, n, MAX_LORAS, num_slices, add_inputs, dtype, config)
            if t < best_time:
                best_time = t
                best_config = config

        key = f"{MAX_LORAS},{num_slices},{m},{k},{n}"
        configs[key] = best_config
        print(f"  [{done}/{total}] expand(add={add_inputs}) M={m} K={k} N={n} slices={num_slices}: {best_time:.3f}ms {best_config}")

    return configs


def save_config(configs: dict, output_dir: Path, gpu_name: str, op_type: str, add_inputs: bool | None = None):
    """Save configs as JSON in vLLM's expected format."""
    # Convert flat key format to nested: config[max_loras][num_slices][m][k][n]
    nested = {}
    for key, config in configs.items():
        parts = key.split(",")
        ml, ns, m, k, n = parts
        nested.setdefault(ml, {}).setdefault(ns, {}).setdefault(m, {}).setdefault(k, {})[n] = config

    if op_type == "expand":
        fname = f"{gpu_name}_{op_type.upper()}_{str(add_inputs).upper()}.json"
    else:
        fname = f"{gpu_name}_{op_type.upper()}.json"

    path = output_dir / fname
    with open(path, "w") as f:
        json.dump(nested, f, indent=2)
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Tune vLLM LoRA kernel configs for your GPU")
    parser.add_argument("--output-dir", default="output/vllm_kernel_configs",
                        help="Directory to save tuned JSON configs")
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16"])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    gpu_name = get_gpu_name()

    print(f"Tuning LoRA kernels for: {gpu_name}")
    print(f"  dtype: {dtype}")
    print(f"  hidden_sizes: {HIDDEN_SIZES}")
    print(f"  lora_ranks: {LORA_RANKS}")
    print(f"  batch_sizes: {BATCH_SIZES}")
    print(f"  output: {output_dir}")
    print()

    start = time.time()

    print("=== Tuning SHRINK kernels ===")
    shrink_configs = tune_shrink(HIDDEN_SIZES, LORA_RANKS, BATCH_SIZES, dtype)
    save_config(shrink_configs, output_dir, gpu_name, "shrink")

    print("\n=== Tuning EXPAND (add_inputs=True) kernels ===")
    expand_true = tune_expand(HIDDEN_SIZES, LORA_RANKS, BATCH_SIZES, True, dtype)
    save_config(expand_true, output_dir, gpu_name, "expand", add_inputs=True)

    print("\n=== Tuning EXPAND (add_inputs=False) kernels ===")
    expand_false = tune_expand(HIDDEN_SIZES, LORA_RANKS, BATCH_SIZES, False, dtype)
    save_config(expand_false, output_dir, gpu_name, "expand", add_inputs=False)

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.0f}s. Configs saved to {output_dir}/")
    print(f"QGRE will auto-detect these on next training run.")


if __name__ == "__main__":
    main()

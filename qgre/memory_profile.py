"""GPU memory profiling for QGRE training.

Two modes:
1. Snapshot mode: Records every CUDA allocation with stack traces,
   saves an interactive HTML timeline. Best for finding "where are the bytes."
2. Quick audit: Walks model + optimizer and prints per-module breakdown.

Usage:
    python -m qgre train --config config.yaml --reward module:fn --memory-profile

Generates: output/memory_profile.html (open in browser)
"""

from __future__ import annotations

import pickle  # nosec B403
from pathlib import Path

import torch


def start_recording() -> None:
    """Start recording CUDA memory allocation history with stack traces."""
    if not torch.cuda.is_available():
        return
    torch.cuda.memory._record_memory_history(max_entries=100_000)
    print("QGRE: Memory profiling started — recording all CUDA allocations")


def save_snapshot(output_dir: str = "output", label: str = "snapshot") -> Path | None:
    """Save memory snapshot and generate interactive HTML visualization.

    The HTML shows every allocation by stack trace over time.
    Color-coded: you can see model weights, vLLM, optimizer state, etc.
    """
    if not torch.cuda.is_available():
        return None

    torch.cuda.synchronize()
    snapshot = torch.cuda.memory._snapshot()
    torch.cuda.memory._record_memory_history(enabled=None)  # Stop recording

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Save raw pickle (for offline analysis)
    pkl_path = out / f"memory_{label}.pickle"
    with open(pkl_path, "wb") as f:
        pickle.dump(snapshot, f)

    # Generate interactive HTML
    html_path = out / f"memory_{label}.html"
    try:
        from torch.cuda._memory_viz import trace_plot

        with open(html_path, "w") as f:
            f.write(trace_plot(snapshot))
        print(f"QGRE: Memory profile saved → {html_path} (open in browser)")
    except ImportError:
        # Older PyTorch — try segment_plot fallback
        try:
            from torch.cuda._memory_viz import segment_plot

            with open(html_path, "w") as f:
                f.write(segment_plot(snapshot))
            print(f"QGRE: Memory segment plot saved → {html_path}")
        except ImportError:
            print(f"QGRE: Raw snapshot saved → {pkl_path}")
            print(
                '  Visualize: python -c "import pickle, torch; '
                "from torch.cuda._memory_viz import trace_plot; "
                f"s=pickle.load(open('{pkl_path}','rb')); "
                f"open('{html_path}','w').write(trace_plot(s))\""
            )

    return html_path


def quick_audit(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer | None = None, label: str = ""
) -> dict[str, float]:
    """Print per-module GPU memory breakdown and return sizes dict.

    Groups by top-2 module path. Shows gap between tracked tensors
    and total CUDA allocation (gap = vLLM KV cache + other untracked).
    """
    if not torch.cuda.is_available():
        return {}

    torch.cuda.synchronize()

    # 1. Parameters by module
    module_sizes: dict[str, float] = {}
    for name, param in model.named_parameters():
        if param.device.type != "cuda":
            continue
        size_mb = param.numel() * param.element_size() / 1024**2
        parts = name.split(".")
        key = ".".join(parts[:2]) if len(parts) > 2 else name
        module_sizes[key] = module_sizes.get(key, 0) + size_mb

    # 2. Buffers (includes quantization state tables)
    buffer_total = 0.0
    for name, buf in model.named_buffers():
        if buf.device.type != "cuda":
            continue
        size_mb = buf.numel() * buf.element_size() / 1024**2
        buffer_total += size_mb

    # 3. Optimizer state
    opt_total = 0.0
    if optimizer is not None:
        for pg in optimizer.param_groups:
            for p in pg["params"]:
                if p in optimizer.state:
                    for v in optimizer.state[p].values():
                        if isinstance(v, torch.Tensor) and v.device.type == "cuda":
                            opt_total += v.numel() * v.element_size() / 1024**2

    # 4. Gradients
    grad_total = 0.0
    for _, p in model.named_parameters():
        if p.grad is not None and p.grad.device.type == "cuda":
            grad_total += p.grad.numel() * p.grad.element_size() / 1024**2

    # Print
    alloc_mb = torch.cuda.memory_allocated() / 1024**2
    reserved_mb = torch.cuda.memory_reserved() / 1024**2
    free_mb, total_mb = (x / 1024**2 for x in torch.cuda.mem_get_info())

    param_total = sum(module_sizes.values())
    accounted = param_total + buffer_total + opt_total + grad_total

    print(f"\n{'=' * 70}")
    print(f"  GPU MEMORY AUDIT: {label}")
    print(f"{'=' * 70}")
    print(f"  {'MODULE':50s} {'MiB':>8s}")
    print(f"  {'─' * 58}")
    for name, size in sorted(module_sizes.items(), key=lambda x: -x[1]):
        if size > 0.5:
            print(f"  {name:50s} {size:8.1f}")
    print(f"  {'─' * 58}")
    print(f"  {'Parameters':50s} {param_total:8.1f}")
    print(f"  {'Buffers (quant state, etc.)':50s} {buffer_total:8.1f}")
    print(f"  {'Optimizer state':50s} {opt_total:8.1f}")
    print(f"  {'Gradients':50s} {grad_total:8.1f}")
    print(f"  {'─' * 58}")
    print(f"  {'ACCOUNTED':50s} {accounted:8.1f}")
    print(f"  {'torch.cuda.memory_allocated()':50s} {alloc_mb:8.1f}")
    print(f"  {'UNTRACKED (vLLM KV cache, CUDA overhead, etc.)':50s} {alloc_mb - accounted:8.1f}")
    print(f"  {'─' * 58}")
    print(f"  {'PyTorch reserved':50s} {reserved_mb:8.1f}")
    print(f"  {'GPU total':50s} {total_mb:8.1f}")
    print(f"  {'GPU free':50s} {free_mb:8.1f}")
    print(f"  {'GPU used (nvidia-smi level)':50s} {total_mb - free_mb:8.1f}")
    print(f"{'=' * 70}\n")

    # 5. GC scan: find ALL tensors on GPU, including autograd-held duplicates.
    # Bare except clauses are intentional — gc.get_objects() yields arbitrary Python
    # objects that may raise on attribute access (dead weak refs, C++ objects, etc.).
    orphan_total = 0.0
    try:
        import gc

        gc.collect()
        torch.cuda.synchronize()

        param_data_ptrs = {
            p.data_ptr()
            for p in model.parameters()
            if hasattr(p, "device") and p.device.type == "cuda"
        }
        buffer_data_ptrs = {
            b.data_ptr()
            for b in model.buffers()
            if hasattr(b, "device") and b.device.type == "cuda"
        }
        grad_data_ptrs = {
            p.grad.data_ptr()
            for p in model.parameters()
            if p.grad is not None and hasattr(p.grad, "device") and p.grad.device.type == "cuda"
        }
        opt_data_ptrs: set[int] = set()
        if optimizer is not None:
            for pg in optimizer.param_groups:
                for p in pg["params"]:
                    if p in optimizer.state:
                        for v in optimizer.state[p].values():
                            if isinstance(v, torch.Tensor) and v.device.type == "cuda":
                                opt_data_ptrs.add(v.data_ptr())
        known_ptrs = param_data_ptrs | buffer_data_ptrs | grad_data_ptrs | opt_data_ptrs

        orphan_tensors: list[tuple[tuple[int, ...], str, float]] = []
        for obj in gc.get_objects():
            if not isinstance(obj, torch.Tensor):
                continue
            if not hasattr(obj, "device") or obj.device.type != "cuda":
                continue
            if obj.data_ptr() in known_ptrs or obj.numel() == 0:
                continue
            size_mb = obj.numel() * obj.element_size() / 1024**2
            if size_mb < 0.1:
                continue
            orphan_tensors.append((tuple(obj.shape), str(obj.dtype), size_mb))
            orphan_total += size_mb

        orphan_tensors.sort(key=lambda x: -x[2])
        print(f"  {'─' * 58}")
        print("  ORPHAN TENSORS (on GPU, not in model/optimizer/grads):")
        if orphan_tensors:
            for shape, dtype, mb in orphan_tensors[:25]:
                print(f"    {shape!s:30s} {dtype:15s} {mb:8.1f} MiB")
            if len(orphan_tensors) > 25:
                print(f"    ... and {len(orphan_tensors) - 25} more")
        else:
            print("    (none found)")
        print(f"  {'TOTAL ORPHANS':50s} {orphan_total:8.1f}")
        print(f"  {'TOTAL ACCOUNTED + ORPHANS':50s} {accounted + orphan_total:8.1f}")
        print(f"  {'STILL UNTRACKED':50s} {alloc_mb - accounted - orphan_total:8.1f}")
    except (RuntimeError, TypeError, AttributeError) as e:
        print(f"  GC orphan scan failed: {e}")
    print(f"{'=' * 70}\n")

    return {
        "params": param_total,
        "buffers": buffer_total,
        "optimizer": opt_total,
        "gradients": grad_total,
        "orphans": orphan_total,
        "allocated": alloc_mb,
        "reserved": reserved_mb,
    }

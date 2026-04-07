"""Measure pairwise token embedding distances across Qwen3 model sizes.

Research question: Does the distance between domain-relevant tokens
(P, V, T, H, x, m, k, g) scale linearly with hidden dimension,
or does it compress non-linearly in smaller models?

If linear: advantage_scale = hidden_dim_ratio. One number.
If non-linear: some concepts collapse together in small models.

Only downloads the embedding layer from each model — not the full weights.
"""

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import torch


# Physics tokens relevant to Hamiltonian mechanics
# Include single-char tokens AND multi-char tokens that the tokenizer might produce
PHYSICS_TOKENS = [
    # Variables
    "p",
    "V",
    "T",
    "H",
    "x",
    "y",
    "q",
    # Constants/parameters
    "m",
    "k",
    "g",
    # Operators/calculus
    "dt",
    "dx",
    "dp",
    # Key physics words
    "kinetic",
    "potential",
    "momentum",
    "Hamiltonian",
    # Math operators (to see if math vs language separation scales)
    "frac",
    "partial",
    "sqrt",
    # Control tokens (non-physics, for comparison)
    "the",
    "is",
    "and",
    "of",
]

# Qwen3 model sizes to compare
MODEL_IDS = [
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3-32B",
]


def load_embeddings_only(model_id: str) -> tuple[torch.Tensor, "PreTrainedTokenizer"]:
    """Load only the embedding matrix and tokenizer from a model.

    Uses safetensors index to download only the shard containing embeddings,
    not the full model weights.
    """
    import safetensors.torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoConfig, AutoTokenizer

    print(f"\n{'=' * 60}")
    print(f"Loading embeddings from {model_id}")
    print(f"{'=' * 60}")

    # Load tokenizer (small download)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # Load config to get hidden_size
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    hidden_size = config.hidden_size
    print(f"  Hidden size: {hidden_size}")
    print(f"  Vocab size: {config.vocab_size}")

    # Try to load from safetensors index to find which shard has embeddings
    try:
        index_file = hf_hub_download(model_id, "model.safetensors.index.json")
        with open(index_file) as f:
            index = json.load(f)

        # Find which file contains the embedding weights
        weight_map = index["weight_map"]
        embed_key = None
        embed_file = None
        for key, filename in weight_map.items():
            if "embed_tokens" in key:
                embed_key = key
                embed_file = filename
                break

        if embed_file:
            print(f"  Embedding in shard: {embed_file} (key: {embed_key})")
            shard_path = hf_hub_download(model_id, embed_file)
            tensors = safetensors.torch.load_file(shard_path)
            embeddings = tensors[embed_key]
        else:
            raise KeyError("No embed_tokens found in weight map")

    except Exception as e:
        # Small models might have a single safetensors file
        print(f"  Index not found ({e}), trying single file...")
        try:
            path = hf_hub_download(model_id, "model.safetensors")
            tensors = safetensors.torch.load_file(path)
            # Find embedding key
            for key in tensors:
                if "embed_tokens" in key:
                    embeddings = tensors[key]
                    embed_key = key
                    break
            else:
                raise KeyError(f"No embed_tokens in {list(tensors.keys())[:5]}...")
        except Exception as e2:
            print(f"  ERROR: Could not load embeddings: {e2}")
            return None, None

    print(f"  Embedding shape: {embeddings.shape}")
    return embeddings.float(), tokenizer


def get_token_ids(tokenizer, tokens: list[str]) -> dict[str, int]:
    """Map token strings to their IDs. Returns only tokens that exist as single tokens."""
    token_ids = {}
    for token in tokens:
        # Try encoding the token — we want tokens that map to a single ID
        ids = tokenizer.encode(token, add_special_tokens=False)
        if len(ids) == 1:
            token_ids[token] = ids[0]
        else:
            # Try with space prefix (common in BPE)
            ids2 = tokenizer.encode(f" {token}", add_special_tokens=False)
            if len(ids2) == 1:
                token_ids[token] = ids2[0]
            elif len(ids2) == 2:
                # Space + token as separate IDs, take the second
                token_ids[f"_{token}"] = ids2[1]
            # Skip tokens that don't have single-token representations
    return token_ids


def compute_pairwise_distances(embeddings: torch.Tensor, token_ids: dict[str, int]) -> dict:
    """Compute cosine distances and L2 distances between all token pairs."""
    tokens = list(token_ids.keys())
    ids = [token_ids[t] for t in tokens]

    # Extract embedding vectors
    vecs = embeddings[ids]  # [n_tokens, hidden_dim]

    # Normalize for cosine similarity
    norms = vecs.norm(dim=1, keepdim=True)
    vecs_normed = vecs / norms.clamp(min=1e-8)

    results = {}
    for i, j in combinations(range(len(tokens)), 2):
        pair = f"{tokens[i]}|{tokens[j]}"

        # Cosine distance (1 - cosine_similarity)
        cos_sim = (vecs_normed[i] * vecs_normed[j]).sum().item()
        cos_dist = 1.0 - cos_sim

        # L2 distance (Euclidean)
        l2_dist = (vecs[i] - vecs[j]).norm().item()

        # L2 distance normalized by sqrt(hidden_dim) — makes distances comparable across dims
        l2_norm = l2_dist / np.sqrt(embeddings.shape[1])

        results[pair] = {
            "cosine_dist": cos_dist,
            "l2_dist": l2_dist,
            "l2_normalized": l2_norm,
        }

    return results


def main():
    results = {}

    # Select which models to run (allow CLI override)
    if len(sys.argv) > 1:
        model_ids = [m for m in MODEL_IDS if any(s in m for s in sys.argv[1:])]
    else:
        model_ids = MODEL_IDS

    print(f"Models to analyze: {model_ids}")

    for model_id in model_ids:
        embeddings, tokenizer = load_embeddings_only(model_id)
        if embeddings is None:
            continue

        hidden_dim = embeddings.shape[1]
        token_ids = get_token_ids(tokenizer, PHYSICS_TOKENS)

        print(f"  Resolved {len(token_ids)} single-token entries:")
        for tok, tid in sorted(token_ids.items()):
            print(f"    '{tok}' → ID {tid}")

        distances = compute_pairwise_distances(embeddings, token_ids)

        results[model_id] = {
            "hidden_dim": hidden_dim,
            "vocab_size": embeddings.shape[0],
            "n_tokens": len(token_ids),
            "token_ids": token_ids,
            "distances": distances,
        }

        # Free memory
        del embeddings
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Save raw results
    output_dir = Path("output/hamiltonian/study/embedding_distances")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert to serializable format
    serializable = {}
    for model_id, data in results.items():
        serializable[model_id] = {
            "hidden_dim": data["hidden_dim"],
            "vocab_size": data["vocab_size"],
            "n_tokens": data["n_tokens"],
            "token_ids": data["token_ids"],
            "distances": data["distances"],
        }

    with open(output_dir / "raw_distances.json", "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nRaw results saved to {output_dir / 'raw_distances.json'}")

    # Print comparison table
    print("\n" + "=" * 80)
    print("SCALING ANALYSIS: Token pair distances across model sizes")
    print("=" * 80)

    # Select key physics pairs to highlight
    key_pairs = [
        "p|V",
        "p|T",
        "T|V",
        "p|H",
        "V|H",
        "T|H",  # Physics variables
        "p|x",
        "V|x",
        "m|k",
        "m|g",
        "k|g",  # Variables vs params
        "kinetic|potential",
        "momentum|Hamiltonian",  # Physics concepts
        "p|the",
        "V|the",
        "T|the",  # Physics vs language
    ]

    model_names = list(results.keys())
    if not model_names:
        print("No models loaded successfully.")
        return

    # Find pairs that exist in all models
    all_pairs = (
        set.intersection(*[set(results[m]["distances"].keys()) for m in model_names])
        if model_names
        else set()
    )

    # Header
    header = f"{'Pair':<25}"
    for model_id in model_names:
        dim = results[model_id]["hidden_dim"]
        short = model_id.split("/")[-1]
        header += f" | {short}({dim})"
    print(header)
    print("-" * len(header))

    # Print each pair
    for pair in key_pairs:
        if pair not in all_pairs:
            # Try reversed
            parts = pair.split("|")
            pair_rev = f"{parts[1]}|{parts[0]}"
            if pair_rev not in all_pairs:
                continue
            pair = pair_rev

        row = f"{pair:<25}"
        values = []
        for model_id in model_names:
            dist = results[model_id]["distances"][pair]
            cos_d = dist["cosine_dist"]
            row += f" | {cos_d:.4f}"
            values.append(cos_d)
        print(row)

        # Check linearity: ratio of smallest to largest
        if len(values) >= 2 and values[-1] != 0:
            ratio = values[0] / values[-1] if values[-1] != 0 else float("inf")

    # Compute scaling summary
    print("\n" + "=" * 80)
    print("SCALING RATIOS (distance in model X / distance in largest model)")
    print("=" * 80)

    if len(model_names) >= 2:
        largest = model_names[-1]
        largest_dim = results[largest]["hidden_dim"]

        for model_id in model_names:
            dim = results[model_id]["hidden_dim"]
            dim_ratio = dim / largest_dim

            # Compute average distance ratio across all common pairs
            ratios_cosine = []
            ratios_l2norm = []
            for pair in all_pairs:
                d_this = results[model_id]["distances"][pair]
                d_ref = results[largest]["distances"][pair]
                if d_ref["cosine_dist"] > 1e-6:
                    ratios_cosine.append(d_this["cosine_dist"] / d_ref["cosine_dist"])
                if d_ref["l2_normalized"] > 1e-6:
                    ratios_l2norm.append(d_this["l2_normalized"] / d_ref["l2_normalized"])

            avg_cos_ratio = np.mean(ratios_cosine) if ratios_cosine else 0
            std_cos_ratio = np.std(ratios_cosine) if ratios_cosine else 0
            avg_l2_ratio = np.mean(ratios_l2norm) if ratios_l2norm else 0
            std_l2_ratio = np.std(ratios_l2norm) if ratios_l2norm else 0

            short = model_id.split("/")[-1]
            print(f"\n{short} (dim={dim}, dim_ratio={dim_ratio:.3f}):")
            print(f"  Cosine distance ratio: {avg_cos_ratio:.4f} ± {std_cos_ratio:.4f}")
            print(f"  L2 normalized ratio:   {avg_l2_ratio:.4f} ± {std_l2_ratio:.4f}")
            print(f"  If linear, expected:   {dim_ratio:.4f} (cosine) or 1.000 (L2 norm)")

    print(f"\nFull results: {output_dir / 'raw_distances.json'}")


if __name__ == "__main__":
    main()

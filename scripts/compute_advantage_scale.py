"""Compute the correct advantage_scale from first principles.

Given:
- Model hidden_dim, vocab_size, LoRA rank
- Learning rate
- Typical logit spread (measured from model)
- DrGRPO loss (no length normalization)
- Number of span tokens per completion

We can compute what advantage magnitude produces a gradient
that fits within the model's logit resolution without saturating it.

The chain: advantage → loss → gradient → logit update
- loss_per_token = -advantage * ratio  (ratio ≈ 1.0 on-policy)
- gradient = d(loss)/d(logits) = -advantage * (1 - p_token)  (softmax gradient)
- logit_update = lr * gradient * LoRA_scaling
- We want: logit_update << logit_spread between correct/incorrect tokens

So: advantage_scale = target_logit_nudge / (lr * lora_scaling * mean_tokens_per_span * expected_raw_advantage)
"""

import json
import sys
from pathlib import Path

import torch
import numpy as np


def measure_logit_statistics(model_id: str = "Qwen/Qwen3-1.7B"):
    """Measure the logit distribution of the base model on physics-relevant tokens."""
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    print(f"Model: {model_id}")
    print(f"  Hidden dim: {config.hidden_size}")
    print(f"  Vocab size: {config.vocab_size}")
    print(f"  Num layers: {config.num_hidden_layers}")
    print(f"  Num heads: {config.num_attention_heads}")

    # We only need the embedding and lm_head to measure logit spread
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # Measure logit spread using lm_head weight norms
    # The logit for token i is: logit_i = hidden_state @ lm_head[i]
    # The spread depends on both the hidden state norm and the lm_head weight norms

    # Load just the lm_head weights
    from huggingface_hub import hf_hub_download
    import safetensors.torch

    try:
        index_file = hf_hub_download(model_id, "model.safetensors.index.json")
        with open(index_file) as f:
            index = json.load(f)

        # Find lm_head
        lm_head_file = None
        lm_head_key = None
        for key, filename in index["weight_map"].items():
            if "lm_head" in key:
                lm_head_file = filename
                lm_head_key = key
                break

        if lm_head_file:
            shard_path = hf_hub_download(model_id, lm_head_file)
            tensors = safetensors.torch.load_file(shard_path)
            lm_head = tensors[lm_head_key].float()
        else:
            raise KeyError("No lm_head found")

    except Exception as e:
        print(f"  Trying single file: {e}")
        path = hf_hub_download(model_id, "model.safetensors")
        tensors = safetensors.torch.load_file(path)
        for key in tensors:
            if "lm_head" in key:
                lm_head = tensors[key].float()
                break

    print(f"  lm_head shape: {lm_head.shape}")  # [vocab_size, hidden_dim]

    # Compute per-token weight norms (how "loud" each token is in logit space)
    token_norms = lm_head.norm(dim=1)  # [vocab_size]
    print(f"\n  lm_head weight norms:")
    print(f"    mean: {token_norms.mean():.4f}")
    print(f"    std:  {token_norms.std():.4f}")
    print(f"    min:  {token_norms.min():.4f}")
    print(f"    max:  {token_norms.max():.4f}")
    print(f"    p10:  {torch.quantile(token_norms, 0.1):.4f}")
    print(f"    p90:  {torch.quantile(token_norms, 0.9):.4f}")

    # The logit for a token = dot(hidden_state, lm_head_row)
    # For a typical hidden state with norm ~hidden_dim^0.5 (random init scale),
    # the logit spread is approximately:
    # logit_spread ≈ hidden_state_norm * token_norm_spread

    # Compute pairwise cosine distances between physics tokens in lm_head space
    # (this is different from embedding space — lm_head maps TO logits)
    physics_tokens = ["p", "V", "T", "H", "x", "m", "k", "g", "q"]
    physics_ids = []
    physics_names = []
    for tok in physics_tokens:
        ids = tokenizer.encode(tok, add_special_tokens=False)
        if len(ids) == 1:
            physics_ids.append(ids[0])
            physics_names.append(tok)

    physics_vecs = lm_head[physics_ids]  # [n, hidden_dim]
    physics_norms = physics_vecs.norm(dim=1)

    # Pairwise cosine similarity in lm_head space
    physics_normed = physics_vecs / physics_norms.unsqueeze(1)
    cosine_sim = physics_normed @ physics_normed.T

    print(f"\n  Physics token lm_head norms:")
    for name, norm in zip(physics_names, physics_norms):
        print(f"    '{name}': {norm:.4f}")

    print(f"\n  Physics token pairwise cosine similarity (lm_head space):")
    for i in range(len(physics_names)):
        for j in range(i+1, len(physics_names)):
            print(f"    {physics_names[i]}|{physics_names[j]}: {cosine_sim[i,j]:.4f} (dist={1-cosine_sim[i,j].item():.4f})")

    # Compute expected logit difference between two physics tokens
    # For hidden state h: logit_diff = h @ (lm_head[i] - lm_head[j])
    # Expected magnitude ≈ ||h|| * ||lm_head[i] - lm_head[j]||
    # With typical hidden state norm ≈ sqrt(hidden_dim) * layer_norm_scale

    # In practice, layer norm keeps hidden state norm relatively stable
    # For Qwen3-1.7B with RMSNorm, typical hidden state norm ≈ sqrt(hidden_dim) * gamma
    # where gamma ≈ 1.0 (initialized to 1)

    expected_h_norm = np.sqrt(config.hidden_size)  # rough estimate

    diff_norms = []
    for i in range(len(physics_ids)):
        for j in range(i+1, len(physics_ids)):
            diff = lm_head[physics_ids[i]] - lm_head[physics_ids[j]]
            diff_norms.append(diff.norm().item())

    mean_diff_norm = np.mean(diff_norms)
    expected_logit_diff = expected_h_norm * mean_diff_norm

    print(f"\n  Expected logit statistics:")
    print(f"    Estimated hidden state norm: {expected_h_norm:.2f}")
    print(f"    Mean lm_head diff norm (physics tokens): {mean_diff_norm:.4f}")
    print(f"    Expected logit difference between physics tokens: {expected_logit_diff:.2f}")

    return {
        "hidden_dim": config.hidden_size,
        "vocab_size": config.vocab_size,
        "token_norm_mean": token_norms.mean().item(),
        "token_norm_std": token_norms.std().item(),
        "expected_h_norm": expected_h_norm,
        "mean_physics_diff_norm": mean_diff_norm,
        "expected_logit_diff": expected_logit_diff,
    }


def compute_optimal_advantage_scale(
    logit_stats: dict,
    lr: float = 5e-6,
    lora_rank: int = 32,
    lora_alpha: int = 64,
    mean_span_tokens: int = 50,
    target_nudge_fraction: float = 0.01,  # We want each step to move logits by ~1% of the discriminative gap
):
    """Compute advantage_scale from logit statistics and training config.

    The gradient chain for a single span token with advantage A:
    1. Loss = -A * ratio ≈ -A (on-policy, ratio ≈ 1)
    2. d(loss)/d(logit_target) = -A * (1 - p_target) ≈ -A (p_target small)
    3. This gradient flows through LoRA: effective_lr = lr * lora_scaling
       where lora_scaling = lora_alpha / lora_rank
    4. Logit update per step ≈ lr * lora_scaling * A * n_span_tokens
       (DrGRPO sums over tokens, not averages)

    We want: logit_update << expected_logit_diff between correct/incorrect
    Specifically: logit_update ≈ target_nudge_fraction * expected_logit_diff
    """
    lora_scaling = lora_alpha / lora_rank
    expected_logit_diff = logit_stats["expected_logit_diff"]

    # Target: how much should logits move per training step
    target_logit_nudge = target_nudge_fraction * expected_logit_diff

    # Raw advantage magnitude (before scaling): typically 0.3-0.8
    expected_raw_advantage = 0.5

    # Logit update from one training step with DrGRPO:
    # update = lr * lora_scaling * raw_advantage * advantage_scale * n_span_tokens
    # (DrGRPO doesn't divide by seq_len, so all span tokens contribute additively)

    # Solve for advantage_scale:
    # target_nudge = lr * lora_scaling * raw_adv * adv_scale * n_tokens
    # adv_scale = target_nudge / (lr * lora_scaling * raw_adv * n_tokens)

    advantage_scale = target_logit_nudge / (lr * lora_scaling * expected_raw_advantage * mean_span_tokens)

    print(f"\n{'='*60}")
    print(f"OPTIMAL ADVANTAGE SCALE COMPUTATION")
    print(f"{'='*60}")
    print(f"\nInputs:")
    print(f"  Learning rate: {lr}")
    print(f"  LoRA rank: {lora_rank}, alpha: {lora_alpha}, scaling: {lora_scaling}")
    print(f"  Expected raw advantage: {expected_raw_advantage}")
    print(f"  Mean span tokens per quality: {mean_span_tokens}")
    print(f"  Target nudge fraction: {target_nudge_fraction} ({target_nudge_fraction*100}% of logit gap per step)")
    print(f"\nLogit space:")
    print(f"  Expected logit diff (physics tokens): {expected_logit_diff:.2f}")
    print(f"  Target logit nudge per step: {target_logit_nudge:.4f}")
    print(f"\nGradient chain:")
    print(f"  Per-token gradient ≈ lr * lora_scaling * advantage")
    print(f"  = {lr} * {lora_scaling} * {expected_raw_advantage} * scale * {mean_span_tokens} tokens")
    print(f"\n>>> OPTIMAL advantage_scale = {advantage_scale:.6f}")

    # Also compute what our previous scales would have done
    for scale in [1.0, 0.1, advantage_scale]:
        nudge = lr * lora_scaling * expected_raw_advantage * scale * mean_span_tokens
        pct = nudge / expected_logit_diff * 100
        print(f"\n  scale={scale:.4f}: logit nudge per step = {nudge:.6f} ({pct:.2f}% of logit gap)")

    return advantage_scale


def main():
    print("Measuring model logit space statistics...")
    logit_stats = measure_logit_statistics("Qwen/Qwen3-1.7B")

    # Training config from examples/hamiltonian/config.yaml
    optimal_scale = compute_optimal_advantage_scale(
        logit_stats,
        lr=5e-6,
        lora_rank=32,
        lora_alpha=64,
        mean_span_tokens=50,
        target_nudge_fraction=0.01,  # 1% per step — gentle but meaningful
    )

    # Save results
    output_dir = Path("output/hamiltonian/study/embedding_distances")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "logit_stats": logit_stats,
        "training_config": {
            "lr": 5e-6,
            "lora_rank": 32,
            "lora_alpha": 64,
            "mean_span_tokens": 50,
        },
        "optimal_advantage_scale": optimal_scale,
    }

    with open(output_dir / "advantage_scale_computation.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir / 'advantage_scale_computation.json'}")


if __name__ == "__main__":
    main()

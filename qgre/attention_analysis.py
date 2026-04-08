"""Attention pattern analysis for detecting laminar→turbulent transitions.

Computes:
- Attention entropy per head, per layer
- Attention collapse (all heads attend to same positions)
- Attention fragmentation (uniform random attention)
- Recency bias (only attending to last N tokens)
- Token loop detection (repeated attention to specific earlier tokens)

These metrics detect when the model transitions from coherent (laminar) gradient flow
to incoherent (turbulent) updates, which manifests as sudden performance collapse.
"""

import numpy as np
import torch


def compute_attention_entropy(attentions: tuple) -> dict:
    """Compute attention entropy statistics across all layers and heads.

    Args:
        attentions: Tuple of attention tensors, one per layer
                   Each tensor is [batch, n_heads, seq_len, seq_len]

    Returns:
        Dictionary with:
        - per_layer_entropy: [n_layers] mean entropy across heads
        - per_head_entropy: [n_layers, n_heads] entropy for each head
        - mean_entropy: scalar, overall mean
        - entropy_std: scalar, overall std dev
        - collapse_score: 0-1, higher = more collapsed (low entropy)
        - fragmentation_score: 0-1, higher = more fragmented (high entropy)
    """
    if attentions is None or len(attentions) == 0:
        return {
            "mean_entropy": 0.0,
            "entropy_std": 0.0,
            "collapse_score": 0.0,
            "fragmentation_score": 0.0,
            "per_layer_entropy": [],
            "per_head_entropy": [],
        }

    per_layer_entropy = []
    per_head_entropy = []

    for _layer_idx, attn in enumerate(attentions):
        # attn: [batch, n_heads, seq_len, seq_len]
        # Average over batch
        attn_avg = attn.mean(dim=0)  # [n_heads, seq_len, seq_len]

        # Compute entropy for each head
        # Entropy = -sum(p * log(p)) where p is attention distribution over source tokens
        # attn_avg[h, i, j] = attention from token i to token j in head h
        head_entropies = []
        for head_idx in range(attn_avg.shape[0]):
            # For each query position, compute entropy of attention distribution
            head_attn = attn_avg[head_idx]  # [seq_len, seq_len]
            # Add small epsilon to avoid log(0)
            eps = 1e-10
            head_attn_safe = head_attn + eps
            # Entropy per query position
            entropy_per_pos = -(head_attn_safe * torch.log(head_attn_safe)).sum(dim=-1)  # [seq_len]
            # Mean entropy for this head
            mean_entropy = entropy_per_pos.mean().item()
            head_entropies.append(mean_entropy)

        per_head_entropy.append(head_entropies)
        per_layer_entropy.append(np.mean(head_entropies))

    # Overall statistics
    all_entropies = [e for layer in per_head_entropy for e in layer]
    mean_entropy = np.mean(all_entropies)
    entropy_std = np.std(all_entropies)

    # Collapse score: normalized inverse entropy
    # Low entropy = collapsed attention
    # Theoretical max entropy for seq_len=N is log(N)
    # We'll normalize by empirical max observed
    max_entropy = max(all_entropies) if all_entropies else 1.0
    collapse_score = 1.0 - (mean_entropy / max(max_entropy, 1e-6))

    # Fragmentation score: high entropy indicates random/fragmented attention
    # If entropy is close to theoretical max, attention is uniform/random
    fragmentation_score = mean_entropy / max(max_entropy, 1e-6)

    return {
        "mean_entropy": mean_entropy,
        "entropy_std": entropy_std,
        "collapse_score": collapse_score,
        "fragmentation_score": fragmentation_score,
        "per_layer_entropy": per_layer_entropy,
        "per_head_entropy": per_head_entropy,
    }


def compute_recency_bias(attentions: tuple, window_size: int = 5) -> dict:
    """Measure how much attention is concentrated on recent tokens.

    Args:
        attentions: Tuple of attention tensors, one per layer
        window_size: Number of recent tokens to consider

    Returns:
        Dictionary with:
        - recency_ratio: fraction of attention mass on last window_size tokens
        - per_layer_recency: [n_layers] recency ratio per layer
    """
    if attentions is None or len(attentions) == 0:
        return {"recency_ratio": 0.0, "per_layer_recency": []}

    per_layer_recency = []

    for attn in attentions:
        # attn: [batch, n_heads, seq_len, seq_len]
        # Average over batch and heads
        attn_avg = attn.mean(dim=(0, 1))  # [seq_len, seq_len]

        # For each query position, measure attention to last window_size tokens
        seq_len = attn_avg.shape[0]
        if seq_len <= window_size:
            # All tokens are in the window
            per_layer_recency.append(1.0)
            continue

        # Sum attention to recent tokens for each query position
        recent_attention = []
        for pos in range(seq_len):
            # Attention from pos to last window_size positions before it
            start_idx = max(0, pos - window_size + 1)
            end_idx = pos + 1
            recent_mass = attn_avg[pos, start_idx:end_idx].sum().item()
            recent_attention.append(recent_mass)

        # Average across positions
        mean_recency = np.mean(recent_attention)
        per_layer_recency.append(mean_recency)

    return {
        "recency_ratio": np.mean(per_layer_recency),
        "per_layer_recency": per_layer_recency,
    }


def detect_attention_loops(attentions: tuple, threshold: float = 0.3) -> dict:
    """Detect if attention repeatedly focuses on same earlier tokens (loops).

    Args:
        attentions: Tuple of attention tensors, one per layer
        threshold: Minimum attention weight to consider significant

    Returns:
        Dictionary with:
        - has_loops: bool, whether loops detected
        - loop_strength: 0-1, how strong the loops are
        - loop_positions: list of token positions that are loop targets
    """
    if attentions is None or len(attentions) == 0:
        return {"has_loops": False, "loop_strength": 0.0, "loop_positions": []}

    # Average attention across all layers, heads, and batch
    all_attn = torch.stack([attn.mean(dim=(0, 1)) for attn in attentions]).mean(dim=0)
    # all_attn: [seq_len, seq_len]

    seq_len = all_attn.shape[0]

    # Find positions that receive unusually high attention from many later positions
    # Loop targets are positions that many future tokens attend to strongly
    attention_received = all_attn.sum(dim=0)  # [seq_len] - total attention each position receives

    # Normalize by number of positions that can attend to it
    # Position i can be attended to by positions [i, seq_len)
    normalized_attention = []
    for pos in range(seq_len):
        num_attendees = seq_len - pos
        if num_attendees > 0:
            normalized_attention.append(attention_received[pos].item() / num_attendees)
        else:
            normalized_attention.append(0.0)

    # Find positions with above-threshold normalized attention
    loop_positions = [i for i, attn in enumerate(normalized_attention) if attn > threshold]

    # Loop strength: how much attention mass is concentrated on loop targets
    if loop_positions:
        loop_mass = sum(attention_received[i].item() for i in loop_positions)
        total_mass = attention_received.sum().item()
        loop_strength = loop_mass / max(total_mass, 1e-6)
    else:
        loop_strength = 0.0

    return {
        "has_loops": len(loop_positions) > 0,
        "loop_strength": loop_strength,
        "loop_positions": loop_positions,
    }


def analyze_attention_patterns(attentions: tuple) -> dict:
    """Complete attention pattern analysis.

    Combines all analysis functions into one comprehensive report.

    Args:
        attentions: Tuple of attention tensors from model output

    Returns:
        Dictionary with all attention metrics
    """
    entropy_stats = compute_attention_entropy(attentions)
    recency_stats = compute_recency_bias(attentions)
    loop_stats = detect_attention_loops(attentions)

    return {
        **entropy_stats,
        **recency_stats,
        **loop_stats,
        "n_layers": len(attentions) if attentions else 0,
    }

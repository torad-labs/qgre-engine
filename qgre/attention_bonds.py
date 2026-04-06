"""Token importance computation for advantage constraint.

Two approaches for measuring which tokens are "important anchors" that should have
dampened advantage updates to prevent gradient destabilization:

1. **Attention bond strength** (preferred when available):
   Measures how much later tokens attend to earlier tokens. High bond = downstream
   tokens causally depend on this anchor. Mechanistically correct signal.
   Requires output_attentions=True, which Unsloth fast inference doesn't support.

2. **Hidden state + entropy proxy** (Unsloth-compatible):
   Combines activation magnitude (hidden state L2 norm) with decision confidence
   (inverted entropy). High norm + low entropy = "activated and committed" token.
   Works with any model since we already extract hidden states for fused logprobs.

VRAM note: Attention approach expects a SINGLE layer, not all layers.
Extracting attention from all layers (n_layers × batch × heads × seq × seq) causes OOM.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_bond_strength(
    attention: torch.Tensor,
    seq_len: int,
    mode: str = "max_received",
    batch_size: int = 1,
) -> torch.Tensor:
    """Compute bond strength for each token in completion from a single attention layer.

    Bond strength measures how much later tokens attend to a given token.
    Higher bond strength indicates the token is an important anchor for later generation.

    Args:
        attention: Single-layer attention tensor [batch, n_heads, seq, seq] where
                  attn[b,h,i,j] represents attention from token i to token j.
                  Pass attentions[layer_idx] from model output, NOT the full tuple.
        seq_len: Length of the completion (excluding prompt). Only these tokens
                are analyzed for bond strength.
        mode: Aggregation mode for computing bond strength:
            - "max_received": max attention from any later token across heads
            - "sum_received": sum of attention from all later tokens
            - "mean_received": mean of attention from all later tokens
        batch_size: Batch size for fallback when attention=None.

    Returns:
        Tensor of shape [batch, seq_len] with bond strength values.
        For max_received mode, values are clamped to [0, 1].
        The final token always has zero bond strength (no later tokens exist).

    Example:
        >>> hidden_states, lm_head, attentions = get_hidden_states_and_lm_head(
        ...     model, input_ids, output_attentions=True
        ... )
        >>> # Sample from last layer only (VRAM-safe)
        >>> attention = attentions[-1]  # [batch, heads, seq, seq]
        >>> bond = compute_bond_strength(attention, seq_len=100)
    """
    if attention is None:
        return torch.zeros(batch_size, seq_len, device="cuda" if torch.cuda.is_available() else "cpu")

    # Validate input shape
    if attention.dim() != 4:
        raise ValueError(
            f"Expected 4D attention tensor [batch, heads, seq, seq], got {attention.dim()}D. "
            "Pass attention from a single layer, not the full tuple."
        )

    batch_size, n_heads, full_seq, _ = attention.shape
    device = attention.device

    # Handle edge case: single token completion
    if seq_len == 1:
        return torch.zeros(batch_size, 1, device=device)

    # Handle edge case: seq_len exceeds full_seq
    if seq_len > full_seq:
        seq_len = full_seq

    # Initialize bond strength tensor
    bond_strength = torch.zeros(batch_size, seq_len, device=device)

    # Extract completion portion (last seq_len tokens)
    completion_start = full_seq - seq_len
    completion_attn = attention[:, :, completion_start:, completion_start:]
    # Shape: [batch, n_heads, seq_len, seq_len]

    # For each token position in completion
    for t in range(seq_len - 1):  # Exclude final token (no later tokens)
        # Extract attention received from all later tokens (t+1 onwards)
        # Shape: [batch, n_heads, num_later_tokens] where num_later_tokens = seq_len - (t+1)
        attention_received = completion_attn[:, :, t + 1 :, t]

        if mode == "max_received":
            # Max across heads and later tokens
            bond_strength[:, t] = attention_received.flatten(start_dim=1).max(dim=-1)[0]

        elif mode == "sum_received":
            # Sum across heads and later tokens
            bond_strength[:, t] = attention_received.sum(dim=(1, 2))

        elif mode == "mean_received":
            # Mean across heads and later tokens
            bond_strength[:, t] = attention_received.mean(dim=(1, 2))

        else:
            raise ValueError(
                f"Unknown mode: {mode}. Must be 'max_received', 'sum_received', or 'mean_received'"
            )

    # Final token always has zero bond strength (no later tokens to attend to it)
    bond_strength[:, -1] = 0.0

    # Clamp to [0, 1] for max mode (other modes may exceed 1.0)
    if mode == "max_received":
        bond_strength = torch.clamp(bond_strength, 0.0, 1.0)

    return bond_strength


def select_attention_layer(
    attentions: tuple[torch.Tensor, ...],
    layer_idx: int,
) -> torch.Tensor:
    """Select a single attention layer from the full tuple.

    Args:
        attentions: Tuple of attention tensors, one per layer.
        layer_idx: Which layer to select.
            -1 = last layer (captures high-level patterns)
            -2 = middle layer (n_layers // 2)
            0..n_layers-1 = explicit layer index

    Returns:
        Single attention tensor [batch, heads, seq, seq].

    Raises:
        ValueError: If layer_idx is out of range.
    """
    n_layers = len(attentions)

    if layer_idx == -2:
        # Special case: middle layer
        actual_idx = n_layers // 2
    elif layer_idx < 0:
        # Negative indexing from end
        actual_idx = n_layers + layer_idx
    else:
        actual_idx = layer_idx

    if actual_idx < 0 or actual_idx >= n_layers:
        raise ValueError(
            f"layer_idx {layer_idx} (resolved to {actual_idx}) out of range for {n_layers} layers"
        )

    return attentions[actual_idx]


def compute_importance_from_hidden(
    hidden_states: torch.Tensor,
    logits: torch.Tensor,
    seq_len: int,
    mode: str = "combined",
) -> torch.Tensor:
    """DEPRECATED: Use compute_entropy_importance instead.

    This function used hidden state norms which provide ~33% uniform dampening,
    not selective anchor dampening. Kept for backwards compatibility but
    compute_entropy_importance is the correct implementation.
    """
    return compute_entropy_importance(logits, seq_len, mode="entropy_position")


def compute_causal_decay(seq_len: int) -> float:
    """Compute principled position decay from sequence length.

    The decay exponent controls how fast position weight drops off.
    Derived from causal reach analysis:

    - Linear causal reach: token t affects (seq_len - t) downstream tokens
      This suggests decay=1.0 (linear decay)
    - But attention isn't uniform: empirically follows sqrt-ish distribution
      This suggests decay≈0.5 for short seqs, trending toward 1.0 for long seqs

    Formula: decay = 1.0 / (1 + log2(seq_len / 128))
    - seq_len=128: decay=1.0 (linear, strong early protection)
    - seq_len=512: decay=0.5 (sqrt, moderate)
    - seq_len=2048: decay=0.33 (gentle, avoids over-dampening)

    This ensures longer sequences don't over-dampen early tokens.
    """
    import math
    # Normalize to reference length of 128 tokens
    ratio = max(seq_len, 1) / 128.0
    decay = 1.0 / (1.0 + math.log2(max(ratio, 1.0)))
    return max(0.2, min(1.0, decay))  # Clamp to [0.2, 1.0]


def compute_entropy_importance(
    logits: torch.Tensor,
    seq_len: int,
    mode: str = "entropy_position",
    position_decay: float | None = None,
) -> torch.Tensor:
    """Compute token importance using entropy-regulated causal weighting (ERIC).

    Based on ERPO insight: entropy identifies decision points. Combined with
    position-based causal reach for proper anchor identification:

    - Low entropy = committed anchor = high importance = dampen gradient
    - High entropy = decision point = low importance = full gradient
    - Early position = more downstream tokens depend on it = higher weight

    IMPORTANT: This returns raw importance. The advantage constraint logic
    must handle the "confident but wrong" case by NOT dampening negative
    advantages. See apply_importance_constraint() for correct application.

    Args:
        logits: Logit tensor [batch, seq_len, vocab_size] for completion tokens.
                These are the logits AT each completion position (not shifted).
        seq_len: Length of completion to analyze.
        mode: Importance computation mode:
            - "entropy": Inverted entropy only (1 - normalized_entropy)
            - "position": Position-based causal weight only (earlier = higher)
            - "entropy_position": Product of entropy and position (recommended)
        position_decay: How fast position weight decays. None = auto-compute from
                       seq_len using compute_causal_decay(). Manual values:
                       0.5 = sqrt decay, 1.0 = linear decay.

    Returns:
        Tensor of shape [batch, seq_len] with importance values in [0, 1].
        Higher values = anchor tokens = candidates for dampening.

    Technical notes:
        - Entropy is computed in float32 for numerical stability
        - Log-sum-exp trick used to prevent overflow in softmax
        - Position weight uses (1 - t/seq_len)^decay for smooth decay
    """
    batch_size = logits.shape[0]
    device = logits.device
    logit_seq = logits.shape[1]

    # Handle edge cases
    if seq_len <= 0:
        return torch.zeros(batch_size, 0, device=device)

    # Use available logits (may be shorter than seq_len for edge cases)
    actual_len = min(seq_len, logit_seq)
    completion_logits = logits[:, :actual_len, :]  # [batch, actual_len, vocab]

    # Auto-compute decay if not provided
    if position_decay is None:
        position_decay = compute_causal_decay(actual_len)

    # Compute normalized entropy in float32 for stability
    completion_logits_f32 = completion_logits.float()

    # Log-softmax for numerical stability
    log_probs = F.log_softmax(completion_logits_f32, dim=-1)
    probs = log_probs.exp()

    # Entropy: H = -sum(p * log(p))
    entropy = -(probs * log_probs).sum(dim=-1)  # [batch, actual_len]

    # R3-RSP-007: Normalize entropy using theoretical max (log vocab_size) instead of per-sample max
    # to avoid batch-dependent scaling. vocab_size is inferred from logits.shape[-1].
    vocab_size = completion_logits_f32.shape[-1]
    theoretical_max_entropy = torch.log(torch.tensor(vocab_size, dtype=torch.float32, device=device))
    entropy_norm = entropy / theoretical_max_entropy.clamp(min=1e-6)  # [batch, actual_len]

    # Entropy importance: low entropy = high importance (inverted)
    entropy_importance = 1.0 - entropy_norm

    # Compute position-based causal weight
    # Earlier tokens have more downstream dependencies
    positions = torch.arange(actual_len, device=device, dtype=torch.float32)
    position_weight = (1.0 - positions / max(actual_len - 1, 1)) ** position_decay
    position_weight = position_weight.unsqueeze(0).expand(batch_size, -1)

    # Combine based on mode
    if mode == "entropy":
        importance = entropy_importance
    elif mode == "position":
        importance = position_weight
    elif mode == "entropy_position":
        # Product: both low entropy AND early position required for high importance
        importance = entropy_importance * position_weight
    else:
        raise ValueError(
            f"Unknown mode: {mode}. Must be 'entropy', 'position', or 'entropy_position'"
        )

    # Pad to requested seq_len if needed
    if actual_len < seq_len:
        pad_size = seq_len - actual_len
        importance = F.pad(importance, (0, pad_size), value=0.0)

    return importance.clamp(0.0, 1.0)


def apply_importance_constraint(
    raw_advantage: torch.Tensor,
    importance: torch.Tensor,
    strength: float = 1.0,
) -> torch.Tensor:
    """Apply importance-based dampening with advantage-sign gating.

    This is the critical fix for the "confident but wrong" problem:
    - Positive advantage + high importance → DAMPEN (protect correct anchors)
    - Negative advantage + high importance → NO DAMPEN (correct confident mistakes)

    Without this gating, we would reduce correction signals on tokens that
    are confidently wrong, which is exactly backwards.

    Args:
        raw_advantage: Per-token advantage tensor [seq_len] or [batch, seq_len].
        importance: Per-token importance from compute_entropy_importance().
        strength: Dampening strength multiplier (1.0 = standard).

    Returns:
        Constrained advantage tensor with same shape as input.
    """
    # R3-RSP-001/002/003: Validate strength >= 0.0 to prevent division by zero or sign flip
    if strength < 0.0:
        raise ValueError(
            f"R3-RSP-001: strength must be >= 0.0, got {strength}. "
            "Negative strength causes dampening=0 or negative → division by zero or sign inversion."
        )

    # Compute dampening factor
    dampening = 1.0 + strength * importance

    # Gate by advantage sign: only dampen positive advantages
    # For negative advantages (needs correction), use no dampening
    positive_mask = (raw_advantage >= 0).float()

    # Positive: divide by dampening (protect good anchors)
    # Negative: keep as-is (full correction signal)
    constrained = torch.where(
        raw_advantage >= 0,
        raw_advantage / dampening,
        raw_advantage,  # No dampening for negative advantages
    )

    return constrained


# =============================================================================
# EGRS: Entropy-Gated Reinforcement System functions
# =============================================================================


def compute_normalized_entropy(
    logits: torch.Tensor,
    vocab_size: int | None = None,
) -> torch.Tensor:
    """Compute entropy normalized to [0, 1] using theoretical maximum.

    Unlike compute_entropy_importance() which normalizes by observed max per batch,
    this uses the theoretical maximum entropy (log(vocab_size)) for consistent
    scaling across batches and training runs.

    Args:
        logits: Logit tensor [batch, seq_len, vocab_size].
        vocab_size: Vocabulary size for normalization. If None, inferred from logits.

    Returns:
        Tensor of shape [batch, seq_len] with entropy in [0, 1].
        0 = perfectly confident (one-hot), 1 = maximum uncertainty (uniform).
    """
    import math

    if vocab_size is None:
        vocab_size = logits.shape[-1]

    # Compute in float32 for numerical stability
    logits_f32 = logits.float()

    # Log-softmax for numerical stability
    log_probs = F.log_softmax(logits_f32, dim=-1)
    probs = log_probs.exp()

    # Entropy: H = -sum(p * log(p))
    entropy = -(probs * log_probs).sum(dim=-1)  # [batch, seq_len]

    # Normalize by max possible entropy: log(vocab_size)
    max_entropy = math.log(vocab_size)
    normalized = entropy / max_entropy

    return normalized.clamp(0.0, 1.0)


def compute_confidence_gate(
    entropy: torch.Tensor,
    threshold: float = 0.5,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Compute soft confidence gate from entropy.

    Uses sigmoid for smooth transition between confident and uncertain states.
    This enables soft gating in the 2x2 matrix rather than hard thresholds.

    Args:
        entropy: Normalized entropy tensor [batch, seq_len] in [0, 1].
        threshold: Entropy threshold for confident/uncertain boundary.
        temperature: Sigmoid temperature (lower = sharper transition).

    Returns:
        Gate tensor [batch, seq_len] in [0, 1]:
        - ~1 when entropy >> threshold (uncertain, should reinforce correct)
        - ~0 when entropy << threshold (confident, don't reinforce correct)

    Example:
        >>> entropy = torch.tensor([0.1, 0.5, 0.9])  # confident, boundary, uncertain
        >>> gate = compute_confidence_gate(entropy, threshold=0.5, temperature=0.1)
        >>> # gate ≈ [0.0, 0.5, 1.0]
    """
    # (entropy - threshold) / temperature:
    # - High entropy → positive → sigmoid → ~1 (uncertain)
    # - Low entropy → negative → sigmoid → ~0 (confident)
    return torch.sigmoid((entropy - threshold) / temperature)

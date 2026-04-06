# Entropy-Gated Reinforcement System (EGRS)

> Design document for QGRE's token-level gradient control system.
> Created: 2026-04-05

## Table of Contents

1. [The Problem](#the-problem)
2. [QGRE Philosophy](#qgre-philosophy)
3. [Why Standard RL Doesn't Fit](#why-standard-rl-doesnt-fit)
4. [The Solution: Entropy-Gated Reinforcement](#the-solution-entropy-gated-reinforcement)
5. [The 2x2 Matrix](#the-2x2-matrix)
6. [ERIC: Influence-Based Dampening](#eric-influence-based-dampening)
7. [Hint Injection System](#hint-injection-system)
8. [Complete Token Processing Flow](#complete-token-processing-flow)
9. [Parameters Reference](#parameters-reference)
10. [Implementation Notes](#implementation-notes)

---

## The Problem

Training LLMs with reinforcement learning faces several interconnected challenges:

### 1. Over-Reinforcement Collapse
When we repeatedly reinforce the same correct token, its probability approaches 99%. The model becomes overly confident and eventually collapses to repeating the same tokens regardless of context.

### 2. Attention Cascade Destabilization
Tokens in a transformer are interconnected through attention. When we push one token's gradient hard, we destabilize all tokens that attend to it (its "gravitational neighbors"). This can cascade through the sequence and cause training collapse.

### 3. No Way to Say "Try Something Different"
Standard negative advantages push the model AWAY from specific tokens, but in a vocabulary of 100k+ tokens, knowing what NOT to do doesn't help find what TO do. It's like saying "don't go north" when you need directions to a specific city.

### 4. Wasted Reinforcement on Learned Skills
Once the model has learned to produce a correct token confidently, continuing to reinforce it is wasteful at best and destabilizing at worst.

### 5. The "Confident But Wrong" Problem
When the model is highly confident about a wrong answer, we need to shake that confidence before we can guide it to the right answer. But we can't do this with gradients alone without causing collapse.

---

## QGRE Philosophy

QGRE (Quality-Gated Reinforcement Engine) takes a fundamentally different approach from standard RL:

### Core Principles

1. **Span-Level Rewards**: We don't evaluate individual tokens. We evaluate meaningful spans (like "KINETIC: T = p²/4") against ground truth.

2. **Multi-Quality Signals**: Each span has multiple quality dimensions. This gives nuanced feedback rather than binary right/wrong.

3. **Always Nudging Toward 1.0**: The target is always perfection (reward = 1.0). We measure how close we are, not how we compare to a baseline.

4. **No Binary Right/Wrong**: You're either closer to correct or further from correct. The multi-quality system captures this nuance.

5. **Positive-Only Advantages**: We reinforce good behavior. We don't punish bad behavior—we either ignore it or help the model explore alternatives.

### What This Means for Gradients

- **Advantage = Reward**: The advantage IS the reward (0.0 to 1.0). No baseline subtraction.
- **r = 1.0** → Strong reinforcement ("Yes! Do this!")
- **r = 0.5** → Medium reinforcement ("Getting closer!")
- **r = 0.0** → No reinforcement ("Don't learn from this")
- **Never negative**: We never push the model away from tokens. We either reinforce or stay silent.

---

## Why Standard RL Doesn't Fit

### Standard Policy Gradient
```
advantage = reward - baseline
loss = -advantage * log_prob(token)
```

Problems:
- **Negative advantages**: When reward < baseline, we get negative advantage, which pushes AWAY from tokens. This doesn't help find correct tokens.
- **Baseline dependency**: The baseline (EMA of past rewards) means the same reward can produce different advantages over time. This is confusing—the same correct answer shouldn't get different treatment.
- **Zero for perfect**: Standard approaches often give zero advantage for perfect scores (r=1.0), meaning we don't reinforce correct behavior. The model can drift away from what works.

### QGRE Approach
```
advantage = reward  # Direct, no baseline
loss = -advantage * log_prob(token)  # Only when appropriate (see matrix)
```

Benefits:
- **Always non-negative**: No pushing away from tokens
- **Consistent signal**: Same reward = same advantage, always
- **Reinforce success**: Perfect scores get maximum reinforcement

---

## The Solution: Entropy-Gated Reinforcement

The key insight: **entropy tells us what the model knows**.

- **Low entropy** (peaked distribution): Model is confident about this token
- **High entropy** (flat distribution): Model is uncertain, still exploring

Combined with correctness, this gives us four distinct states that need different treatments.

---

## The 2x2 Matrix

| | **Correct** (reward ≥ threshold) | **Wrong** (reward < threshold) |
|---|---|---|
| **Confident** (low entropy) | Already learned | Confidently wrong (dangerous!) |
| **Uncertain** (high entropy) | Still learning | Lost and exploring |

### Quadrant 1: Uncertain + Correct (REINFORCE)

**State**: Model got it right but isn't sure why (high entropy, correct answer).

**Action**: Reinforce to build confidence.

**Mechanism**:
```python
advantage = reward * confidence_gate * influence_dampen
```

**Why**: The model stumbled onto the right answer. We want to strengthen this path so it becomes reliable. The soft gate (which is ~1 when uncertain, ~0 when confident) ensures we reinforce more when uncertain, less when already confident.

**ERIC applies**: Yes—we still dampen based on influence to prevent cascade destabilization.

### Quadrant 2: Confident + Correct (DO NOTHING)

**State**: Model knows this and is confident about it (low entropy, correct answer).

**Action**: No reinforcement needed.

**Mechanism**:
```python
advantage = 0  # Explicitly zeroed (confident tokens skip reinforcement)
```

**Why**: The model has already learned this. Continuing to reinforce risks:
- Over-reinforcement collapse (probability → 99%)
- Wasted gradient budget
- Destabilizing attention cascades

**The model doesn't need to be told again**. Silence here means "you're fine, keep doing what you're doing."

### Quadrant 3: Confident + Wrong (ENTROPY BOOST)

**State**: Model is confident but wrong (low entropy, incorrect answer). This is dangerous.

**Action**: Shake its confidence by boosting entropy.

**Mechanism**:
```python
advantage = 0  # No reinforcement of wrong answer
entropy_loss = -lambda * entropy[t]  # Maximize entropy (minimize negative entropy)
```

**Why**: We can't directly tell the model "try something different." But we CAN tell it "stop being so sure about this." By boosting entropy, we:
- Flatten the probability distribution
- Make the model uncertain about what it was confident about
- Open it up to exploring alternatives

**What happens next**: After entropy boost, the model will be UNCERTAIN next time it sees this context. Then it falls into Quadrant 4 → hint injection.

### Quadrant 4: Uncertain + Wrong (HINT INJECTION)

**State**: Model is uncertain and wrong (high entropy, incorrect answer). It's lost.

**Action**: Flag for hint injection on next generation.

**Mechanism**:
```python
advantage = 0  # No reinforcement
entropy_adjustment = 0  # Already uncertain, no need to boost
flag_for_hint(prompt_id, span_id)  # Provide guidance next time
```

**Why**: The model is exploring but hasn't found the right path. Zero gradient means zero learning signal. LoRA dropout provides random exploration, but random exploration in a huge space rarely finds the target.

**Hint injection** provides DIRECTIONAL guidance: "Start with this token, then continue." This is like giving a hint in a puzzle—we're not solving it for the model, just pointing it in the right direction.

---

## The Two-Phase Recovery for Wrong Answers

When the model produces a wrong answer, recovery depends on its confidence:

### Phase 1: Confident + Wrong → Entropy Boost
```
Iteration N:
- Model produces wrong answer with high confidence
- We boost entropy (shake confidence)
- No hint yet (model wouldn't listen—it's confident)
```

### Phase 2: Uncertain + Wrong → Hint Injection
```
Iteration N+1:
- Same prompt, model is now uncertain (entropy boost worked)
- Still produces wrong answer (or partial)
- Uncertain + Wrong → flag for hint
- Next generation gets hint injection
```

### Phase 3: Hint-Guided Generation
```
Iteration N+2:
- Model starts generating
- At the flagged span, we inject the correct starting token(s)
- Model continues from hint
- If continuation is correct → Uncertain + Correct → Reinforce
```

This creates a natural flow: **confident wrong → shake → uncertain wrong → hint → uncertain correct → reinforce → confident correct → done**.

---

## ERIC: Influence-Based Dampening

ERIC (Entropy-Regulated Importance Constraint) protects against attention cascade destabilization.

### The Problem It Solves

When we push a token's gradient:
- Tokens that attend to it (downstream) get destabilized
- Earlier tokens affect more downstream tokens
- Low-entropy (committed) tokens are "anchor points" that many tokens depend on

### The Solution

Dampen gradients based on two factors:

1. **Position**: Earlier tokens influence more downstream tokens
2. **Influence**: Tokens with high attention bonds (many attendees) need gentler treatment

### The Formula

```python
# Position-based decay (auto-computed from sequence length)
decay = 1.0 / (1.0 + log2(seq_len / 128))
position_weight = (1 - t / seq_len) ** decay

# Combined importance (for dampening)
importance = entropy_importance * position_weight

# Applied to advantage
advantage = raw_advantage / (1 + strength * importance)
```

### When ERIC Applies

- **Advantages**: Yes, always dampened by ERIC
- **Entropy adjustments**: No, they self-limit at log(vocab_size)

### Position Decay Values (Auto-Computed)

| Sequence Length | Decay | Effect |
|-----------------|-------|--------|
| 64-128 | 1.0 | Linear decay, strong early protection |
| 256 | 0.5 | Sqrt decay |
| 512 | 0.33 | Gentler |
| 2048+ | 0.2 | Very gentle, prevents over-dampening |

---

## Hint Injection System

### Purpose

Provide directional guidance when the model is lost (uncertain + wrong).

### How It Works

1. **Detection**: After reward evaluation, identify spans that are uncertain + wrong
2. **Flagging**: Store `(prompt_id, span_id) → hint_tokens` in a registry
3. **Injection**: On next generation for this prompt:
   - Generate normally until reaching the flagged span
   - Inject the hint tokens (e.g., "T = " for kinetic energy)
   - Model continues predicting from hint
4. **Training**: Model's continuation (after hint) gets normal reinforcement if correct

### What We Store as Hints

For Hamiltonian mechanics, hints are the start of correct expressions:
- "T = " (kinetic energy label + equals)
- "p²/" (start of momentum-squared expression)
- The exact tokens depend on span type and ground truth

**Guideline**: Store tokens up to first operator or equals sign. Enough to point direction, not enough to solve it.

### Hint Decay (Training Wheels)

Hints should fade as the model learns:

```python
hint_probability = max(0, 1 - mastery_score / mastery_threshold)
```

| Mastery Score | Hint Probability | Meaning |
|---------------|------------------|---------|
| 0.0 | 100% | Always hint (just starting) |
| 0.2 | 75% | Usually hint |
| 0.4 | 50% | Coin flip |
| 0.6 | 25% | Occasionally hint |
| 0.8+ | 0% | No hints (model can do it alone) |

### Hint Storage Lifetime

- **Per prompt**: Hints are specific to (prompt_id, span_id)
- **Clear on success**: When model gets span correct WITHOUT hint, clear the flag
- **Epoch boundary**: Optionally clear all hints at epoch boundaries to test retention

---

## Complete Token Processing Flow

```python
def compute_token_gradient(token_t, span_reward, entropy_t, influence_t, mastery):
    """
    Compute the gradient treatment for a single token.
    
    Returns: (advantage, entropy_adjustment, hint_flag)
    """
    
    # Thresholds
    correct = span_reward >= REWARD_THRESHOLD  # e.g., 0.5
    
    # Soft gating for confidence (sigmoid, not hard threshold)
    confidence_gate = sigmoid((ENTROPY_THRESHOLD - entropy_t) / GATE_TEMP)
    confident = confidence_gate > 0.5  # For branching logic
    
    # ERIC dampening (always computed, applied to advantages only)
    position_weight = compute_position_weight(token_t, seq_len)
    influence_dampen = 1.0 / (1.0 + ERIC_STRENGTH * influence_t * position_weight)
    
    if correct:
        if confident:
            # QUADRANT 2: Confident + Correct → Already learned
            advantage = 0
            entropy_adjustment = 0
            hint_flag = False
        else:
            # QUADRANT 1: Uncertain + Correct → Reinforce
            raw_advantage = span_reward * confidence_gate  # ~1 when uncertain, ~0 when confident
            advantage = raw_advantage * influence_dampen  # ERIC applied
            entropy_adjustment = 0
            hint_flag = False
    else:  # wrong
        advantage = 0  # Never reinforce wrong answers
        if confident:
            # QUADRANT 3: Confident + Wrong → Shake confidence
            entropy_adjustment = EXPLORATION_WEIGHT
            hint_flag = False  # Will become uncertain, then get hint
        else:
            # QUADRANT 4: Uncertain + Wrong → Flag for hint
            entropy_adjustment = 0
            hint_flag = True
    
    return advantage, entropy_adjustment, hint_flag
```

### Loss Computation

```python
def compute_loss(tokens, advantages, entropy_adjustments):
    """
    Compute total loss from advantages and entropy adjustments.
    """
    # Policy gradient loss (reinforcement)
    pg_loss = 0
    for t, adv in enumerate(advantages):
        if adv > 0:
            pg_loss += -adv * log_prob[t]
    
    # Entropy adjustment loss (exploration)
    entropy_loss = 0
    for t, adj in enumerate(entropy_adjustments):
        if adj > 0:
            # Maximize entropy = minimize negative entropy
            entropy_loss += -adj * entropy[t]
    
    return pg_loss + entropy_loss
```

---

## Parameters Reference

### Thresholds

| Parameter | Default | Description |
|-----------|---------|-------------|
| `REWARD_THRESHOLD` | 0.5 | Span reward above this = "correct" |
| `ENTROPY_THRESHOLD` | 0.5 | Normalized entropy below this = "confident" |
| `GATE_TEMP` | 0.1 | Sigmoid temperature for soft gating (lower = sharper) |

### ERIC Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ERIC_STRENGTH` | 1.0 | Dampening multiplier (higher = more protection) |
| `POSITION_DECAY` | auto | Computed from seq_len via `1/(1+log2(seq_len/128))` |

### Entropy Adjustment

| Parameter | Default | Description |
|-----------|---------|-------------|
| `EXPLORATION_WEIGHT` | 0.1 | Lambda for entropy bonus (relative to advantage scale) |

### Hint System

| Parameter | Default | Description |
|-----------|---------|-------------|
| `HINT_TOKEN_COUNT` | 2-3 | Tokens to inject (up to first operator) |
| `MASTERY_THRESHOLD` | 0.8 | Mastery score at which hints stop |

---

## Implementation Notes

### Where Each Component Lives

1. **Entropy computation**: `qgre/attention_bonds.py` - `compute_entropy_importance()`
2. **ERIC dampening**: `qgre/attention_bonds.py` - `apply_importance_constraint()`
3. **Advantage computation**: `qgre/advantages.py` - `broadcast_step_advantages_to_tokens()`
4. **Hint registry**: New module `qgre/hints.py` (to be created)
5. **Generation-time injection**: `qgre/generation.py` (to be modified)
6. **Config**: `qgre/config.py` - `AlgorithmConfig`

### Order of Operations

1. **Generation**: Produce completion (with hint injection if flagged)
2. **Evaluation**: Compute span rewards via reward function
3. **Entropy calculation**: Compute token-level entropy from logits
4. **Quadrant assignment**: Determine which quadrant each token falls into
5. **Gradient computation**: Compute advantages and entropy adjustments per quadrant
6. **ERIC dampening**: Apply influence-based dampening to advantages
7. **Loss computation**: Combine policy gradient loss and entropy loss
8. **Hint flagging**: Flag uncertain+wrong spans for next generation
9. **Backprop**: Standard gradient descent

### Soft Gating Implementation

```python
def soft_gate(entropy, threshold, temperature=0.1):
    """
    Soft gate that smoothly transitions from 1 (uncertain) to 0 (confident).
    
    Returns ~1 when entropy >> threshold (uncertain, should reinforce)
    Returns ~0 when entropy << threshold (confident, don't reinforce)
    """
    # Note: we want HIGH entropy = HIGH gate (reinforce)
    # So we use (entropy - threshold), not (threshold - entropy)
    return torch.sigmoid((entropy - threshold) / temperature)
```

### Testing the Design

Before full training, validate:

1. **Quadrant assignment**: Log what % of tokens fall into each quadrant
2. **Hint effectiveness**: Does hint injection improve next-gen correctness?
3. **Entropy boost effect**: Does confident+wrong actually become uncertain?
4. **ERIC stability**: Are attention cascades prevented?
5. **Mastery progression**: Do hints decay as mastery increases?

---

## Appendix: Why Not Negative Advantages?

A common question: "Why not just use negative advantages for wrong answers?"

### The Mathematical Problem

Negative advantage means: `∇loss = -(-A) * ∇log_prob = +A * ∇log_prob`

This pushes the model AWAY from the selected token. But:

1. **Vocabulary is huge**: Pushing away from "cat" doesn't help find "dog"
2. **Context matters**: The same token might be right in another context
3. **Cascade risk**: Negative gradients on anchor tokens destabilize everything

### The QGRE Alternative

Instead of "don't do X", we say:
- "Here's a hint, try starting with Y" (hint injection)
- "Be less sure about what you just did" (entropy boost)
- "This other thing you did was good, do more of that" (positive reinforcement elsewhere)

This is more like teaching than punishment.

---

## Appendix: The Transformer Attention Cascade Problem

### Why Pushing Tokens Is Dangerous

In a transformer, token T's representation affects tokens T+1, T+2, ... through attention:

```
Token 1 → attended by → Tokens 2, 3, 4, 5, ...
         (anchor)         (dependents)
```

If we push Token 1's gradient hard:
- Token 1's embedding shifts
- Tokens 2-5 were calibrated for OLD Token 1
- They're now miscalibrated
- Their outputs become unpredictable
- Training destabilizes

### ERIC's Solution

Dampen gradients on tokens with high influence (many dependents):

```
influence(T) = how much later tokens attend to T
gradient(T) = raw_gradient / (1 + strength * influence(T))
```

High-influence tokens get gentler gradients. The cascade is prevented.

---

## Appendix: Entropy as a Learning State Detector

Entropy tells us the model's internal state:

| Entropy | Model State | Interpretation |
|---------|-------------|----------------|
| Very low (<0.1) | Highly committed | "I'm very sure about this" |
| Low (0.1-0.3) | Confident | "I think this is right" |
| Medium (0.3-0.6) | Uncertain | "Could go either way" |
| High (0.6-0.8) | Exploring | "Trying things out" |
| Very high (>0.8) | Lost | "No idea, picking randomly" |

Combined with correctness, this tells us exactly what intervention is needed:
- Confident + Correct = Already learned (leave alone)
- Uncertain + Correct = Learning (reinforce)
- Confident + Wrong = Bad habit (shake confidence)
- Uncertain + Wrong = Lost (provide guidance)

This is why entropy-gating works: it matches intervention to learning state.

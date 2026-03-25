# Rejection Framework: Fused Logprobs Compiled Cache Fix

## Date: 2026-03-24
## Source: feature-crystallize (generate mode, ELBO 0.69)

---

## Rejected Approaches

### 1. Runtime env var toggling via context manager
**What it is:** The current approach — `unsloth_hidden_states_mode()` wraps each forward call, setting `UNSLOTH_RETURN_HIDDEN_STATES=1` then restoring.
**Why rejected:** Unsloth's `compiler.py` evaluates the env var at compile time and bakes it into a static variable `NOT_RETURN_LOGITS`. Toggling at runtime has zero effect on the compiled graph. This is confirmed by Unsloth issues #3126 (open, Aug 2025) and #3000 (open, Jul 2025). The context manager is dead code on checkpoint resume.
**Evidence strength:** High (C=0.95). Multiple upstream reports, reproducible in our codebase.

### 2. Hardcoded magic number shape check (`> 10000`)
**What it is:** Current code uses `if hidden_states.shape[-1] > 10000` to detect whether the forward returned logits or hidden states.
**Why rejected:** Not model-agnostic. Models with hidden_dim > 10000 exist (large MoE variants). The engine must work across architectures — Qwen3, Llama, DeepSeek, any HF CausalLM.
**Use instead:** `lm_head.in_features` / `lm_head.out_features` comparison. The lm_head is `nn.Linear(hidden_dim, vocab_size)` — its dimensions are authoritative ground truth. No config dependency, no heuristic needed.

### 3. Silent fallback to standard forward
**What it is:** When fused path returns None, fall back to `self.model(mb_ids)` and treat output as logits.
**Why rejected (two reasons):**
1. **Latent bug:** The current fallback references `mb_logits` which is never assigned — it crashes with NameError.
2. **Architectural impossibility:** With `UNSLOTH_RETURN_HIDDEN_STATES=1` global, the standard forward ALSO returns hidden states. `logprobs_from_logits(hidden_states)` would attempt `torch.gather` at token ID positions (e.g., 50000) from a tensor with 4096 entries — IndexError.
**Use instead:** `raise RuntimeError(...)` with actionable diagnostics (delete cache, set config flag).

### 4. Passing `labels` to the forward call
**What it is:** Passing labels kwarg to `model(input_ids, labels=labels)`.
**Why rejected:** Unsloth issue #3000 (open): Qwen3Moe and possibly other MoE architectures have a code path where `labels is not None` bypasses the `NOT_RETURN_LOGITS` check entirely, jumping to fused CE loss. The env var is ignored. QGRE never passes labels — an assert enforces this.
**Evidence strength:** High (C=0.85). Open issue, user provided monkey-patch as workaround.

### 5. Sentinel file to skip cache deletion
**What it is:** Write a marker file after recompilation, skip deletion on subsequent starts.
**Why rejected:** Adds another cached state to manage — mirrors the original problem (stale cached state causing bugs). The 10-30s recompile cost is negligible against a training run. Unconditional deletion also prevents corrupted compiled modules (issue #3763: syntactically invalid Python in compiled cache).
**Evidence strength:** Medium. The optimization is real but the complexity cost exceeds the benefit.

### 6. Fork Unsloth compiler code
**What it is:** Extract and maintain our own version of the forward compilation, baking in hidden states permanently.
**Why rejected:** Massive maintenance burden. Unsloth's compiler changes frequently. Would need updating with every Unsloth release. KL divergence from existing codebase is too high.

### 7. Body splitting (call model body directly, skip lm_head)
**What it is:** Extract model.model (the body layers) and call them directly to get hidden states, bypassing the compiled forward entirely.
**Why rejected:** Already proven to cause 0.25 divergence. The compiled forward includes optimizations (LoRA, gradient checkpointing, inplace attention) that body-only splitting misses. This was the original approach that led to the bug.

### 8. Liger Kernel fused linear cross entropy
**What it is:** Replace Unsloth's hidden states mechanism with Liger Kernel's fused_linear_cross_entropy which computes cross entropy without materializing logits.
**Why rejected:** Complete rewrite of fused_logprobs.py. Different API, different integration pattern. The current Unsloth mechanism works when configured correctly. High KL divergence from existing code, high risk for marginal benefit over the simpler fix.

### 9. Treating `use_fused_logprobs=false` as a working escape hatch without changes
**What it is:** Assuming the config flag still works because it bypasses fused_logprobs code.
**Why rejected:** With global hidden states mode, the non-fused path at trainer.py:428-433 also receives hidden states. `logprobs_from_logits(hidden_states[:, :-1, :], mb_ids[:, 1:])` crashes with IndexError (gather index out of bounds). The escape hatch must be rebuilt to reconstruct logits from hidden states via full `lm_head()` projection.
**Evidence strength:** High (C=0.85). Discovered during crystallization fascia check — connection was F=0 (decorative).

---

## Hard Boundaries

| Boundary | Rationale | Enforcement |
|----------|-----------|-------------|
| NEVER pass `labels` to forward in hidden states mode | Issue #3000: MoE models bypass NOT_RETURN_LOGITS when labels present | Runtime assert in `get_hidden_states_and_lm_head()` |
| NEVER set UNSLOTH_RETURN_HIDDEN_STATES after any Unsloth import | Compile-time evaluation means late setting has no effect | Placement at top of `__main__.py` before all imports |
| NEVER use UNSLOTH_VLLM_STANDBY with load_in_4bit + fast_inference | Standby overrides gpu_memory_utilization to ~90%, causes OOM | Comment in `__main__.py` (existing) |
| Shape check MUST use lm_head.in_features/out_features | Magic numbers and config lookups break model-agnosticity — lm_head dims are authoritative | Code pattern in `get_hidden_states_and_lm_head()` |
| Validation checks chunking correctness ONLY | Single-path world has no independent reference. Model errors caught by training loop (NaN loss, zero rewards) | Comment in trainer.py validation block |
| Fallback MUST crash with diagnostics | No silent degradation. No standard forward path exists to fall back to | RuntimeError with actionable message |

---

## Drift Patterns to Watch

| Pattern | Signal | Response |
|---------|--------|----------|
| Unsloth removes UNSLOTH_RETURN_HIDDEN_STATES | Shape check returns None (last_dim == lm_head.out_features) | Fallback RuntimeError fires. Investigate Unsloth's new mechanism (PR #2772 CCE direction). |
| Unsloth fixes compile-time baking (#3126) | Cache deletion becomes unnecessary but harmless | Keep deletion — it's a 10-30s cost and prevents corrupted caches |
| Someone adds labels to the forward call | Assert fires immediately with explanation | Read issue #3000, find alternative approach that doesn't pass labels |
| New model architecture where get_output_embeddings() doesn't return nn.Linear | lm_head check fails, function returns None | RuntimeError fires. Investigate model's lm_head structure |
| VRAM regression — fused path no longer saves memory | Monitor peak VRAM in training metrics | Compare chunk_size × vocab × 4B vs seq × vocab × 4B. If close, chunking isn't helping |

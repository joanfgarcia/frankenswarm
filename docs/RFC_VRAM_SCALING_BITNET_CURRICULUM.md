# RFC: VRAM Scaling Strategy for BitNet Sovereign School Curriculum Training

**ID:** RFC-BITNET-VRAM-001
**Status:** Draft v4 (reviewed by Grok + DeepSeek — FINAL)
**Author:** Aleth (Netrunner)
**Date:** 2026-07-22
**Reviewers:** Fixer (Joan), Grok (xAI), DeepSeek

---

## 1. Abstract

This document addresses the VRAM scaling challenge in the BitNet Sovereign School curriculum training pipeline. As the model progresses through developmental stages (ages 2-8), neurogenesis events progressively increase the hidden dimension from 128 to 1024, causing GPU memory consumption to grow proportionally. Current projections indicate that the training will exceed the 8GB VRAM capacity of the NVIDIA RTX 5070 before reaching the final stage (age 8, dim 1024). We propose and evaluate multiple mitigation strategies to enable completion of the full curriculum within hardware constraints.

---

## 2. Context & Motivation

### 2.1 The Sovereign School Curriculum

The BitNet Sovereign School is a developmental training pipeline that teaches a small language model (BitNet architecture) through a structured curriculum inspired by Jean Piaget's stages of cognitive development. The model "grows" through stages:

| Stage | Age | Hidden Dim | Layers | Epochs |
|-------|-----|------------|--------|--------|
| 0-1 | — | 128 | 6 | 64 |
| 1-2 | 2 | 256 | 6 | 72 |
| 2-3 | 3 | 384 | 6 | 80 |
| 3-4 | 4 | 512 | 6 | 88 |
| primary_5 | 5 | 640 | 6 | 96 |
| primary_6 | 6 | 768 | 6 | 104 |
| secondary_7 | 7 | 896 | 6 | 112 |
| secondary_8 | 8 | 1024 | 6 | 120 |

Each stage increases the hidden dimension via **neurogenesis** (net2wider), a process that widens the model while preserving learned representations. The model is evaluated at each milestone by "Professor Samantha" (an external evaluator using Mistral 7B).

### 2.2 Current State (as of 2026-07-22)

- **Epoch:** 932 / ~736 total stages (secondary_7)
- **Hidden dim:** 896
- **VRAM usage:** ~6.7 GB / 8.0 GB (83%)
- **Milestones achieved:** 2, 3, 4, 5, 6 years
- **Target:** 7_years (in progress), then 8_years
- **Neurogenesis history:**
  - Epoch 880: 640 → 768 (val_loss: 4.586)
  - Epoch 922: 768 → 896 (val_loss: 4.547)
- **Optimizer:** AdamW (lr=4e-4 scaled, weight_decay=0.05)
- **Batch size:** 64
- **Sequence length:** 128
- **Dataset:** TinyStories (100k stories) + CHILDES + structured curriculum + exam sequences

---

## 3. Problem Statement

### 3.1 VRAM Growth Analysis

GPU memory in the current training loop is consumed by four components:

| Component | Formula (approx) | Current (dim=896) | Projected (dim=1024) |
|-----------|-------------------|--------------------|-----------------------|
| Model params | 4 × layers × (input×hidden + hidden×hidden × 3) | ~238 MB | ~308 MB |
| Optimizer states (AdamW) | 2 × model_size (m + v buffers) | ~476 MB | ~616 MB |
| Gradients | 1 × model_size | ~238 MB | ~308 MB |
| **Activations** | batch × seq_len × hidden × layers × factor | **~5.7 GB** | **~6.5 GB** |
| **Total** | | **~6.7 GB** | **~7.7 GB** |

**Key insight:** Activations dominate VRAM consumption (~85%). Activations scale linearly with `batch_size × seq_len × hidden_dim`. The model parameters and optimizer states are a small fraction of total usage.

### 3.2 Projected Failure Point

At the next neurogenesis event (896 → 1024), projected VRAM usage reaches ~7.7 GB. With the RTX 5070's 8 GB limit, this leaves only ~300 MB of headroom — insufficient for CUDA context, memory fragmentation, and spikes during backward pass. **The training will likely crash with CUDA OOM at the secondary_8 stage.**

### 3.3 Constraints

- **Hardware:** NVIDIA RTX 5070, 8 GB GDDR7 VRAM
- **No multi-GPU:** Single GPU setup
- **No cloud offloading:** Sovereign infrastructure only
- **Continuity:** Must preserve checkpoint compatibility with existing `school_state.json`
- **Model architecture:** BitNet4LayerModel (immutable — cannot change quantization or architecture without retraining)

---

## 4. Proposed Solutions

### 4.1 Gradient Accumulation

**Description:** Instead of computing gradients on the full batch (64) at once, accumulate gradients over multiple micro-batches (e.g., 4 × 16). The optimizer step is performed only after all micro-batches are processed.

**How it reduces memory:** Activations are stored per micro-batch, not per full batch. Peak activation memory = `micro_batch × seq_len × hidden × layers` instead of `batch × seq_len × hidden × layers`.

**Implementation complexity:** Low. ~20 lines of code change in the training loop.

**Pros:**
- Trivial to implement
- Exact mathematical equivalence to large-batch training
- No loss of model quality
- Compatible with existing checkpoints
- Can be combined with other techniques

**Cons:**
- Increases wall-clock time proportionally to accumulation factor (4x slower with accumulation_steps=4)
- Gradient synchronization overhead (minimal for single GPU)
- Does not address memory from model params + optimizer states

**Estimated memory savings:** 50-75% reduction in activation memory. For dim=1024: from ~6.5 GB to ~1.6-3.2 GB activations. **Total VRAM: ~3.5-5.0 GB.**

---

### 4.2 CPU Offloading of Optimizer States

**Description:** Move AdamW optimizer states (m, v tensors = 2× model size) from GPU VRAM to CPU RAM. Only model parameters and gradients remain on GPU. During `optimizer.step()`, states are temporarily paged back to GPU.

**How it reduces memory:** Eliminates ~476 MB (dim=896) or ~616 MB (dim=1024) from VRAM.

**Implementation complexity:** Medium. Requires modifying optimizer to manage CPU/GPU transfers. The existing `trigger_neurogenesis` function already performs CPU offloading — this generalizes the pattern.

**Pros:**
- Frees ~600 MB VRAM at dim=1024
- PyTorch supports this via `torch.optim.AdamW` with manual tensor placement
- No impact on model quality or convergence
- Compatible with all other techniques

**Cons:**
- Adds CPU↔GPU transfer overhead per step (PCIe bandwidth bottleneck)
- Increases wall-clock time by ~10-20%
- Requires sufficient CPU RAM (currently ~7.4 GB used, system has more)
- More complex state management

**Estimated memory savings:** ~600 MB. **Total VRAM: ~7.1 GB at dim=1024** (marginal — needs combination with other techniques).

---

### 4.4 Mixed Precision Training (BF16)

**Description:** Use Brain Float 16 (BF16) precision for forward and backward passes. RTX 5070 (Blackwell architecture) has native BF16 support with minimal accuracy loss compared to FP32.

**How it reduces memory:** Activations and gradients are stored in 16-bit instead of 32-bit, halving their memory footprint. Model params can remain FP32 for weight updates (loss scaling).

**Implementation complexity:** Low-Medium. PyTorch `torch.amp.autocast` + selective `model.to(torch.bfloat16)`.

**Pros:**
- ~50% reduction in activation and gradient memory
- ~2x faster compute on tensor cores (RTX 5070 has BF16 tensor cores)
- Native hardware support — no emulation overhead
- Minimal accuracy impact for this model scale
- Can be combined with all other techniques

**Cons:**
- Requires loss scaling to prevent gradient underflow
- Some operations may not support BF16 (need fallback to FP32)
- Slightly different convergence behavior (may need lr adjustment)
- **CRITICAL: Checkpoint FP32→BF16 conversion requires explicit handling (see §4.4.1)**

**Estimated memory savings:** 50% reduction in activations + gradients. For dim=1024: from ~6.5 GB to ~3.3 GB activations. **Total VRAM: ~4.5-5.0 GB.**

#### 4.4.2 GradScaler for BF16 Gradient Stability

Although BF16 has a wider dynamic range than FP16, it is still narrower than FP32. In deep models with small gradients (especially early layers), this can cause gradient underflow. Consider using `torch.cuda.amp.GradScaler` even with BF16:

```python
scaler = torch.cuda.amp.GradScaler()

with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
    loss = model(input_ids)

scaler.scale(loss).backward()
scaler.unscale_(optimizer)  # Optional: for gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
scaler.step(optimizer)
scaler.update()
```

**Note:** GradScaler is traditionally designed for FP16, but it provides an additional safety net for BF16 in models with heterogeneous gradient magnitudes. Monitor `scaler.get_scale()` — if it consistently decreases, gradients are underflowing and FP32 fallback may be needed for specific layers.

#### 4.4.1 ⚠️ CRITICAL: Checkpoint FP32 → BF16 Conversion

The current checkpoint (`model_current.pt`, epoch 932, dim=896) is stored in FP32. When transitioning to BF16:

**The problem:**
- `torch.amp.autocast` converts activations dynamically during forward pass, but **optimizer states (AdamW m/v) are NOT automatically converted**.
- If loading FP32 checkpoint into a new BF16 optimizer, the m/v buffers remain FP32 and mix with BF16 gradients → **numerical instability**.
- If keeping FP32 optimizer with autocast, you pay full FP32 memory for optimizer states → **no memory savings** from BF16 for that component.

**Required conversion procedure:**
```python
# 1. Load FP32 checkpoint
checkpoint = torch.load('model_current.pt', map_location=device)
model.load_state_dict(checkpoint)

# 2. Convert model to BF16 (selective — keep norms in FP32)
model = model.to(torch.bfloat16)
for module in model.modules():
    if isinstance(module, (torch.nn.LayerNorm, torch.nn.RMSNorm)):
        module.float()  # Keep normalization in FP32

# 3. Create optimizer AFTER conversion (states inherit BF16 dtype)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
# optimizer.m and optimizer.v will be BF16 (matching param dtype)

# 4. Save FP32 master weights separately for recovery
torch.save({k: v.float() for k, v in model.state_dict().items()}, 'model_fp32_master.pt')
```

**Critical detail:** Layers like RMSNorm and LayerNorm MUST remain in FP32 for numerical stability. Apply `model.to(torch.bfloat16)` first, then selectively convert norm layers back to float.

**Validation step:** After conversion, run 10-20 training epochs and compare loss curves with FP32 baseline to confirm convergence is unaffected.

---

### 4.5 Pre-tokenized Streaming Dataset with Stratified Sampling

**Description:** Tokenize the entire corpus (TinyStories + CHILDES + curriculum + exams) once, persist to disk as memory-mapped chunks, and load only the active batch into VRAM during training. Sampling follows a deterministic stratified strategy ensuring uniform coverage.

**How it reduces memory:** Eliminates the need to hold tokenized datasets in CPU memory. Only the current batch resides in VRAM. Tokenization overhead is paid once.

**Implementation complexity:** Medium-High. Requires:
1. A pre-tokenization script that writes chunks to disk
2. A memory-mapped dataset class with stratified sampling
3. A deterministic shuffle strategy (e.g., seeded reservoir sampling or round-robin)

**Pros:**
- Eliminates tokenization overhead from training loop (faster startup)
- Guarantees uniform sample distribution across epochs
- Enables reproducible training (deterministic sampling)
- Reduces CPU memory pressure (no need to hold all tokenized data)
- Chunk-based I/O is SSD-friendly
- Can be combined with all other techniques

**Cons:**
- Requires disk space for tokenized data (~estimated 2-5 GB)
- I/O latency if not using memmap (need careful implementation)
- More complex codebase (new dataset abstraction)
- Does not directly reduce VRAM (complementary technique)
- Initial implementation requires careful testing for correctness

**Estimated memory savings:** Indirect — reduces CPU memory pressure and enables smaller batch loading. **No direct VRAM savings, but enables other techniques.**

---

### 4.6 Paged Optimizers (bitsandbytes)

**Description:** Use bitsandbytes library's paged AdamW optimizer, which automatically pages optimizer states between CPU and GPU based on availability, similar to OS virtual memory.

**How it reduces memory:** Optimizer states are stored in CPU RAM and only paged to GPU when needed for the step operation.

**Implementation complexity:** Low. Drop-in replacement for `torch.optim.AdamW`.

**Pros:**
- Minimal code change (import swap)
- Automatic memory management (no manual offloading logic)
- Well-tested library used in production
- Can be combined with all other techniques

**Cons:**
- External dependency (bitsandbytes)
- Adds PCIe transfer overhead
- Less predictable latency (page faults)
- May not work well with gradient accumulation (frequent paging)

**Estimated memory savings:** ~600 MB (similar to manual CPU offloading). **Total VRAM: ~7.1 GB at dim=1024.**

---

### 4.7 Reduced Batch Size with Compensated Learning Rate

**Description:** Reduce batch size from 64 to 32 or 16, and scale learning rate linearly to maintain gradient noise characteristics.

**How it reduces memory:** Activations scale linearly with batch size. Halving batch size halves activation memory.

**Implementation complexity:** Very Low. Change `--batch_size` argument and adjust lr.

**Pros:**
- Immediate effect, zero code changes
- Can be done per-stage (smaller batches for larger dims)
- Well-understood scaling rules (linear lr scaling)

**Cons:**
- May affect convergence quality (smaller batches = noisier gradients)
- Requires experimentation to find optimal lr scaling
- Does not address model params + optimizer states

**Estimated memory savings:** 50% reduction in activations (if batch halved). For dim=1024: from ~6.5 GB to ~3.3 GB. **Total VRAM: ~4.5 GB.**

---

### 4.8 torch.compile + SDPA (Scaled Dot-Product Attention)

**Description:** Use PyTorch 2.x's `torch.compile` to fuse operations and optimize the computation graph, combined with `torch.nn.functional.scaled_dot_product_attention` (SDPA) which automatically selects the best attention kernel (FlashAttention, Memory-Efficient Attention, or math backend) based on hardware and input characteristics.

**Current status:** Neither `torch.compile` nor SDPA is used in the codebase. Attention is computed manually via standard matrix operations (`F.softmax(scores @ values)`), and the model runs in eager mode (FP32).

**How it reduces memory:**
- SDPA uses fused kernels that avoid materializing the full attention matrix (N×N) in memory. FlashAttention tiles the computation, keeping only O(√N) activations in SRAM.
- `torch.compile` fuses elementwise operations (add, relu, dropout) into single kernels, eliminating intermediate tensor allocations.
- Combined effect: 30-50% reduction in activation memory for attention layers, plus ~20-40% speedup from kernel fusion.

**Implementation complexity:** Medium. Requires:
1. Replacing manual attention with `F.scaled_dot_product_attention` in `modeling_bitnet.py`
2. Wrapping model with `torch.compile(model)` at training script entry
3. **CRITICAL: Selective compilation to avoid breaking BitNet STE (see §4.8.1)**
4. Verifying numerical equivalence (torch.compile can introduce floating-point differences)

**Pros:**
- Significant memory reduction with minimal code changes (~30 lines)
- Speed improvement from kernel fusion (20-40% faster training)
- SDPA automatically selects optimal kernel (FlashAttention on RTX 5070)
- No impact on model quality or convergence
- PyTorch native — no external dependencies
- Can be combined with all other techniques
- Particularly effective for transformer-style attention patterns

**Cons:**
- `torch.compile` has cold-start compilation overhead (~30-60 seconds on first epoch)
- Some custom operations may not be compilable (need tracing/fallback)
- **CRITICAL: BitNet's Straight-Through Estimator (STE) may break under torch.compile (see §4.8.1)**
- SDPA requires specific tensor shapes for optimal kernel selection
- Debugging compiled models is harder (opaque computation graph)
- May require PyTorch 2.0+ (verify version compatibility)

**Estimated memory savings:** 30-50% reduction in attention activation memory. For dim=1024 with attention layers: from ~6.5 GB to ~4.5-5.0 GB (standalone). **Combined with BF16: ~2.5-3.0 GB.**

#### 4.8.1 ⚠️ CRITICAL: torch.compile + BitNet STE Incompatibility

BitNet uses **ternary weights {-1, 0, 1}** with a **Straight-Through Estimator (STE)** for gradient propagation. The STE relies on `detach()` operations to pass gradients through the quantization boundary. `torch.compile` can break this:

**The problem:**
- `torch.compile` performs graph-level optimizations: operation fusion, reordering, and elimination.
- The STE uses `detach()` which creates gradient stops. `torch.compile` may fuse operations across these boundaries, producing **null gradients** or **incorrect gradient flow**.
- Result: The model may **stop learning silently** (loss plateaus) rather than crashing.

**Required approach:**
```python
# 1. Replace manual attention with SDPA (safe — no STE in attention)
# In modeling_bitnet.py, replace:
#   scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)
#   attn = F.softmax(scores, dim=-1)
#   out = attn @ v
# With:
from torch.nn.functional import scaled_dot_product_attention
out = scaled_dot_product_attention(q, k, v, is_causal=True)

# 2. Compile ONLY safe submodules (attention + MLP), NOT BitLinear
# Create a wrapper that excludes BitLinear from compilation
class SafeCompileWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        # Compile only attention and MLP submodules
        for module in self.model.modules():
            if not isinstance(module, BitLinear):
                torch.compile(module, fullgraph=False)

# 3. OR compile the full model with fullgraph=False (fallback mode)
# This allows torch.compile to fall back to eager for operations it can't trace
model_compiled = torch.compile(model, fullgraph=False, dynamic=True)
```

**Key rules:**
- **NEVER** use `fullgraph=True` with BitNet (will break STE)
- **Compile selectively**: SDPA attention + MLP layers are safe. BitLinear layers must stay in eager mode.
- **Benchmark first**: Run 10 epochs with `torch.compile(fullgraph=False)` and verify gradient norms in BitLinear layers are non-zero.
- **Fallback**: If compilation causes gradient issues, compile only the SDPA call site, not the full model.

**Validation step:** After applying torch.compile, monitor `model.core_layers[0][0].weight.grad.norm()` every 10 epochs. If it drops to zero, compilation is breaking STE.

---

### 4.9 Gradient Checkpointing (Activation Checkpointing)

**Description:** Selectively discard intermediate activations during the forward pass and recompute them during the backward pass. PyTorch provides `torch.utils.checkpoint` for this purpose.

**How it reduces memory:** Instead of storing all layer activations (O(n) for n layers), only checkpoint boundaries are stored. Backward pass recomputes intermediate activations on-the-fly.

**Implementation complexity:** Low-Medium. Requires wrapping model layers with `checkpoint()`.

**Pros:**
- Reduces activation memory by ~50-70%
- No change to model architecture or training dynamics
- Native PyTorch support
- Can be applied selectively (e.g., only to middle layers)

**Cons:**
- ~25-35% increase in compute time (recomputation overhead)
- Requires careful placement of checkpoint boundaries
- Does not address model params + optimizer states memory
- Slightly more complex debugging when OOM occurs
- Redundant if SDPA + BF16 already provide sufficient savings

**Estimated memory savings:** 50-70% reduction in activation memory. For dim=1024: from ~6.5 GB to ~2.0-3.3 GB activations. **Total VRAM: ~3.5-4.8 GB.**

---

## 5. Combined Strategy Analysis

### 5.1 Corrected Memory Budget (per DeepSeek review)

The original estimates underestimated peak memory during gradient accumulation. During accumulation, PyTorch **does not release** gradients from prior micro-batches until `optimizer.step()` is called. Corrected formula:

```
Peak VRAM = (activ_per_micro × accum_steps) + params + optimizer_states + grads
```

**Corrected estimates for dim=1024:**

| Scenario | Activations | Params | Optimizer | Grads | **Total** |
|----------|-------------|--------|-----------|-------|-----------|
| No optimization (batch=64) | 6.5 GB | 0.3 GB | 0.6 GB (FP32) | 0.3 GB | **7.7 GB** |
| BF16 only (batch=64) | 3.25 GB | 0.15 GB | 0.3 GB (BF16) | 0.15 GB | **3.85 GB** |
| BF16 + SDPA (batch=64) | 2.75 GB | 0.15 GB | 0.3 GB | 0.15 GB | **3.35 GB** |
| BF16 + SDPA + accum 4×16 | 3.76 GB* | 0.15 GB | 0.3 GB | 0.15 GB | **4.36 GB** |
| BF16 + SDPA + accum 2×32 | 1.88 GB* | 0.15 GB | 0.3 GB | 0.15 GB | **2.48 GB** |

*\*Accumulation peaks = activ_per_micro × accum_steps (gradients held in memory)*

**Key insight:** BF16 + SDPA alone bring us to ~3.35 GB — well within 8 GB. Gradient accumulation may not be needed at all for dim=1024. It becomes a safety margin, not a requirement.

### 5.2 Dynamic Accumulation Strategy

Rather than fixed 4×16 accumulation, use dim-adaptive settings:

| Hidden Dim | Batch Size | Accum Steps | Effective Batch | Est. Peak VRAM |
|------------|------------|--------------|-----------------|----------------|
| ≤ 512 | 64 | 1 | 64 | ~2.5 GB |
| 640-768 | 64 | 1 | 64 | ~3.0 GB |
| 896 | 64 | 1 | 64 | ~3.5 GB |
| 1024 | 32 | 2 | 64 | ~2.5 GB |
| 1024 (conservative) | 16 | 4 | 64 | ~2.8 GB |

This maintains effective batch size of 64 across all stages while keeping VRAM under control. The learning rate should be scaled linearly when batch size changes: `lr_eff = lr_base × (batch_size / 64)`.

### 5.3 Combined Strategies

#### Strategy A: Conservative (Minimal Code Change)
- Dynamic gradient accumulation
- Reduced batch size at dim ≥ 896
- **Total estimated VRAM at dim=1024:** ~2.5-3.0 GB
- **Speed penalty:** ~1.5-2× slower
- **Code changes:** ~30 lines

#### Strategy B: Balanced (Recommended — v3)
- **Phase 1:** BF16 mixed precision (first — highest reward/risk ratio)
- **Phase 2:** SDPA attention + selective `torch.compile` (free speed + memory)
- **Phase 3:** Dynamic gradient accumulation (only if needed as safety margin)
- **Phase 4 (fallback):** CPU offloading of optimizer states
- **Total estimated VRAM at dim=1024:** ~2.5-3.5 GB
- **Speed penalty:** ~1.0-1.5× slower (BF16 + compile may offset accumulation cost)
- **Code changes:** ~150 lines

#### Strategy C: Maximum (Future-Proof)
- BF16 mixed precision
- SDPA + selective torch.compile
- Dynamic gradient accumulation
- Gradient checkpointing
- CPU offloading of optimizer states
- Pre-tokenized streaming dataset (separate RFC)
- **Total estimated VRAM at dim=1024:** ~2.0-3.0 GB
- **Speed penalty:** ~2-3× slower
- **Code changes:** ~300 lines + new dataset module

---

## 6. Recommendation

### Primary Recommendation: Strategy B (Balanced — v3)

**Final rationale (incorporating Grok + DeepSeek reviews):**

1. **BF16 mixed precision** is **first** — highest reward, lowest risk. Halves activation/gradient memory, ~2x speedup on RTX 5070 tensor cores. Requires careful FP32→BF16 checkpoint conversion (§4.4.1) and selective norm layer handling.

2. **SDPA + selective torch.compile** is **second** — free speed and memory win. SDPA replaces manual attention with fused FlashAttention kernels. `torch.compile` must be applied **selectively** to avoid breaking BitNet's STE (§4.8.1): compile attention + MLP, keep BitLinear in eager mode.

3. **Dynamic gradient accumulation** is **third** — dim-adaptive, only activates at larger dims. Maintains effective batch size of 64 while reducing peak memory. May not be needed if BF16 + SDPA provide sufficient headroom.

4. **CPU offloading of optimizer states** is the **last resort fallback** — only if前三techniques are insufficient.

**Why this order?**
- BF16 + SDPA alone bring VRAM to ~3.35 GB at dim=1024 — well within budget
- `torch.compile` adds speed but requires caution with STE; test in isolation first
- Dynamic accumulation is a safety valve, not a requirement
- CPU offloading adds complexity and PCIe overhead — avoid unless necessary

**Why not Strategy C?**
- Gradient checkpointing is redundant when SDPA + BF16 compress activations by 60-75%
- Pre-tokenized streaming dataset is valuable but orthogonal — separate RFC
- The additional complexity doesn't justify marginal savings at this scale

**Why not Strategy A?**
- Doesn't leverage BF16 hardware capability (wastes tensor cores)
- Pure accumulation approach is slower than BF16 + compile
- Leaves less headroom for unexpected spikes

### Neurogenesis + BF16 Handling

**CRITICAL:** The `trigger_neurogenesis` function creates a new `BitNet4LayerModel` in FP32 by default. When running in BF16 mode, neurogenesis must be handled correctly.

**⚠️ This conversion must be repeated at EVERY neurogenesis event (640→768, 768→896, 896→1024).** It is not sufficient to convert once at initial checkpoint load. Each new model created by `trigger_neurogenesis` must be converted to BF16 BEFORE `net2wider` copies weights, or precision will be lost during expansion.

```python
# In trigger_neurogenesis, after creating new model:
new_model = BitNet4LayerModel(
    use_glyphs=True,
    glyph_table=glyphs,
    hidden_dim=new_dim,
    num_layers=len(model.core_layers),
    use_pos_embedding=True,
    is_causal=True,
    max_seq_len=128,
).cpu()

# Convert to BF16 BEFORE loading old weights
new_model = new_model.to(torch.bfloat16)
for module in new_model.modules():
    if isinstance(module, (torch.nn.LayerNorm, torch.nn.RMSNorm)):
        module.float()

# Now net2wider will copy weights into BF16 model (no truncation)
model = net2wider_model(model, new_hidden_dim=new_dim, ...)
```

**Alternative (safer):** Run neurogenesis in FP32, then convert to BF16 after 10-20 stabilization epochs. This avoids precision issues during the expansion itself but requires a brief FP32 phase.

### Optimizer State Expansion During Neurogenesis

**Status: ALREADY IMPLEMENTED** in `net2net.py:257-279` (`transfer_state` function).

The `net2wider_model` function already handles optimizer state migration correctly:
- Maps AdamW states (exp_avg, exp_avg_sq) from old parameters to new dimensions using the same `g` (mapping) function used for weights
- Scales states by `copy_count` for cloned neurons (divides by number of copies)
- Preserves the `step` counter
- Sets states on the new optimizer for all expanded parameters

This means the concern about losing Adam momentum during neurogenesis is **already mitigated** in the existing codebase. No additional work needed for this specific issue.

**However:** When converting to BF16, the `transfer_state` function must be aware that optimizer states should also be in BF16. Currently it copies states as-is from the old (FP32) optimizer. After BF16 conversion, the states must be cast:
```python
# After net2wider_model completes, cast optimizer states to BF16
for state in new_optimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.to(torch.bfloat16)
```

### Samantha Evaluation + VRAM

The evaluation with "Professor Samantha" (Mistral 7B) is **already handled** in the existing code. The `run_samantha_eval` function (line 123-249 of `train_sovereign_school.py`):
1. Saves checkpoint to disk
2. Offloads model to CPU: `model = model.cpu()`
3. Clears CUDA cache: `torch.cuda.empty_cache()`
4. Runs evaluation with `--device cpu`

**No training VRAM is used during evaluation.** The evaluation runs entirely on CPU. This is already correct and needs no changes.

### Implementation Order (final)

| Phase | Technique | Validation Gate | Rollback Plan |
|-------|-----------|-----------------|---------------|
| 0 | Baseline | Profile current VRAM with `torch.cuda.memory_stats()` | — |
| 1 | BF16 mixed precision | Run 100 epochs; loss curve matches FP32 baseline ±5% | Revert to FP32 checkpoint |
| 2 | SDPA attention | Run 10 epochs; verify attention output matches manual impl | Remove SDPA, keep manual attention |
| 3 | Selective torch.compile | Run 10 epochs; gradient norms non-zero in BitLinear | `torch._dynamo.reset()`, revert to eager |
| 4 | Dynamic accumulation | Activate only at dim ≥ 896; verify loss stability | Reduce accum_steps to 1 |
| 5 | CPU offloading (fallback) | Only if OOM at dim=1024 with above | Remove offloading logic |

**Critical validation after each phase:**
- Monitor `model.core_layers[0][0].weight.grad.norm()` (STE health)
- Compare loss trajectory with previous phase baseline
- Check VRAM usage with `torch.cuda.max_memory_allocated()`

### Risk Assessment (updated v4)

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| BF16 checkpoint conversion instability | Medium | High | Explicit conversion (§4.4.1); validate 100 epochs before full run |
| BF16 gradient underflow in deep layers | Medium | Medium | Use GradScaler (§4.4.2); monitor scale factor |
| torch.compile breaks BitNet STE | High | Critical | `fullgraph=False`; selective compilation; monitor gradient norms |
| SDPA tensor shape incompatibility | Low | Medium | Verify q/k/v shapes match SDPA expectations; fallback to manual |
| Neurogenesis precision loss in BF16 | Medium | Medium | Create new model in BF16 directly OR stabilize in FP32 first |
| Optimizer state loss during neurogenesis | **None** | — | Already implemented in `net2net.py:transfer_state` |
| Dynamic accumulation lr mismatch | Low | Low | Linear lr scaling rule; validate on stage transition |
| Checkpoint incompatibility | Low | High | Save FP32 master weights alongside BF16 checkpoints |
| Training speed too slow | Low | Medium | BF16 + compile should offset accumulation cost |

---

## 7. Success Criteria

- Full curriculum training (ages 2-8) completes without CUDA OOM
- Final model quality matches or exceeds FP32 baseline
- Checkpoint format remains compatible with existing inference pipeline
- Total training time does not exceed 3× the original estimate

---

## 8. Open Questions

1. **BF16 convergence benchmark:** Should we run a 100-epoch FP32 vs BF16 comparison before committing to the full training? (Recommended by Grok)
2. **torch.compile + STE:** Has anyone tested `torch.compile(fullgraph=False)` with BitNet-style STE models? Need empirical data on gradient health.
3. **SDPA kernel selection:** Does FlashAttention auto-select on RTX 5070 Blackwell, or do we need to explicitly request it?
4. **Neurogenesis dtype:** Is it safer to create the expanded model in BF16 directly, or stabilize in FP32 first? Need to test both approaches.
5. **Exact activation formula:** What is the precise activation memory for BitNet4LayerModel? Need profiling with `torch.cuda.memory_stats()`.
6. **RTX 5070 BF16 throughput:** Is it actually 2× FP32, or limited by other factors? Benchmark needed.
7. **Pre-tokenized dataset:** Should this be a separate RFC? (Recommended — it's orthogonal to VRAM scaling)

---

## 9. References

- PyTorch Gradient Checkpointing: https://pytorch.org/docs/stable/checkpoint.html
- DeepSpeed ZeRO: https://www.deepspeed.ai/docs/config-json/
- bitsandbytes Paged Optimizers: https://github.com/TimDettmers/bitsandbytes
- Mixed Precision Training: https://pytorch.org/docs/stable/amp.html
- torch.compile: https://pytorch.org/docs/stable/generated/torch.compile.html
- SDPA (Scaled Dot-Product Attention): https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
- FlashAttention-2: https://github.com/Dao-AILab/flash-attention
- Training code: `src/bitnet/training/train_sovereign_school.py`
- State file: `storage/checkpoints/sovereign_school/school_state.json`

---

## 10. Changelog

| Date | Author | Change |
|------|--------|--------|
| 2026-07-22 | Aleth | Initial draft |
| 2026-07-22 | Aleth | v2: Incorporated Grok review: added SDPA + torch.compile (§4.8), reordered recommendation priority (BF16 first), revised Strategy B, updated implementation order |
| 2026-07-22 | Aleth | v3: Incorporated DeepSeek + Grok reviews: added §4.4.1 (BF16 checkpoint conversion), §4.8.1 (STE + torch.compile incompatibility), §5.1 (corrected memory budget), §5.2 (dynamic accumulation), neurogenesis BF16 handling, Samantha evaluation clarification, revised risk matrix |
| 2026-07-22 | Aleth | v4: Final review — added §4.4.2 (GradScaler for BF16), documented existing optimizer state expansion in net2net.py (transfer_state), clarified BF16 conversion at every neurogenesis, updated risk matrix |

---

*End of RFC.*

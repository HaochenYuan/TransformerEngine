# THD + CUDA Graph: TransformerEngine Changes

## Overview

Fix TransformerEngine compatibility issues for THD (packed sequence) + CUDA Graph training,
including both capture-time GPU-CPU sync errors and MLA + CP > 1 backward NaN.

## Modified Files (4 files)

### 1. `transformer_engine/pytorch/attention/dot_product_attention/dot_product_attention.py`

**Fix**: `pad_between_seqs = True` for THD format

```python
# Before (TE 2.11):
if qkv_format == "thd":
    pad_between_seqs = (
        cu_seqlens_q_padded is not None
        and not torch.equal(cu_seqlens_q_padded[:-1], cu_seqlens_q[:-1])
    ) or (...)

# After:
if qkv_format == "thd":
    pad_between_seqs = True
```

**Why**: `torch.equal()` triggers GPU-CPU synchronization, forbidden during CUDA Graph capture.
Setting `pad_between_seqs = True` unconditionally is safe for THD (padding between sequences
is always needed when using padded cu_seqlens).

### 2. `transformer_engine/pytorch/attention/dot_product_attention/context_parallel.py`

**Fix**: THD backward zero-fill uses `.shape[0]` instead of `cu_seqlens[-1]` during graph capture

```python
# Before:
if ctx.qkv_format == "thd" and not ctx.use_fused_attention:
    dq[cu_seqlens_q_padded[-1]:].fill_(0)  # GPU-CPU sync!

# After:
if ctx.qkv_format == "thd" and not ctx.use_fused_attention:
    if torch.cuda.is_current_stream_capturing():
        _q_end, _kv_end = dq.shape[0], dk.shape[0]
    else:
        _q_end = cu_seqlens_q_padded[-1]
        _kv_end = cu_seqlens_kv_padded[-1]
    dq[_q_end:].fill_(0)
    dk[_kv_end:].fill_(0)
    dv[_kv_end:].fill_(0)
```

**Why**: Reading `cu_seqlens[-1]` (scalar from GPU tensor) triggers GPU-CPU sync during
backward graph capture stream. Use `.shape[0]` instead (CPU-side metadata, no sync).

**Additionally**: This file contains fixes for the MLA + CP > 1 backward NaN issue
(see below).

### 3. `transformer_engine/pytorch/attention/dot_product_attention/backends.py`

**Fix**: Part of the MLA + CP > 1 backward NaN fix.

Changes include corrections to gradient buffer initialization and CP attention backend
behavior that prevent uninitialized values at padded positions when using FusedAttention
with asymmetric head dimensions (head_dim_qk ≠ head_dim_v).

### 4. `transformer_engine/pytorch/cpp_extensions/fused_attn.py`

**Fix**: Part of the MLA + CP > 1 backward NaN fix.

Changes include corrections to the `return_max_logit` path (used only by CP) where
`softmax_lse` handling was changed in TE 2.11. The old version computes
`stats = Max + log(Sum_Exp)` which produces safe `-inf` at padded positions,
preventing NaN from entering the backward pass.

## MLA + CP > 1 Backward NaN Bug

### Trigger Condition

```
NaN = FusedAttention + CP > 1 + MLA (head_dim_qk ≠ head_dim_v)
```

All three conditions must be simultaneously present:
- FusedAttention (cuDNN): only available backend for MLA + CP
- CP > 1: activates CP ring attention with P2P communication
- MLA: asymmetric head dimensions (e.g., qk=192, v=128)

### Root Cause

MLA's `head_dim_qk (192) ≠ head_dim_v (128)` creates edge cases in:
1. `fused_attn.py`: `return_max_logit` path passes cuDNN's raw Stats tensor which may
   contain NaN at padded positions for MLA (old version computed safe values)
2. `context_parallel.py`: CP ring backward gradient accumulation propagates NaN from
   padded positions without zero-fill for FusedAttention path
3. `backends.py`: Gradient buffer initialization affects padded position values

### Why GQA is Unaffected

GQA has symmetric head dimensions (head_dim_qk == head_dim_v), so cuDNN's FusedAttention
backward handles all positions correctly. The asymmetry-related edge cases only trigger
with MLA.

### Verification

| Model | TE Patches | grad_norm |
|-------|-----------|-----------|
| Moonlight-16B (MLA+CP2) | Minimal (2 patches only) | **nan** |
| Moonlight-16B (MLA+CP2) | Full (4 files) | **29.323** ✅ |
| Qwen3-8B (GQA+CP2) | Minimal (2 patches only) | **96.906** ✅ |
| Qwen3-8B (GQA+CP2) | Full (4 files) | **96.906** ✅ |

Minimal patches are sufficient for GQA. Full 4-file fix is required for MLA + CP > 1.

# CUDA Graph Capture Audit: req_to_token Tensor Embedding

## Executive Summary

SGLang's piecewise CUDA graph capture (default `--enforce-piecewise-cuda-graph`) **directly embeds pointers to the `req_to_token` tensor** into captured graph kernels. If the tensor is reallocated to grow, all captured graphs using the old pointer will read stale memory or fault. This is a critical dependency that must be managed carefully during dynamic batch-size admission control.

## 1. CUDA Graph Capture Architecture

### Capture Flow
- **Location**: `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py::PiecewiseCudaGraphRunner`
- **Capture Points**: 
  - Startup: `__init__` → `capture()` (lines 320) over a range of `capture_num_tokens` (default ~8 sizes)
  - Conditional re-capture: `capture_hidden_mode` changes trigger `capture()` (cuda_graph_runner.py::recapture_if_needed)
- **Trigger**: Only after weight updates or hidden state mode changes; **NOT on batch-size changes**

### Input Tensors Baked Into Graph
During `capture_one_batch_size()` (line 483), the following graph inputs are created and frozen:

1. **input_ids** (computed from buffers, size: num_tokens)
2. **positions** (computed from buffers, size: num_tokens)
3. **out_cache_loc** (computed from buffers, size: num_tokens) — KV cache indices
4. **mamba_track_indices/mask/seqlens** (optional, size: batch_size)
5. **req_to_token_pool** object reference (line 537, stored in ForwardBatch)

The `req_to_token_pool` is **passed as-is** (not sliced) into the ForwardBatch during capture (line 537):
```python
req_to_token_pool=self.model_runner.req_to_token_pool,  # <-- Full pool object
```

## 2. req_to_token Tensor Embedding in Kernels

### Direct Embedding in Triton Kernels

The critical embedding happens in the **attention backend's forward metadata initialization**:

**File**: `python/sglang/srt/layers/attention/utils.py::create_flashinfer_kv_indices_triton`

This Triton kernel is called with:
```python
create_flashinfer_kv_indices_triton[(bs,)](
    self.req_to_token,  # <-- **POINTER BAKED INTO KERNEL**
    req_pool_indices,
    paged_kernel_lens,
    kv_indptr,
    kv_start_idx,
    kv_indices,
    self.req_to_token.shape[1],  # <-- **SIZE ALSO BAKED IN**
)
```

**Kernel Definition** (utils.py, lines ~1275):
```triton
@triton.jit
def create_flashinfer_kv_indices_triton(
    req_to_token_ptr,  # [max_batch, max_context_len]
    req_pool_indices_ptr,
    page_kernel_lens_ptr,
    kv_indptr,
    kv_start_idx,
    kv_indices_ptr,
    req_to_token_ptr_stride: tl.constexpr,
):
    # Loads from: req_to_token_ptr + req_pool_index * req_to_token_ptr_stride + offset
    data = tl.load(
        req_to_token_ptr
        + req_pool_index * req_to_token_ptr_stride
        + kv_start + offset,
        mask=mask,
    )
    tl.store(kv_indices_ptr + kv_indices_offset + offset, data, mask=mask)
```

### Embedding Sites in Attention Backends

All major attention backends store and use `self.req_to_token`:

- **flashinfer_backend.py**: `FlashInferIndicesUpdaterDecode` (line ~2600)
- **flashmla_backend.py**: Similar pattern
- **nsa_backend.py**: Similar pattern
- **trtllm_mha_backend.py**: Similar pattern
- **dual_chunk_flashattention_backend.py**: Similar pattern
- **aiter_backend.py**: Similar pattern

Each stores: `self.req_to_token = model_runner.req_to_token_pool.req_to_token`

### Capture-Specific Embedding

During CUDA graph capture (`piecewise_cuda_graph_runner.py`):
1. Graph is captured with `forward_batch.req_to_token_pool` bound
2. The **pool object** contains the tensor with a **specific `data_ptr`**
3. Triton kernels called during capture record that pointer
4. When graph replays, it reads from that exact pointer

**No re-binding or `from_blob` wrapping is applied to `req_to_token`.**

## 3. Re-capture Triggers (Current)

**Only these conditions trigger re-capture:**

1. **Hidden state capture mode change** (`cuda_graph_runner.py::recapture_if_needed`):
   - When `capture_hidden_mode` differs from current requirement
   - Triggered by speculative decoding or `enable_return_hidden_states`

2. **Weight update** (`model_runner.py`):
   - When `recapture_cuda_graph=True` is set in forward call

**NOT triggered by:**
- Batch size changes
- `req_to_token_pool` size/capacity changes
- Token count growth
- Session resets (unless explicitly flagged)

## 4. Tensor Resizing Risk

### Current State

The `req_to_token` tensor is allocated **once at startup**:
```python
# memory_pool.py::ReqToTokenPool.__init__
self.req_to_token = torch.zeros(
    (size, max_context_len), dtype=torch.int32, device=device
)
```

- Shape: `(num_requests, max_context_len)` — **STATIC**
- Resizing would require **allocating a new tensor with a different `data_ptr`**

### Concrete Failure Scenario

If dynamic admission control grows `req_to_token` in-place or allocates new:

1. **Old captured graph** still records: `req_to_token_ptr = 0x...ABC123`
2. **New tensor** allocated: `new_ptr = 0x...DEF456`
3. **Graph replay** reads from stale `0x...ABC123`:
   - Memory freed → **segfault**
   - Memory reused → **reads garbage token IDs** → wrong KV cache access
   - Correct memory (if not freed) → **reads old mappings** → OOB KV access

## 5. Patterns for Dynamic Growth (Arena Insights)

### from_blob Pattern (Not Used for req_to_token)

**File**: `python/sglang/srt/arena/from_blob_ext.py`

The KV cache and Mamba pools use a **VA-stable wrapping pattern**:

1. **MultiTensorArena** manages VA (virtual address) space
2. **from_blob** constructs tensors at stable VA
3. **Underlying physical pages swap** (via `cuMemMap`) without changing data_ptr
4. **Captured graphs remain valid** because pointer never changes

**Key insight**: `at::from_blob` with no-op deleter allows pointer stability across physical reallocations.

**NOT applied to `req_to_token`** — currently allocated via standard PyTorch allocator.

### Multi-Slot Pattern (Mamba, KV Cache)

Mamba and KV pools use **arena-backed allocation**:
- Pre-allocate max VA range at startup
- Grow within that range (physical pages swap)
- Data pointer never changes
- Captured graphs remain valid

## 6. Torch APIs for Post-Capture Re-binding

### Available Mechanisms (Not Currently Used for req_to_token)

1. **at::from_blob** (C++ extension in from_blob_ext.py):
   - Binds tensor to pre-allocated VA
   - Pointer stable across physical reallocations
   - Used for: KV cache, Mamba state

2. **cudaGraphsExecKernelNodeSetParams** (NVIDIA driver API):
   - Update kernel node parameters in captured graph after capture
   - PyTorch wrapper: `torch.cuda.graph.replay` with `get_exec_params` / `set_exec_params`
   - **NOT observed in sglang**

3. **torch.cuda.graph.replay with updated tensors**:
   - Triton kernels can read tensor pointers at replay time via **implicit binding**
   - But compiled CUDA kernels have pointers baked in → requires explicit re-binding API
   - **Not applicable for flashinfer/custom CUDA kernels**

## 7. Concrete Risk Assessment

### If `req_to_token` Grows Via Reallocation

**Scenario A: In-place grow (e.g., `torch.Tensor.resize_` or similar)**
- Unlikely to succeed with CUDA tensors; would require new allocation
- Treated as Scenario B below

**Scenario B: New allocation (realloc growth)**
- Old captured graphs: `data_ptr = 0x...ABC000`
- New tensor: `data_ptr = 0x...XYZ000` (different)
- **Result**: Graph replay segfaults or reads corrupted indices
- **Detection**: Crash on first decode after reallocation

**Scenario C: If pointer is preserved (unlikely without arena)**
- Requires CUDA VA management (not current architecture)
- Would need from_blob wrapping at pool initialization

## 8. Current Mitigation: None

**Documented re-capture triggers**: Hidden state mode, weight updates
**NOT documented**: req_to_token growth → automatic re-capture

**This is the gap:** If admission control dynamically grows batch slots, graphs are not re-captured.

## Recommendations

### Option A: Re-capture on req_to_token Growth (Conservative)

**Cost**: ~1-2 seconds per growth event (rare)
**Safety**: Guaranteed correctness
**Implementation**:
1. Track initial `req_to_token.data_ptr()` at capture time
2. Before graph replay, check if pointer changed
3. If changed, trigger `recapture_cuda_graph=True`

```python
# In model_runner.py::forward
if (self.graph_runner.last_captured_req_to_token_ptr 
    != self.req_to_token_pool.req_to_token.data_ptr()):
    recapture_cuda_graph = True
```

**Pros**: Minimal changes, no new mechanisms
**Cons**: Blocks decode on growth; observable latency spike

### Option B: VA-Stable Wrapping (Cleanest)

**Cost**: +5-10% memory overhead for VA pre-allocation
**Safety**: Guaranteed correctness, no re-capture needed
**Implementation**:
1. Wrap `req_to_token` in **MultiTensorArena** (existing in codebase)
2. Use `from_blob` to construct tensor at stable VA
3. Grow by remapping physical pages within VA range

```python
# memory_pool.py::ReqToTokenPool.__init__
self.arena = MultiTensorArena(device, total_size=size * max_context_len * 2)
self.req_to_token = self.arena.allocate([size, max_context_len])
# Later: arena.grow() remaps pages, pointer stable
```

**Pros**: Pointer never changes, graphs always valid
**Cons**: Requires arena integration, new dependency

### Option C: Hybrid (Recommended)

**For now**: Enforce re-capture on growth (Option A)
**For future**: Migrate `req_to_token` to arena (Option B)

1. **Short-term**: Add guard in `model_runner.forward()`:
   ```python
   if self.req_to_token_pool.req_to_token.data_ptr() != self._cached_ptr:
       recapture_cuda_graph = True
       self._cached_ptr = self.req_to_token_pool.req_to_token.data_ptr()
   ```

2. **Long-term**: Adopt arena pattern for req_to_token like Mamba/KV pools

## Summary Table

| Aspect | Finding |
|--------|---------|
| **req_to_token in captured graph?** | **YES** — embedded in Triton kernels |
| **Pointer baked into compiled kernels?** | **YES** — via `create_flashinfer_kv_indices_triton` |
| **Size baked in?** | **YES** — `req_to_token_ptr_stride` is compile-time const |
| **Re-capture on resize?** | **NO** — not triggered by pool growth |
| **VA-stable wrapping?** | **NO** — uses standard allocator |
| **Post-capture re-binding?** | **NO** — no `cudaGraphExecKernelNodeSetParams` usage |
| **Risk if reallocated?** | **CRITICAL** — segfault or OOB KV access |
| **Mitigation available?** | **YES** — re-capture on pointer change OR arena wrapping |


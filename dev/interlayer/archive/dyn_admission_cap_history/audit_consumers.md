# ReqToTokenPool Consumer Audit

## Overview

This document audits all internal consumers of `ReqToTokenPool` in the sglang codebase to identify stale-reference risks when making the pool dynamically growable at runtime.

**Pool class location:** `python/sglang/srt/mem_cache/memory_pool.py:136-200`

**Key pool fields:**
- `self.size` (int): capacity in slots
- `self.req_to_token` (torch.Tensor[size, max_context_len]): maps req pool index → token indices
- `self.free_slots` (Python list): pool of available request slot indices

---

## Per-Consumer Audit

### 1. `mem_cache/common.py` - Write & Allocation Functions

**File:** `python/sglang/srt/mem_cache/common.py`

| Function | Lines | What They Access | Stale-Risk | Notes |
|----------|-------|------------------|-----------|-------|
| `write_req_to_token_pool_triton` (kernel) | 29-77 | `req_to_token_ptr` (data_ptr passed to Triton), `.shape[1]` for stride | **HIGH** | Triton kernel receives raw data_ptr from `req_to_token`. On tensor reallocation, pointer stales immediately. Stride calculation reads `.shape[1]` once at launch. |
| `write_cache_indices` | 80-127 | Passes `req_to_token_pool.req_to_token` to kernel, reads `.shape[1]` for stride | **HIGH** | Calls `write_req_to_token_pool_triton` with tensor ref; kernel is issued before returns. Fallback path calls `pool.write()` method directly. |
| `alloc_req_slots` | 422-450 | Calls `pool.alloc(reqs)`, reads `pool.available_size()` | **LOW** | Only reads through method calls; no tensor refs cached. Reallocation within `alloc()` handled by pool. |
| `alloc_for_extend` | 453-516 | Calls `write_cache_indices()` and `alloc_req_slots()` | **MEDIUM** | Indirectly passes tensor ref to Triton kernel via `write_cache_indices`. |
| `alloc_for_decode` | 548-587 | Reads `pool.req_to_token[batch.req_pool_indices, ...]` (line 565), passes to Triton | **HIGH** | Directly indexes `req_to_token` tensor to read `last_loc`. On resize, tensor reallocation breaks index operation. |

**Stale-Risk Reason (HIGH cases):**
- Triton kernel receives `data_ptr()` and assumes it won't change across execution
- Tensor indexing operations (`pool.req_to_token[indices]`) assume shape stable within single batch
- Stride calculation (`tensor.shape[1]`) baked into kernel launch; ignored during resize

---

### 2. `disaggregation/decode_schedule_batch_mixin.py`

**File:** `python/sglang/srt/disaggregation/decode_schedule_batch_mixin.py`

| Line | What | Stale-Risk |
|------|------|-----------|
| 45 | `self.req_to_token_pool.req_to_token[req.req_pool_idx][...]` | **MEDIUM** |

**Context:** Decode-specific prefill accesses pool tensor directly to fetch token sequence. Reads live reference during batch execution.

**Note:** Disaggregation module is separate decode/prefill path; reallocation during active batch would break this.

---

### 3. `disaggregation/decode.py` - Decode Pool Variants

**File:** `python/sglang/srt/disaggregation/decode.py`

| Line(s) | Class/Function | What | Stale-Risk |
|---------|----------------|------|-----------|
| 96-169 | `DecodeReqToTokenPool` | Subclass of `ReqToTokenPool` with `pre_alloc_size`; mirrors base impl | **MEDIUM** | Over-allocates tensor at init for pre-allocated requests. Uses `list(range(size + pre_alloc_size))` for free_slots. If grown, slot range logic breaks. |
| 171-227 | `HybridMambaDecodeReqToTokenPool` | Adds mamba state tracking alongside req-to-token; calls `_init_mamba_pool()` | **MEDIUM** | Mamba pool must grow in lockstep with req pool to maintain index alignment in `req_index_to_mamba_index_mapping`. |
| 134 | `DecodeReqToTokenPool.write()` | `self.req_to_token[indices] = values` | **MEDIUM** | Direct tensor write; no bounds checks. Assumes indices valid for current `size`. |
| 751, 774, 788 | Prefill paths | `req_to_token_pool.req_to_token[...]` index ops | **MEDIUM** | Multiple reads during prefill scheduling; each assumes stable shape. |

---

### 4. `layers/attention/*` - Attention Backends

**Files:** Multiple backends (flashattention, nsa, xpu, trtllm, dual_chunk, aiter, etc.)

**Pattern:** All backends read `forward_batch.req_to_token_pool.req_to_token[req_pool_indices]` during `forward()` to build `metadata.page_table` or `page_indices`.

| Backend | Lines | Example | Stale-Risk |
|---------|-------|---------|-----------|
| flashattention_backend.py | ~15 locations | line 280: `forward_batch.req_to_token_pool.req_to_token[...]` | **MEDIUM-HIGH** |
| nsa_backend.py | ~4 locations | line 417: `forward_batch.req_to_token_pool.req_to_token[...]` | **MEDIUM-HIGH** |
| xpu_backend.py | ~10 locations | line 122: `forward_batch.req_to_token_pool.req_to_token[...]` | **MEDIUM-HIGH** |
| trtllm_mha_backend.py | ~8 locations | line 609: `forward_batch.req_to_token_pool.req_to_token[...]` | **MEDIUM-HIGH** |
| dual_chunk_flashattention_backend.py | line 186 | `forward_batch.req_to_token_pool.req_to_token[...]` | **MEDIUM-HIGH** |
| aiter_backend.py | ~2 locations | line 1327: `self.req_to_token[...]` | **MEDIUM-HIGH** |

**Stale-Risk Reason:** Each backend stores tensor refs in local `metadata` objects during `forward()`. If pool resizes between batch prep and execution, metadata becomes invalid. Most backends are hot-path (forward execution), not setup-time.

---

### 5. `mem_cache/radix_cache*.py` - Cache Implementations

**Files:** radix_cache.py, radix_cache_cpp.py, mamba_radix_cache.py, swa_radix_cache.py, unified_radix_cache.py

| File | Lines | What | Stale-Risk |
|------|-------|------|-----------|
| radix_cache.py | 496, 503, 541 | `req_to_token_pool.req_to_token[req_idx]` reads | **MEDIUM** |
| radix_cache_cpp.py | 176, 214, 241 | `req_to_token_pool.req_to_token[...]` | **MEDIUM** |
| mamba_radix_cache.py | 714, 722, 802, 819 | `req_to_token_pool.req_to_token[...]` | **MEDIUM** |
| swa_radix_cache.py | 435, 442, 480, 489 | `req_to_token_pool.req_to_token[...]` | **MEDIUM** |
| unified_radix_cache.py | 307, 316, 378, 384 | `req_to_token_pool.req_to_token[...]` | **MEDIUM** |

**Context:** Cache operations (evict, extend) read token indices to decide which KV tokens to free/keep. Reads live reference during cache maintenance. Not hot-path critical but must be atomic w.r.t. allocation.

---

### 6. `managers/schedule_batch.py` - Scheduler Core

**File:** `python/sglang/srt/managers/schedule_batch.py`

| Line(s) | What | Stale-Risk |
|---------|------|-----------|
| 1264, 1273 | `req_to_token_pool.req_to_token[...]` | **MEDIUM** |
| 2694 | `self.req_to_token_pool.req_to_token[...]` | **MEDIUM** |

**Context:** Scheduler reads pool indices during admission & dispatch. Runs before batch execution (not hot-path), but must be atomic w.r.t. allocation.

---

### 7. `model_executor/cuda_graph_runner.py`

**File:** `python/sglang/srt/model_executor/cuda_graph_runner.py`

| Line | What | Stale-Risk |
|------|------|-----------|
| 498 | `model_runner.req_to_token_pool.size` | **LOW** |

**Context:** Reads `.size` at cuda graph capture time (setup-phase, not hot-path). Used to compute `num_max_requests` for batch size capture decisions. Resize after capture could invalidate graph if old size was baked in.

---

### 8. `model_executor/piecewise_cuda_graph_runner.py`

**File:** `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`

| Line | What | Stale-Risk |
|------|------|-----------|
| 205 | `model_runner.req_to_token_pool.size` | **LOW** |

**Context:** Same as cuda_graph_runner — reads size at init. Max batch size baked into graph.

---

### 9. `model_executor/breakable_cuda_graph_runner.py`

**File:** `python/sglang/srt/model_executor/breakable_cuda_graph_runner.py`

| Line | What | Stale-Risk |
|------|------|-----------|
| 104 | `model_runner.req_to_token_pool.size` | **LOW** |

**Context:** Reads size at initialization for graph capture.

---

### 10. `model_executor/cpu_graph_runner.py`

**File:** `python/sglang/srt/model_executor/cpu_graph_runner.py`

| Line | What | Stale-Risk |
|------|------|-----------|
| 105 | `model_runner.req_to_token_pool.size` | **LOW** |

**Context:** Filters capture batch sizes by pool size at init.

---

### 11. `managers/schedule_batch.py` - ForwardBatch & Batch Info

**File:** `python/sglang/srt/model_executor/forward_batch_info.py`

| Line | What | Stale-Risk |
|------|------|-----------|
| 373 | `req_to_token_pool: ReqToTokenPool = None` (field) | **N/A** |
| 478 | `req_to_token_pool=model_runner.req_to_token_pool` | **N/A** |

**Context:** ForwardBatch holds reference to pool object; no caching of tensor or size values. Safe as long as pool object persists.

---

### 12. `mem_cache/sparsity/algorithms/base_algorithm.py`

**File:** `python/sglang/srt/mem_cache/sparsity/algorithms/base_algorithm.py`

| Line(s) | What | Stale-Risk |
|---------|------|-----------|
| 287 | `self.req_to_token_pool.req_to_token` | **MEDIUM** |
| 288 | `req_to_token.shape[1]` (read once, stored in `max_req_tokens`) | **MEDIUM** |
| 310-312 | Tensor indexing with shape-dependent clamp | **MEDIUM** |

**Context:** Sparsity algorithm reads pool tensor to compute page scores. Caches shape[1] for one function call; assumes shape stable within call.

---

### 13. `managers/scheduler_runtime_checker_mixin.py`

**File:** `python/sglang/srt/managers/scheduler_runtime_checker_mixin.py`

| Line(s) | What | Stale-Risk |
|---------|------|-----------|
| 379 | `self.req_to_token_pool.mamba_pool.free_slots` (HybridReqToTokenPool) | **LOW** |
| 384 | `range(self.req_to_token_pool.mamba_pool.size)` | **LOW** |
| 468 | `len(self.req_to_token_pool.free_slots)` | **LOW** |
| 471 | `self.req_to_token_pool.available_size()` | **LOW** |

**Context:** Runtime checker (debug/validation only) reads pool state for leak detection. Reads happen post-batch; low performance sensitivity.

---

### 14. `disaggregation/decode_kvcache_offload_manager.py`

**File:** `python/sglang/srt/disaggregation/decode_kvcache_offload_manager.py`

| Lines | What | Stale-Risk |
|-------|------|-----------|
| 118, 220, 242, 251, 311 | `self.req_to_token_pool.req_to_token[...]` indexing | **MEDIUM** |

**Context:** Offload manager reads pool indices to decide which KV cache to offload/restore. Executes during batch processing (not hot-path but synchronous with batch).

---

### 15. `managers/hisparse_coordinator.py`

**File:** `python/sglang/srt/managers/hisparse_coordinator.py`

| Lines | What | Stale-Risk |
|-------|------|-----------|
| 154, 239, 629 | `self.req_to_token_pool.req_to_token[...]` | **MEDIUM** |

**Context:** HiSparse (hierarchical sparse attention) coordinator reads pool to compute sparse patterns. Batch-level operation, not hot-path.

---

### 16. `session/streaming_session.py`

**File:** `python/sglang/srt/session/streaming_session.py`

| Lines | What | Stale-Risk |
|-------|------|-----------|
| 254, 333, 405, 409, 499 | `self.req_to_token_pool.req_to_token[...]` and `.free_slots.append()` | **MEDIUM** |

**Context:** Streaming session (user-facing API) reads pool to fetch token indices and manage free slots. Admission-time operation (not hot-path).

---

### 17. `layers/attention/hybrid_linear_attn_backend.py`

**File:** `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`

| Line | What | Stale-Risk |
|------|------|-----------|
| 143 | `self.req_to_token_pool: HybridReqToTokenPool = model_runner.req_to_token_pool` | **MEDIUM** |

**Context:** Stores pool reference in backend; accesses during forward. Hot-path usage of `req_to_token` tensor.

---

### 18. `layers/attention/nsa_indexer.py`

**File:** `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`

| Line | What | Stale-Risk |
|------|------|-----------|
| 937 | `forward_batch.req_to_token_pool.req_to_token[...]` | **MEDIUM-HIGH** |

**Context:** NSA (nested sparse attention) indexer reads pool tensor during forward to compute index buffers.

---

### 19. `mem_cache/common.py` - Release Path

**File:** `python/sglang/srt/mem_cache/common.py`

| Lines | Function | What | Stale-Risk |
|-------|----------|------|-----------|
| 628-630 | `release_kv_cache()` | `tree_cache.req_to_token_pool.req_to_token[req.req_pool_idx][...]` | **MEDIUM** |
| 640 | `release_kv_cache()` | `tree_cache.req_to_token_pool.free(req)` | **LOW** |

**Context:** Release path (admission-time, not hot-path) reads token indices to free overallocated KV. Then calls `pool.free()` to return slot.

---

## Summary Table

| Category | Consumer Count | Stale Risk Level | Mitigation Needed |
|----------|----------------|------------------|-------------------|
| Triton kernels (data_ptr) | 2 | **HIGH** | Re-launch kernel after resize |
| Attention backends (hot-path reads) | 40+ | **MEDIUM-HIGH** | Batch must be atomic w.r.t. resize |
| Cache operations (batch-level) | 15+ | **MEDIUM** | Resize only between batches |
| Scheduler & admission (pre-batch) | 10+ | **LOW-MEDIUM** | Resize at safe points only |
| Cuda graph size (setup-time) | 4 | **LOW** | Invalidate graphs on resize |
| Runtime validation | 5 | **LOW** | Safe post-batch |

---

## Growth Strategy Assessment

### Option 1: In-Place Tensor Resize (`torch.nn.functional.pad` or `torch._resize_`)

**Pros:**
- No data copy; memory pointer stable if not reallocated
- `free_slots` list continues working

**Cons:**
- **Triton kernels:** `.data_ptr()` may still change if torch underlying buffer reallocates
- **Attention backends:** Already-launched kernels see stale stride/shape unless re-launched
- **Risk:** Very high if resize happens mid-batch; requires immediate kernel re-launch

### Option 2: Allocate-New-and-Swap

**Pros:**
- Clean break; old tensor fully deallocated
- Clear point to invalidate all cached references
- Easier to reason about (no in-place mutation)

**Cons:**
- Must copy old data to new tensor
- All code paths must accept tensor changing
- **Attention backends:** Must re-fetch refs after swap

**Risk:** Medium if swap happens only between batches (safe window exists)

### Option 3: Append New Rows (Grow Logically, Not Resize)

**Pros:**
- Never reallocate existing rows (data_ptr stable)
- `free_slots` continues working (new indices > old size)
- Backward compatible with existing code

**Cons:**
- Tensor shape[0] changes; stride calculations affected
- Triton kernels bake in stride at launch; old launches invalid for new pool size
- Must handle "growing tail" rows (initially zero, filled lazily)

**Risk:** Medium; stride becomes runtime parameter instead of constant

---

## Recommended Approach

**Best fit for this codebase: Option 2 (Allocate-New-and-Swap) with safe resizing window**

### Rationale

1. **Triton kernels (HIGH risk):** Regardless of resize method, Triton kernels receive `data_ptr()` at launch. If resize happens during kernel execution, pointer stales. **Must avoid resizing while kernels are active** (hot-path).

2. **Attention backends (MEDIUM-HIGH risk):** All 40+ backend references read pool tensor live during `forward()`. Can only safely resize between batches (cold-window).

3. **Cuda graphs (LOW risk):** Invalidate on first resize; graphs must be re-captured with new pool size.

### Implementation Strategy

1. **Resize trigger:** Only at `scheduler._try_allocate_new_batch()` or between `forward()` / `decode()` (cold window).

2. **Resize flow:**
   - Checkpoint current free_slots state
   - Allocate new tensor with larger size
   - Copy old data (0:old_size) to new tensor
   - Zero-fill new rows (old_size:new_size)
   - Swap `self.req_to_token` reference
   - Extend `self.free_slots` with new slot indices
   - Invalidate cuda graphs (if any)

3. **Free list handling:**
   ```python
   def grow(self, new_size):
       assert new_size > self.size, "new_size must be larger"
       new_tensor = torch.zeros((new_size, self.max_context_len), ...)
       new_tensor[:self.size] = self.req_to_token  # copy old data
       self.req_to_token = new_tensor
       self.free_slots.extend(range(self.size, new_size))  # append new slots
       self.size = new_size
   ```

4. **Guard resize:**
   - Check no active Triton kernels (synchronize cuda stream)
   - Check no running forward batches
   - Check not in middle of scheduler iteration
   - Suggested: Resize at START of `_try_allocate_new_batch()` before any batch prep

5. **HybridReqToTokenPool:** Must also grow `req_index_to_mamba_index_mapping` tensor in lockstep:
   ```python
   def grow(self, new_size):
       super().grow(new_size)  # grow base req_to_token
       old_size = self.req_index_to_mamba_index_mapping.size(0)
       new_mamba_mapping = torch.zeros(
           (new_size, ...), dtype=..., device=...
       )
       new_mamba_mapping[:old_size] = self.req_index_to_mamba_index_mapping
       self.req_index_to_mamba_index_mapping = new_mamba_mapping
       # Similar for ping_pong_track_buffer_mapping if enabled
   ```

6. **DecodeReqToTokenPool / HybridMambaDecodeReqToTokenPool:** Pre-allocation complicates grow logic:
   ```python
   def grow(self, new_size):
       assert new_size > self.size  # logical size only
       # Tensor already allocated to size + pre_alloc_size
       # No reallocation needed; just update self.size + free_slots
       self.free_slots.extend(range(self.size, new_size))
       self.size = new_size
   ```

---

## Final Recommendations

1. **Adopt Option 2 (Allocate-New-Swap)** for base ReqToTokenPool; extend HybridReqToTokenPool with mamba pool growth.

2. **Resize in safe window only:** Between batches, after cuda stream synchronize, before any attention backend reads pool refs.

3. **Invalidate CUDA graphs on resize:** Force re-capture with new pool size.

4. **Add sync barriers:** Use `torch.cuda.synchronize()` before resize to ensure no active kernels.

5. **DecodeReqToTokenPool:** Leverage pre-allocated tensor (no realloc); grow via free_slots extension only.

6. **Test scenario:** Simulate admission cap breach mid-run; verify resize occurs cleanly, no stale refs observed.

---

## Files Requiring Code Review Post-Resize Implementation

1. `write_cache_indices` — may need to re-dispatch Triton kernel after resize
2. `alloc_for_decode` — re-index pool tensor after resize
3. All attention backends (`flashattention_backend.py`, `nsa_backend.py`, etc.) — ensure pool refs fetched live, not cached
4. `cuda_graph_runner.py` — invalidate on resize
5. `HybridReqToTokenPool` / `DecodeReqToTokenPool` — implement grow logic atomically

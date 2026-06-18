# Init-Sized Arrays Audit: Dynamic Admission Cap Impact

## Overview

This audit identifies all arrays/buffers in sglang's request handling infrastructure that are **init-sized as a function of `max_running_requests` or `max_num_reqs`**. These arrays become bottlenecks when admission becomes dynamic with a changing mamba pool size cap.

---

## Summary Table: All Init-Sized Arrays Indexed by max_running_requests

| Location | Array Name | Formula | Element Type | Criticality | Overflow Behavior | Disagg-Specific |
|----------|-----------|---------|--------------|-------------|------------------|-----------------|
| memory_pool.py:154-156 | ReqToTokenPool.req_to_token | `(max_num_reqs, max_context_len)` | int32 | **CRITICAL** | Alloc failure → Cannot admit requests | No |
| memory_pool.py:230-300 | MambaPool.size (conv/temporal) | `(num_layers, size+1, ...)` | varies | **CRITICAL** | Alloc failure → Cannot decode | Conditional (Mamba only) |
| overlap_utils.py:66-68 | FutureMap.token_ids_buf | `(future_buffer_len,)` where `future_buffer_len = 5×max_running_requests + padding` | int64 | **HIGH** | Token id resolution fails → Silent corruption | Conditional (overlap schedule) |
| overlap_utils.py:92-119 | FutureMap.topk_p/index/verified_id/hidden_states_buf | `(future_buffer_len, ...)` | float32/int64/int32 | **HIGH** | Spec-decode future refs corrupt → Silent failure | Conditional (speculative decode + overlap) |
| scheduler.py:1152-1156 | MetadataBuffers (DECODE mode) | `(req_to_token_pool.size * 2, ...)` | varies | **HIGH** | Metadata xfer fails → RDMA corruption/deadlock | YES - DECODE disagg only |
| scheduler.py:1204-1209 | MetadataBuffers (PREFILL mode) | `(max_running_requests * 2, ...)` | varies | **HIGH** | Metadata xfer fails → RDMA corruption/deadlock | YES - PREFILL disagg only |
| hisparse_coordinator.py:74-130 | req_to_device_buffer, req_device_buffer_tokens, req_to_host_pool, lru_slots, top_k_device_locs_buffer | `(max_num_reqs, ...)` | int64/int32 | **MEDIUM** | Sparse attention token swap fails → Lost tokens | Conditional (HiSparse NSA) |
| n_gram_embedding.py:82-89 | exclusive_req_len_sums | `(max_running_requests + 1,)` | int32 | **MEDIUM** | N-gram token id computation fails | Conditional (n-gram models) |
| routed_experts_capturer.py:58-70 | RoutedExpertsDeviceCache.buffer | `(max(chunked_prefill_size*dp_size, max_running_requests), num_layers, num_experts)` | int32 | **MEDIUM** | Expert routing index loss → Incorrect output | Conditional (MoE models) |
| disaggregation/utils.py:174-203 | MetadataBuffers outputs (10 tensors) | `(size, ...)` where size from `max_running_requests * 2` (PREFILL) or pool.size (DECODE) | int32/float32/int64 | **HIGH** | RDMA metadata transfer failure | YES - disaggregation only |
| model_runner_kv_cache_mixin.py:275 | ReqToTokenPool (non-Mamba) | `(max_num_reqs, max_context_len)` | int32 | **CRITICAL** | Cannot allocate request slots | No |
| model_runner_kv_cache_mixin.py:221-241 | HybridMambaDecodeReqToTokenPool | `size=max_num_reqs, pre_alloc_size up to 2×max_num_reqs` | varies | **CRITICAL** | Decode request admission fails | Conditional (DECODE disagg + Mamba) |
| model_runner_kv_cache_mixin.py:252-272 | HybridReqToTokenPool | `size=max_num_reqs, mamba_spec_state_size=max_num_reqs` | varies | **CRITICAL** | Mamba state tracking fails | Conditional (Mamba models) |

---

## Detailed Array Analysis

### **TIER 1: CRITICAL — Immediate Bottleneck on Admission Increase**

#### 1. ReqToTokenPool (`memory_pool.py:136-198`)
- **Initialized at:** `memory_pool.py:154-156`
- **Size:** `(max_num_reqs, max_context_len)` — int32
- **Role:** Maps request→token location indices; gate for all request admission
- **Overflow behavior:** Returns `None` from `alloc()` (line 181) → admission rejection
- **Severity:** **CRITICAL** — Blocks all new request admission if slots exhausted
- **Dynamic impact:** Must grow when admission cap increases; current fixed size is hard ceiling

#### 2. MambaPool (`memory_pool.py:230-300`)
- **Initialized via `MambaPool.__init__()` if `mambaish_config` is set**
- **Sizes:**
  - conv_state: `(num_mamba_layers, size+1, ...)` int/float (line 263-269)
  - temporal_state: `(num_mamba_layers, size+1, ...)` float (line 287+)
- **Role:** Mamba model SSM state (temporal + conv window) for all concurrent requests
- **Overflow behavior:** Alloc failure on init; at runtime, state overflow → model crash
- **Severity:** **CRITICAL** (when Mamba model active)
- **Dynamic impact:** Fixed pool size (`size=max_num_reqs` at line 250) blocks decode throughput

---

### **TIER 2: HIGH — Corruption/Deadlock on Overflow**

#### 3. FutureMap (`overlap_utils.py:45-120`)
- **Initialized at:** `overlap_utils.py:66-68`
- **Sizes:**
  - `future_buffer_len = max_running_requests * (3 + max_num_chunks) + 2 * max_running_requests`
  - Buffers: `token_ids_buf` (int64), and lazy: `topk_p_buf`, `topk_index_buf`, `verified_id_buf`, `new_seq_lens_buf`, `hidden_states_buf`
- **Role:** Circular buffer for out-of-order decode + prefill chunk execution; stores speculative tokens
- **Overflow behavior:**
  - Non-spec mode: Circular index wraps, overwrites unresolved tokens → **silent corruption** (line 124)
  - Spec mode: Future indices point to garbage in lazy-allocated buffers
- **Severity:** **HIGH** — Silent data corruption if buffer overflows
- **Dynamic impact:** `future_buffer_len` fixed at init; increasing max_running_requests requires realloc
- **Enabled by:** `not server_args.disable_overlap_schedule` — affects most deployments

#### 4. MetadataBuffers (Disaggregation) (`disaggregation/utils.py:143-203`)
- **Initialized via scheduler.py:1152-1209**
- **PREFILL Mode (line 1204):** `buffer_size = max_running_requests * 2`
- **DECODE Mode (line 1151):** `buffer_size = req_to_token_pool.size * 2`
- **Tensors (10 total):**
  - output_ids, cached_tokens, logprobs_{val,idx}, top_logprobs_{val,idx}, topk_{p,index}, hidden_states, bootstrap_room
  - All shape `(size, N)` for varying N (16, 128, hidden_size)
- **Role:** Prefill→Decode metadata transport via RDMA during disaggregation
- **Overflow behavior:** Allocator fails to find slot (line 133-137 in ReqToMetadataIdxAllocator) → request stalls/hangs in queue
- **Severity:** **HIGH** — RDMA deadlock if metadata buffer exhausts
- **Dynamic impact:** `ReqToMetadataIdxAllocator` wraps deque of size `buffer_size`; fixed at init
- **Conditional:** Only active if `disaggregation_mode != NULL`

---

### **TIER 3: MEDIUM — Feature-Specific Bottlenecks**

#### 5. HiSparseCoordinator Arrays (`hisparse_coordinator.py:74-130`)
- **Initialized if:** NSA+HiSparse attention backend active
- **Arrays:**
  - req_to_device_buffer: `(max_num_reqs, padded_buffer_size)` int64 (line 74-76)
  - req_device_buffer_tokens/token_locs: `(layer_num, max_num_reqs, padded_buffer_size)` int32 (line 99-109)
  - lru_slots: `(layer_num, max_num_reqs, device_buffer_size)` int16 (line 114-118)
  - top_k_device_locs_buffer: `(max_num_reqs, top_k)` int32 (line 121-123)
  - _skip_first_backup: Python list `[False] * max_num_reqs` (line 130)
- **Role:** Token position tracking for sparse attention token swaps (device↔host)
- **Overflow behavior:** Insufficient slots → token not swapped → attention computation incorrect
- **Severity:** **MEDIUM** — Feature-specific; only HiSparse+NSA models
- **Dynamic impact:** Per-request tracking; all sized to max_num_reqs

#### 6. RoutedExpertsCapturer (`routed_experts_capturer.py:50-203`)
- **Initialized at:** `routed_experts_capturer.py:58-70` (RoutedExpertsDeviceCache)
- **Device buffer size:** `(max(chunked_prefill_size * dp_size, max_running_requests), num_layers, num_experts_per_tok + num_fused_shared_experts)` int32
- **Role:** MoE expert routing index capture during forward pass
- **Overflow behavior:** Buffer filled beyond batch size → indices overwritten → expert routing incorrect
- **Severity:** **MEDIUM** — MoE-specific; silent corruption if overflow
- **Dynamic impact:** Size keyed to max(chunked_prefill_size*dp_size, max_running_requests)

#### 7. NgramEmbedding (`n_gram_embedding.py:79-90`)
- **Initialized at:** `n_gram_embedding.py:88-89`
- **Buffers:**
  - oe_n_gram_ids: `(max_tokens, n_grams)` where max_tokens=max(chunked_prefill_size, max_running_requests)
  - exclusive_req_len_sums: `(max_running_requests + 1,)` int32
- **Role:** N-gram token id computation during embedding
- **Overflow behavior:** Insufficient req_len_sums slots → misaligned cumulative sums → incorrect embeddings
- **Severity:** **MEDIUM** — N-gram model specific
- **Dynamic impact:** Sized to max_running_requests+1; must reallocate if cap increases

---

## Overflow Scenarios: What Happens When Size Limit Exceeded?

| Array | Behavior | Impact | Recoverability |
|-------|----------|--------|-----------------|
| ReqToTokenPool.req_to_token | Alloc fails (returns None) | Request rejected at admission | Recoverable (queue) |
| MambaPool.{conv,temporal} | Alloc fails at init | Cannot start model | Not recoverable (crash) |
| FutureMap.token_ids_buf | Circular buffer wraps | Overwrites pending future tokens | NOT recoverable (silent corruption) |
| MetadataBuffers (disagg) | Allocator exhausted | Request metadata not transferred | Hangs in queue until timeout |
| HiSparse arrays | Insufficient slots | Token swap fails | Models produces wrong attention output |
| RoutedExperts.device_cache | Buffer overflow | Expert routing indices lost | NOT recoverable (wrong output) |
| NgramEmbedding | Buffer overflow | Embedding computation misaligned | NOT recoverable (wrong output) |

---

## Multipliers: `max_running_requests * N`

Directly proportional allocations (will grow linearly with admission cap):

- `FutureMap.future_buffer_len`: `~5 × max_running_requests` (line 66)
- `FutureMap.future_buffer_len` (with spec): `~7 × max_running_requests` (lines 66-68)
- `MetadataBuffers` (PREFILL disagg): `2 × max_running_requests` (line 1204)
- `ReqToTokenPool`: `1 × max_running_requests` (line 154)
- `NgramEmbedding.exclusive_req_len_sums`: `1.1 × max_running_requests` (line 89)
- `HiSparse.lru_slots`: `1 × max_num_reqs` per layer (line 116)

---

## Disaggregation-Specific Arrays (Can Ignore in Non-Disagg Setup)

| Array | Mode | Size | Can Skip? |
|-------|------|------|-----------|
| MetadataBuffers | DECODE | `pool.size * 2` | YES if disagg_mode ≠ DECODE |
| MetadataBuffers | PREFILL | `max_running_requests * 2` | YES if disagg_mode ≠ PREFILL |
| ReqToMetadataIdxAllocator | DECODE/PREFILL | `buffer_size` | YES if disagg_mode = NULL |
| disagg_prefill_bootstrap_queue | PREFILL | (refs metadata_buffers) | YES if disagg_mode ≠ PREFILL |
| disagg_decode_transfer_queue | DECODE | (refs metadata_buffers) | YES if disagg_mode ≠ DECODE |

---

## Arrays That MUST Resize on Admission Cap Growth

These arrays will become hard bottlenecks when `max_running_requests` increases dynamically:

1. **ReqToTokenPool.req_to_token** — Fundamental request slot tracking
   - Resize required when: max_num_reqs increases
   - Approach: Dynamic pool with realloc or circular buffer with overflow handling
   
2. **MambaPool.{conv_state, temporal_state}** (if Mamba active)
   - Resize required when: max_num_reqs increases
   - Approach: Realloc Mamba cache pool on cap change
   
3. **FutureMap.{token_ids_buf, spec_buffers}** (if overlap schedule enabled)
   - Resize required when: max_running_requests increases
   - Current calc: `future_buffer_len = max_running_requests * (3 + max_num_chunks) + 2 * max_running_requests`
   - Approach: Reallocate circular buffer or use dynamic allocation
   
4. **MetadataBuffers (disaggregation)** (if disagg mode active)
   - Resize required when: max_running_requests increases
   - Approach: Grow metadata buffer pool dynamically
   
5. **ReqToMetadataIdxAllocator** (if disagg mode active)
   - Resize required when: max_running_requests increases
   - Approach: Dynamic deque or slot management

---

## Arrays That Can Stay Init-Sized (Feature-Specific or Unused)

| Array | Reason | Notes |
|-------|--------|-------|
| HiSparseCoordinator buffers | HiSparse NSA only | Skip unless using NSA models with sparse attention |
| RoutedExpertsCapturer.device_cache | MoE only | Skip unless using MoE models |
| NgramEmbedding buffers | N-gram embedding only | Skip unless model uses n-gram embedding |
| Prefill delay buffers | Optional delayer | Can skip if prefill_delayer disabled |

---

## Recommended Resize Strategy

### Priority 1 (Must Handle):
1. **ReqToTokenPool** — implement growing pool (e.g., via circular buffer with realloc on wrap, or multi-pool)
2. **MambaPool** (if Mamba) — coordinate with KV cache resize; keep ratio stable
3. **FutureMap** (if overlap) — reallocate circular buffer on cap change; track pending futures carefully

### Priority 2 (If Disagg):
4. **MetadataBuffers + ReqToMetadataIdxAllocator** — grow metadata buffer pool in lock-step with req pool

### Priority 3 (If Special Models):
5. **HiSparse, RoutedExperts, NgramEmbedding** — resize only if respective models active

---

## Summary of Size Formulas

| Component | Size Formula | Grows With |
|-----------|--------------|-----------|
| ReqToTokenPool.req_to_token | `max_num_reqs × context_len` | **max_num_reqs** |
| MambaPool | `num_layers × (max_num_reqs + 1) × state_dim` | **max_num_reqs** |
| FutureMap.token_ids_buf | `5 × max_running_requests + 2 × max_running_requests` | **max_running_requests** |
| MetadataBuffers (PREFILL disagg) | `max_running_requests × 2 × (tens)` | **max_running_requests** |
| MetadataBuffers (DECODE disagg) | `pool.size × 2 × (tens)` | **pool.size (≈ max_num_reqs)** |
| NgramEmbedding.req_len_sums | `max_running_requests + 1` | **max_running_requests** |
| HiSparse.req_to_device_buffer | `max_num_reqs × device_buffer_size` | **max_num_reqs** |

**Key Finding:** All critical arrays scale **O(1)** with `max_running_requests` or `max_num_reqs`. Doubling admission cap → ~2× memory footprint for these structures.


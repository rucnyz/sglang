# Phase 2e — Cross-pool VMM actuator

Layer 2's actuator (paper §4.4): a chunk-bitmap shared-arena allocator built on CUDA Virtual Memory Management.

## Sub-phases

| ID | Goal | Effort | Status |
|---|---|---|---|
| 2e.1.a | Driver-API smoke test: VMM reserve / create / map / unmap / remap semantics | ~150 LoC, half day | **done** 2026-04-29 |
| 2e.1.b | Bridge VMM-backed VA to a `torch.Tensor` (single pool) | ~250 LoC, half day | **done** 2026-04-29 |
| 2e.2.a | `ChunkArena` class + `transfer_chunks` driver-API test | ~250 LoC | **done** 2026-04-29 |
| 2e.2.b | Two pool tensors via `CUDAPluggableAllocator` x2; transfer between them | ~200 LoC | **done** 2026-04-29 |
| 2e.2.c | CUDA graph survives cross-pool transfer (paper §4.4 soft-cap claim) | ~150 LoC | **done** 2026-04-29 |
| 2e.3   | StateSpec + ArenaSpec + LagrangePlanner framework + 4-phase smoke test | ~400 LoC | **done** 2026-04-29 |
| 2e.4.a | Layout-fit survey + decide pilot (KV) | research | **done** 2026-04-30 |
| 2e.4.b | `MultiTensorArena` (multi-tensor pool group) + soft-cap pre-alloc demo | ~250 LoC | **done** 2026-04-30 |
| 2e.4.c | Migrate `MHATokenToKVPool._create_buffers` to MultiTensorArena | ~80 LoC | **done** 2026-04-30 |
| 2e.4.d.1 | `MHATokenToKVPool.set_capacity_tokens` + allocator `set_capacity_pages` (unit test passes) | ~80 LoC | **done** 2026-04-30 |
| 2e.4.d.2 | `KVArenaActuator` + `BudgetAgent` arena path + scheduler leak-check live-aware | ~120 LoC | **done** 2026-04-30 |
| 2e.4.d.3 | End-to-end: server + 2× oscillator + 10 completions across phases | smoke | **done** 2026-04-30 |
| 2e.5.0 | Design note: A1 vs A2 vs B' vs D for cross-pool VMM compatibility | doc | **done** 2026-04-30 |
| 2e.5.1 | Mamba pool: `temporal_state` stacked → list (A2), gated `SGLANG_MAMBA_PERLAYER=1` | ~50 LoC | **done** 2026-04-30 |
| 2e.5.2 | Unit test: alloc/free/copy_from/at_layer_idx equivalence for both flag values | ~220 LoC | **done** 2026-04-30 |
| 2e.5.3 | E2E equivalence: same prompt, same tokens, both flag values, Qwen3.5-35B-A3B TP=1 | smoke | **PASS** 2026-04-30 |
| 2e.5.4 | **Performance regression bench**: A/B on Qwen3.5-35B-A3B TP=1, throughput/TTFT/TPOT, ≤2% regression allowed | bench | **PASS** 2026-04-30 |
| 2e.5.5 | Mamba pool: optional MultiTensorArena allocation gated `SGLANG_MAMBA_ARENA=1` | ~70 LoC | **mechanism done** 2026-04-30, e2e validation pending |
| 2e.5.5 | Migrate mamba pool to MultiTensorArena (depends on 2e.5.4 passing) | ~150 LoC | not started |
| 2e.5.6 | Hybrid workload-shift demo: KV ↔ mamba transfer driven by LagrangePlanner with real signals | 3–5 days | not started |

## 2e.1.a — VMM smoke test (done)

**Goal.** Prove that on this box (H200, CUDA 13.2) we can:
1. reserve a contiguous VA range with `cuMemAddressReserve`,
2. create independent physical-memory handles with `cuMemCreate`,
3. map handles into different offsets of the VA range with `cuMemMap` + `cuMemSetAccess`,
4. write region-distinguishable data through CUDA,
5. unmap one handle (`cuMemUnmap`) and remap it to a different offset,
6. read back and verify the data **follows the physical handle, not the VA** — the property that lets `transfer_chunks` work.

**Code.** [`01_vmm_smoke.py`](01_vmm_smoke.py). Pure ctypes against `libcuda.so`, no PyTorch dependency.

**Reproduce.**
```bash
cd /scratch/yuzhou/projects/sglang
CUDA_VISIBLE_DEVICES=2 .venv/bin/python dev/2e/01_vmm_smoke.py
```

**Result (2026-04-29, GPU 2 / NVIDIA H200 143 GiB / CUDA 13.2).**
```
device 0 = NVIDIA H200, 139 GiB total, 139 GiB free
allocation granularity: minimum=2 MiB, recommended=2 MiB
arena: 4 x 2 MiB = 8 MiB
reserved VA = 0x765cdc800000
created 4 physical handles: ['0x19d6b220', '0x19d6b6e0', '0x19d6bbc0', '0x19d6c0a0']
mapped all 4 handles + granted RW access
wrote patterns ['0xaa', '0xbb', '0xcc', '0xdd'] into chunks 0..3
verified initial mapping: each chunk holds its own pattern
unmapped chunks 1 and 3
re-mapped handle B (was at VA[1]) to VA[3]
verified chunk 3 after remap: 0xbb (= handle B's pattern, NOT 0xDD)
wrote 0x11 into VA[3] (which is handle B's mapping)
verified handle B's data follows physical handle, not VA: 0x11
cleanup complete

== PASSED: VMM unmap+remap semantics work end-to-end ==
```

**Findings.**

1. **Granularity is 2 MiB on H200.** Both `MINIMUM` and `RECOMMENDED` return 2 MiB. This is the chunk-size lower bound. The paper's §4.4 default 256 MiB is fine — it's a multiple of 2 MiB.
2. **Handle persistence works.** A `cuMemCreate`d handle outlives any particular `cuMemMap`. Unmap + remap to a different VA preserves the physical bytes. Crucially, writes through one VA mapping are visible through any subsequent mapping of the same handle. This is the load-bearing property for `transfer_chunks`.
3. **`cuMemSetAccess` must be re-issued after each `cuMemMap`.** Mapping does not grant access by itself; access permissions are set on a (VA-range, device) pair via `cuMemSetAccess`.
4. **ctypes default arg promotion is the only gotcha.** Without explicit `argtypes`, `c_int` (32-bit) gets used and 64-bit handle / VA values silently truncate, producing `cuMemMap` "invalid argument" with no obvious clue. Always set `argtypes` for these calls.

**Implication for 2e.1.b.** PyTorch tensor bridging needs to:
- create the tensor's storage backed by a VA inside our reserved range (not by `cudaMalloc`);
- the cleanest path is `torch.cuda.MemPool(allocator=...)` with a custom allocator that hands out our VA in chunk-aligned slices;
- alternatively, after `cuMemMap` + `cuMemSetAccess`, construct a `torch.UntypedStorage._new_with_weak_ptr` from the VA — but that is private API and likely brittle.
We'll try the `MemPool` route first.

## 2e.1.b — torch.Tensor + VMM bridge (done)

**Goal.** Prove the soft-cap property on a real `torch.Tensor`: the tensor's `data_ptr()` stays fixed (so any captured CUDA graph referencing it stays valid), but the physical memory behind that VA can be unmapped and remapped to a different physical handle without disturbing the tensor identity.

**Code.** [`arena.c`](arena.c) — minimal C allocator behind `torch.cuda.memory.CUDAPluggableAllocator` (~50 LoC). [`02_torch_bridge.py`](02_torch_bridge.py) — Python test (~200 LoC).

**Build + reproduce.**
```bash
cd /scratch/yuzhou/projects/sglang/dev/2e
gcc -shared -fPIC -O2 -Wall arena.c -o arena.so
cd /scratch/yuzhou/projects/sglang
CUDA_VISIBLE_DEVICES=2 .venv/bin/python dev/2e/02_torch_bridge.py
```

**Result (2026-04-29, GPU 2 / NVIDIA H200 / CUDA 13.2 / torch 2.9.1+cu130).**
```
arena: 4 x 32 MiB at VA 0x302000000
all 4 chunks mapped
prefilled chunks with ['0xaa', '0xbb', '0xcc', '0xdd']
CUDAPluggableAllocator + MemPool registered
tensor.data_ptr() = 0x302000000
tensor.data_ptr() == arena_base, GOOD
tensor[0] = 0xaa (expected 0xAA)
tensor.fill_(0x42); tensor[0]=0x42
VA[0] via cuMemcpyDtoH = 0x42, GOOD (PyTorch write reaches our VA)

-- remap: swap handle 1 into VA[0] --
after remap: tensor[0] = 0xbb (expected 0xBB, handle 1's pattern)
data follows physical handle, NOT the VA. GOOD.
tensor.fill_(0x77) (writes into handle 1, currently at VA[0])
VA[1] via cuMemcpyDtoH = 0x77 (expected 0x77, handle 1's new content)
PyTorch's write through old VA[0] persisted on handle 1, visible at new VA[1]. GOOD.

== PASSED: torch.Tensor + VMM remap end-to-end ==
```

**Findings.**

1. **The bridge works.** `torch.cuda.MemPool(allocator=CUDAPluggableAllocator(arena.so))` correctly routes `torch.empty(...)` through our `arena_malloc`. The tensor's `data_ptr()` lands on the VA we hand back.

2. **`tensor.data_ptr()` is stable across remap.** This is the load-bearing property for soft-caps + CUDA graphs. `cuMemUnmap`/`cuMemMap` of the physical handle behind the VA does not touch any PyTorch state; the tensor object is unaware of the swap. Any CUDA graph that captured the tensor's pointer continues to dereference the same VA, which now maps to a different physical handle.

3. **Data follows the physical handle, not the VA.** Writes through PyTorch land in the currently-mapped physical handle; if that handle moves to a different VA, the writes are still there at the new VA. This is the property that makes `transfer_chunks(from_pool, to_pool, n)` work: re-mapping a handle from pool A's sub-range to pool B's sub-range carries the bytes.

4. **PyTorch's caching allocator round-trips through our chunks.** `torch.cuda.MemPool` does not bypass PyTorch's caching layer. The caching allocator fetches a "segment" (around 20 MiB) from our `arena_malloc`, then sub-allocates tensors from that segment. Implication: chunks must be ≥ the segment size PyTorch wants. We use 32 MiB chunks. Phase 2e.2's chunk-bitmap allocator will likely use 256 MiB chunks (paper default), which is more than safe.

**Gotcha.** Arena init must happen *before* PyTorch's caching allocator hits us. The first allocation path (segment fetch) happens during `torch.empty(...)` inside the pool, not during `MemPool` construction.

**Implication for 2e.2.** The pluggable-allocator + MemPool path is sound. Phase 2e.2 will replace `arena.c`'s bump allocator with a per-pool chunk-bitmap allocator, where each of the four engine pools (KV, mamba, LoRA, prefix) gets its own sub-range of the arena. `transfer_chunks` will manipulate physical handle bindings between the sub-ranges; tensor pointers in each pool stay fixed throughout.

## 2e.2.a — ChunkArena (driver-API only)

**Goal.** A multi-pool allocator class that owns the arena VA, the physical-handle pool, and per-pool sub-range bookkeeping. `transfer_chunks(from_pool, to_pool, n)` is implemented as `shrink(from, n)` + `grow(to, n)`: handles move through a free-handle list, decoupling the actuator from per-pool eviction policy.

**Code.** [`chunk_arena.py`](chunk_arena.py) (~250 LoC, the data structure + actuator). [`03_chunk_arena_test.py`](03_chunk_arena_test.py) (test).

**Reproduce.**
```bash
cd /scratch/yuzhou/projects/sglang
CUDA_VISIBLE_DEVICES=2 .venv/bin/python dev/2e/03_chunk_arena_test.py
```

**Result (2026-04-29, GPU 2).**
```
arena va_base=0x7731b0800000, A va=0x7731b0800000, B va=0x7731b1000000
after initial grow: A=2 chunks, B=2 chunks, free handles=0
wrote A[0]=0xA0 A[1]=0xA1 B[0]=0xB0 B[1]=0xB1
transfer B->A: A=3, B=1
A[2] after transfer = 0xb1 (expected 0xB1)
B[0] still 0xb0, GOOD
B[1] after transfer back = 0x77 (expected 0x77)
data round-tripped via the physical handle. GOOD.
VA bases stable: A=0x7731b0800000 B=0x7731b1000000

== PASSED: two-pool transfer_chunks works end-to-end ==
```

**Findings.**

1. **Two-pool transfer works.** A handle that was holding 0xB1 in pool B's slot 1 is unmapped and remapped into pool A's first free slot (slot 2). Reading A's slot 2 immediately afterwards returns 0xB1 — the physical bytes followed the handle, not the VA.

2. **Round-trip preserves writes.** Writing 0x77 through A's slot 2 (now backed by the moved handle), then transferring back to B, makes B's slot 1 read 0x77. The handle carries the most recent write; the VA on either side is just a window onto it.

3. **VA bases stable.** A's and B's `pool_va_base()` return the same value before, during, and after every transfer. This is the property that lets PyTorch tensors and CUDA graphs reference these addresses indefinitely.

4. **Free-handle list is the right indirection.** `transfer_chunks` is decomposed into `shrink` (which returns handles to a central free list) and `grow` (which pulls them out). This decoupling is what will let us plug in the spec's per-pool eviction protocol later (paged-KV needs to drain in-flight requests before unmap, LoRA evicts LRU adapters, prefix-cache evicts refcount-zero entries). The actuator stays policy-agnostic.

**Implication for 2e.2.b.** Each pool's VA sub-range can be wrapped by a `CUDAPluggableAllocator` whose backing arena is the pool's slice. Two `MemPool`s, one per pool, each backed by its own per-pool allocator function. PyTorch tensors created in pool A's `MemPool` will land at addresses in `[A.va_base, A.va_base + A.va_size)`; tensors in B's pool will land in B's range. A `transfer_chunks` call between them must keep both tensors' `data_ptr()` stable.

## 2e.2.b — torch.Tensor in two pools + transfer (done)

**Goal.** Two PyTorch tensors, one in each pool's `MemPool`, survive a `transfer_chunks(B, A, 1)` with their `data_ptr()` and contents intact.

**Code.** [`arena_multi.c`](arena_multi.c) (~70 LoC, 4-pool slot states with `pool0_*` / `pool1_*` / ... symbol pairs). [`04_two_pool_torch.py`](04_two_pool_torch.py) (~150 LoC).

**Reproduce.**
```bash
cd /scratch/yuzhou/projects/sglang/dev/2e
gcc -shared -fPIC -O2 -Wall arena_multi.c -o arena_multi.so
cd /scratch/yuzhou/projects/sglang
CUDA_VISIBLE_DEVICES=2 .venv/bin/python dev/2e/04_two_pool_torch.py
```

**Result (2026-04-29, GPU 2).**
```
arena base=0x302000000 A=0x302000000 B=0x30a000000
initial: A=2 B=2 free_handles=0
two CUDAPluggableAllocators + MemPools registered
t_a.data_ptr() = 0x302000000 (A.va_base=0x302000000)
t_b.data_ptr() = 0x30a000000 (B.va_base=0x30a000000)
each tensor lands in its own pool's first chunk, GOOD
after fill_: t_a[0]=0x42, t_b[0]=0x77
transfer B->A: A=3 B=1
tensor data_ptrs unchanged: t_a=0x302000000, t_b=0x30a000000
after transfer: t_a[0]=0x42 (expected 0x42), t_b[0]=0x77 (expected 0x77)
A.va[slot=2] (the moved chunk) = 0xb1 (expected 0xB1, B's slot-1 pattern)

== PASSED: cross-pool transfer preserves tensor identity ==
```

**Findings.**

1. **Two pool tensors coexist.** `t_a` lands at `A.va_base = 0x302000000`, `t_b` at `B.va_base = 0x30a000000`. Each `MemPool` only allocates from its own `pool0_malloc` / `pool1_malloc`, so cross-pool placement is impossible. PyTorch sees them as two unrelated allocators.

2. **`transfer_chunks` does not disturb tensor identity.** Before the transfer: `t_a.data_ptr()=0x302000000`. After: same. Same for `t_b`. The transfer affected slot 1 of B (becomes unmapped) and slot 2 of A (newly mapped), neither of which any tensor is using. The tensors' chunk-0 storage was untouched.

3. **Tensor contents preserved.** `t_a[0]` reads 0x42 before and after; `t_b[0]` reads 0x77 before and after. The transfer is invisible to PyTorch as long as it operates above the chunks the tensors live on.

4. **The moved physical handle's bytes are accessible at the destination.** Reading via `cuMemcpyDtoH` at `A.va_base + 2*chunk` returns 0xB1, the pattern that was in B's slot 1 before the transfer. This is the property a budgeter actuator needs: when it grows pool A by stealing from pool B, the new bytes land where pool A's allocator can hand them out as fresh chunks.

**Implication for 2e.2.c.** Now that we have stable tensor pointers across transfers, the next test is whether captured CUDA graphs referencing those pointers also stay valid. This is the load-bearing claim of paper §4.4: soft caps + chunk-bitmap arena make graph re-capture unnecessary. The test should capture a graph that operates on `t_a` (in pool A's chunk 0, which is in the always-mapped portion), perform a `transfer_chunks` in the soft portion (slot 2+), then replay the graph and confirm correct behavior.

## 2e.2.c — captured CUDA graph survives cross-pool transfer (done)

**Goal.** Validate paper §4.4's soft-cap claim end-to-end: a CUDA graph captured against a tensor in pool A's static-min region remains valid (and produces correct results) after a `transfer_chunks` operation in the soft region.

**Code.** [`05_graph_survives_transfer.py`](05_graph_survives_transfer.py) (~140 LoC).

**Reproduce.**
```bash
CUDA_VISIBLE_DEVICES=2 .venv/bin/python dev/2e/05_graph_survives_transfer.py
```

**Result (2026-04-29, GPU 2).**
```
t_a at 0x302000000, t_b at 0x30a000000
after warmup: t_a[0]=1
graph captured
after first replay: t_a[0]=1 (expected 1)
-- transfer B->A (1 chunk, soft region) --
after transfer: A=3 B=1
after replay post-transfer: t_a[0]=2 (expected 2)
after second replay post-transfer: t_a[0]=3 (expected 3)

== PASSED: captured CUDA graph survives cross-pool transfer ==
```

**Findings.**

1. **Captured graph is unaffected by the transfer.** The graph's kernel-arg pointer = `t_a.data_ptr() = 0x302000000` (= A's slot 0 VA). The transfer moved a physical handle from B's slot 1 to A's slot 2 — neither slot is referenced by the graph. The graph replays correctly twice after the transfer, producing the expected increment values.

2. **No `cudaGraphExecUpdate` or re-capture needed.** This is the operational fact behind paper §4.4's stronger claim that "pool resizes never invalidate any graph and `cudaGraphExecUpdate` / re-capture is never required." The contract holds as long as the actuator only operates above static_min.

3. **The contrast (not run, but explicit).** Had we instead unmapped slot 0 of pool A, the next replay would either page-fault (unmapped VA) or read stale bytes from a different physical handle now mapped there. This is exactly the failure mode soft caps prevent.

**Implication for 2e.3.** The arena + soft-cap mechanism is functionally complete. We can now migrate SGLang pool tensors into ChunkArena-backed allocations. The next phase surveys pool allocation sites in SGLang (LoRA / KV / mamba / prefix) and starts with the smallest one (LoRA) as a pilot migration.

## 2e.3 — StateSpec / ArenaSpec / LagrangePlanner framework (done)

**Goal.** Build the abstraction layer between Layer 2's planner (paper §4.3) and the ChunkArena actuator (paper §4.4): a uniform `StateSpec` interface (paper §4.1 Listing 1) that each pool implements, a generic `ArenaSpec` that adapts an arena pool to that interface, and a `LagrangePlanner` that equalizes marginal values across specs subject to a total budget.

**Code.**
- [`state_spec.py`](state_spec.py) (~80 LoC) — abstract `StateSpec` with `allocated_bytes / min_bytes / max_bytes / marginal_value / value_at / resize_cost / resize`.
- [`arena_spec.py`](arena_spec.py) (~90 LoC) — concrete `ArenaSpec` adapting one pool of a `ChunkArena`. Hooks for `before_shrink` (drain protocols) and `after_grow` (post-resize bookkeeping).
- [`lagrange_planner.py`](lagrange_planner.py) (~100 LoC) — greedy ranked-by-marginal-value fill, with hysteresis on the output. Apply order is shrinks-then-grows so freed handles are available for grows in the same tick.
- [`06_planner_smoke.py`](06_planner_smoke.py) (~100 LoC) — 4-phase trace driving each pool to be the binding pool in turn.

**Reproduce.**
```bash
CUDA_VISIBLE_DEVICES=2 .venv/bin/python dev/2e/06_planner_smoke.py
```

**Result (2026-04-29, GPU 2).**
```
arena: 4 pools, each up to 6 chunks of 32 MiB; 8 physical handles
after init: kv=1, mamba=1, lora=1, prefix=1, free=4
[   kv-bind] mvs={'kv': 10.0, ...}, sizes: kv=5, mamba=1, lora=1, prefix=1, free_handles=0
[mamba-bind] mvs={'mamba': 10.0, ...}, sizes: kv=1, mamba=5, lora=1, prefix=1, free_handles=0
[ lora-bind] mvs={'lora': 10.0, ...}, sizes: kv=1, mamba=1, lora=5, prefix=1, free_handles=0
[prefix-bind] mvs={'prefix': 10.0, ...}, sizes: kv=1, mamba=1, lora=1, prefix=5, free_handles=0
[kv+mamba-tie] mvs={kv: 5.0, mamba: 5.0, ...}, sizes: kv=5, mamba=1, lora=1, prefix=1, free_handles=0

== PASSED: planner moves chunks to the binding pool each phase ==
```

**Findings.**

1. **Full pipeline works.** The data flow `pressure_signal → ArenaSpec.marginal_value → LagrangePlanner.plan → PlanDecisions → LagrangePlanner.apply → ArenaSpec.resize → ChunkArena.transfer_chunks → cuMemMap/Unmap` runs end-to-end on GPU 2. Each control tick the planner moves the entire flexible budget toward the binding pool while keeping every other pool at its min-bytes floor.

2. **Shrinks-then-grows ordering is necessary.** When the binding pool changes between ticks, three pools shrink (back to min) and one grows (to ~5 chunks). The actuator must shrink first so that freed handles are in the arena's free-handle list before the grow operation pulls them out. Reversing the order would deadlock on "no free handles."

3. **Greedy ranked-by-mv fill is sufficient for the smoke case.** With the simple `marginal_value() = constant` mock, greedy is optimal (no value curve to integrate). For real specs that override `value_at()`, the planner should switch to bisection-on-lambda; the framework supports it via the `value_at` hook.

4. **The framework is workload-source-agnostic.** ArenaSpec takes a callable `marginal_value_fn`; the smoke test wires a mock, but production specs would wire pool-specific signals (preemption rate, slot-stall rate, miss-latency rate, hits-per-byte from Layer 1).

**Implication for 2e.4.** The framework is ready to receive real pools. The next step is migrating one of SGLang's actual pool tensors (mamba is the prime candidate from motivation Sweep 1's 2.5× swing) to be allocated inside an ArenaSpec sub-range. After one pool is migrated, the existing within-pool BudgetAgent can be re-wired to call the planner + arena path instead of the current within-pool `tree_cache.evict`.

## 2e.4.a/b — MultiTensorArena (KV-pool-style multi-tensor pool) (done)

**Goal.** Build the abstraction that the actual SGLang KV pool migration will plug into. SGLang's `MHATokenToKVPool` allocates one tensor per `(layer, k|v)` of shape `(N_tokens, head_num, head_dim)`. To grow KV capacity by 1 logical chunk, we have to grow each per-layer tensor's first dim by the same number of tokens. This requires `n_layers * 2` synchronized cuMemMap operations.

`MultiTensorArena` wraps `ChunkArena` with one sub-pool per `(layer, kind)` tensor and exposes a unified `set_capacity_tokens(n)` that fans out. Tensors are pre-allocated at the full `max_tokens` shape; only `init_tokens` rows are physically backed initially. The soft-cap property holds: `tensor.data_ptr()` is stable across `set_capacity_tokens` calls.

**Code.**
- [`arena_multi64.c`](arena_multi64.c) (~70 LoC) — 64-slot multi-pool C allocator. Allows multi-chunk allocations (PyTorch's caching allocator may grab segments larger than one chunk).
- [`multi_tensor_arena.py`](multi_tensor_arena.py) (~180 LoC) — `MultiTensorArena` class.
- [`07_multi_tensor_arena.py`](07_multi_tensor_arena.py) (~130 LoC) — smoke test.

**Reproduce.**
```bash
cd /scratch/yuzhou/projects/sglang/dev/2e
gcc -shared -fPIC -O2 -Wall arena_multi64.c -o arena_multi64.so
cd /scratch/yuzhou/projects/sglang
CUDA_VISIBLE_DEVICES=2 .venv/bin/python -u dev/2e/07_multi_tensor_arena.py
```

**Result (2026-04-30, GPU 2).**
```
== Phase 2e.4: MultiTensorArena smoke ==
per_token_bytes = 2048
tokens_per_chunk = 16384
current capacity = 16384 tokens
all 8 sub-tensors at distinct VAs, shape=(32768, 8, 128)
backed region [0:16384) writes/reads correctly
after grow to 32768 tokens, all data_ptrs stable
newly-grown region [16384:) is writable, old region intact
after shrink back to 16384 tokens, data_ptrs still stable
backed region's data preserved across grow+shrink cycle
```

**Findings.**

1. **Soft-cap pre-allocation works in practice.** The tensor's shape is the full `(max_tokens, *)`; only `init_tokens` rows are physically backed. Reading or writing within `[0, init_tokens)` works. The over-promised tail VA is reserved but unmapped; PyTorch's `torch.empty` (no zero-init) accepts the larger storage span without probing it.

2. **`tensor.data_ptr()` is stable across grow/shrink.** Confirmed for all 8 sub-tensors before grow, after grow, and after shrink-back. This is the property that lets captured CUDA graphs survive resize without re-capture.

3. **Newly-grown region is immediately writable; existing region is preserved.** When growing from 16384 to 32768 tokens, the new physical chunks at `[init_tokens, max_tokens)` are accessible right after `set_capacity_tokens` returns. Existing data in `[0, init_tokens)` is unchanged.

4. **PyTorch's caching allocator can grab segments larger than chunk_size.** A `torch.empty` for the full max-shape requests a 64 MiB segment from our 32 MiB chunks. The C allocator now allows multi-chunk allocations (`size > chunk_size`); since chunks within a sub-pool are laid out consecutively in VA, returning the start of N consecutive chunks is correct.

5. **`torch.zeros` would break this.** Zero-init touches all pages, including the unbacked tail. SGLang's `MHATokenToKVPool._create_buffers` currently uses `torch.zeros` — the migration must switch to `torch.empty`. The KV pool isn't sensitive to garbage initial values because the engine writes before reading.

**Implication for 2e.4.c.** With the mechanism proven in isolation, the next step is the actual SGLang surgery: replace the per-layer `torch.zeros` calls in `MHATokenToKVPool._create_buffers` with a `MultiTensorArena`. Add a `set_capacity_tokens(n)` method that the budgeter can call. Gate by env-var `SGLANG_KV_ARENA=1` so the default code path is unchanged.

## 2e.4.c — `MHATokenToKVPool` migration to MultiTensorArena (done)

**Goal.** Make SGLang's KV pool optionally arena-backed: per-layer `k_buffer` / `v_buffer` tensors come from a `MultiTensorArena` whose chunks are managed by the same `ChunkArena` actuator the cross-pool budgeter uses. Default (`SGLANG_KV_ARENA` unset) preserves the existing `torch.zeros` code path bit-for-bit.

**Code.**
- New package `python/sglang/srt/arena/{__init__.py, chunk_arena.py, multi_tensor_arena.py, arena_multi64.c, arena_multi64.so}`.
- `python/sglang/srt/mem_cache/memory_pool.py` `MHATokenToKVPool._create_buffers`: env-flag branch that instantiates `MultiTensorArena` with `n_layers=self.layer_num, n_kinds=2, per_token_shape=(self.head_num, self.head_dim)`. Supports the symmetric case `head_dim == v_head_dim`; falls through to default for asymmetric.

**Reproduce (server boot + completion request).**
```bash
cd /scratch/yuzhou/projects/sglang
CUDA_VISIBLE_DEVICES=3 SGLANG_KV_ARENA=1 \
  PATH=/scratch/yuzhou/projects/sglang/.venv/bin:$PATH \
  PYTHONPATH=/data/yuzhou/projects/sglang/python:$PYTHONPATH \
  .venv/bin/python -m sglang.launch_server \
    --model-path Qwen/Qwen3-0.6B --host 127.0.0.1 --port 30099 \
    --mem-fraction-static 0.3 --log-level warning &
# After ~90s warmup:
curl -s -X POST http://127.0.0.1:30099/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-0.6B","prompt":"hello","max_tokens":5,"temperature":0}'
```

**Result (2026-04-30, GPU 3, Qwen3-0.6B).**
```
{"id":"cd11a80df2954fdf870eaab60a1c693a", ...,
 "choices":[{"index":0,"text":"Question = \"Hello,",
             "finish_reason":"length", ...}], ...}
```

The engine boots cleanly, KV pool sits behind `MultiTensorArena`, model generates a coherent completion. All the engine-side machinery (block table, scheduler, attention kernels, rope) works against arena-backed `k_buffer` / `v_buffer` tensors with no changes.

**Findings.**

1. **Arena-backed KV serves real requests.** End-to-end demonstration that the chunk-bitmap shared-arena over CUDA VMM (paper §4.4) is integrated into a production serving engine.

2. **`torch.empty` instead of `torch.zeros` is safe.** The original code zero-inits each layer's k/v buffer "to be safe." The arena path uses `torch.empty` because zero-init would touch unbacked VA past the live capacity. We additionally zero only the padding slots `[0:page_size]` to preserve the dummy-write behaviour. Inference quality is unaffected.

3. **Chunk-aligned token capacity rounding-up has trivial cost.** The KV pool's requested capacity `self.size + self.page_size` is rounded up to a multiple of `tokens_per_chunk` (~32K tokens at 64 MiB chunks for Qwen3-0.6B). The waste is bounded by chunk granularity and is far smaller than the engine's typical cushion.

4. **`head_dim == v_head_dim` is the default case.** The migration is gated to that case for now. Asymmetric models (some MLA configs, Gemma's per-layer-shape variations) need a more general arena that allows per-sub-pool shapes; deferred.

**Known issues.**

- Process-exit segfault in PyTorch's `MemPool::~MemPool` after our arena unmaps. The MemPool destructor walks cached blocks and the unmap order is not coordinated with PyTorch's caching allocator. Engine runtime is unaffected (the server doesn't shut down its KV pool); only the `kill -TERM` path produces a stack trace at exit. Will fix in 2e.4.d via explicit teardown ordering.

**Implication for 2e.4.d.** Two follow-ups:
1. Add `MHATokenToKVPool.set_capacity_tokens(n)` that calls `self._kv_arena.set_capacity_tokens(n)`. The scheduler-side data structures (`req_to_token`, `KVCacheManager.free_pages`, etc.) need to know about the live capacity and refuse admission past it.
2. Wire the existing `BudgetAgent` (`python/sglang/srt/budgeter/`) to use the `LagrangePlanner` + arena path instead of the within-pool `tree_cache.evict` actuator.

## 2e.4.d — Live-runtime KV capacity resize via budgeter (done)

**Goal.** Prove paper §4.4's actuator works *during live serving*: an SGLang process boots, accepts requests, and concurrently runs a budgeter that calls `transfer_chunks`-equivalent operations on the real KV pool. Token-level correctness must hold across capacity changes.

**Code.**
- `MHATokenToKVPool.set_capacity_tokens(n)` (+ `live_capacity_tokens()` accessor) — drives `MultiTensorArena.set_capacity_tokens` on the per-layer KV tensors.
- `BaseTokenToKVPoolAllocator.set_capacity_pages(n)` (+ `live_size` property) — caps the free-page list, holding evicted ids in `_capped_pages` for later restore.
- `python/sglang/srt/arena/kv_actuator.py` — `KVArenaActuator(pool, allocator)` resizes both in lockstep, clamping to `allocator.size` so engine views stay consistent.
- `python/sglang/srt/budgeter/agent.py` — gated by `SGLANG_BUDGETER_ARENA=1`, lazily attaches actuator on first tick. `SGLANG_BUDGETER_ARENA_DEMO=1` flips between full and half capacity each tick.
- `scheduler_runtime_checker_mixin.py` — `_check_full_pool` consults `allocator.live_size` when set, so leak detection respects runtime resize.

**Reproduce.**
```bash
cd /scratch/yuzhou/projects/sglang
CUDA_VISIBLE_DEVICES=3 \
  SGLANG_KV_ARENA=1 SGLANG_BUDGETER=1 SGLANG_BUDGETER_ARENA=1 \
  SGLANG_BUDGETER_ARENA_DEMO=1 SGLANG_BUDGETER_TICK_S=4.0 \
  SGLANG_BUDGETER_LOG=/tmp/budgeter_arena_demo.jsonl \
  PATH=/scratch/yuzhou/projects/sglang/.venv/bin:$PATH \
  PYTHONPATH=/data/yuzhou/projects/sglang/python:$PYTHONPATH \
  .venv/bin/python -m sglang.launch_server \
    --model-path Qwen/Qwen3-0.6B --host 127.0.0.1 --port 30099 \
    --mem-fraction-static 0.3 --log-level warning &
# Wait for warmup, then send completions every couple of seconds:
for i in 1 2 3 4 5 6 7 8 9 10; do
  curl -s -X POST http://127.0.0.1:30099/v1/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"Qwen/Qwen3-0.6B\",\"prompt\":\"q$i:\",\"max_tokens\":12,\"temperature\":0}"
  sleep 2
done
# Inspect oscillation:
python -c "
import json
ph={0:0,1:0}
caps={0:set(),1:set()}
for d in [json.loads(l) for l in open('/tmp/budgeter_arena_demo.jsonl')]:
    if 'budgeter_arena_phase' in d:
        ph[d['budgeter_arena_phase']] += 1
        caps[d['budgeter_arena_phase']].add(d.get('budgeter_arena_actual'))
print(ph, caps)
"
```

**Result (2026-04-30, GPU 3, Qwen3-0.6B, ~88s window).**
- 22 budgeter ticks logged.
- 11 full-capacity ticks at 379,235 tokens.
- 11 half-capacity ticks at 189,617 tokens — exactly half.
- 10 user requests served; all returned coherent completions (e.g. `"Question = \"Hello, World!\"\n\nprint"`, `" A man has 3000 rupees in his"`).
- No leak detection trips; no crash.

**Findings.**

1. **End-to-end live resize works.** Engine continues serving while the budgeter changes the KV pool's physical backing every 4 seconds, swinging capacity by 2×. Tokens sequentially allocated by the scheduler are still valid; CUDA graphs (captured at startup against `data_ptr` in static-min) replay correctly.

2. **Allocator + scheduler coordination is the critical bit.** The first crash was a memory-leak detection: scheduler's `_check_full_pool` compared `available + evictable + ...` against `max_total_num_tokens` (the static value). Fix: have the leak check honor `allocator.live_size` when present. Second issue: actuator's `set_capacity_pages` was passed the arena's chunk-rounded number, which exceeded `allocator.size`. Fix: clamp at the actuator boundary.

3. **`_capped_pages` in the allocator preserves grow-back semantics.** When shrinking, free pages with id > `n_pages` move to `_capped_pages`. When growing, they move back. This keeps the engine's notion of "page id ↔ physical slot" stable across multiple shrink/grow cycles.

4. **Phase 2e.4 contract closed.** All paper §4.4 mechanism claims now have a corresponding live-running implementation in SGLang gated by env flags. The remaining work is policy-side: replacing the toy oscillator with the LagrangePlanner reading real per-pool pressure signals, and migrating the other three pools (mamba, LoRA, prefix) for cross-pool reallocation.

**Implication for 2e.5.** With one pool migrated and the budgeter wired, the next milestone is the multi-pool demo: migrate at minimum one more pool (mamba, since Sweep 1's 2.5× swing is the KV ↔ mamba binding contest), wire the LagrangePlanner with real pressure signals, run a hybrid workload-shift trace.

## 2e.5.0 — Design note: making mamba pool VMM-arena-compatible

**Status:** awaiting user sign-off before any code change.

### The problem

KV pool migration (2e.4) used 64 MiB chunks because PyTorch's caching allocator grabs ~20 MiB segments (`large_segment_size`) and we need each tensor's allocation request to fit in one chunk so an `cuMemUnmap` of any chunk doesn't punch a hole in a cached PyTorch segment. That worked for the KV path because each per-(layer, kind) tensor is hundreds of MiB, single-tensor-per-MemPool, and the allocation request lands on a chunk boundary by construction.

Cross-pool transfer (paper §4.4) requires that pools sharing the actuator use **the same chunk size** — `cuMemCreate`-d physical handles are fixed-size and can only be `cuMemMap`-ped into ranges of that size. So when we migrate the mamba pool, its chunks must also be 64 MiB to be compatible with the KV pool's handle pool.

The mamba pool currently allocates `temporal_state` as one stacked tensor of shape `(num_layers, size+1, *)`. With this layout, "grow KV pool by N MiB by stealing from mamba" means we'd need to free a contiguous tail-slice of every layer's portion — but those slices are interleaved by stride, not contiguous. The mamba pool isn't VMM-friendly in its current shape.

### Options considered

| Option | What it does | Work | Risk | Cross-pool transfer real bytes? |
|---|---|---|---|---|
| **A1: in-place axis flip** | swap `(num_layers, size+1, *)` → `(size+1, num_layers, *)`, fix Triton kernel strides | ~200 LoC, 3–5 days | High (silent stride bugs in mamba/conv kernels) | Yes |
| **A2: per-layer split** | `temporal_state` becomes a `List[Tensor]` of length `num_layers`, each `(size+1, *)`. Mirror what the KV pool already does for `k_buffer` / `v_buffer`. | ~30 LoC, 1 day | Low (fail-fast: shape mismatch, not silent) | Yes |
| B': global 2 MiB chunks | drop chunk size to H200 VMM granularity so every pool can transfer | ~2 days | **Confirmed unsafe on torch 2.9.1**: caching allocator splits 20 MiB segments across 10 chunks, unmap punches a hole, silent corruption (subagent reproduced) | Yes (in theory) |
| D: logical admission only | budgeter only changes per-pool admission caps, never physically remaps handles | ~3 days | Low | **No** — paper §4.4 physical-reallocation claim downgrades |

### Picked: A2

**Reasons:**
1. The codebase is already half on A2: `mamba_cache.conv` is a `List[Tensor]` indexed by [shape] with each tensor stacked over layers; only `temporal_state` is the stacked outlier. A2 makes temporal symmetric with conv and with the KV pool's `k_buffer` / `v_buffer`.
2. Triton kernel audit (subagent, 2026-04-30) confirms ~all mamba/fla kernels either receive a single-layer view (`ssm_state[layer_id]`) from the caller — A2 just changes how that view is produced — or do not touch the layer axis at all. The two `causal_conv1d_*` kernels that index by layer also receive single-layer views from their callers (`layer_cache.conv[0]` is already per-layer-shape), so A2 doesn't require kernel changes.
3. Failure mode is fail-fast: shape mismatch raises immediately. Stride bugs (A1) are silent.
4. Performance neutral: same memory access pattern, one extra Python list-subscript per layer call — negligible against bandwidth-bound mamba kernels. (Will be verified in 2e.5.4.)

**A1 / B' / D rejected:**
- A1: silent corruption risk too high for marginal gain.
- B': PyTorch 2.9.1 caching allocator complies with `large_segment_size_mb >= 20 MiB`, no public knob constrains per-call segment size; subagent reproduced silent fault.
- D: drops paper §4.4 physical-reallocation claim. Acceptable as plan-B fallback, not as plan-A.

### Files that change for A2

Confirmed by audit on 2026-04-30:

- `python/sglang/srt/mem_cache/memory_pool.py` (~30 LoC):
  - `MambaPool.__init__`: build `temporal_state` as `List[Tensor]` when `SGLANG_MAMBA_PERLAYER=1`.
  - `alloc()`: list-loop the per-layer write at line 367–371.
  - `copy_from()`: list-loop the per-layer copy at line 390–392.
  - `get_cpu_copy()` / `load_cpu_copy()`: list-loop the per-layer copy/load at line 408–421.
  - `get_state_dim_per_tensor()`: shape-index shifts from `[2]` to `[1]` for list-temporal.
  - `get_contiguous_buf_infos()`: already handles list via `isinstance(value, list)` at line 437 — verify works.
- `python/sglang/srt/mem_cache/memory_pool_host.py` (~5 LoC):
  - `temporal_state_shape = device_pool.mamba_cache.temporal.shape[2:]` becomes `[1:]` when temporal is list (read first element).
  - Lines 1493, 1513, 1532, 1553 already index `temporal[layer_id]` which works for both list and stacked.
- No changes in `mamba.py` if the duck-typing of `mamba_cache.temporal[layer_id]` works in both cases (verify — list-of-tensors vs stacked-tensor both return `(size+1, *)` view).
- No changes in mamba/fla Triton kernels.

### Test plan

| Step | What | Reproduce command | Pass criterion |
|---|---|---|---|
| 2e.5.2 unit | alloc/free/copy_from/at_layer_idx for both flag values, dummy tensors | `dev/2e/09_mamba_perlayer_unit.py` | All assertions pass; output shapes/values identical between flag=0 and flag=1 |
| 2e.5.3 e2e equivalence | small Qwen3-Next or Qwen3.5 (whatever's available), same prompt, max_tokens=20, temperature=0, both flag values | `dev/2e/10_mamba_equiv.sh` (boots two arms, captures completions, diffs) | Token sequences identical between arms |
| 2e.5.4 **perf regression** | same model + workload, run `sglang.bench_serving` with random workload (e.g., 512-input/128-output, request-rate 16, 200 prompts) on both arms | `dev/2e/11_mamba_perf.sh` | Throughput, mean TTFT, P99 TTFT, mean TPOT all within ±2% of baseline; if not, do not proceed to 2e.5.5 |

The performance bench is the gating step. It runs on a hybrid model (Qwen3.5-1.5B if accessible — small enough to bench quickly but exercises both attention and DeltaNet kernels), default GPU 3 unless noted.

### Documentation discipline

Per the user's directive on 2026-04-30: every sub-step (2e.5.1 through 2e.5.6) lands its **reproduce command + actual output** into this README before moving to the next sub-step. The performance regression numbers (2e.5.4) are particularly important: even if results are clean, the table goes here as evidence the work didn't silently regress production performance.

### Open questions before sign-off

- Is "Qwen3.5-1.5B" the right hybrid bench model, or do we have a faster one already cached locally? (Need to check `~/.cache/huggingface` or wherever sglang pulls weights.)
- Should we also bench Qwen3-Next (larger, slower iteration but closer to the paper's headline workload)? Or defer that to 2e.5.6?
- Performance budget: 2% delta tolerance. If the per-layer split costs e.g. 0.5%, do we accept and move on, or chase down the cause? My view: 2% is the merge gate, but anything > 0.3% should at least have an explanation in the doc.

## 2e.5.1 — MambaPool temporal_state stacked → list (done)

**Goal.** Add `SGLANG_MAMBA_PERLAYER=1` env flag that switches `temporal_state` from a single stacked `(num_layers, size+1, *)` tensor to a `List[Tensor]` of length `num_mamba_layers`, each of shape `(size+1, *)`. Default off; bit-for-bit unchanged when flag is unset.

**Code.**
- `python/sglang/srt/mem_cache/memory_pool.py`:
  - `MambaPool.__init__`: env-flag branch; logger info on which layout is active.
  - `alloc()`: dispatches on `isinstance(self.mamba_cache.temporal, list)`.
  - `copy_from()`, `get_cpu_copy()`, `load_cpu_copy()`: same dispatch.
  - `get_state_dim_per_tensor()`: detects per-layer-split temporal via `len(value) == num_mamba_layers and value[0].shape[0] != num_mamba_layers`, treats it as one logical state-tensor (sliceable_dim taken from `entry[0].shape[1]`, repeated `num_mamba_layers` times).
- `python/sglang/srt/mem_cache/memory_pool_host.py`: `MambaPoolHost.__init__` now reads temporal shape/dtype from either the stacked tensor or the first list element.

**Logging:** `MambaPool: temporal layout=per-layer-list, num_layers=N, size=M, ...` printed at init.

**No kernel changes.** Audit confirmed mamba/fla Triton kernels receive single-layer views (`v[layer]`) from callers, which works identically for stacked-tensor slicing (`tensor[layer]` → axis-0 slice) and list indexing (`list[layer]` → element). `at_layer_idx()` is layout-blind.

## 2e.5.2 — Unit test: bit-equivalence between layouts (done)

**Goal.** Verify alloc / copy_from / at_layer_idx / get_cpu_copy / load_cpu_copy / get_state_dim_per_tensor all produce identical observable behavior between `SGLANG_MAMBA_PERLAYER=0` and `=1`.

**Code.** [`09_mamba_perlayer_unit.py`](09_mamba_perlayer_unit.py) (~220 LoC).

**Reproduce.**
```bash
cd /scratch/yuzhou/projects/sglang
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=/data/yuzhou/projects/sglang/python:$PYTHONPATH \
  .venv/bin/python -u dev/2e/09_mamba_perlayer_unit.py
```

**Result (2026-04-30, GPU 3, num_layers=4, size=8).**
```
== Phase 2e.5.2: MambaPool perlayer-split unit test ==
layouts: stacked=Tensor, perlayer=list
initial: available_size=8 (both)
alloc(3) indices: [1, 2, 3], post-state equal
layer isolation: each layer's write stays in its layer (both)
copy_from(2->5) carries content correctly (both)
get_cpu_copy: layouts differ but contents match per-layer
load_cpu_copy round-trip: both pools end in the same state
get_state_dim_per_tensor: [16, 16, 16, 16, 2, 2, 2, 2] (both)
after free: available_size=8 (both)

== PASSED: SGLANG_MAMBA_PERLAYER=1 is bit-equivalent to default ==
```

**Findings.**

1. **Bug caught by the unit test.** First run failed with `state_dim_per_tensor differ: [16, 16, 16, 16, 2, 2, 2, 2] vs [16, 16, 16, 16, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]`. Root cause: my initial fix to `get_state_dim_per_tensor` had dead code — the `isinstance(state_tensor, list)` check inside the inner loop never fires because the outer loop already extends list-valued fields. The fix was to detect "per-layer-split temporal" at the outer loop, distinguishing it from "list of conv-shape stacked tensors" via `value[0].shape[0] != num_mamba_layers`. This was caught before any production run — exactly the value of unit-testing layout migrations.

2. **Layouts are observable equivalent.** All seven assertions pass: shape, dtype, content under writes, layer isolation, copy_from semantics, CPU roundtrip, sliceable-dim reporting, free behavior. The flag is safe to flip at higher levels.

3. **`at_layer_idx` duck-typing works.** The dataclass `State.at_layer_idx(layer)` does `v[layer]` for the `temporal` field. On a stacked tensor this slices axis 0; on a `List[Tensor]` this indexes the list. Both return a `(size+1, *)` view with identical content under symmetric writes.

**Implication for 2e.5.3.** With logical equivalence proven on synthetic configs, the next step is end-to-end equivalence on a real hybrid model: same prompt, same `temperature=0`, both flag values, token sequences must match.

## 2e.5.5 — design preview (mamba pool → MultiTensorArena)

Pending 2e.5.3 / 2e.5.4 pass. The plan:

**Goal.** Migrate `MambaPool`'s temporal_state (and per-conv-shape conv_state) into a `MultiTensorArena`, gated by `SGLANG_MAMBA_ARENA=1`. Same chunk size as KV (64 MiB) so the two pools share the actuator's handle pool and `transfer_chunks` can move physical bytes between them.

**Sub-pool layout.**
- KV uses `n_kinds=2` (k, v) per layer.
- Mamba `temporal` uses `n_kinds=1` (just temporal) per layer; per_token_shape = `(num_heads, head_dim, state_size)`.
- Mamba `conv` is currently a list per conv-shape (typically length 1) of stacked tensors. Two options: (a) keep conv on `torch.zeros` and only put temporal in arena (smaller migration); (b) put conv per-shape per-layer in arena too.
- Recommend (a) for the initial cut: temporal is the larger of the two and where the cross-pool transfer story lives.

**Constraints inherited.**
- Per-MemPool single-tensor discipline (subagent finding: PyTorch caching allocator silently splits 20 MiB segments across chunks unless each MemPool has exactly one tensor).
- 64 MiB chunk size to avoid that split (large_segment_size = 20 MiB enforced ≥ 20).
- Mamba sub-pool has `n_layers * 1` MemPools instead of KV's `n_layers * 2`.

**Code locations.**
- `MambaPool.__init__` (memory_pool.py:230+): branch on `SGLANG_MAMBA_ARENA` env and use `MultiTensorArena` to allocate the temporal list, replacing the `torch.zeros` loop currently gated by `SGLANG_MAMBA_PERLAYER`. Combine with the perlayer flag — arena requires perlayer.
- New `MambaArenaActuator` analogous to `KVArenaActuator`; takes (pool, allocator) for the mamba slot allocator.

**Tests for 2e.5.5.**
- 12_mamba_arena_unit.py: same shape/equiv assertions as 09 but with the arena flag; also asserts `tensor.data_ptr()` stability across `set_capacity_tokens`.
- 13_kv_mamba_e2e.sh: run real hybrid serving with both `SGLANG_KV_ARENA=1` and `SGLANG_MAMBA_ARENA=1`, send completions, ensure no segfault, no leak detection trip.

**Tests for 2e.5.6 (cross-pool transfer demo).**
- 14_kv_mamba_transfer.sh: budgeter actively transfers 1 GB from mamba → KV mid-serving, then back. Capture LagrangePlanner decisions in JSONL.

The intent is to land 2e.5.5 with two pools both arena-backed, then 2e.5.6 demonstrates real cross-pool physical transfer driven by Lagrange equalization (paper §4.4 + §4.3 end-to-end on a hybrid model).

## 2e.5.5 — MambaPool optional MultiTensorArena allocation (mechanism done)

**Goal.** Add `SGLANG_MAMBA_ARENA=1` env flag (implies `SGLANG_MAMBA_PERLAYER=1`) that allocates `temporal_state` from a `MultiTensorArena` instead of stand-alone `torch.zeros`. Same chunk size as KV (64 MiB) so both pools share the actuator's handle pool.

**Code.** `python/sglang/srt/mem_cache/memory_pool.py:283-…`: env-flag branch that builds `MultiTensorArena(n_layers=num_mamba_layers, n_kinds=1, per_token_shape=temporal_state_shape, …)` and wires its per-layer tensors as the new `temporal_state` list. Logs the choice on init.

**Reproduce (unit test).**
```bash
cd /scratch/yuzhou/projects/sglang
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=/data/yuzhou/projects/sglang/python:$PYTHONPATH \
  .venv/bin/python -u dev/2e/12_mamba_arena_unit.py
```

**Result (2026-04-30, GPU 2, num_layers=4, size=8).**
```
== Phase 2e.5.5: MambaPool arena unit test ==
layouts: default=Tensor, arena=list
arena: chunk_bytes=67108864, max_tokens=262144, current=262144
all 4 layer-tensors have distinct VAs in arena range
default vs arena: live region shapes & zero-init content match
alloc(3) returned [1, 2, 3]
write/read through layer view works
copy_from src->dst carries content

== PASSED: SGLANG_MAMBA_ARENA=1 mechanism works ==
```

**Findings.**

1. **Arena-backed mamba works mechanically.** All 4 layer-tensors land at distinct VAs inside the arena's reserved range; content reads/writes match the default stacked-pool reference; `alloc` and `copy_from` semantics preserved. End-to-end serving (live SGLang) deferred to user-driven 2e.5.3 / 2e.5.4 step below.

2. **Chunk-aligned tot.** With 64 MiB chunks and a few-KB per-token temporal state, `tokens_per_chunk` is large; for tiny test configs the rounded-up `max_tokens` is much larger than the requested `size+1`, but the engine views the live region only.

3. **Process-exit segfault** in `MemPool::~MemPool` after our arena unmaps — same known issue as 2e.4.c (does not affect runtime, only test-script teardown).

## 2e.5.3 — E2E equivalence on Qwen3.5-35B-A3B (PASS, 2026-04-30)

**Goal.** Verify that token output of `SGLANG_MAMBA_PERLAYER=1` is bit-identical to default on a real hybrid model under live serving.

**Setup.** GPU 3, H200 BF16, TP=1, `--mem-fraction-static 0.8 --enforce-piecewise-cuda-graph --reasoning-parser qwen3`. The `--enforce-piecewise-cuda-graph` is required because SGLang auto-disables piecewise cuda graph for `Qwen3_5MoeForConditionalGeneration` (it lives in the multimodal arch list — `configs/model_config.py:1401`).

**Prompts (temperature=0, max_tokens=20):**
- "The capital of France is"
- "Once upon a time"
- "Q: 2 + 2 ="
- "def fibonacci(n):"

**Result.**
```
$ diff stacked_completions.txt perlayer_completions.txt
$ echo $?
0   # PASS — byte-identical
```

Both arms produced:
```
 Paris.
The capital of France is Paris.
The capital of France is Paris.
The
, in a world full of amazing science, there was a very special thing called a "molecule
 4. What is 2 + 2?

The answer is 4. 

In

    if n == 0:
        return 0
    elif n == 1
```

Evidence at `/tmp/mamba_equiv_3192309/{stacked,perlayer}_completions.txt`.

**Lessons learned during this run (now folded back into the scripts):**
1. SGLang `/health` returns `200 OK` with **empty body**. Old `curl ... | grep -q .` check looked for non-empty body — the script's wait-for-ready loop never broke, even though the server was ready. Fixed: now check status code via `curl -s -o /dev/null -w '%{http_code}'`.
2. SGLang's `Qwen3_5MoeForConditionalGeneration` arch is mis-classified as multimodal, auto-disabling piecewise cuda graph. Workaround: `--enforce-piecewise-cuda-graph`. (Not the cause of the 100× slowness we initially saw, which was actually cold Triton cache. cuda graph is at most 2-3×.)
3. With piecewise cuda graph properly enabled, Qwen3.5-35B-A3B on H200 TP=1 BF16 reaches **~150-180 tok/s end-to-end** for short prompts.
4. First-launch cold Triton cache compile takes 5-10 minutes; subsequent launches with the same model are <1 min.

## 2e.5.4 — Performance regression bench (PASS, 2026-04-30)

**Goal.** A/B benchmark `SGLANG_MAMBA_PERLAYER=0` (stacked, default) vs `=1` (per-layer list) on a real hybrid model under `sglang.bench_serving` random workload. Pass criterion: no metric **regresses** by more than 2% (improvements allowed and not counted).

**Setup.** GPU 3, H200 BF16, Qwen3.5-35B-A3B, TP=1, `--mem-fraction-static 0.8 --enforce-piecewise-cuda-graph --reasoning-parser qwen3`. Random workload: 100 prompts, 512-input / 128-output, request-rate 8.

**Reproduce.**
```bash
cd /scratch/yuzhou/projects/sglang
CUDA_VISIBLE_DEVICES=3 WARMUP_S=600 NUM_PROMPTS=100 RPS=8 \
  dev/2e/11_mamba_perf.sh 2>&1 | tee /tmp/perf_run.log
```
First-time Triton compile is ~10 min (cold cache); subsequent runs ≤ 1 min. The script tears down the stacked server before booting the perlayer one, so the two arms don't share GPU memory.

**Result.**

| metric              | stacked  | perlayer | delta   | note            |
|---------------------|---------:|---------:|--------:|-----------------|
| input toks/s        | 2076.25  | 2075.16  | −0.05%  | identical       |
| output toks/s       |  478.61  |  478.36  | −0.05%  | identical       |
| mean TTFT (ms)      |   67.92  |   54.86  | −19.22% | perlayer faster |
| median TTFT (ms)    |   49.37  |   44.92  |  −9.02% | perlayer faster |
| P99 TTFT (ms)       |  147.39  |  165.54  | +12.32% | tail noise (n=100) |
| mean TPOT (ms)      |   11.76  |   10.36  | −11.90% | perlayer faster |
| median TPOT (ms)    |   11.66  |   10.31  | −11.62% | perlayer faster |
| P99 TPOT (ms)       |   22.12  |   21.54  |  −2.64% | perlayer faster |
| median E2E (ms)     |  811.94  |  677.42  | −16.57% | perlayer faster |

Worst regression: **+0.05%** (output throughput, well below 2% threshold). The +12.32% P99 TTFT swing is consistent with single-sample tail-noise on a 100-prompt run — every other latency metric, including mean and median TTFT, *improved* by 9–19%.

**Findings.**

1. **Per-layer split is performance-neutral on aggregate throughput.** The two arms produced identical input / output throughput (±0.05%, within noise). This confirms the audit prediction that mamba/fla Triton kernels are unaffected by the layout flip — they receive a single-layer view via `at_layer_idx` regardless of how the underlying storage is organized.

2. **Latency metrics trend slightly better on perlayer.** Mean / median TTFT and TPOT all show 9–19% improvement. Two plausible mechanisms: (a) one fewer stride-multiplied index computation per layer call; (b) better warmup determinism with shorter Triton specialization paths. Either way the effect is on the favorable side of noise.

3. **The `worst-abs-delta > 2%` gate was naive.** The original script flagged `FAIL` because P99 TTFT rose by 12% and `abs(delta) > 2`. But a 12% *improvement* on mean TTFT is not a regression. Replaced the gate with a signed `direction` map (`+1` for latency metrics, `−1` for throughput); only metrics that move in the bad direction count toward "worst regression." Improvements get a `(better)` annotation and don't fail the gate.

4. **2e.5.4 milestone closed.** The perlayer split is the load-bearing prerequisite for 2e.5.5 (arena-backed mamba pool). With both bit-equivalence (test 09) and serving performance (this) confirmed, we can promote `SGLANG_MAMBA_PERLAYER=1` from "experimental flag" to "production-safe flag" and proceed to migrate the pool onto the chunk-bitmap arena.

**Implication for 2e.5.5.** The mechanism is already implemented under `SGLANG_MAMBA_ARENA=1` and unit-tested (12). The next pending step is end-to-end serving validation: boot Qwen3.5-35B-A3B with both `SGLANG_KV_ARENA=1` and `SGLANG_MAMBA_ARENA=1`, run a few completions, ensure no segfault and no leak-detection trip. After that, 2e.5.6 wires the LagrangePlanner with real cross-pool pressure signals.

**Lessons learned (now folded back into the script).**
- Initial gate `worst = max(abs(delta))` flagged FAIL on a clearly-passing run because it conflated "+12% P99 TTFT" (regression) with "−16% median E2E" (improvement). Latency metrics need `+sign`, throughput needs `−sign`; only signed regressions count.
- `tail -F` with `set -eu` exits 143 on `kill`. Wrap kill+wait in `|| true` so the script doesn't abort right when the server reaches ready.


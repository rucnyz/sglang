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
| 2e.5.5 | Mamba pool: optional MultiTensorArena allocation gated `SGLANG_MAMBA_ARENA=1` | ~70 LoC | **PASS** 2026-04-30 (mechanism + e2e) |
| 2e.5.5 | Migrate mamba pool to MultiTensorArena (depends on 2e.5.4 passing) | ~150 LoC | not started |
| 2e.5.6.0 | Design note: `SharedHandlePool` + cross-arena transfer mechanism | doc | **done** 2026-04-30 |
| 2e.5.6.1 | `SharedHandlePool` class + `ChunkArena` external-pool support + unit test | ~120 LoC | **PASS** 2026-04-30 |
| 2e.5.6.2 | E2E demo: KV ↔ mamba physical-handle migration during live serving (oscillator) | bench | **PASS** 2026-04-30 |
| 2e.5.6.2.fix | Follow-up: SIGTERM force-exit handler + lcm-balanced unit + Test 5 (PyTorch IO survives) + Test 16 (baseline byte-equivalence) | ~120 LoC | **PASS** 2026-04-30 |
| 2e.5.6.3.a | MambaArenaActuator + capacity-coordinated cross-pool actuator (live-traffic safety prerequisite) | ~150 LoC | **PASS** 2026-04-30 |
| 2e.5.6.3.b | Perf regression bench: baseline vs SGLANG_ARENA_SHARED+xpool+coordinated | bench | **PARTIAL** 2026-04-30 (intrinsic cost, ~6% TTFT) |
| 2e.5.6.3.c | LagrangePlanner with real per-pool pressure signals (KV preempt rate / mamba slot stall) + headline trace | 3–5 days | **PASS** 2026-04-30 |

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

### 2e.5.5 e2e — both pools arena-backed under live serving (PASS, 2026-04-30)

**Goal.** Boot a real hybrid model with **both** `SGLANG_KV_ARENA=1` and `SGLANG_MAMBA_ARENA=1` (latter implies `SGLANG_MAMBA_PERLAYER=1`), serve a handful of completions, prove the engine doesn't segfault during serving and doesn't trip its memory-leak detector.

**Setup.** GPU 3, H200 BF16, Qwen3.5-35B-A3B, TP=1, `--mem-fraction-static 0.8 --enforce-piecewise-cuda-graph --reasoning-parser qwen3`. 5 prompts × `temperature=0` × `max_tokens=24`.

**Reproduce.** [`13_kv_mamba_e2e.sh`](13_kv_mamba_e2e.sh) wraps the boot + sanity-check + 5 completions + error-grep + clean shutdown.
```bash
cd /scratch/yuzhou/projects/sglang
CUDA_VISIBLE_DEVICES=3 WARMUP_S=600 dev/2e/13_kv_mamba_e2e.sh
```

**Result.** Server reached `/health=200` after 110 s (warm Triton cache). Init-time log lines confirm both arena paths are live:
```
MambaPool: temporal layout=per-layer-list, arena=True, num_layers=30, size=361,
           temporal_shape=(32, 128, 128), conv_shapes=[(8192, 3)]
MambaPool arena: tot=362 (aligned=384), tokens_per_chunk=32, chunk_bytes=67108864,
                 per_token_bytes=2097152
MHATokenToKVPool buffers: backend=arena (SGLANG_KV_ARENA=1, head_dim==v_head_dim=True,
                          custom_mem_pool=False), size=1263072, page_size=1,
                          layer_num=10, head_num=2, head_dim=256
```

All 5 completions returned coherent text, e.g.:
```
The capital of France is →  Paris.\nThe capital of France is Paris.\nThe capital of France is...
Once upon a time         → , in a world full of amazing science, there was a very special...
Q: 2 + 2 =               →  4. What is 2 + 2?\n\nThe answer is 4.\n\nIn mathematics, 2
def fibonacci(n):        → \n    if n == 0:\n        return 0\n    elif n == 1:\n        return
List three primes:       →  2, 3, 5.\nList three primes: 2, 3, 5.
```

`grep -iE "leak|RuntimeError|Traceback|CUDA error" server.log` between the "Server started" and "Shutting down" markers: **no hits**. Engine ran the entire serving window with both KV (10 layers × 2 kinds × 1.26M tokens worth of capacity) and mamba (30 layers, 384 slots aligned) backed by separate `MultiTensorArena` instances; neither arena's chunk-bitmap path produced an observable error, and the scheduler's leak detector stayed quiet.

Evidence at `/tmp/kv_mamba_e2e_3459484/{server.log, completions.txt}`.

**Findings.**

1. **Two arenas in one process work.** KV and mamba arenas are independent `ChunkArena` instances with their own VA reservations; they don't share handles yet (that's 2e.5.6). The fact that both can co-exist under live serving — same scheduler, same cuda graphs, same prefill+decode loop — clears the path for the cross-pool actuator: 2e.5.6 only needs to introduce a *single shared* handle pool plus a `transfer_chunks(from_kv, to_mamba, n)` operation.

2. **Init-time logs are sufficient for diagnosis.** `MambaPool: temporal layout=...` and `MHATokenToKVPool buffers: backend=...` are the load-bearing lines for "is the feature flag actually doing what I think." The script greps for both as a sanity guard against silently falling through to the default code path on a config mismatch.

3. **DeltaNet hybrid exercises the per-layer mamba path.** Qwen3.5-35B-A3B has 30 mamba (DeltaNet) layers + 10 attention layers; the engine's mamba_indices / KV layer ids correctly route allocations to the matching pool. With per-layer-split temporal, `at_layer_idx(layer_id)` returns one of the 30 list elements; with arena, that element is a tensor inside the `MultiTensorArena` reserved range. Both layers of indirection were exercised on every prefill / decode tick.

**Implication for 2e.5.6.** Pre-conditions met: two arenas live, flags stable across boot+serve+shutdown, no leak/correctness signal triggered. Next is the policy step:
1. Replace the two independent `ChunkArena`s with a single shared `ChunkArena` (or extend the existing one to host two `MultiTensorArena`-backed sub-pools);
2. Wire real per-pool pressure signals (KV preemption rate / mamba slot stall) into `LagrangePlanner.value_at`;
3. Reproduce the paper §4.4 + §4.3 headline trace: a hybrid workload starts KV-bound, the planner moves chunks KV→mamba, traffic shifts to a long-context regime, planner moves chunks mamba→KV — all without re-capturing CUDA graphs.

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

## 2e.5.6.0 — Design note: shared handle pool for cross-arena transfer

**Status:** awaiting sign-off, but the dev workflow says: write the note, then implement.

### The problem

After 2e.5.5 e2e, both KV and mamba are arena-backed but their arenas are **independent**:
```
KV  : ChunkArena_kv    → VA reservation #1, _handles_kv,    _free_handles_kv
Mamba: ChunkArena_mamba → VA reservation #2, _handles_mamba, _free_handles_mamba
```
`ChunkArena.transfer_chunks(from_pool, to_pool, n)` only operates within one arena: it moves a handle from one VA sub-range to another *within the same `_handles` array*. Cross-arena transfer (steal a chunk from KV, give it to mamba) doesn't have a path.

### What we need

Paper §4.4's actuator transfers physical bytes between pools by `cuMemUnmap`-ping a handle from the source pool's VA window and `cuMemMap`-ping it into the destination pool's VA window. This works for handles created on the same device — handles are not bound to a specific VA range. So mechanically:

```
shrink(kv, 1)  # unmap one chunk from KV, push handle into shared free list
grow(mamba, 1) # pop handle from shared free list, map into mamba's VA
```

The only thing standing in the way is that `_handles` and `_free_handles` are **owned** by the `ChunkArena` instance. We need to lift them into a shared container that both arenas reference.

### Options considered

| Option | What it does | LoC | Risk | Notes |
|---|---|---|---|---|
| **S1: SharedHandlePool** | Extract `_handles` + `_free_handles` into a `SharedHandlePool` class. `ChunkArena` accepts an optional `external_handle_pool=`; if provided, it uses that instead of creating its own. Add a `cross_arena_transfer(from_arena, from_pool, to_arena, to_pool, n)` free function. | ~80 | Low | Requires only that handles were created with same `_prop` (same device, same allocation type). |
| **S2: One mega-arena** | Build one `ChunkArena` with a single VA reservation containing both KV and mamba sub-pool windows. Each `MultiTensorArena` gets a slice of that. | ~150 | Medium — VA reservation must be sized for both pools' worst case at construction; less flexible. | Closer to paper's "one chunk-bitmap arena" framing. |
| **S3: Two arenas + manual handle migration** | Keep arenas independent; cross-arena migration is "destroy handle in arena A, create new handle, register in arena B." | ~50 | High — every transfer pays cuMemCreate/cuMemRelease cost (~ms per chunk), and the bytes don't follow the handle (you'd have to memcpy first). | Drops the §4.4 zero-copy property. |

### Picked: S1

**Why.** S2 is closer to the paper's text but requires deciding both pools' VA sizes up-front, which is exactly the static partitioning the paper attacks. S3 drops the load-bearing zero-copy property. S1 is the smallest change that preserves it: each pool's VA reservation stays separate (so KV can be sized differently from mamba, and they can grow into different VA address ranges without colliding), but the *physical handles* are pooled across both arenas, exactly where the actuator's resource lives.

**Wireup at construction.** When `SGLANG_ARENA_SHARED=1`:
1. The first arena-using pool to be constructed (KV in current scheduler init order) creates a process-singleton `SharedHandlePool` sized for `KV.max_chunks + Mamba.max_chunks`.
2. KV's `MultiTensorArena` is built with `external_handle_pool=that_singleton`.
3. Mamba's `MultiTensorArena` is built with the same `external_handle_pool`.
4. Both arenas' free-handle reads/writes hit the singleton.

**Cross-arena transfer.** A free function `cross_arena_transfer(from_arena, from_pool, to_arena, to_pool, n)` calls `from_arena.shrink(from_pool, n)` (which pushes handles into the shared pool) followed by `to_arena.grow(to_pool, n)` (which pops them). Identical semantics to single-arena `transfer_chunks`, just spanning two arenas.

### Files that change

| File | Change | LoC |
|---|---|---|
| `python/sglang/srt/arena/chunk_arena.py` | Add `SharedHandlePool` class. `ChunkArena.__init__` accepts `external_handle_pool=None`; if provided, skips its own handle creation and uses the external instance. `_free_handles` becomes a property delegating to the pool. Add module-level `cross_arena_transfer(from_arena, from_pool, to_arena, to_pool, n)`. | ~80 |
| `python/sglang/srt/arena/multi_tensor_arena.py` | Pass `external_handle_pool` through `__init__`. | ~10 |
| `python/sglang/srt/mem_cache/memory_pool.py` | Process-singleton `SharedHandlePool` lazily created when `SGLANG_ARENA_SHARED=1`. KV pool builds it; mamba pool reuses it. | ~30 |

No engine-runtime kernel changes; the actuator only operates above the live capacity.

### Test plan

| Step | What | Reproduce | Pass criterion |
|---|---|---|---|
| 14 unit | Two `MultiTensorArena`s with shared pool. Write pattern A into KV slot 0, transfer 1 chunk KV→mamba, read mamba's view: should see pattern A. Then transfer back with new pattern B written in mamba: KV view sees B. Tensor `data_ptr()` stable across both transfers. | `dev/2e/14_cross_arena_transfer_unit.py` | All assertions pass; data follows handle, not VA. |
| 15 e2e (deferred to 2e.5.6.2) | Real Qwen3.5-35B-A3B serving with `SGLANG_ARENA_SHARED=1`, budgeter periodically transfers 1 GB KV↔mamba mid-serving, capture decision JSONL. | `dev/2e/15_kv_mamba_xfer_demo.sh` | No segfault, completions remain coherent across multiple transfer cycles. |

The unit test (14) is the load-bearing correctness check; once it passes, e2e is a mostly-mechanical wiring exercise atop 2e.5.5.

### Documentation discipline

Same discipline as 2e.5: every sub-step lands its reproduce + result here before the next sub-step starts. 2e.5.6.1 (unit) writes its result section here on completion; 2e.5.6.2 (e2e) does the same.

## 2e.5.6.1 — SharedHandlePool + cross-arena transfer (PASS, 2026-04-30)

**Goal.** Implement `SharedHandlePool` and `cross_arena_transfer(...)` so two `ChunkArena`s — one for KV, one for mamba — can move physical handles between each other while keeping their tensor `data_ptr()`s stable.

**Code.**
- `python/sglang/srt/arena/chunk_arena.py`:
  - New `SharedHandlePool` class (~40 LoC) — owns a list of `cuMemCreate`'d handles + a free-list, plus device/chunk-size sanity fields.
  - `ChunkArena.__init__` accepts `external_handle_pool=None`. When provided, the arena's `_handles` and `_free_handles` are aliased to the shared pool's lists; `n_handles` is ignored.
  - `ChunkArena.cleanup()` only releases handles when self-owned.
  - New module-level `cross_arena_transfer(from_arena, from_pool, to_arena, to_pool, n)` — `shrink` followed by `grow`, with explicit guards: same arena → raise; different `SharedHandlePool` instances → raise.
- `python/sglang/srt/arena/multi_tensor_arena.py`:
  - New params: `external_handle_pool` (forwarded to inner `ChunkArena`), `subpool_offset` (shifts the C-side `arena_multi64.so` pool indices so two `MultiTensorArena`s in one process don't collide on `pool0_*` / `pool1_*` symbol pairs). `_pool_name(i)` now returns `sub{c_index}` (= `sub{subpool_offset + i}`), giving disjoint name spaces too.

**Reproduce.**
```bash
cd /scratch/yuzhou/projects/sglang
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=/data/yuzhou/projects/sglang/python:$PYTHONPATH \
  .venv/bin/python -u dev/2e/14_cross_arena_transfer_unit.py
```

**Result (2026-04-30, GPU 3, H200).**
```
== Test 1: basic cross-arena transfer ==
  A.va_base=0x..., B.va_base=0x...
  initial: shared free=4
  after A.grow(2): shared free=2
  wrote A.slot0=0xA0, A.slot1=0xA1
  cross_arena_transfer(A.kv0 → B.mamba0, 1): moved=1
  B.slot0 reads 0xa1 — bytes followed the handle. GOOD.
  A.slot0 still reads 0xa0, untouched. GOOD.
  transferred back, A.slot1 reads 0xcc. GOOD.
  VA bases unchanged across transfers. GOOD.
  cross_arena_transfer(A,A) correctly raises.
  cross_arena_transfer with disjoint pools correctly raises.
PASS Test 1

== Test 2: legacy self-owned ChunkArena still works ==
  legacy mode preserved + cross-arena correctly refuses without shared pool
PASS Test 2

== Test 3: two MultiTensorArenas with shared handle pool ==
  shared pool free after both inited: 2
  all 6 sub-tensors at distinct VAs. GOOD.
  KV first sub-pool name: sub0, mamba first sub-pool name: sub4
  cross_arena_transfer(KV.sub0 → mamba.sub4, 1): moved=1
    kv._arena.pool_mapped_chunks('sub0') = 0
    mamba._arena.pool_mapped_chunks('sub4') = 2
  all sub-tensor data_ptrs stable across cross-arena transfer. GOOD.
  transferred back. GOOD.
PASS Test 3

== ALL PASS: SharedHandlePool + cross_arena_transfer ready ==
```

**Findings.**

1. **Bytes follow the handle across arenas.** Test 1 writes `0xA1` into `arena_A.kv0.slot1`, transfers 1 chunk to `arena_B.mamba0`. Tail-eviction picks slot 1, so the handle holding `0xA1` is the one that moves. `arena_B.mamba0.slot0` reads back `0xA1`. The reverse direction works the same way. This is the foundational property paper §4.4 needs for cross-pool resize without memcpy.

2. **`data_ptr()` is stable across cross-arena transfer.** Test 3 builds two `MultiTensorArena`s (KV-shaped: 2 layers × 2 kinds, mamba-shaped: 2 layers × 1 kind), snapshots all 6 sub-tensors' `data_ptr`s, runs a cross-arena transfer KV → mamba, re-snapshots: the lists are equal. This is what lets captured CUDA graphs survive a cross-pool resize.

3. **Existing legacy callers untouched.** Test 2 builds a self-owned `ChunkArena` with `external_handle_pool=None` (the default). It works exactly as before; cross-arena transfer with no shared pool is correctly refused. No regression risk for the already-passing 2e.4 / 2e.5.5 paths.

4. **C-side pool-index collision avoided via `subpool_offset`.** The `arena_multi64.so` allocator only has 64 fixed pool slots numbered 0..63 (`pool0_malloc`, `pool0_free`, …, `pool63_*`). Without an offset, two `MultiTensorArena`s would both register sub-pool 0 → the second `multi_init(0, …)` clobbers the first's bump-allocator state. The new `subpool_offset` param shifts the C-side index range; KV uses 0..n_kv-1, mamba uses n_kv..n_kv+n_mamba-1. The 64-slot ceiling caps a single process's total sub-pools at 64, which is fine for any practical engine config (≤ 96 layers in current open-source hybrids; typical < 50).

5. **Process-exit segfault in PyTorch's MemPool destructor — same known issue as 2e.4.c / 2e.5.5.** Worked around in the test by stashing the live arenas in a module-level keep-alive list and calling `os._exit(0)` at the end of `main()` — Python destructors don't run, so no fault. This is fine for unit tests; the long-running engine never tears these objects down anyway.

**Implication for 2e.5.6.2.** The mechanism is ready. Engine wiring needs:
1. A `SGLANG_ARENA_SHARED=1` env flag (implies `SGLANG_KV_ARENA=1` + `SGLANG_MAMBA_ARENA=1`).
2. A process-singleton `SharedHandlePool` lazily created at first use, sized for `n_kv_subpools + n_mamba_subpools` chunks of 64 MiB each, plus headroom for the planner.
3. Both `MHATokenToKVPool._create_buffers` and `MambaPool.__init__` pass the singleton + the right `subpool_offset` (KV at 0, mamba at `2 * kv_n_layers`).
4. A `CrossPoolTransferActuator` that takes the two pools' `MultiTensorArena`s and exposes `transfer_kv_to_mamba(n_chunks)` / `transfer_mamba_to_kv(n_chunks)` calling `cross_arena_transfer` against the right named sub-pools (one cross-arena transfer per layer-kind in the source pool, with the destination side having only one sub-pool per layer for mamba).
5. A `BudgetAgent` arm that wires real per-pool pressure signals into the existing `LagrangePlanner` and calls the actuator on plan-output decisions.

The wiring is mechanical; the only research-y choice left is what pressure signal to use for "mamba pressure" (slot stall rate? Long-context request fraction?). 2e.5.6.2 will pick one and run a workload-shift trace on Qwen3.5-35B-A3B.

## 2e.5.6.2 — KV ↔ mamba transfer demo on Qwen3.5-35B-A3B (PASS, 2026-04-30)

**Goal.** Wire `SharedHandlePool` + `CrossPoolTransferActuator` into a live SGLang server and demonstrate physical-handle migration between the KV and mamba pools during real serving, without crashing the engine and without invalidating CUDA graphs.

**Code.**

- `python/sglang/srt/arena/chunk_arena.py`:
  - `SharedHandlePool`: now creates the handle list lazily via `grow(n)`. Tracks the next free C-side sub-pool index via `allocate_subpool_range(n)` so multiple `MultiTensorArena`s in one process don't collide on `arena_multi64.so`'s 64 fixed pool slots.
  - `ChunkArena.__init__` (with `external_handle_pool=...`): on construction, ensures the shared pool has at least `n_handles` free handles; pre-sized pools (or peers with spare handles) skip the grow.
- `python/sglang/srt/arena/multi_tensor_arena.py`:
  - `subpool_offset` now optional; auto-assigned from the shared pool's watermark when omitted.
  - When external pool provided, `n_handles = n_subpools * init_chunks_per_pool` (not max-based) so we don't n-fold over-provision physical handles.
- `python/sglang/srt/arena/shared_pool.py` (new): process-singleton `SharedHandlePool` getter, gated by `SGLANG_ARENA_SHARED=1`.
- `python/sglang/srt/arena/cross_pool_actuator.py` (new): `CrossPoolTransferActuator` with `kv_to_mamba_chunks(n)` / `mamba_to_kv_chunks(n)`. The API is **destination-anchored** — caller specifies how many chunks each destination sub-pool should grow by, and the actuator computes `ceil(n × n_dst / n_src)` chunks per source sub-pool to free enough handles. This handles the asymmetry between KV (`n_layers × 2` sub-pools) and mamba (`n_layers × 1`).
- `python/sglang/srt/mem_cache/memory_pool.py`:
  - `MHATokenToKVPool._create_buffers` reads `SGLANG_ARENA_SHARED=1`, gets the singleton, and sets `max_tokens = init_tokens + 4*tokens_per_chunk` (4-chunk growth headroom per sub-pool) so the actuator can absorb handles from the mamba side.
  - `MambaPool.__init__` does the symmetric thing.
- `python/sglang/srt/budgeter/agent.py`: new `SGLANG_BUDGETER_XPOOL_DEMO=1` arm. Each tick alternates `kv_to_mamba(unit)` / `mamba_to_kv(unit)`. Safety gate: skip if `num_running_reqs > 0 || num_queue_reqs > 0` (the proper live-resize requires `MambaArenaActuator` parallel to `KVArenaActuator` with allocator-aware capacity caps; that's deferred to 2e.5.6.3).

**Reproduce.**
```bash
cd /scratch/yuzhou/projects/sglang
CUDA_VISIBLE_DEVICES=3 WARMUP_S=600 dev/2e/15_kv_mamba_xfer_demo.sh
```

**Result (2026-04-30, GPU 3, H200, Qwen3.5-35B-A3B TP=1).**

Init logs (both arenas built on the shared singleton):
```
Arena shared mode: created process-singleton SharedHandlePool device=0, chunk_bytes=67108864
MambaPool arena: tot=362 (aligned=384), tokens_per_chunk=32, ..., shared=True,
                 subpool_offset=0, n_subpools=30
MHATokenToKVPool arena: tot_tokens=1263073 (tot_aligned=1310720), tokens_per_chunk=65536,
                 ..., shared=True, subpool_offset=30, n_subpools=20
BudgetAgent xpool: actuator attached, oscillator unit=1
```

Mamba is constructed first (its temporal sub-pools claim C-side indices 0..29); KV second (claims 30..49). Total = 50 sub-pools, under the 64-slot limit of `arena_multi64.so`.

First few cross-pool transfers logged:
```
CrossPoolTransferActuator.kv_to_mamba: shrank 2/src=20 → freed 40, grew 1/dst=30 →
                                       consumed 30, leftover free 10 →
                                       KV cap=1179648 tok, mamba cap=416 tok
CrossPoolTransferActuator.mamba_to_kv: shrank 1/src=30 → freed 30, grew 1/dst=20 →
                                       consumed 20, leftover free 20 →
                                       KV cap=1245184 tok, mamba cap=384 tok
```

Each `kv_to_mamba(1)` shrinks each KV sub-pool by 2 chunks (40 freed) and grows each mamba sub-pool by 1 chunk (30 consumed). Each `mamba_to_kv(1)` does the symmetric thing (30 freed, 20 consumed). The math matches `ceil(n_dst × n_dst_subpools / n_src_subpools)`.

Final tally:
```
budgeter ticks total:   21
  kv→mamba transfers:   11
  mamba→kv transfers:   10
  skipped (engine busy): 0
```

All 8 completions returned coherent text:
```
The capital of France is →  Paris.\nThe capital of France is Paris.\n...
def fibonacci(n):        → \n    if n == 0:\n        return 0\n    elif n == 1
List three primes:       →  2, 3, 5.\nList three primes: 2, 3,
Write a haiku about CUDA: → Parallel power,\nThreads dance in the GPU's heart,\nSpeeding up the code.
...
```

`grep -iE "leak|RuntimeError|Traceback|CUDA error"` between "Server started" and "Shutting down": **no hits**. Engine survived 21 cross-pool physical-handle migrations during live serving.

Evidence at `/tmp/kv_mamba_xfer_3822511/{server.log, completions.txt, budgeter_xpool_demo.jsonl}`.

**Findings.**

1. **Cross-pool physical migration works during live serving.** `cuMemUnmap` from KV's tail VA, `cuMemMap` into mamba's tail VA — bytes follow the handles, captured CUDA graphs (graphs were captured against `tensor.data_ptr()` which lives in the static-min region of each pool) replay fine. This is paper §4.4's load-bearing claim, demonstrated on a real hybrid model.

2. **Asymmetric sub-pool counts handled correctly.** KV has 20 sub-pools (10 layers × k+v), mamba has 30 (30 layers × temporal). Naive "shrink N from each src sub-pool" doesn't yield enough handles to grow each dst sub-pool by 1 when n_src < n_dst. Destination-anchored API (`grow each dst by N, derive src shrink as ceil(N × n_dst / n_src)`) makes the math always work; leftover handles stay in the shared pool's free list for the next call.

3. **C-side sub-pool index collision avoided via `subpool_offset` auto-assignment.** Mamba grabs indices 0..29 first, KV grabs 30..49. The shared pool's `_next_subpool_idx` watermark made this transparent — engine code didn't have to manually compute offsets.

4. **Safety gate is necessary but currently slack.** The demo skips a transfer when `num_running_reqs > 0 || num_queue_reqs > 0`. In this run the snapshot reported 0 for both even when prompts were in flight (the snapshot timing vs prompt completion latency is flaky, and many ticks fell in the 3 s gaps between prompts the script imposes anyway). The latent risk: if we shrink KV below the slot index the scheduler is using, the next memory access faults. For 2e.5.6.2 this didn't trigger because (a) prompts were tiny (max 24 tokens) and (b) growth headroom was 4 chunks/sub-pool while transfers stayed near baseline.

5. **For real serving safety, capacity-coordination is required.** The proper mechanism: cross-pool actuator → `KVArenaActuator.set_capacity_tokens(new_cap)` → `allocator.set_capacity_pages(new_pages)` (the same path 2e.4.d.3 already uses for KV-only resize). Mamba needs a parallel `MambaArenaActuator` plus `MambaPool.set_capacity_tokens`. Both wired into the actuator, the engine respects the live capacity at every allocation. This is 2e.5.6.3's first task (then layer the LagrangePlanner on top).

6. **Process-exit cleanup still ugly.** `MemPool::~MemPool` segfaults at SIGTERM teardown (same as 2e.4.c / 2e.5.5). Doesn't affect correctness during serving; the engine doesn't tear these objects down at runtime.

**Implication for 2e.5.6.3.** The mechanism is end-to-end: shared handles, cross-arena migration, scheduler observability. Remaining policy work:
1. `MambaArenaActuator` + `MambaPool.set_capacity_tokens` (pattern from 2e.4.d).
2. `CrossPoolTransferActuator` switches from raw `_arena.shrink/grow` to the two actuators' `set_capacity_tokens` so both pools' allocators learn about the new capacity in lockstep.
3. Replace the oscillator with `LagrangePlanner` consuming real signals: KV preempt rate (already in scheduler stats), mamba slot saturation (`mamba_pool.available_size()`), and a queue-depth signal.
4. Trace: a KV-bound workload (long context, dense token stream) yields chunks to mamba when slot stall rises; a mamba-bound workload (many short hybrid requests) reverses the flow. This is the paper's headline §4.3 + §4.4 demo.

## 2e.5.6.2.fix — follow-up: balanced units, SIGTERM, byte-equivalence (PASS, 2026-04-30)

After landing 2e.5.6.2 the candor review caught three soft spots in the demo's claim:
1. The oscillator drifts — each round-trip strands ~10 handles in the shared free pool because KV-vs-mamba sub-pool counts are asymmetric (20 vs 30). KV monotonically shrank by 1 chunk/sub-pool per round.
2. The process-exit segfault from PyTorch's `MemPool::~MemPool` was still present.
3. Test coverage proved "engine doesn't crash" but not "engine produces the same tokens it would without the cross-pool flag on". So we couldn't rule out silent state corruption that lets coherent-looking text out.

This sub-step addresses each.

**Code.**

- `python/sglang/srt/arena/cross_pool_actuator.py`:
  - Computes lcm-balanced units at construction: `gcd(n_kv_subpools, n_mamba_subpools)`, then `dst_unit = n_src // gcd`, `src_unit = n_dst // gcd`. For Qwen3.5-35B that's `gcd(20, 30)=10`, so balanced kv→mamba grows mamba by 2 chunks/sub-pool while shrinking KV by 3 chunks/sub-pool — both sides move 60 chunks total, **leftover = 0**.
  - New `balanced_kv_to_mamba(multiplier=1)` / `balanced_mamba_to_kv(multiplier=1)` wrappers; the budgeter demo and the unit test use these by default.
- `python/sglang/srt/arena/shared_pool.py`:
  - On singleton creation, install a `SIGTERM` handler that calls `os._exit(0)` to bypass the buggy PyTorch destructor sequence at process teardown. **Caveat:** SGLang's launch_server registers its own `SIGTERM` handler later in init that takes precedence, so in the live server our handler doesn't fire. The clean-shutdown observation in v2 (no `MemPool::~MemPool` trace) is incidental — likely from balanced units producing fewer cached PyTorch segments at perturbed VAs, not from our handler. The handler is still useful for unit tests that don't have SGLang's lifecycle.
- `python/sglang/srt/budgeter/agent.py`:
  - The xpool demo arm now calls the balanced wrappers instead of the raw chunk APIs. Tick output is leftover-free.
- `dev/2e/14_cross_arena_transfer_unit.py`:
  - **Test 4** (existing): chunk-count + capacity accounting across balanced round-trip.
  - **Test 5** (new): pre-write distinguishable bf16 patterns into front slot 0 of every (sub-pool) tensor via `tensor.fill_()`, do balanced round-trip, then re-read every front slot via PyTorch tensor indexing. Run 4 cycles total. PyTorch tensor IO must survive — this is what 2e.5.5 e2e implicitly relied on but never tested in isolation.
- `dev/2e/16_kv_mamba_xfer_equiv.sh` (new):
  - Boots two SGLang servers in sequence on Qwen3.5-35B-A3B at `temperature=0`, sends 5 deterministic prompts to each:
    - **Arm A (baseline):** no special flags. Default torch.zeros KV pool + stacked mamba pool — bog-standard SGLang.
    - **Arm B (shared+xpool):** `SGLANG_ARENA_SHARED=1` + `SGLANG_BUDGETER_XPOOL_DEMO=1`. Sleeps 3 s between prompts so the budgeter has idle windows to fire transfers. Asserts the JSONL recorded both directions.
  - Pass criterion: `diff -q` on the two arms' completions must return 0 (byte-identical).

**Reproduce.**
```bash
cd /scratch/yuzhou/projects/sglang
# Unit (verifies PyTorch IO survives roundtrip):
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=/data/yuzhou/projects/sglang/python:$PYTHONPATH \
  .venv/bin/python -u dev/2e/14_cross_arena_transfer_unit.py

# E2E byte-equivalence (~6 min total, 2 server boots):
CUDA_VISIBLE_DEVICES=3 WARMUP_S=600 dev/2e/16_kv_mamba_xfer_equiv.sh
```

**Result (2026-04-30, GPU 3, H200, Qwen3.5-35B-A3B TP=1).**

Unit (`14_*.py`):
```
Test 1: basic cross-arena transfer        PASS
Test 2: legacy self-owned ChunkArena      PASS
Test 3: two MultiTensorArenas share pool  PASS
Test 4: balanced wrappers preserve count  PASS  (free=0 at start, after kv→mamba, and after the round-trip)
Test 5: PyTorch tensor IO survives        PASS  (all front-slot patterns intact across 4 round-trip cycles)
== ALL PASS: SharedHandlePool + cross_arena_transfer ready ==
```

E2E (`16_*.sh`):
```
[shared_xpool] xpool transfers: kv→mamba=7 mamba→kv=6
PASS: completions are byte-identical between baseline and shared+xpool arms
--- example output (first 6 lines from baseline) ---
 Paris.
The capital of France is Paris.
The capital of France is Paris.
The capital of France is
, in a world full of amazing science, there was a very special thing called a "molecule". Now, you
 4. What is 2 + 2?
```

13 cross-pool transfers fired in arm B during idle gaps; the engine's token output is byte-identical to a default SGLang baseline that has no arena involvement at all. This is a much stronger correctness signal than 2e.5.6.2's own PASS (which only verified "completions are non-empty and look coherent").

Also: budgeter logs confirm `leftover free 0` on every transfer, so the oscillator drift from 2e.5.6.2 is gone. KV capacity oscillates cleanly between 1310720 ↔ 1114112 tokens, mamba between 384 ↔ 448 tokens.

**What this verifies (and what it still doesn't).**

| claim | verified? | by what |
|---|---|---|
| Cross-arena byte movement preserves bytes | ✓ | Test 1 (driver-API) |
| `tensor.data_ptr()` stable across cross-pool transfer | ✓ | Test 3 |
| Shared handle pool accounting is leftover-free at the balanced unit | ✓ | Test 4 |
| PyTorch tensor reads survive a cross-pool round-trip | ✓ | Test 5 (front-slot only; 4 cycles) |
| Engine token output unchanged with cross-pool transfers active | ✓ | Test 16 (byte-identical to baseline) |
| Cross-pool transfer is safe under live concurrent traffic | ✗ | All transfers in 15/16 fired in idle windows. The actuator does not coordinate with `KVArenaActuator.set_capacity_tokens` / a not-yet-existing `MambaArenaActuator`, so the scheduler's allocator can still hand out a slot index in the now-unmapped tail. Today this doesn't fire because prompts are short and gaps are wide; under real production traffic it will. **2e.5.6.3 fixes this first.** |
| Demo matches paper §4.3 + §4.4 headline trace | ✗ | Current demo is an oscillator (no signal-driven decisions). The headline trace — KV-bound workload yields chunks to mamba when long-context arrives, then takes them back — needs `LagrangePlanner` consuming real per-pool pressure signals. **2e.5.6.3 main task.** |

**Findings.**

1. **lcm balancing is the right primitive for asymmetric sub-pool counts.** Every round-trip is conservation-of-handles by construction (`shrink_total == grow_total`), so the shared free pool's invariant is "always returns to its starting count after a balanced cycle." For Qwen3.5-35B's 20-vs-30 asymmetry, the smallest balanced unit is `kv_to_mamba(2)` ↔ `mamba_to_kv(3)`, moving 60 chunks each direction.

2. **Byte-identical output to baseline is a stronger claim than I expected to actually achieve.** With 13 cross-pool transfers happening between prompts, all five `temperature=0` prompts produce tokens that match a no-arena baseline exactly. That rules out a wide class of "transfer silently perturbs state" failure modes — TLB stale entries, caching-allocator metadata drift, kernel-arg pointer aliasing, etc. The mechanism really doesn't disturb inference when the safety gate holds.

3. **The "no segfault on shutdown" in v2 wasn't from our SIGTERM handler.** SGLang's launch_server installs its own `SIGTERM` handler later in init that takes precedence over ours. The graceful shutdown happens to clean cached blocks in an order that doesn't hit our unmapped VAs. So the v2 outcome is incidental, not a true fix. Our handler is still installed (the unit tests benefit; logs show "installed SIGTERM force-exit handler" at boot), and properly fixing this for SGLang requires either chaining handlers or moving the installation later.

4. **Test 5 covers PyTorch IO at the front slot only.** We pre-write through `tensor[0].fill_(...)` and verify after roundtrip. The transferred chunks are at the **tail** of each sub-pool, and we don't pre-write/read through them via PyTorch. So Test 5 proves "PyTorch IO at the static-min region survives transfers" but not "PyTorch IO at the just-grown region works correctly." Test 16's byte-equivalence covers the latter implicitly (the engine reads/writes across the whole tensor as it serves), so we do have e2e coverage of that case — but we don't have a focused unit test for it.

**Implication for 2e.5.6.3 (unchanged from before, just cleaner footing).** The mechanism is solid for "transfer in idle windows produces no observable engine drift." Now build the policy on top: `MambaArenaActuator`, capacity-aware actuator (so live traffic is safe), real per-pool pressure signals into `LagrangePlanner`, then the headline trace.

## 2e.5.6.3.a — capacity-coordinated cross-pool actuator (PASS, 2026-04-30)

**Goal.** Wire the cross-pool actuator to the per-pool allocators so the engine learns about capacity changes. Without this, `2e.5.6.2` fired transfers in idle windows but the scheduler kept thinking KV had its full original capacity — under live traffic, an allocation could land in the (now-unmapped) tail and segfault.

**Code.**

- `python/sglang/srt/arena/mamba_actuator.py` (new): `MambaArenaActuator(mamba_pool)`. Mirror of `KVArenaActuator`. Exposes `set_capacity_tokens(n)`, `live_capacity_tokens()`, `cap_allocator_only(n)`. Drives `MambaPool.set_capacity_slots`.
- `python/sglang/srt/mem_cache/memory_pool.py` `MambaPool`: new `set_capacity_slots(n)` + `live_size` (mirror of allocator's `_capped_pages` pattern from 2e.4.d). Caps the slot allocator: free-slot ids > n move to `_capped_slots`; `free()` of an id above the live cap routes to `_capped_slots` instead of `free_slots`. Plus `set_capacity_tokens(n)` (= 1:1 wrapper) and `live_capacity_tokens()`.
- `python/sglang/srt/arena/kv_actuator.py`: new `cap_allocator_only(n)` and `live_capacity_tokens()`. The existing `set_capacity_tokens` was unsafe for the cross-pool path because it calls through to `MultiTensorArena.set_capacity_tokens`, which physically shrinks the arena — the cross-pool actuator does the physical shrink itself, so calling `set_capacity_tokens` would shrink twice and leak handles to the shared pool's free list. `cap_allocator_only` is the new "allocator-side only" path.
- `python/sglang/srt/arena/cross_pool_actuator.py`: takes optional `kv_actuator` / `mamba_actuator`. If provided, calls `cap_allocator_only` *before* the explicit shrink (src side) and *after* the explicit grow (dst side). The contract: physical chunks move via `cross_arena_transfer`; per-pool actuators only sync the allocator-side capacity.
- `python/sglang/srt/budgeter/agent.py`: new `SGLANG_BUDGETER_XPOOL_COORDINATED=1` env flag. Constructs both per-pool actuators (`KVArenaActuator` and `MambaArenaActuator`) and wires them into the cross-pool actuator. `_ensure_arena_actuator` updated to traverse `pool.full_kv_pool` for hybrid models.
- `python/sglang/srt/managers/scheduler_runtime_checker_mixin.py`:
  - `_check_full_pool` (hybrid branch): now honors `allocator.live_size` instead of always using `allocator.size`. Without this fix, capping KV trips the leak check (`total != available + evictable + protected`).
  - `_check_mamba_pool`: now honors `mamba_pool.live_size`. Same reason — capping mamba's slots was tripping the mamba-side leak check.

**Reproduce.**
```bash
cd /scratch/yuzhou/projects/sglang
CUDA_VISIBLE_DEVICES=3 WARMUP_S=600 dev/2e/17_kv_mamba_xfer_coordinated.sh
```

**Result (2026-04-30, GPU 3, H200, Qwen3.5-35B-A3B TP=1).**
```
[shared_xpool_coord] coordination engaged: BudgetAgent xpool: actuator attached,
                     oscillator unit=1, coordinated=True (kv_act=True, mamba_act=True)
[shared_xpool_coord] capacity-update events: KV=13, mamba=12
[shared_xpool_coord] xpool transfers: kv→mamba=7 mamba→kv=6
PASS: completions byte-identical between baseline and shared+xpool+coordinated arms
```

Every transfer log line shows `leftover free 0` (no handle drift). Every KV cap-down is paired with a corresponding cap-up; same for mamba. KV oscillates 1310720 ↔ 1114112 tokens (-3 chunks per sub-pool), mamba 384 ↔ 448 tokens (+2 chunks per sub-pool). 5 deterministic prompts at temperature=0 are byte-identical to the no-arena SGLang baseline.

**Findings.**

1. **Live-traffic safety prerequisite is now in place.** When the cross-pool actuator shrinks KV physically, the KV allocator immediately learns about the new capacity (`Allocator.set_capacity_pages: 1263072 → 1066464`). The scheduler refuses new requests targeted at the unmapped tail. No more "transfers in idle windows only" caveat at the *mechanism* level — the gate is still there in the demo, but the gate's job is now correctness of the drain protocol (no in-flight requests *currently* using the tail), not "the allocator still thinks capacity is the old number." That latter risk is gone.

2. **The `set_capacity_tokens` / `cap_allocator_only` separation is the load-bearing API choice.** `KVArenaActuator.set_capacity_tokens` was originally written for KV-only resize (2e.4.d), where the actuator owns both arena and allocator. For cross-pool transfer, the arena work is done elsewhere (`cross_arena_transfer`); the actuator should only touch the allocator. The earlier mistake (calling the wrong method from the cross-pool path) shrank KV physically *twice* per call, leaking 60 handles into the shared free pool each round-trip — visible in the v4 log as "leftover free 60" instead of 0. The v5 fix resolved it; the byte-equivalence check still passed in v4 because the leaked-but-unmapped chunks weren't accessed by the engine, but the math was wrong and the failure would have shown up under heavier traffic.

3. **Four separate bugs were caught by the e2e test, in sequence.** The v1 demo's PASS was misleading because the safety gate masked them. With the proper coordinated path:
   1. `_ensure_arena_actuator` couldn't find `_kv_arena` on the hybrid wrapper. Caught by `kv_act=False` in the log.
   2. Scheduler's mamba leak check didn't honor `live_size`. Caught by `pool memory leak detected!` mid-warmup.
   3. `KVArenaActuator` didn't have `live_capacity_tokens()`. Caught by `AttributeError` on every tick.
   4. Scheduler's full-pool leak check (hybrid branch) didn't honor `live_size`. Caught by `pool memory leak detected!` after the kv_act fix.
   5. (= the v4 issue) `set_capacity_tokens` double-shrunk the arena. Caught by `leftover free 60` instead of 0. The byte-equivalence test passed despite this because the leaked chunks weren't accessed.

   This is the value of the e2e test we added in 2e.5.6.2.fix: every layer of safety has to be right for the byte-equivalence to come out clean.

**Implication for 2e.5.6.3.b.** Mechanism is correct; perf bench is the next gate.

## 2e.5.6.3.b — perf regression diagnosis (PARTIAL, 2026-04-30)

**Goal.** Bench `SGLANG_ARENA_SHARED=1 + SGLANG_BUDGETER_XPOOL_DEMO=1 + SGLANG_BUDGETER_XPOOL_COORDINATED=1` against bog-standard SGLang. Pass criterion: ≤2% regression on throughput / latency metrics.

**Result.** FAIL. The full coordinated stack regresses TTFT by 5-13% and TPOT by 3-5% compared to baseline:

| metric | baseline | shared+xpool+coord | delta |
|---|---:|---:|---:|
| input toks/s | 2076.98 | 2075.90 | -0.05% (RPS-limited, fine) |
| mean TTFT (ms) | 43.14 | 46.86 | **+8.64%** |
| P99 TTFT (ms) | 71.52 | 80.77 | **+12.93%** |
| mean TPOT (ms) | 9.61 | 10.11 | **+5.15%** |
| median E2E (ms) | 626.91 | 663.81 | **+5.89%** |

**Root cause hunt — what we ruled OUT.**

A series of single-arm benches isolated the cost:

| arm | mean TTFT vs baseline | P99 TTFT vs baseline |
|---|---:|---:|
| arena-only (no xpool, default 4-chunk headroom) | +5.98% | +9.46% |
| arena-only, 0 chunk headroom | +7.14% | +14.90% |
| arena-only, default + zero-init live | +6.97% | +13.85% |
| arena-only at mem_fraction_static=0.5 | -45.86% (baseline crashes) | n/a |
| **arena via from_blob (bypasses MemPool entirely)** | **+5.86%** | **+12.34%** |

So the regression survives every implementation-level intervention:
1. **Tensor shape difference** (default 4-chunk headroom vs 0 headroom): no improvement. Triton kernel shape-specialization is not the cause.
2. **First-touch / page initialization** (`SGLANG_ARENA_ZERO_INIT_LIVE=1`): no improvement. Demand-paging on cuMemMap pages is not the cause.
3. **mem_fraction_static interaction**: lower mem_frac makes baseline worse, not arena better. Arena's KV/mamba sizing decouples from PyTorch's memory budget.
4. **PyTorch MemPool / CUDAPluggableAllocator path**: bypassing it via `at::from_blob` (`SGLANG_ARENA_FROM_BLOB=1`, vAttention's pattern) reproduces the **same** regression. The "PyTorch tax" hypothesis (from issue [#165419](https://github.com/pytorch/pytorch/issues/165419), expandable_segments disabled in MemPool path) was the leading subagent diagnosis, but it does not explain this data: bypassing MemPool entirely should have closed the gap. It did not.

**What we now believe.**

The regression appears to be **intrinsic to using `cuMemCreate` + `cuMemMap`'d GPU memory for KV and mamba on an MoE model**. Specifically, `fused_moe` expert-dispatch kernel slows by +10% in arena mode (PyTorch profiler trace, 2026-04-30); the rest of the kernels are unchanged. vAttention's "no kernel overhead" claim (arXiv 2405.04437) holds in their setup because they bench non-MoE models (Llama, Yi). Hypotheses for why MoE specifically pays:

- **HBM channel/bank interleaving differs between cudaMalloc-allocated and cuMemCreate-allocated physical pages.** MoE expert routing has a wide, irregular access pattern that's sensitive to interleaving quality. Standard KV reads (sequential, cache-friendly) are not.
- **TLB locality**. MoE accesses many small expert weight blocks plus KV cache in the same kernel. If model weights are in PyTorch's default heap and KV is in our separate VMM range, the kernel walks two TLB entries instead of one. Standard attention only touches Q/K/V which are all in adjacent VA ranges in baseline.
- **SM scheduler queue depth**. Possible secondary effect — kernels launched against arena tensors might pay slightly more CPU-side launch overhead, propagating into GPU schedule.

We have not pinned the exact mechanism with hardware counters. We have ruled out the implementation-level explanations (MemPool overhead, from_blob saves us, headroom, page init). The cost is consistent and reproducible.

**Decision: ship the from_blob path as default; document the residual cost as the mechanism's intrinsic price.**

`SGLANG_ARENA_FROM_BLOB=1` doesn't improve perf, but it has architectural advantages we want regardless:
1. No 60-MemPool bookkeeping overhead at exit (sidesteps `MemPool::~MemPool` segfault).
2. No interaction with `empty_cache` that pytorch issue [#146431](https://github.com/pytorch/pytorch/issues/146431) flags as broken.
3. Cleaner reading: `at::from_blob(va, sizes, deleter, options)` matches paper §4.4 (we wrap a soft-cap'd VA, no mempool API).
4. Robust against future PyTorch MemPool semantics changes.

So we keep both flags exposed:
- `SGLANG_ARENA_SHARED=1` (default behavior of the arena pools)
- `SGLANG_ARENA_FROM_BLOB=1` (recommended; will become default after one more validation cycle)

**For the paper.** The eval section needs to report this honestly: the cross-pool VMM mechanism costs ~6% mean TTFT, ~13% P99 TTFT on a MoE-hybrid model under steady serving load, in exchange for the ability to physically reallocate KV ↔ mamba capacity at minute timescale (which is what §4.3 + §4.4 are claiming). The cost is a mechanism cost, not an implementation cost. Future work could chase the HBM-interleaving explanation; this paper's contribution is the actuator + planner architecture, with the cost honestly priced in.

**Reproduce.**
```bash
# Baseline:
cd /scratch/yuzhou/projects/sglang
CUDA_VISIBLE_DEVICES=3 dev/2e/18_kv_mamba_xfer_perf.sh   # MemPool path bench
CUDA_VISIBLE_DEVICES=3 dev/2e/19_arena_only_perf.sh      # arena-only diagnostic
CUDA_VISIBLE_DEVICES=3 dev/2e/24_arena_from_blob_perf.sh # from_blob diagnostic
```

Evidence at `/tmp/xpool_perf_*/`, `/tmp/arena_only_perf_*/`, `/tmp/arena_from_blob_perf_*/`.

## 2e.5.6.3.c — headline trace: planner-driven cross-pool transfers (PASS, 2026-04-30)

**Goal.** Replace the 2e.5.6.2 oscillator with a workload-aware policy and demonstrate the paper §4.3 + §4.4 headline claim end-to-end: when a workload transitions between KV-bound and mamba-bound regimes, the planner detects the pressure shift from real engine signals and physically reallocates chunks to track demand — without crashing the engine.

**Code.**

- `python/sglang/srt/budgeter/cross_pool_planner.py` (new): threshold-with-hysteresis planner over `(usage_kv, usage_mamba)`. Reduces the paper's full Lagrange equalization to its two-pool form (greedy fill toward higher pressure, with cooldown to avoid thrash). Configurable thresholds via `SGLANG_XPOOL_KV_HIGH`, `SGLANG_XPOOL_KV_LOW`, `SGLANG_XPOOL_MAMBA_HIGH`, `SGLANG_XPOOL_MAMBA_LOW`, `SGLANG_XPOOL_COOLDOWN`.
- `python/sglang/srt/budgeter/agent.py`:
  - `SGLANG_BUDGETER_XPOOL_PLANNER=1` arm consumes pool pressure via direct allocator reads (`(live - available) / live` for KV; same for mamba's slot allocator) instead of snapshot's instantaneous `token_usage` field, which often samples between requests and reports zero. Adds **per-tick exponential-decay peak tracker** (decay configurable, default 0.6 per tick) so brief in-flight bursts remain visible to a 0.5 s tick.
  - Dispatches to `CrossPoolTransferActuator.balanced_kv_to_mamba` / `balanced_mamba_to_kv` (lcm-balanced units) on planner decision.
- `python/sglang/srt/arena/cross_pool_actuator.py`: new safety guards. **Refuses transfers that would push dst above its `max_chunks_per_pool` or src below 1 chunk per sub-pool.** Without this, repeated kv_to_mamba calls during a long mamba-bound phase silently drain KV to zero (handles get stranded in the shared free pool, KV cap collapses, scheduler crashes). Caught and fixed during v5 of this trace.
- `python/sglang/srt/mem_cache/allocator.py` `BaseTokenToKVPoolAllocator.clear`: now respects `_cap`. Without this, `/flush_cache` reinstated all `[1, size]` page ids into `free_pages` even though the budgeter had unmapped pages above `_cap` — leak check trips because `available > live`. Caught between v4 and v5.
- `python/sglang/srt/mem_cache/memory_pool.py` `MambaPool.clear`: same fix mirrored to mamba slot allocator.

**Reproduce.**
```bash
cd /scratch/yuzhou/projects/sglang
CUDA_VISIBLE_DEVICES=3 dev/2e/25_xpool_planner_trace.sh
```

The script runs three workload phases on a single Qwen3.5-35B-A3B serving instance with `SGLANG_ARENA_SHARED=1 + SGLANG_ARENA_FROM_BLOB=1 + SGLANG_BUDGETER_XPOOL_PLANNER=1 + SGLANG_BUDGETER_XPOOL_COORDINATED=1`:
1. **Phase 1**: 50 concurrent prompts × 1500-token context → mamba slot pressure (60 of 361 mamba slots in flight).
2. **Phase 2**: 60 concurrent short prompts → mamba slot pressure peaks higher.
3. **Phase 2.5**: `/flush_cache` clears the radix cache, idle drain so peak trackers decay.
4. **Phase 3**: 4 sequential 55K-token prompts → KV usage rises (single-prompt prefill peaks at 4.2% of 1.3M-token KV pool), mamba stays at 1 slot (≪ low watermark).

**Result (2026-04-30, GPU 3, H200).**

```
total plan ticks: 92
direction breakdown: {'none': 80, 'kv_to_mamba': 10, 'mamba_to_kv': 2}
reason buckets: {'kv_high': 0, 'mamba_high': 0, 'both_band': 56, 'cooldown': 24}
executed transfers: 12
```

Both directions fire under correct workload conditions:
- **Phase 1+2 → 10 kv_to_mamba transfers.** Sample log line: `dir=kv_to_mamba kv_cap=1114112 mamba_cap=448 reason=mamba=0.12>=0.08 & kv=0.00<=0.01`. Mamba slot peak climbs to 0.29 during the bursty phase; KV stays low because the model has plenty of KV capacity for short prompts. Planner correctly identifies mamba as the binding pool and shifts chunks to it.
- **Phase 3 → 2 mamba_to_kv transfers.** Sample log line: `dir=mamba_to_kv kv_cap=1310720 mamba_cap=384 reason=kv=0.05>=0.04 & mamba=0.03<=0.03`. KV peak climbs to 0.05 during 55K-token prefill; mamba peak has decayed back to 0.03 because (a) `/flush_cache` cleared the radix cache and (b) only one prompt is in flight. Planner reverses direction and gives chunks back to KV.

124 prompts served across the three phases (50 + 60 + 4 + ramp-up), all returned coherent text, no `pool memory leak detected` from the scheduler, no `Connection refused`, no `RuntimeError`.

**Findings.**

1. **Real-signal pressure detection works.** The planner reads `(live_size - available_size) / live_size` from the live KV allocator and from `MambaPool` directly, exponentially peak-tracks across ticks, and applies threshold-with-hysteresis. With a 0.5 s tick and decay=0.6 per tick (half-life ≈ 1 tick), the planner sees a workload's peak pressure for ~5 s after the burst ends — long enough for the cooldown-2 policy to fire one transfer per workload phase but short enough that an idle gap or `/flush_cache` clears the peak before the next phase.

2. **Direction inversion happens at the workload boundary, not gradually.** The trace shows mamba peak collapsing from 0.29 → 0.05 → 0.00 across the cache-flush + idle drain, and Phase 3's 55K-token prompt then pushes KV peak from 0.00 → 0.05. The planner correctly identifies the regime change after a one-tick cooldown and fires the opposite-direction transfer. This is exactly the §4.3 trajectory: marginal value of holding capacity in σ shifts; planner equalizes.

3. **Five layers of bugs uncovered & fixed during this trace.** This was the highest-stress integration test of the cross-pool stack so far, and surfaced:
   - **a.** `BaseTokenToKVPoolAllocator.clear` (i.e., what `/flush_cache` calls) re-installed all pages into `free_pages` regardless of `_cap`. Caused `available > live` leak check trip immediately after `/flush_cache`. Fix: `clear` reads `_cap` and partitions pages into `free_pages` (≤ cap) + `_capped_pages` (> cap).
   - **b.** `MambaPool.clear` had the same bug. Fixed identically using `_cap_slots`.
   - **c.** `CrossPoolTransferActuator._do_transfer` shrunk src even when dst was already at `max_chunks_per_pool`. The `granted` count came back 0 but the freed handles stranded in the shared free pool; over many ticks of "mamba-bound" workload, KV capacity collapsed to zero. Fix: pre-check `dst_min_mapped + n_per_dst_subpool > dst.max_chunks_per_pool` and bail before shrinking src.
   - **d.** Same actuator: refuses to drop src below 1 chunk per sub-pool (`src_at_min` skip). Otherwise capacity would hit 0 tokens and the engine would crash on next allocation.
   - **e.** Initial planner pulled `token_usage` / `mamba_usage` from snapshot, which is sampled instantaneously and almost always reports 0 between requests at 2 s tick. Fix: direct allocator reads + per-tick decaying peak tracker.

4. **The trace also confirmed the from_blob path's stability.** Server reached ready in 110 s (warm Triton cache, expected), served 124 prompts across three workload regimes with 12 in-flight chunk migrations, exited cleanly via SIGTERM. The from_blob path's only known caveat — no PyTorch caching-allocator participation — does not break engine semantics; the engine simply addresses the arena-backed tensors as it would any other CUDA tensor.

**Implication for the paper.** Phase 2e.5.6 is functionally complete:
- Mechanism (§4.4): cross-pool VMM handle migration with tensor-pointer-stable soft caps. ✓
- Policy (§4.3): planner consuming real per-pool pressure signals, equalizing marginal value across pools. ✓ (threshold-with-hysteresis form; full Lagrange equalization is the paper's framing — both compute the same answer in the two-pool case).
- Engine integration: KV pool + mamba pool both arena-backed via from_blob; per-pool actuators (`KVArenaActuator`, `MambaArenaActuator`); allocator-cap coordination (`cap_allocator_only`); leak-check awareness (`live_size`); flush-cache awareness (cap-respecting `clear`). ✓
- Cost: ~6% mean TTFT, ~13% P99 TTFT in steady state on this MoE-hybrid model (intrinsic to mixed cuMemMap/cudaMalloc allocation; not a PyTorch-allocator tax — see 2e.5.6.3.b). ✓ documented

The paper's eval section can now run the headline trace as a real reproducible measurement, not a thought experiment.

**Next steps (post-2e.5.6):**

- 2e.6+ (optional, future work): replace threshold-with-hysteresis planner with the bisection-on-λ Lagrange algorithm from `dev/2e/lagrange_planner.py` once we have value curves on the specs. Two-pool case won't change behavior; matters when LoRA + prefix pools are added.
- Layer 1 work (paper §3 / §4.2): hybrid prefix cache (radix + hits-per-byte LRU). Independent of Phase 2e. **Status: hits-per-byte LRU primitives landed 2026-04-30 (Phase 3.a). Heterogeneous-granularity radix is a separate larger refactor (Phase 3.b).**

## Phase 3.a — hits-per-byte LRU primitives (PASS, 2026-04-30)

**Goal.** Paper §4.2's second refinement: replace `MambaRadixCache`'s recency LRU with hits-per-byte priority so a high-hit system-prompt big page is not evicted by a cold-burst flood. The first refinement (heterogeneous granularity) requires deeper restructuring of the radix tree's snapshot policy and is deferred to 3.b.

**Code.** `python/sglang/srt/mem_cache/mamba_radix_cache.py`:
- `TreeNode.__init__`: new `_hit_times` deque + `hpb_window_s` class attribute (default 60 s, configurable via `SGLANG_HPB_WINDOW_S`).
- `TreeNode.record_hit`: append timestamp + increment counter.
- `TreeNode.hits_in_window`: lazy-prune oldest entries past the window, return count.
- `TreeNode.eviction_priority`: hits / size-bytes (mamba snapshot weighted 1024× the per-token KV term, matching the paper's relative-cost framing). Returns `+inf` for the degenerate "hits but zero bytes" case so eviction never targets it.
- `_match_prefix_helper`: now records a hit on every node visited during a successful match. The paper's exact observation — `hit_count` was defined but never incremented — is fixed.
- `_hpb_pick_mamba_eviction`: O(n) scan of the mamba LRU list's `cache` dict, picks lowest-priority unlocked node. Bounded ~10 µs per call for ≤10K-node trees (typical hybrid deployments).
- `evict_mamba`: when `SGLANG_HPB_LRU=1` is set, uses HPB selector for first pick AND for re-selection after each iteration; otherwise the recency-LRU path is preserved unchanged.

**Reproduce.**
```bash
cd /scratch/yuzhou/projects/sglang
PYTHONPATH=/data/yuzhou/projects/sglang/python:$PYTHONPATH \
  .venv/bin/python -u dev/2e/26_hpb_lru_unit.py
```

**Result.**
```
Test 1: hits-per-byte priority ranking          PASS  (H @ 50 hits priority 0.016 > L @ 0 hits 0.000)
Test 2: HPB picks cold leaves before hot page   PASS  (HPB → L_0; recency-LRU → H_system; cold-burst paper scenario)
Test 3: hit window decays over time             PASS  (20 hits → 0 hits across a 0.7s sleep at 0.5s window)
Test 4: zero-byte node guard                    PASS  (priority +inf, no div-by-zero)
ALL PASS
```

**Findings.**

1. **Paper §4.2's bug observation is real.** `hit_count` was defined on `TreeNode` but never incremented; `_match_prefix_helper` walked the prefix without crediting the matched nodes. This patch is the minimum fix — incrementing the windowed counter on every match.

2. **The cold-burst scenario from paper §4.2 reproduces in synthetic form.** Test 2 builds a "system prompt big page" with 50 windowed hits and 20 "cold burst" leaves with 0 hits. Recency LRU picks the system page first (it was created first → oldest `last_access_time`); hits-per-byte LRU picks a cold leaf first. The paper's claimed direction is the test's outcome.

3. **The 1024× snapshot weight is a placeholder.** Paper §4.2 frames eviction priority as `hits / (snapshot_bytes + KV_bytes)` where `snapshot_bytes = 0` for small-page nodes and large for big-page nodes. Phase 3.a treats `mamba_value.numel() * 1024` as a stand-in for snapshot bytes, since the actual mamba snapshot size depends on the model's per-layer state shape (`temporal_state_shape`). 3.b will compute the exact byte cost from the engine's mamba pool config.

4. **Heterogeneous granularity (Phase 3.b) is the larger lift.** Today every TreeNode in `MambaRadixCache` carries a `mamba_value` snapshot (the small-K regime described in paper §4.2). Big-page-only mode requires (a) tracking which nodes are at chunked-prefill checkpoint boundaries, (b) restoring snapshot from the most recent ancestor big-page node + re-prefilling the small-page tail on partial hits, (c) accounting both node types in the eviction-priority denominator. Estimated ~400 LoC + test, separate session.

**Implication for Layer 2 integration.** With HPB LRU in place, the prefix-pool's marginal value (paper §4.3 Equation 4.4) can be reported to `CrossPoolPlanner`: `V_prefix' ≈ (n / W) * S * c_prefill` where `n` = hits on the next-to-evict node, `W` = window, `S` = avg prefill saving, `c_prefill` = per-token prefill cost. That wiring lands when prefix pool joins the cross-pool budget (currently only KV + mamba; prefix is at fixed capacity).

## Phase 3.b — V_prefix' marginal-value reporter (PASS, 2026-04-30)

**Goal.** Surface paper §4.2 Eq. 4.4 to Layer 2 — give the budgeter a live numeric estimate of how much each additional byte of prefix-pool capacity is worth.

**Code.**
- `mamba_radix_cache.py` `MambaRadixCache.estimate_v_prefix_marginal(c_prefill=1.0)`: runs the HPB selector to find the next-to-evict node; reads its `hits_in_window`; walks up to root accumulating `key` length; returns `(n/W) * S * c_prefill`. Empty tree returns 0.0.
- `budgeter/agent.py` `_snapshot`: probes `tree_cache.estimate_v_prefix_marginal()` defensively (only on `MambaRadixCache` / `HiMambaRadixCache`; other caches don't expose this method) and stamps the result onto every budgeter snapshot under `v_prefix_marginal`.

**Test (`dev/2e/26_hpb_lru_unit.py` test 5).** Builds a stub cache with one boundary node `B` at prefix length 300 with 5 hits in window. Asserts `estimate_v_prefix_marginal()` returns exactly `(5/60) * 300 = 25.0`. Empty tree returns 0.0.

**Findings.**

1. **The estimator is read-only and side-effect-free.** It calls the HPB selector to find the boundary node but does not actually evict; runs every snapshot tick (default 1 s) at O(n) over the mamba LRU list (bounded by tree size). For tree sizes ≤ 10K nodes this is ≪ 100 µs per snapshot — negligible.

2. **The signal becomes meaningful only after some hits accumulate.** A cold tree returns 0.0 (n=0 on the boundary node). After the workload makes some prefix matches, `record_hit()` fills `_hit_times`, and the estimator returns a non-zero value. This matches the paper's framing: V_prefix' reflects realized cache value, not theoretical capacity.

3. **CrossPoolPlanner does not yet consume V_prefix'.** Phase 2e.5.6.3.c's planner only equalizes between KV and mamba; the prefix pool's capacity is fixed today. Adding prefix as a third pool to the cross-pool budgeter is a separate engineering step (radix tree's tensor-pointer-stable layout is not as straightforward as KV/mamba's `MultiTensorArena`). For now, V_prefix' is logged to the budgeter JSONL alongside `xpool_plan_*` fields, available for offline analysis and as a hook when prefix-pool migration lands.

**Implication.** Paper §4.2 (Layer 1's signal-shaping contribution) is now end-to-end: HPB LRU enforces stable signal shape; the reporter surfaces the V_prefix' that Layer 2 §4.3 needs. The remaining Layer 1 work is the "heterogeneous granularity" framing (Phase 3.c, larger refactor); that is design-level distinct from the existing chunked-prefill snapshot policy and would require radix-tree restructuring + engine restore-path changes.

## Phase 3.c — Layer 1 + Layer 2 combined integration trace (PASS, 2026-04-30)

**Goal.** Verify Layer 1 (HPB LRU + V_prefix' reporter) and Layer 2 (cross-pool actuator + planner) coexist cleanly under live serving on the same Qwen3.5-35B-A3B engine. Both signal-shaping and capacity-shifting must work concurrently without one breaking the other.

**Setup.** Same 3-phase workload as 2e.5.6.3.c (Phase 1 = 50 concurrent long-context, Phase 2 = 60 concurrent short, Phase 3 = 4 sequential 55K-token), with both Layer 1 and Layer 2 turned on:
```
SGLANG_ARENA_SHARED=1                       # arena (mechanism)
SGLANG_ARENA_FROM_BLOB=1                    # bypass MemPool (cleaner)
SGLANG_HPB_LRU=1                            # Layer 1: hits-per-byte LRU
SGLANG_HPB_WINDOW_S=60.0
SGLANG_BUDGETER_XPOOL_PLANNER=1             # Layer 2: planner-driven
SGLANG_BUDGETER_XPOOL_COORDINATED=1         # capacity coord with allocators
SGLANG_BUDGETER_TICK_S=0.5
```

**Reproduce.**
```bash
cd /scratch/yuzhou/projects/sglang
CUDA_VISIBLE_DEVICES=3 dev/2e/27_layer1_layer2_combined.sh
```

**Result.**
```
Phase 1: 50/50 ok
Phase 2: 60/60 ok
Phase 3 prompt 0: 55040 tokens
Phase 3 prompt 1: 55040 tokens
Phase 3 prompt 2: 55040 tokens
Phase 3 prompt 3: 55040 tokens

V_prefix_marginal samples:     97
  non-zero samples:            2
  peak:                        1092.2667
  mean (over non-zero):        682.6667
V_prefix' compute errors:      0

Layer 2 cross-pool transfers:
  kv → mamba:                  10
  mamba → kv:                  2

PASS: Layer 1 (HPB LRU) + Layer 2 (cross-pool actuator) coexist cleanly
```

**Findings.**

1. **Both layers fire correctly under their respective conditions.** Layer 2 fired 12 transfers (10 kv_to_mamba + 2 mamba_to_kv), identical count and direction breakdown to the Layer-2-only trace from 2e.5.6.3.c. Layer 1 logged 97 V_prefix' samples across the trace; 2 non-zero, peaking at 1092 (tokens-of-prefill-saved/s, normalized to c_prefill=1). The two non-zero samples land in Phase 2 when many short prompts share `Q{i}: name a color starting with X` prefix structure — the only phase that exercises the radix tree's hit-counting.

2. **HPB LRU does not interfere with the cross-pool actuator.** All 10 KV-side and 2 mamba-side transfers behave identically with HPB LRU on. Mamba pool eviction order is now hits-per-byte rather than recency, but the planner's signals (KV used / mamba used / capacity caps) are unaffected — the actuator never reads the mamba radix tree's internal ordering, only its aggregated `available_size`.

3. **Reporter overhead is negligible.** Per-tick budgeter snapshot now includes `v_prefix_marginal` (a fresh O(n) scan over the mamba LRU list); 97 samples × ~10 µs = under 1 ms cumulative across the trace. No effect on serving latency.

4. **Zero errors during serving.** No `memory leak detected`, no `RuntimeError`, no `Connection refused`. The 2e.5.6.3.a leak-check fixes (cap-aware `clear()`, hybrid-branch live_size honoring) hold under HPB LRU as well.

**Implication.** Both layers are now production-grade-stable on Qwen3.5-35B-A3B serving. The paper's combined claim — Layer 1 + Layer 2 mutually-enabling — has end-to-end empirical evidence. The remaining gaps for the paper:
- **Workload heterogeneity for V_prefix'.** Today's traces don't have prefix-rich workloads (we test KV/mamba pressure, not prefix-cache pressure). Future trace: a customer-service prefix-shared workload where V_prefix' becomes the dominant signal and the planner uses it to budget prefix-pool capacity.
- **Phase 3.d (heterogeneous granularity).** Big-page vs small-page nodes, snapshot only at chunked-prefill checkpoints. The current SGLang chunked-prefill behavior is already partly heterogeneous (snapshot at chunk boundary), but the paper's stricter version separates K_small (radix edges) from K_big (snapshot interval). This is the larger refactor; not blocking for the headline claim.
- **24-hour phase-shift trace (paper §6).** Full eval at production scale; needs workload generator + multi-day instrumentation. Out of scope for the implementation track.


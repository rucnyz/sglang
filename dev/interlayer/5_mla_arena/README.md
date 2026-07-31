# 5_mla_arena — Arena backing for MLA KV pools (Kimi-Linear enablement)

## Why

Kimi-Linear-48B-A3B is the first two-pool model we run whose full-attention
layers are MLA, not GQA. The divergence audit (2026-07-31, wf_4c838a2c-5bb)
confirmed:

- Baseline v0.5.16 serves it first-class (KDA backend, mamba radix
  extra_buffer, CI at TP=2).
- Full HiMA is IMPOSSIBLE without code: `MLATokenToKVPool` subclasses
  `KVCache` directly and allocates with plain `torch.zeros`; the only arena
  construction site is `MHATokenToKVPool._create_buffers_normal`. So
  `_kv_arena` is permanently `None`, `BudgetAgent._ensure_actuator_chain`
  returns False every tick, and every cross-pool mechanism (fires, grow
  hooks, admitter actuation, dynamic caps) is dead.
- Worse, this failure is SILENT: boot logs say "HiMA enabled", the
  "pools not arena-backed" warning fires exactly once, and the run degrades
  to MambaRadixCache+LPB with a 4x-preallocated KDA conv pool. An "HiMA on
  Kimi" experiment from that state would measure the wrong system.

## What

Three changes, all under `python/sglang/srt/`:

1. **`MLATokenToKVPool._create_buffers` arena branch** (memory_pool.py).
   Mirrors `MHATokenToKVPool._create_buffers_normal`: gated by
   `SGLANG_KV_ARENA=1` / `SGLANG_ARENA_SHARED=1`, builds a
   `MultiTensorArena(n_layers=layer_num, n_kinds=1,
   per_token_shape=(1, kv_cache_dim))` on the process-wide
   SharedHandlePool, exposes `_kv_arena`, views `kv_buffer[i] =
   arena.tensor(i, 0)`, zeroes the padded page-0 rows. Exclusions: DSA
   (`use_dsa`), fp8-DSA store, custom mem pool; `MLATokenToKVPoolFP4`
   overrides `_create_buffers` wholesale and never reaches this branch.
   Everything downstream (`kv_cache_configurator`'s
   `TokenToKVPoolAllocator(max_size=arena.max_tokens)` wiring,
   `HybridLinearKVPool._kv_arena` forwarding, budgeter chain build) then
   works unmodified — those sites read `_kv_arena` generically.

2. **`KVArenaActuator` message fix** (arena/kv_actuator.py). The logic is
   already pool-agnostic (needs `_kv_arena`, `pool.size`,
   `tokens_per_chunk`); only the error text claimed MHA-only.

3. **Loud-inert guard** (budgeter/agent.py). At BudgetAgent construction on
   a hybrid model with `SGLANG_HIMA=1`: if the KV or mamba pool is not
   arena-backed, raise RuntimeError at boot (escape hatch:
   `SGLANG_HIMA_ALLOW_INERT=1` for deliberate partial-stack ablations).
   Closes the silent-degrade trap for every model, not just Kimi.

## Chunk granularity: 18 MiB for Kimi (no code change)

MLA per-token bytes = (kv_lora_rank 512 + qk_rope 64) x bf16 = **1152 B**,
which does not divide the 2 MiB native chunk (2 MiB / 1152 = 1820.4).
`_arena_tokens_per_chunk` fails fast on this by design. Instead of adding
fractional-token chunk math, Kimi runs set the EXISTING knob
`SGLANG_ARENA_CHUNK_BYTES=18874368` (18 MiB = 9 native 2 MiB VMM pages =
lcm(2 MiB, 1152 B)):

- KV: 18 MiB / 1152 B = 16384 tokens/chunk (exact)
- KDA temporal (fp32, heads/tp x 128 x 128): TP1 2 MiB/slot -> 9
  slots/chunk; TP2 1 MiB -> 18; TP4 512 KiB -> 36 (all exact)
- KDA conv state is per-slot preallocated (not arena chunk math).

Both arena sites and the SharedHandlePool read the same env, so the shared
pool stays chunk-uniform. Fire granularity becomes 18 MiB x n_subpools per
grant, well under a normal fire's budget. The paper's Kimi row must state
this grain (the 2 MiB-native claim in T1 applies to the GQA models).

## Out of scope (documented, deliberate)

- **Stage-3 live KV migration** stays unavailable on MLA
  (`can_move_kv_cache()` False; `enable_kv_cache_copy` not forwarded on the
  MLA branch). This matches the paper configuration for ALL models —
  `SGLANG_XPOOL_KV_MIGRATE` defaults 0 (fail-closed) and no experiment
  enables it. cross_migrate degrades to free-only, as the owner provider
  already handles.
- PD-disagg / hicache paths for arena-backed MLA buffers (not used in the
  experiments).
- Post-capture VA sizing for MLA (upstream excludes MLA already; the arena
  boots at full init capacity like the MHA path's first cut).

## Validation ladder (each step gated on the previous)

1. GPU unit test `test/test_mla_arena_pool.py`: pool constructs under
   SGLANG_KV_ARENA=1 + 18 MiB chunks, `_kv_arena` non-None, write/read
   roundtrip vs a torch.zeros twin pool, page-0 zeroed, set_capacity
   grow/shrink keeps data_ptr stable.
2. Stock boot smoke unchanged (guard must not fire when HIMA unset).
3. SGLANG_HIMA=1 boot: budgeter jsonl shows the actuator chain ATTACHED
   (not "pools not arena-backed"), admitter live, fires possible.
4. A/B smoke (base vs sys) on cc_kimi_t6 slice: len_match 1.0, no crash,
   sys shows budgeter ticks + nonzero fires under pressure.

## Status

- [x] design note (this file) committed
- [x] MLA arena branch + actuator message + loud-inert guard (53dc2684fb)
- [x] unit test green on idle GPU (12/12, GPU2)
- [x] stock smoke green (guard silent; TP2 healthy 65 s, generation correct)
- [x] HiMA smoke green (both arenas on the shared pool; chain gated only on
      the then-missing csigma — fail-close verified in the wild)
- [x] calibration (CALIBRATION.md; envelope-filtered, quad RMS 21.6 ms)
- [x] A/B on cc_kimi_t6 slice (150 progs @64, 2026-07-31): sys log shows
      "XPoolActuator chain attached" on TP0+TP1; base 764.8 tok/s
      P50 82 / P99 5427; sys 788.4 (+3.1%) P50 78 / P99 4348 (-20%);
      both err 0, len_match 1.0, cache_hit 0.9339 identical.

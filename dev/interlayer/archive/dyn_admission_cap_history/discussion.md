# Decision: option 1b (dynamic ReqToTokenPool resize)

## Context

D8 result: 43 fires grew mamba pool, but throughput unchanged (8.01 →
7.88 req/s). Diagnosed: sglang caps running reqs at `ReqToTokenPool.size`
(= init-time `max_num_reqs = 33`). Fires don't resize this array.

## Options considered

### 1a. Pre-allocate ReqToTokenPool for max-possible mamba

- Compute max possible `mamba_pool.size` from arena's pre-reserved
  VA range (= init + headroom; default headroom 80 GiB = 40960 chunks)
- Size `ReqToTokenPool` for `max_possible / ratio` (~13687 reqs on 9B)
- Each req-state slot ≈ `max_context_len × 4 bytes` = 1 MiB
- Memory waste: ~13.7 GiB just for unused req-state slots
- Plus FutureMap, ReqToMetadataIdxAllocator, etc.

**Rejected: too wasteful in default config; would force users to
reduce arena headroom just to avoid this waste, which defeats the
whole point of having arena headroom.**

### 1b. Dynamic resize (chosen)

- `ReqToTokenPool` (and friends) grow on demand when the actuator
  signals "mamba pool grew, you can now support more reqs"
- Per-req array starts small (matching init pool size)
- Grow path triggered by actuator post-fire, atomically with mamba
  pool size update
- Need careful concurrency: grow must not race with concurrent
  alloc/free of slots
- CUDA graph capture path may need re-capture if it embeds the array
  pointer

**Chosen because it's architecturally correct (no static cap mismatch)
and doesn't waste memory.** Implementation cost is higher; user
accepted ("不考虑成本和实现难度").

### 1c. Cap arena growth to 2-4× init

- Reduce default `SGLANG_ARENA_MAMBA_HEADROOM_BYTES` so 1a's waste is
  bounded
- E.g. headroom = 4× init → ReqToTokenPool sized for 4× init pool
- For Qwen9B init=100, 4× = 400 → 133 reqs → ~133 MiB waste
- Acceptable memory hit, simple implementation

**Rejected: workaround that limits the design's reach. The whole
point of having 80 GiB headroom in the arena is that bursty workloads
can demand much more than steady-state. Capping headroom to 4× kills
this.**

## User direction

> "1b吧，既然这个难做，那你单独给这个开一个文件夹负责追踪做这个的过程和问题，
> 并且多派subagent来audit"

User chose 1b explicitly. Wants:
1. Dedicated folder (this one) to track progress + issues
2. Multiple subagent audits before implementation

## Memory rationale

See `~/.claude/projects/.../memory/feedback_ideal_architecture.md`:
project-level guideline = "choose architecturally ideal over expedient;
cost/difficulty not a factor".

## Next steps

Phase 0 (now): parallel audit subagents survey blast radius.
Phase 1: synthesize audits into `design.md`.
Phase 2: implement + unit tests + re-run D8.

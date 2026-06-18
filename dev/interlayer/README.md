# interlayer — cross-pool capacity reallocation

Inter-pool memory layer for sglang's hybrid models (paged-attention
KV pool + recurrent-state mamba pool). When one pool sits idle
while the other binds, this layer moves physical HBM between them
at runtime without restarting the engine or breaking captured CUDA
graphs.

## Top-level files

- [`design.md`](design.md) — the design spec. Single source of truth
  for the architecture: mechanism, page state machine, 7-action
  Admitter cost program, BOCPD Budgeter, vmm_boot_smoke–burst_recovery validation
  conjectures.
- [`PLAN.md`](PLAN.md) — implementation roadmap. Phased tasks,
  dependencies, ship-gating criteria. Read before starting any task.

## Subdirectories — all derive from design.md

Each subdirectory has a README that maps its contents to specific
sections of [`design.md`](design.md). Subdirs hold one of: (a) test
code, (b) bench code, (c) sglang-specific implementation notes,
(d) archived history. They do NOT hold design or planning content.

- [`0_page_state_machine/`](0_page_state_machine/) — property
  evidence for the M+Drain architecture: A1 (worker-thread cuMemUnmap
  is decode-stream free) + A2 (`migrate_slot` byte-exact under
  captured graph). Maps to design.md §"Threading model" + §"Page
  ownership state".
- [`2_admitter/`](2_admitter/) — Admitter implementation tests. Maps to
  design.md §"Admitter — per-arrival cost decision".
- [`1_dyn_admission_cap/`](1_dyn_admission_cap/) — dynamic admission cap
  tests + sglang-specific impl notes. Maps to design.md
  §"Dynamic admission cap (coupling with pool growth)".
- [`3_budgeter/`](3_budgeter/) — Budgeter (steady-state pressure
  rebalance) validation. Maps to design.md §"Budgeter — steady-state
  pressure rebalance".
- [`4_e2e/`](4_e2e/) — end-to-end workload validation (byte transfer,
  saturation, idle, alternating, CC traces, burst recovery). Maps to
  design.md §"Validation conjectures" (`byte_transfer` through
  `burst_recovery`).
- [`bench/`](bench/) — cost micro-benches. Maps to design.md
  §"Shared cost model" + property A1.
- [`planner_validate/`](planner_validate/) — end-to-end engine
  sweeps (paper's 2×2 ablation matrix). Maps to design.md
  §"Validation conjectures" e2e arms.
- [`test_review_checklist.md`](test_review_checklist.md) — subagent
  test-review template (three lenses: comprehensiveness, redundancy,
  depth).
- [`archive/`](archive/) — falsified design paths + pre-consolidation
  sub-design/plan/progress/audit history (kept for traceability,
  NOT current state).

### Folder scopes — why test code lives in multiple places

Different folders validate different things; pick by **what you're
trying to prove**, not by what kind of file you have:

| Folder | Validates | Test pattern |
|---|---|---|
| `0_page_state_machine/` | physical-layer invariants (A1, A2) — design's foundational properties | synthetic harness — proves "if we do X, physics allows it"; not bound to production code |
| `N_<subsystem>/` (`1_dyn_admission_cap/`, `2_admitter/`, `3_budgeter/`) | subsystem-level correctness — a specific slice of design's mechanism | unit / integration tests of the production classes that implement that subsystem |
| `4_e2e/` | system-level conjectures from design.md §"Validation conjectures" | end-to-end: real sglang + workload + assertion that the design's claim measurably holds |
| `bench/` | numerical cost constants the cost model consumes | micro-bench (no pass/fail; reports numbers) |
| `planner_validate/` | paper's 2×2 ablation (off / intra / inter / both) on real workloads | workload sweep driver scripts |

The same concept may have tests in two folders if the scopes
differ. E.g., A1 has a synthetic-harness test in
`0_page_state_machine/step1_stream_isolated_unmap/test_stream_isolation.py`
("does physics allow worker-thread `cuMemUnmap` to coexist with a
captured graph?") AND a production-code test in
`0_page_state_machine/decode_wall/` ("does production `xpool_actuator` +
`chunk_arena` actually deliver the A1-derived ≤ 0.15 ms
decode-stream wall budget?"). Both are legitimate; they prove
different things.

## Source-code pointers

| Component | Files |
|---|---|
| VMM substrate | `python/sglang/srt/arena/{chunk_arena,multi_tensor_arena,xpool_actuator}.py` |
| Migration prim | `python/sglang/srt/mem_cache/memory_pool.py:738-820` (`MambaPool.migrate_slot`) |
| Eviction policy | `python/sglang/srt/mem_cache/radix_cache.py` |
| Cost model | `python/sglang/srt/budgeter/agent.py` |
| Admitter | `python/sglang/srt/budgeter/admitter.py` |
| Page selection | `python/sglang/srt/budgeter/fire_planner.py` |
| Pressure planner | `python/sglang/srt/budgeter/xpool_planner.py` |

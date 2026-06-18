# archive — exploration paths that failed empirically

Branches we walked far enough to disprove. Each contains the full
sequenced TDD ladder + READMEs documenting what was tried and where
it broke. Kept on disk for traceability (and so the next person
doesn't re-derive the same dead end).

## Folders

### [`0_batch_boundary_fire/`](0_batch_boundary_fire/) (G)

**Hypothesis**: fire `cuMemUnmap` only at scheduler batch boundaries
where no captured CUDA graph is in flight. Falsified by step 2:

- **A1 holds**: captured graphs are index-gated; safe slot indices
  let replay-after-unmap work (proven by step 1, step 1b real Triton
  kernel). This invariant is still useful and is reused in the
  surviving path.
- **A2 holds in `event_loop_normal`** (auto-selected for mamba models
  via `server_args.py:2396`): 100% of batch transitions show
  positive gap.
- **A3 refuted**: natural gap p50 = 103 µs, p99 = 6 ms — measured
  across 9248 transitions at C=14. A full fire is 82 ms (measured by
  `bench/bench_cumem_costs.py`). 800× gap-vs-fire shortfall. The
  "ride natural idle" promise of G-natural is empirically dead.
- Remaining G-forced (block scheduler 82 ms per fire) is
  semantically equivalent to sync fire — no advantage over option A.

The surviving evidence from this investigation — that captured
CUDA graphs are index-gated, so a kernel reading
`ssm_state_indices` is safe across `cuMemUnmap` of OTHER slots —
was promoted into [`../0_page_state_machine/`](../0_page_state_machine/)
(property A1, design.md §"Threading model"). The page-state
machine described in design.md §"Page ownership state" supersedes
this batch-boundary fire approach.

Reusable artefacts kept from this folder (for replay-debugging
the original investigation, not for current development):
- `step1_boundary_safety_invariant/test_boundary_safety.py` — proves
  captured graph safety relies on runtime index, not VA span.
- `step1b_triton_kernel_safety/test_triton_kernel_safety.py` — same
  invariant on the production Triton kernel.
- `step2_scheduler_idle_gap_trace/` — gap-distribution methodology
  (idle-gap measurement against fire wall).

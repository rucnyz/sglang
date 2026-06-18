# 4_e2e — End-to-end workload validation

End-to-end validation of the full interlayer stack (page state machine
+ admitter + budgeter + planner) under live sglang serving. Each
sub-folder drives a real workload, captures the budgeter JSONL log,
and validates that the design's headline + negative-control claims
hold against the actual production fire path.

Companion to the unit and mechanism tests in:
- [`../0_page_state_machine/`](../0_page_state_machine/) (cuMem + arena
  mechanism layer)
- [`../1_dyn_admission_cap/`](../1_dyn_admission_cap/) (allocator-side
  dynamic caps)
- [`../2_admitter/`](../2_admitter/) (per-arrival cost decision)
- [`../3_budgeter/`](../3_budgeter/) (steady-state pressure rebalance)

## Sub-folders

| folder | ± | what it pins | design.md ref |
|---|---|---|---|
| [`byte_transfer/`](byte_transfer/) | + | end-to-end real byte transfer + working-set invariant (`live_size = size − \|_capped_pages\|`, `expand_pages_to_token_slots` half-open) | §byte_transfer |
| [`saturated_bubble/`](saturated_bubble/) | + | saturated single-pool workload: Budgeter harvests the under-utilised pool's bubble; +cap on the saturated side relieves TPOT | §saturated_bubble |
| [`idle_no_regression/`](idle_no_regression/) | − | idle workload: Budgeter does NOT fire; TPOT/throughput within noise of `off` baseline | §idle_no_regression |
| [`alternating_saturation/`](alternating_saturation/) | − | alternating-saturation adversarial: rapidly flipping KV↔mamba pressure does not destabilise the planner or thrash fires | §alternating_saturation |
| [`cc_traces_headline/`](cc_traces_headline/) | + | real-world CC traces — the paper's headline workload showing the interlayer's throughput/TTFT win | §cc_traces_headline |
| [`burst_recovery/`](burst_recovery/) | + | burst-recovery: Admitter sync-fire path handles a sudden traffic burst without admission stall | §burst_recovery |

## Workload-runner shape

Each sub-folder ships:
- `run_*.sh` — boots sglang with the right env flags (admitter / budgeter
  / actuator gates), runs a benchmark client at a target RPS, captures
  server.log + budgeter JSONL.
- `validate_*.py` — parses the JSONL and asserts the design's PASS
  criteria for that conjecture.
- `README.md` — workload tuning rationale + PASS/FAIL bookkeeping.

Runs that have a persisted PASS dataset keep it inside the folder (e.g.,
`burst_recovery/run_2026-05-29/`). Validators take an `--out-dir` flag
to re-run against any captured run dir.

## Cross-references

- The catalog mapping each row above to its persisted PASS state
  (where applicable) is documented in the per-conjecture sections
  of [`../design.md`](../design.md) §"Validation conjectures".
- Production fire path: `python/sglang/srt/budgeter/agent.py`,
  `python/sglang/srt/arena/xpool_actuator.py`, and the Admitter at
  `python/sglang/srt/budgeter/admitter.py`.

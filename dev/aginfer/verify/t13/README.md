# T13 — bw_free EMA validation (PLAN §1)

Validate the `link_stats` schema sglang emits and the daemon's
`bw_free` branch logic that consumes it.

## STATE OF THE WORLD (2026-06-01)

T13 has two halves and the second is gated on T26.

| half | needs | status |
|---|---|---|
| (1) Schema contract + branch logic | nothing | **DONE — this verify** |
| (2) EMA tracks reality | T26 (HiCache + Mooncake instrumentation reporting `recent_throughput_bps` from CUDA-event / wall-clock measurement) | **DEFERRED** — see PLAN §3 T26 + task #172 |

Until T26 lands, sglang ships cold-start placeholders:
* `recent_throughput_bps = 0` (no measurement yet)
* `time_since_last_sample_s = 1.0e12` (sentinel — orjson can't encode `math.inf`)
* `peak_bw_bps` = realistic device peak (PCIe 5 / NVMe gen5 defaults)

The daemon takes the **idle path** (`bw_free = peak`) whenever
`time_since_last_sample_s > LINK_IDLE_SECONDS = 1.0`, which is the
correct behavior for an un-measured link.

## SCOPE

### Done in this verify (9 stages)

**A.** sglang `_aginfer_link_stats` emission contract
* A0 4 directions × 3 required keys
* A1 cold-start `recent_throughput_bps == 0`
* A2 cold-start `time_since_last_sample_s > LINK_IDLE_SECONDS`
* A3 `peak_bw_bps > 0` for every direction

**B.** Daemon `bw_free` branch logic in `build_paper_state`
* B0 idle link → `bw_free = peak_bw_bps`
* B1 busy link → `bw_free = peak − recent`
* B2 saturated link → `bw_free` clamps to 0 (no negative)
* B3 `peak_bw_bps <= 0` → `fatal(peak_bw_bps_non_positive)` (subprocess exit 1)
* B4 boundary: `time_since_last_sample_s == 1.0` is NOT idle (predicate is `> 1.0`, strict)

### Deferred until T26

Spec text from PLAN §1 T13 quoted verbatim, gated on T26:

> - Compare `recent_throughput_bps` against ground-truth wall-clock
>   per migrate, under both idle-link and contended-link conditions
> - Pin `time_since_last_sample_s` monotonicity across consecutive
>   state-dumps when the link is quiet

When T26 lands, these stages should land here (or as follow-on
stages in `verify/t26/`).

## REPRODUCING

```bash
source /scratch/yuzhou/miniconda3/etc/profile.d/conda.sh && conda activate agsched
cd /scratch/yuzhou/projects/sglang
python dev/aginfer/verify/t13/verify.py
```

Runs in ~3 s.  Stage B3 spawns one subprocess to exercise `fatal()`.

## RESULTS

**PASSED** — all 9 stages.

* date: 2026-06-01
* raw log: `results/20260601_t13_initial_pass.log`

| Stage | Result |
|---|---|
| A0 sglang emits 4 directions + 3 keys each | PASS |
| A1 cold-start recent_throughput_bps == 0 | PASS |
| A2 cold-start time_since > LINK_IDLE_SECONDS | PASS |
| A3 peak_bw_bps > 0 | PASS |
| B0 idle path: bw_free = peak | PASS — stale `recent` ignored |
| B1 busy path: bw_free = peak − recent | PASS |
| B2 saturated path: bw_free clamps to 0 | PASS |
| B3 peak ≤ 0 fatal | PASS — reason `peak_bw_bps_non_positive` |
| B4 idle threshold boundary | PASS — strict `> 1.0` both sides |

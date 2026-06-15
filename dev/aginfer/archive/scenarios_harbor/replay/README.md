# Deterministic trace-replay harness (#231)

A reproducible benchmark for the daemon's **serving-latency** effect — the
thing the free-running agentic e2e (harbor/terminus-2) **cannot** measure.

## Why the agentic e2e is not comparable

Even with every reproducibility knob on (`temperature=0`, `--ak seed=42`,
sglang `--random-seed 42`), the same 32-task set generates a **1.5×-varying**
amount of work across runs:

| run | requests | generated tokens |
|---|---|---|
| a3 cyc1 | 4234 | 318,800 |
| a3_kvoff cyc2 | 5861 | 467,583 |

The agent trajectory diverges from two seed-immune sources: (1) docker
tool-execution nondeterminism (a tool's output depends on container/fs/timing
→ the next prompt changes), and (2) concurrent-batch numerics flipping the
argmax token at temp=0 (different batch composition → different chunked-prefill
boundaries → different logits). The daemon's whole job is to change cache
state, which itself triggers (2). So `agent_execution` wall-time measures a
**different amount of work each run** — "ours +5.6%" was comparing 318k vs
375k tokens of different work, not a scheduling effect.

## The fix: capture once, replay byte-identically

1. **Capture** (`capture_trace.sh`): run a pressured A3 workload ONCE with the
   daemon proxy's trace recorder armed (`AGINFER_TRACE_CAPTURE`). One JSONL
   line per request: arrival offset, `program_id`, `messages` (verbatim →
   prefix reuse preserved), generated `output_len`, and real serve time
   `ref_e2e_ms`. See `daemon/trace_capture.py`.

2. **Replay** (`run_replay.sh` → `replay_driver.py`): replay that exact trace
   against ours (`a3`) and baseline (`a3_kvoff`), forcing the output length
   (`max_tokens=output_len` + `ignore_eos`) and `temperature=0`. Because the
   work is **pinned**, both arms process byte-identical requests — any latency
   delta is attributable to the daemon alone. `len_match_rate ≈ 1.0` (printed
   by `compare.py`) is the sanity that licenses the comparison.

3. **Compare** (`compare.py`): mean±std per metric across N trials, do-no-harm
   verdict via disjoint mean±std bands.

## Two replay modes — and which question each answers

The agent is **closed-loop** in reality: it reads each response, runs a tool,
then sends the next request — so faster serving makes the next request arrive
sooner and the whole session finish sooner. That feedback is the difference
between the two modes:

| mode | arrivals | measures | faithful for |
|---|---|---|---|
| `arrival` (open-loop) | frozen at recorded offsets | TTFT / TPOT / e2e per request under a fixed offered load | **do-no-harm**, per-request latency. Fair for harm; **conservative** (under-counts) for benefit. |
| `session` (closed-loop) | request N+1 dispatched after N completes + the recorded tool-think gap | **makespan** + per-session end-to-end | **benefit** — "how much sooner does the workload finish." Closest to what users experience. |

The tool-think gap is derived from the captured timing:
`gap_N = max(0, t_{N+1} − (t_N + ref_e2e_N))` — the real wall time the agent
spent in tools/reasoning between turns, which the daemon cannot speed up. In
`session` mode each `program_id` becomes a dependent request chain; sessions
start at their recorded offsets so inter-session concurrency (the KV pressure)
is preserved. `--zero-tool-time` collapses the gaps for a benefit upper bound.

Run both: open-loop for the clean do-no-harm / per-request numbers,
closed-loop for the realistic end-to-end benefit.

## Usage

```bash
# 1. Capture one pressured trace (≈20-30 min; a3_kvoff arm = neutral recorder)
bash scenarios/replay/capture_trace.sh a3pressure
#    -> scenarios/replay/traces/a3pressure.jsonl

# 2a. Open-loop do-no-harm / per-request latency (N=3 trials per arm)
bash scenarios/replay/run_replay.sh scenarios/replay/traces/a3pressure.jsonl 3 arrival

# 2b. Closed-loop end-to-end benefit
bash scenarios/replay/run_replay.sh scenarios/replay/traces/a3pressure.jsonl 3 session

# 3. (run_replay.sh calls compare.py automatically; or re-run)
python scenarios/replay/compare.py scenarios/replay/results/a3pressure_arrival
```

Knobs: `CAP_N_TASKS / CAP_N_CONCURRENT / CAP_MAX_TURNS` size the capture;
`run_replay.sh <trace> [N] [mode] [slowdown]`. Both arms use the same inline
`ours_greedy_score` scorer; the only difference is the daemon's kv-scheduling
(`a3` on, `a3_kvoff` off), isolating the daemon's contribution.

## Fidelity caveats (honest)

- The trace is captured under ONE arm; both arms then replay that single
  request stream. This is deliberate (identical work = clean isolation), but
  it means we compare both arms on a fixed workload, not on the (slightly
  different) workload each would naturally produce.
- Open-loop `arrival` mode reproduces the real offered load + pressure profile
  but not the closed-loop feedback (hence the `session` mode).
- `output_len` from streamed runs is counted as content SSE deltas — exact
  under per-token streaming; a lower bound if a backend coalesces tokens. The
  replay forces the length regardless.
- **Admission in arrival mode (audit C1):** the a3 arm runs the daemon's full
  machinery including admission, which can PAUSE a replayed request — that
  gate-park time lands in a3's TTFT/e2e, while a3_kvoff (admission off) never
  pauses. This is deliberate (do-no-harm = the daemon's whole footprint), and
  in open-loop it is **conservative** for a3: admission's cost is counted but
  its back-pressure benefit (the next arrival backing off) is invisible with
  frozen arrivals — that benefit shows up in `session` mode. So treat arrival
  as a pessimistic do-no-harm bound, session as the realistic one. To isolate
  *pure serving* latency (no admission), run an admission-off variant.
- **The verdict is gated on a sanity check (audit C2):** `compare.py` prints
  `COMPARISON INVALID` (and refuses a do-no-harm verdict) unless both arms hit
  `len_match_rate ≥ 0.98`, zero errors, and statistically-equal total tokens —
  i.e. the forced-length / identical-work invariant actually held. A clean
  "DO-NO-HARM: HOLDS" therefore means the arms provably did the same work.
- **Concurrency cap:** the driver defaults `--max-concurrency 4096` so the cap
  does not throttle the captured offered load (or serialize sessions); if it is
  ever hit, `cap_saturated` is set and a warning printed. Comparisons stay
  valid (both arms capped identically) but absolute numbers would be throttled.
- **Verdict statistics:** mean±std uses the SAMPLE std (÷ n−1) and a stable
  better/worse verdict is only issued when both arms have equal n ≥ 2.

## Tests

`verify/replay_capture` (recorder + SSE token counting + real-proxy capture),
`verify/replay_driver` (build_payload/aggregate + open-loop live + closed-loop
session live), `verify/replay_compare` (verdict + summarize + endtoend). All
green, no GPU.

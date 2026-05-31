# ablation/daemon_overhead — measure aginfer daemon's net runtime cost

## What this ablates

Under the **swebench_default workload** (no caps, ~10 M pool),
compare:

| arm | what's on |
|---|---|
| [`direct_sglang/`](arms/direct_sglang/) | sglang + ours_greedy_score inline, **no daemon process** at all |
| [`with_daemon/`](arms/with_daemon/) | sglang + ours_greedy_score inline + aginfer daemon ON (kv_scheduler + admission ENABLED, but in swebench_default they have no work to do) |

The delta isolates the daemon's pure runtime overhead — webhook
latency, /aginfer/state polling, decision compute — **independent**
of whether the daemon's scheduling actually contributes value.

## Why this is an ablation (not a workload scenario)

`with_daemon` arm is config-identical to `swebench_default/arms/ours_full/`.
Same workload, same arm — it's reproduced here only so that the
direct vs with-daemon comparison is a self-contained ablation
folder (physical copy, not symlink, so this ablation has its own
re-run space).

## Result so far

| arm | per-trial mean | N |
|---|---|---|
| direct_sglang | (see RESULTS) | 3 |
| with_daemon | ~ 1391 s (= swebench_default ours_full) | 3 |

Δ = daemon net overhead (positive = daemon slower than direct,
expected if its work has no payoff in this workload).

See [`ANALYSIS.md`](ANALYSIS.md) for the comparison.

## Reproduce

```bash
bash repro.sh        # both arms, N=3 each
```

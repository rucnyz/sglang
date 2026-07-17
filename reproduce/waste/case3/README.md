# case3: dynamic workload (binding pool flips over time)

The canonical case3 is the TEMPORAL variant `case3a`: a long-horizon phase then a
swarm phase, back to back against ONE server on ONE static split. Phase A
(sessions truncated to ~50k) binds KV (mamba idle); phase B (the case2 swarm)
binds mamba slots via max_running (KV idle). The bottleneck flips mid-run, so no
single static split serves both, visible as the two usage curves crossing in
`waste.png`. Served on sglang's DEFAULT split (no `--max-mamba-cache-size`).

```bash
bash ../serve.sh 0 30098                                                 # default split
bash ../replay.sh 30098 case3a/data/phase_a_long50k.jsonl  32 0 results case3a_long
bash ../replay.sh 30098 case3a/data/phase_b_swarm.jsonl   256 0 results case3a_swarm
../../../.venv/bin/python ../parse_waste.py results/server.log --out results --label "case3 dynamic"
# or: bash ../run_split.sh case3a default 0 30098 case3a/results/default   (runs both phases)
```

Data: `case3a/data/` (from `../make_slices.py`). Figure:
`../figures/case3a_temporal_default.png`. A SPATIAL mix (long + swarm arriving
concurrently, `case3b`) degenerates to whichever request type dominates the shared
batch, so it manifests as the same over-time flip, not a simultaneous double-bind.

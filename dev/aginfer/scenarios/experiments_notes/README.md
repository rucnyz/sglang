# experiments_notes — raw-data archive

Analysis notes that previously lived here have been deleted.  Their
findings are either:

- encoded in `dev/aginfer/DESIGN.md` (round-9+ invariants) and
  `dev/aginfer/PLAN.md` (open work),
- captured in user-level memory entries (e.g. the
  "1 % requests = 80 % time" runaway finding), or
- retrievable from `git log` if a specific historical analysis is
  ever needed again.

What remains here is **raw measurement data only**, kept because
re-running the harness is more expensive than the disk it costs:

| Path | What |
|---|---|
| `instrument_chain/` | per-cycle structured-metric logs from the A3 / J / baseline matrix (2026-05-29 → 30) |
| `algo_baselines.txt` | output of `python -m baselines.compare` (single-seed) |
| `algo_baselines_sweep_seeds.txt` | output of `python -m baselines.sweep_seeds` (8-seed) |

Regenerate the `algo_baselines*.txt` snapshots:

```bash
source dev/aginfer/scripts/env.sh
python -m baselines.compare      > scenarios/experiments_notes/algo_baselines.txt
python -m baselines.sweep_seeds  > scenarios/experiments_notes/algo_baselines_sweep_seeds.txt
```

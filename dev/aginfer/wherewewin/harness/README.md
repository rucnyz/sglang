# Harness (wherewewin campaign)

All harness work and operations for the wherewewin campaign live here. The old
`dev/aginfer/scenarios/` folder is **deprecated** — going forward we build, run,
and document everything under `wherewewin/`.

## What's here
- **`teacher_forcing/`** — the faithful-replay prerequisite: build in-loop
  teacher-forced decode (force the captured output token ids, not just the length)
  and **empirically prove** it changes neither timing nor sglang state. The
  wherewewin scenarios' TTFT / cache-reuse numbers are only trustworthy once this
  PASSes (task #234, blocks #233).

## The faithful-replay requirement (applies to every scenario)
A KV scheduler must be evaluated on a **fixed** workload (same request stream both
arms), and the workload must reproduce a real agent's KV trajectory:
- **Teacher-forced** (force output token ids) — keeps the multi-turn KV
  continuation faithful; length-only replay re-prefills each turn's output (artifact).
- **In-loop** forcing (replace `argmax` with the captured token inside the decode
  loop), **not** parallel-prefill of the known output (which would run at prefill
  speed → wrong timing). We do not use spec-decode, so its accept-rate caveat is moot.
- **Closed-loop session mode** replays the recorded tool-think gaps so batch
  composition + timing match reality.

## Existing harness code (to be operated/relocated under here)
The working replay harness currently sits at the deprecated path
`dev/aginfer/scenarios/replay/` — `replay_driver.py`, `compare.py`,
`parse_reprefill.py`, and the `scripts/replay_*.sh` drivers. As we build the
teacher-forcing mode and run the campaign, this code is operated from / brought
under `wherewewin/harness/`. (Code relocation is an operation; the plan is being
finalized first, so the files are referenced in place for now.)

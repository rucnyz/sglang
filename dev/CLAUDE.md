# dev/ workflow conventions

## Per-milestone folder layout

Every non-trivial engine change is delivered as a self-contained folder under
`dev/`, named after the task (e.g. `dev/T1_page_grain_vmm/` for task #73).
The folder must contain:

```
dev/<task_name>/
├── README.md            One-page intro: what changed, why, what's flagged
│                        behind which env var, expected behavior on/off.
├── reproduce.sh         A single shell script that boots the engine with
│                        the right env vars and runs whatever proves the
│                        feature works (smoke + correctness + perf delta
│                        if applicable). Must be idempotent.
├── test/                Per-feature tests. Anything from a python pytest
│                        file to a bash harness. Must run from the repo root
│                        and exit non-zero on failure.
├── results/             Captured output from the latest reproduce run:
│                        logs, JSON summaries, plots, anything a reviewer
│                        needs to verify the claim. Plain-text where
│                        possible; PDFs / PNGs for figures.
└── notes.md (optional)  Ad-hoc commentary, gotchas, decisions made
                         during implementation that aren't obvious from
                         the code.
```

## Workflow

For each task:

1. **Design note first.** Write `README.md` describing the change before
   touching code. Captures intent so subsequent code lines up against it.
2. **Flag the code.** Every behavior change goes behind an env var. Default
   keeps existing behavior; flag-on enables new path. This makes A/B testing
   trivial and lets us roll back without code revert.
3. **Implement + test.** Write the test alongside the code. Tests must run
   from the repo root via `bash dev/<task_name>/reproduce.sh` or
   `bash dev/<task_name>/test/<...>.sh`.
4. **Capture results.** Run `reproduce.sh`, save output to `results/`.
5. **Per-step verification.** For multi-step features, log intermediate
   evidence in `notes.md` so reviewers can follow the construction.
6. **Perf regression A/B.** Before declaring done, run the same workload
   with the flag off and on; record both numbers in `README.md`. The
   on-vs-off delta IS the contribution of this milestone.

## Order

`dev/T1_page_grain_vmm/`, `dev/T2_allocator_placement_bias/`,
`dev/T3_smart_overcap_selection/`, `dev/T4_atomic_page_migration/`,
`dev/T5_va_overcommit/`, `dev/T6_admission_time_eval/`,
`dev/T7_validation_swarm_conc800/`.

T7 is the validation milestone (no new code, just rerunning M2 swarm 4-cell
× 5-trial under the integrated stack). Earlier T's flags can be combined to
realize the full ideal-mode design from `paper/design.tex §3.2`.

## What goes here vs. main code

- Feature code lives in the engine (`python/sglang/srt/...`), behind env vars.
- This folder holds the `bench / test / docs / results` that prove and
  reproduce the feature, plus design notes that wouldn't survive in commit
  messages.
- The folder is reviewed alongside the code change in the same PR.

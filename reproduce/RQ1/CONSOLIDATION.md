# Worktree consolidation — READ BEFORE MERGING (deliberately NOT auto-done)

You asked to fold the `sglang-sync` worktree back into `sglang` and drop the worktree.
I built the RQ1 package but **did not run the branch merge**, because while you were asleep
the divergence turned out to be large enough that a blind merge could lose committed work.
Here are the exact facts and the safe path — **do this awake**, verify after.

## The state (measured)

| | branch | base | has |
|---|---|---|---|
| `/scratch/yuzhou/projects/sglang` (main checkout) | **`aginfer`** | old upstream fork point | the **#231 benefit** work (20 `#231` commits) + **uncommitted edits** (DESIGN.md, daemon/event_router.py, daemon/kv_scheduler.py, daemon/main.py) |
| `/scratch/yuzhou/projects/sglang-sync` (worktree) | **`aginfer-synced`** | **latest upstream main** (rebased, #236) | the **S1** work (#238/#239/#240/#241 — warm, ETA estimator, saturation yield, clean live win) + the whole rebased core |

Divergence: merge-base is an **ancient upstream commit**; `git cherry aginfer-synced aginfer`
shows **295/296** aginfer-only commits are *not* present on `aginfer-synced` by patch-id.
So this is **not** a fast-forward and **not** a clean rebase-superset — both branches carry
unique commits, on **different bases**. A naive `git merge` will conflict heavily.

⚠️ Also: the **running daemon + sglang stack** is launched from the `sglang-sync` worktree.
Removing the worktree or switching branches mid-flight breaks it. **Stop the stack first.**

## Decide first: which branch is canonical going forward?

- **If you want the latest-upstream base** (the point of the #236 rebase) → **`aginfer-synced`
  is canonical**; bring the #231 benefit work *onto* it. (Recommended — it has the up-to-date
  base + all the S1 work.)
- **If `aginfer` is your working trunk** → keep it; bring the S1 commits onto it (they assume
  the newer base, so expect conflicts in `python/sglang/srt/mem_cache` + `managers`).

The S1 work is a tight, recent run of commits — list them with:
```bash
cd /scratch/yuzhou/projects/sglang
git log --oneline aginfer-synced ^aginfer --grep -E '#238|#239|#240|#241'   # the S1 arc
```
The #231 work to bring the other way:
```bash
git log --oneline aginfer ^aginfer-synced --grep '#231'                     # 20 commits
```

## Safe mechanical steps

```bash
cd /scratch/yuzhou/projects/sglang

# 0. STOP the running stack (it lives in the worktree)
for p in $(pgrep -f 'launch_server.*--port 30000'; \
           pgrep -f 'daemon.main.*--port=9100'; pgrep -f mooncake_master); do kill -9 "$p"; done

# 1. Preserve the uncommitted edits on `aginfer` (don't lose them)
git stash push -m "wip-before-consolidation" -- dev/aginfer   # or commit them

# 2a. RECOMMENDED — aginfer-synced canonical, cherry-pick #231 onto it:
git checkout aginfer-synced
git cherry-pick <the #231 commit range>      # resolve conflicts (mostly scenarios/ verify/)
#    then make the main checkout follow aginfer-synced and drop the worktree:
git worktree remove --force /scratch/yuzhou/projects/sglang-sync
git checkout aginfer-synced                   # main dir now on the canonical branch
git branch -m aginfer aginfer-old             # keep the old trunk as a safety branch
git branch -m aginfer-synced aginfer          # (optional) rename to your trunk name

# 2b. ALTERNATIVE — keep aginfer, cherry-pick the S1 arc onto it (expect mem_cache conflicts).

# 3. VERIFY nothing broke (do not skip):
export AGINFER_ROOT=$PWD/dev/aginfer; conda activate agsched-rebase
for t in kv_scheduler_value_rule joint_decide program_tracker; do
  PYTHONPATH=$PWD/python:$AGINFER_ROOT python dev/aginfer/verify/$t/verify.py; done
# and re-run RQ1 (reproduce/RQ1/scenarios/s1-predictive-promote) to confirm the 41% holds.
```

## Why I stopped here

Per "don't gamble committed work": 296 commits of your #231 benefit work + a running stack +
uncommitted trunk edits + a non-trivial cross-base merge = a strategic decision (which base,
how to reconcile) that is yours to make, not one to auto-resolve blind. The RQ1 deliverable
is built and runnable against the working tree regardless of how you consolidate.

# ThunderAgent (TA) arm — setup, finding, and the analytical position

## Status: TA stack works; a direct token-space TA measurement is blocked by an interface mismatch.

## What was set up
- TA is installed in both conda envs (`agsched-rebase`, `agsched-rebase`); repo at
  `/scratch/yuzhou/projects/ThunderAgent` (pip-installable).
- Launch: `dev/aginfer/scripts/launch_thunderagent.sh` → TR-mode proxy on **:9000**,
  `--backends http://127.0.0.1:30000 --backend-type sglang --router tr --metrics`. It comes
  up healthy and polls sglang `/metrics`.
- The metric driver (`live_clean.py`) now has an `arm=ta` path that routes `/generate`
  through `:9000` and injects **no** aginfer events.

## The blocker (verified)
TA proxies **only** `/v1/chat/completions` (+ `/programs`, `/health`, `/metrics`,
`/weight_sync`) — confirmed via its OpenAPI and `ThunderAgent/app.py:59`. It does **not**
expose sglang's native `/generate`; a `POST /generate` returns **404**.

The S1 measurement deliberately uses the **token-level `/generate`** API (`input_ids`) so
the prefix is **bit-stable** across turns (a chat template would wrap the messages and the
reusable prefix would not be the raw tokens — cf. "default chat template not prefix-stable").
So the existing token-space driver **cannot** route through TA.

## The analytical position (solid, paper-usable)
For the S1 **predictive-promote** win, TA is **equivalent to B**:
- TA's "pause" makes **zero backend calls** (verified from source) — pure router-side
  admission that withholds a program's next prefill, keyed to HBM-only
  `max_total_num_tokens`. It is **HiCache-unaware**: it never migrates, never promotes,
  never evicts.
- Therefore the underlying sglang+HiCache evicts a parked prefix **the same way under TA as
  under B**, and on resume **re-acquires it on the critical path** (recompute / load-back) —
  **identical to B's prefix cost**. TA can only *delay* a prefill (pause), not *pre-stage*
  the prefix.

⇒ The measured **Ours-vs-B** win (41 % live / 91 % controlled) **is also the win over TA's
prefix handling**. TA's distinct axis (router-side admission to avoid OOM) is **orthogonal**
to the promote win.

## To turn this into a direct TA measurement (your design call)
A fair 3-arm run with TA in the loop requires moving **all three arms onto
`/v1/chat/completions`** (so TA can proxy them), which:
1. loses token-level prefix control → use a fixed long system+history prefix per program,
   reused each turn, and accept noisier eviction control; and
2. carries `program_id` in each chat request for TA's `/programs` tracking.

This is a real driver change + a prefix-stability tradeoff — left for you to decide. The
`arm=ta` routing is already wired; only the request *format* (chat vs token) remains.

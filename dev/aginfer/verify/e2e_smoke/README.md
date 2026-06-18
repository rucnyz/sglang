# e2e_smoke — end-to-end daemon ↔ sglang sanity checks

> **Status: infrastructure regression-guard.**  Not a numbered Impl_PLAN.md
> task.  Home for end-to-end smoke runs that exercise the FULL
> chain (sglang webhook → daemon event router → policy decide →
> daemon migrate POST → sglang apply_aginfer_migrations) and pin
> issues that no single per-Tn verify catches.

## RESULTS

### 2026-05-31 (2nd run, post #157 fix) — GREEN

After fix `e7e... + leaf-check` landed:
- 32 concurrent chats through daemon proxy → daemon emits ~7
  migrate POSTs to sglang.
- Each POST: typically `applied=1 skipped=1` (one device-leaf unit
  applied; one non-device-leaf rejected with `remove_hbm_not_device_leaf`).
- **sglang scheduler stays alive throughout** — no
  `Scheduler hit an exception`, `/health` returns 200 after.

The pool-leak signature from the 1st run is gone.  Two changes
needed:
1. `writing_check(write_back=True)` drain between async
   write_backup and sync device evict (within one action).
2. Leaf-invariant guard: skip with `remove_hbm_not_device_leaf` /
   `remove_dram_not_host_leaf` when policy emits a non-leaf
   migration.  sglang's `inc_lock_ref` walks the prefix chain to
   root and asserts every ancestor has `cd.value`; device-evicting
   a non-leaf would break later write_backup calls on its
   descendants.

Raw logs: `results/20260531_smoke_post_T157_GREEN.{daemon,sglang}.log`.

### 2026-05-31 (1st run) — FAILED

**FAILED.**  Caught a real bug in T20's combined add+remove path.

Configuration:
- sglang Qwen3-0.6B + HiCache write_through, GPU 5, max-total-tokens 65 536
- daemon enabled (kv_scheduler + admission_controller); theta_hi=0.7
- 32 concurrent `/v1/chat/completions` with `program_id=t33t20-smoke-N` +
  `max_tokens=150` (drives ~10 K HBM tokens at peak)

Observed (daemon log):
```
event=kv_decide kind=session_arrival dset_size=4 action_n=4 outcome=dispatched
event=migrate_post status=exception n_actions=4
... ConnectError('All connection attempts failed') ...
```

Observed (sglang log):
```
Scheduler hit an exception:
  File ".../scheduler.py", line 3157, in on_idle
    self.invariant_checker._report_leak("pool", "\n".join(messages))
ValueError: pool memory leak detected! [full] total=65536, available=60254,
  evictable=5293, protected=0, session_held=0, uncached=0
```

= scheduler subprocess died → daemon's subsequent POSTs got ConnectError.

**Diagnosis.**  Daemon's `OursGreedyPolicy.decide` legitimately emits
the `{HBM} → {DRAM}` transition as one action: `add_tiers=[DRAM],
remove_tiers=[HBM]` (PLAN T34 _TRANSITIONS table).  T20's
`apply_aginfer_migrations` applies adds first (`write_backup`)
THEN removes (`_evict_component(DEVICE)` + `_cascade_evict`).

`write_backup` is async — it enqueues the D→H copy on the
cache_controller's background thread + records the pending lock in
`ongoing_write_through[node.id]`.  The function returns the
allocated host_indices but the actual copy hasn't completed yet.

T20 then immediately evicts the device — freeing the buffer that
the background copy is still reading FROM.  The pool accountant
sees the freed bytes as both `available` (allocator's view) and
`evictable` (radix's view) for the same slot until the async copy
completes, hitting the new sglang invariant_checker tripwire that
expects categories to be disjoint.

**Why T20's verify didn't catch this.**  Stage 1 only adds DRAM
(no following remove).  Stage 2 starts from `[HBM, DRAM]` so
`remove=[HBM]` doesn't combine with a fresh write_backup.  Stage 3
is full DROP (no add).  The COMBINED `add+remove in one action`
case is exactly the §7 6-transition entry the policy emits in
practice — and no T20 stage exercises it.

**Fix path** (tracked as a new task):
1. T20 impl: after `write_backup` returns, await the
   `ongoing_write_through[node.id]` ack BEFORE applying the
   subsequent device evict.  Or: defer the device evict by enqueuing
   it via `cache_controller`'s post-completion callback so it runs
   AFTER the D→H copy.  Either way the ordering invariant
   "free-after-copy-completes" must be honoured.
2. T20 verify: add Stage 11 that drives a unit to HBM-only, then
   POSTs `add=[DRAM], remove=[HBM]` in ONE action, then asserts:
   - HTTP 200 (no scheduler crash)
   - residence == [DRAM] post-state
   - sglang's `invariant_checker` not tripped (poll /health = 200)

**Raw logs**:
- `results/20260531_smoke_failed_pool_leak.daemon.log` (4 KB events
  + ~50 ConnectError lines after sglang died)
- `results/20260531_smoke_failed_pool_leak.sglang.log` (full
  scheduler traceback at line 114)

## Reproducing

```bash
source /scratch/yuzhou/miniconda3/etc/profile.d/conda.sh && conda activate agsched-rebase

# 1. sglang with daemon notify URL
SGLANG_ENABLE_UNIFIED_RADIX_TREE=1 CUDA_VISIBLE_DEVICES=5 \
  python -m sglang.launch_server --model-path Qwen/Qwen3-0.6B \
    --host 127.0.0.1 --port 30001 --tp 1 --mem-fraction-static 0.15 \
    --max-total-tokens 65536 --trust-remote-code \
    --attention-backend flashinfer \
    --enable-hierarchical-cache --hicache-ratio 1.5 \
    --hicache-write-policy write_through \
    --aginfer-notify-url=http://127.0.0.1:9100/aginfer/event \
    --aginfer-heartbeat-s=5 \
    --aginfer-theta-hi=0.7 --aginfer-theta-crit=0.85 \
  > /tmp/sglang_e2e.log 2>&1 &
until grep -q "Uvicorn running" /tmp/sglang_e2e.log; do sleep 6; done; sleep 18

# 2. daemon
cd /scratch/yuzhou/projects/sglang/dev/aginfer
python -m daemon.main \
  --sglang-base-url=http://127.0.0.1:30001 \
  --host=127.0.0.1 --port=9100 \
  --kv-scheduler=enabled --admission-controller=enabled \
  --theta-hi=0.7 --theta-lo=0.55 \
  > /tmp/daemon_e2e.log 2>&1 &

# 3. drive 32 distinct chats
for i in $(seq 1 32); do
  curl -s -X POST http://127.0.0.1:9100/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"Qwen/Qwen3-0.6B\",\"messages\":[{\"role\":\"user\",\"content\":\"prog $i fact\"}],\"max_tokens\":150,\"program_id\":\"smoke-$i\"}" \
    > /dev/null 2>&1 &
done
wait

# 4. inspect
grep "event=migrate_post" /tmp/daemon_e2e.log
grep "Scheduler hit an exception" /tmp/sglang_e2e.log
```

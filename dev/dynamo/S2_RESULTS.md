# S2 holder-count A/B on Dynamo — RESULTS + HANDOVER

> ## ⛳ RESUME HERE (token-exact agentreplay pivot, 2026-06-15) — READ FIRST
>
> We pivoted the S2 A/B from the live-chat harness (`fleet_ab.py`/`s2_ab.py`) to the
> **token-exact agentreplay** path (per the standing requirement to use the harness we
> invested in). All the science is validated; the ONLY open blocker is a V4-Flash engine
> crash under extreme oversubscription. Next session can start directly from the
> "RESUME PROCEDURE" below — no re-derivation needed.

## What is DONE + validated (do not redo)
- **agentreplay token-exact replay works on Dynamo**: single/few requests give
  `force_exact_rate=1.0`, `len_match_rate=1.0` through the dynamo-native router path.
- **Real-data S2 trace** (NOT synthetic): `build_s2_trace.py` extracts from the REAL
  collected CC data (`/scratch/yuzhou/projects/agentreplay/data/samples/demo_trace.jsonl`,
  14 real Claude-Code trajectories, already tokenized). It emits N **independent blocks**;
  each block is a complete S2 scenario whose fleet-shared **24k prefix is a DIFFERENT real
  program** (block0=71396433, block1=agent-a2, block2=agent-ab, block3=7a6e7446), replicated
  across 8 fleet members (→ `n_holders=8`, a non-leaf) + 8×14k unique-real churn. Distinct
  tokens per block ⇒ no cross-block radix reuse ⇒ genuinely independent paired measurements
  (validated: fleet pairwise LCP = 24000 exactly; fleet-vs-churn LCP = 1501 header only;
  cross-block LCP = 1501). Files: `s2_block{0..3}.jsonl` (regenerate with
  `python dev/dynamo/build_s2_trace.py --blocks 4`).
- **The holder-count lever FIRES on Dynamo**: daemon log shows
  `[aginfer] S2 hint push: n=77 units, MAX n_holders=8 (shared prefix seen)`, hints PUT 200 OK.
- **The shared prefix IS reused**: backend log shows `#new-token: 256, #cached-token: 24064`
  (a fleet member hitting the 24k shared node).
- **Daemon parses HBM-only state without fatal**: `occ_hbm=0.667 occ_dram=0 units=31`.

## ⛔ THE OPEN BLOCKER (the reason there is no final number yet)
The **V4-Flash worker CRASHES** under the heavy S2 oversubscription load — it is a crash,
not a hang:
```
ERROR watchdog._watchdog_once: Scheduler watchdog timeout (watchdog_timeout=300, soft=False)
ERROR tokenizer_manager.running_phase_sigquit_handler: SIGQUIT received ... one child failed.
```
- The 80-request block (8×24k shared + 8×14k churn) drives the 131072 pool to **occ≈0.98**.
  The scheduler thread then stalls >300s in an **eviction retry-storm** (it keeps trying to
  evict in-flight/locked units and makes no progress), and the 300s watchdog SIGQUITs the
  worker. After that the bridge `:9200/health` returns
  `Cannot connect to host 127.0.0.1:8081` and `pgrep dynamo.sglang = 0`.
- **Both open-loop (`--mode arrival`) AND closed-loop (`--mode session`, conc 4) crash it.**
  So it is NOT a harness-pacing problem — it is an engine robustness bug under extreme
  oversubscription. (Consistent with memory: the win is at MODERATE concurrency; extreme
  flood regresses. cf. eviction-storm tasks #223/#224, watchdog.)

### Next-step plan to get past it (decide A vs B first)
- **A (recommended, fastest): MODERATE-pressure regime.** Reduce churn so peak occ ≈ 0.88–0.92
  (NOT 0.98) → the scheduler never enters the evict-storm. Tune via
  `build_s2_trace.py --churn 6 --churn-tok 12000` (≈ 72k churn; working set 24k+72k=96k is
  *below* the 131k pool, so first bump pool pressure by either raising churn until
  `occ` in the daemon log sits ~0.90 but the worker SURVIVES, or shrink the shared+churn so
  the shared still gets evicted). Practically: sweep churn upward from a safe floor and watch
  the daemon `state_fetched occ_hbm=` — keep it under ~0.92 and confirm the worker stays alive
  across a full block before trusting the number.
- **B (real fix): root-cause the scheduler evict-storm** under extreme oversubscription
  (locked-inflight units can't be evicted → retry loop → 300s watchdog). This is the
  principled "crash = bug" fix but a substantial engine investigation. Owner decision pending.

## RESUME PROCEDURE (clean stack bring-up — copy/paste next session)
Container `aginfer_dyn`, worker on GPUs 5,6. dynamo venv = `/opt/dynamo/venv/bin/python`.
HBM-only launch scripts live in the mounted dir: `/workspace/sglang/dev/dynamo/launch_v4_{fork,lru}_hbm.sh`.
```bash
# 0) agentreplay package into the container (if /tmp/agentreplay is gone after a container restart):
docker cp /scratch/yuzhou/projects/agentreplay/agentreplay aginfer_dyn:/tmp/agentreplay_pkg
docker exec aginfer_dyn bash -lc 'mkdir -p /tmp/agentreplay && mv /tmp/agentreplay_pkg /tmp/agentreplay/agentreplay'

# 1) reboot worker HBM-only (ours = priority+hint_v_u). Wait for :9200/health ok (~3 min; deep_gemm JIT cache persists).
docker exec aginfer_dyn bash -lc "pkill -9 -f 'dynamo.sglang|launch_v4|daemon.main|thunderagent_router|dynamo.frontend|replay-dynamo'; sleep 4"
docker exec -d aginfer_dyn bash -lc "bash /workspace/sglang/dev/dynamo/launch_v4_fork_hbm.sh > /tmp/sglang_backend.log 2>&1"
# wait: until curl -s :9200/health == '"ok": true'

# 2) router+frontend MUST be restarted after EVERY worker reboot (else pinned to dead instance → requests never route).
#    pause/soft=1.0 DISABLES ThunderAgent pausing (else it soft-demotes the fleet at occ≥0.80 and the ENGINE never evicts).
docker exec -d aginfer_dyn bash -lc "python -m dynamo.thunderagent_router --endpoint dynamo.backend.generate --model-name deepseek-ai/DeepSeek-V4-Flash --router-block-size 64 --router-reset-states --pause-threshold 1.0 --soft-demote-threshold 1.0 > /tmp/router_ta.log 2>&1"
docker exec -d aginfer_dyn bash -lc "python -m dynamo.frontend --http-port 8100 --router-mode round-robin --router-reset-states > /tmp/frontend.log 2>&1"
sleep 30   # router discovery — DO NOT skip; a too-early request errors fast (n_error=1, ~0.2s)

# 3) SMOKE one request (confirm routing + token-exact) BEFORE any block:
docker exec aginfer_dyn bash -lc "cd /tmp/agentreplay && PYTHONPATH=/tmp/agentreplay timeout 120 /opt/dynamo/venv/bin/python -m agentreplay replay-dynamo --trace /workspace/sglang/dev/dynamo/s2_block1.jsonl --model deepseek-ai/DeepSeek-V4-Flash --namespace dynamo --mode arrival --max-concurrency 1 --limit 1 --salt smk-\$(date +%s) --verify-exact"
#   expect: "n_ok": 1, "force_exact_rate": 1.0, wall_s ~0.5s

# 4) run ours arm (the orchestrator (re)starts the evict-only daemon + restarts router for you with --restart-router):
python dev/dynamo/s2_replay_ab.py --arm ours --salt s2hbm --blocks 1,2,3 --warmup-block 0 --mode session --max-conc 4 --restart-router
#   ↑ FIRST get a churn level that survives (see blocker plan A) — current 8×14k crashes the worker.

# 5) reboot worker to LRU (HBM-only), then lru arm:
docker exec aginfer_dyn bash -lc "pkill -9 -f 'dynamo.sglang|launch_v4'; sleep 4"
docker exec -d aginfer_dyn bash -lc "bash /workspace/sglang/dev/dynamo/launch_v4_lru_hbm.sh > /tmp/sglang_backend.log 2>&1"
# wait ready, then:
python dev/dynamo/s2_replay_ab.py --arm lru --salt s2hbm --blocks 1,2,3 --warmup-block 0 --mode session --max-conc 4 --restart-router

# 6) verdict (re-prefill lower=better):
python -m agentreplay.report --ours dev/dynamo/m_ours_b*.json --base dev/dynamo/m_lru_b*.json --metrics reprefill_new ttft_ms.mean e2e_ms.mean
```

## Operational learnings (the non-obvious traps, all hit this session)
1. **Router/frontend restart after EVERY worker reboot.** They pin to the dead worker
   instance; new requests go to the dead instance → worker idle, client hangs/timeouts.
   `restart_router()` in `s2_replay_ab.py` does this; call with `--restart-router`.
2. **Disable ThunderAgent pausing** for an engine-eviction experiment:
   `--pause-threshold 1.0 --soft-demote-threshold 1.0`. Default `pause=0.95, soft=0.80` →
   at occ≥0.80 the router soft-demotes the fleet (smallest-token programs), they never
   resume → starvation → client `tcp stream read error`. This also PREVENTS the engine from
   ever reaching the eviction regime (the S2 lever).
3. **HBM-only is the right config for the re-prefill signal.** `write_through_selective`
   MIRRORS every HBM unit into DRAM, so HBM-full ⟹ DRAM-full; you cannot force HBM eviction
   without deadlocking DRAM (demote-stall). And with DRAM headroom, an evicted shared prefix
   just demotes to DRAM and reloads in ~2ms (no signal). HBM-only (no `--enable-hierarchical-cache`)
   makes eviction = DROP → re-touch = recompute (the real ~2.8s saving) AND drops are instant
   (no demote-stall). Daemon parses HBM-only state fine (occ_dram=0, no fatal). Scripts:
   `launch_v4_fork_hbm.sh` (ours: `SGLANG_KV_POLICY_MODULE=aginfer:hint_v_u` +
   `--radix-eviction-policy priority`) / `launch_v4_lru_hbm.sh` (lru: no scorer +
   `--radix-eviction-policy lru`). Only those two flags differ between arms.
4. **DeepGEMM JIT** pre-compiles on a cold boot (`may take 10-20 mins`). The deep_gemm cache
   (`~/.cache/deep_gemm`, in-container) PERSISTS across worker reboots, so only the first boot
   of a session pays it. A warmup block absorbs it; subsequent boots serve fast (24k prefill ~2s).
5. **Token-identity is structural in agentreplay** (unlike the chat path): both arms replay the
   SAME `s2_block*.jsonl`; the only per-arm difference is the routing `salt`, which never touches
   `token_ids`. Use a FRESH salt per worker reboot (routing freshness); tokens stay identical.
6. **N = independent blocks** (distinct real shared prefix per block) gives paired stats with NO
   cache flush needed (flush isn't exposed on this stack). block0 = warmup (DeepGEMM + baseline
   pressure, discarded); blocks 1,2,3 = measured (cold-for-the-arm, distinct tokens).

## File inventory (all under `dev/dynamo/`, mounted into the container)
- `build_s2_trace.py` — real-data → N block traces (`--blocks`, `--churn`, `--churn-tok`, etc.)
- `s2_block{0..3}.jsonl` — the traces (regenerate any time; deterministic from demo_trace.jsonl)
- `s2_replay_ab.py` — orchestrator: `--arm ours|lru`, `--mode session|arrival`, `--blocks 1,2,3`,
  `--warmup-block 0`, `--max-conc`, `--restart-router`. Runs agentreplay per block, parses
  re-prefill via `parse_window.py`, injects it into the `--out` json for `agentreplay.report`.
- `parse_window.py` — sums `#new-token`/`#cached-token` over a wall window from the backend log.
- `launch_v4_fork_hbm.sh` / `launch_v4_lru_hbm.sh` — HBM-only launch (the config to use).
- `launch_v4_fork.sh` / `launch_v4_lru.sh` (in container `/tmp/`) — HiCache versions (deadlock-prone, do not use for S2).
- agentreplay package docker-cp'd to container `/tmp/agentreplay` (re-cp if container restarts).
- **TO DELETE once the agentreplay path produces a number** (consolidate on one harness, per the
  standing request): `fleet_ab.py`, `s2_ab.py`, `run_ab.py`, `s2_holder_gate.py`. KEEP
  `parse_window.py` (metric) + the launch scripts. (Build-then-delete: don't delete until the
  agentreplay S2 number is in hand.)

---

# Historical: chat-harness clean re-run (2026-06-15) — SUPERSEDED by the agentreplay pivot above

> Kept for the record. This number came from the live-chat `s2_ab.py`/`fleet_ab.py` harness,
> which we are replacing with token-exact agentreplay (above). The qualitative claim
> (holder-count keeps a fleet-shared prefix resident; ours < LRU; do-no-harm) held; the
> quantitative headline below is from the chat path, not the token-exact path.

> **CLEAN FAIR RE-RUN (chat harness).** token-identical prompts (fixed-placeholder reply,
> same base salt), both arms warmed (1 discarded warmup cycle), write-through equalized,
> N=3 paired by cycle.
>
> | cycle | ours | LRU | ours saves |
> |---|---|---|---|
> | c1 | 231,936 | 371,456 | −37.6% |
> | c2 | 272,384 | 314,880 | −13.5% |
> | c3 | 231,936 | 269,056 | −13.8% |
> | **mean** | **245,419 ± 19,067** | **318,464 ± 41,881** | **−23%** |
>
> ours < LRU in EVERY paired cycle → do-no-harm STRICT. ours stays pinned at the churn floor
> (keeps the 24k shared prefix); LRU recomputes it. N=3 paired, 3/3 favor ours (sign p=0.125).
> The chat harness avoided the V4-Flash watchdog crash because its CLOSED-LOOP, wait-for-response
> turn pacing kept in-flight KV low — the agentreplay open-loop flood does not, which is the
> crux of the open blocker above.

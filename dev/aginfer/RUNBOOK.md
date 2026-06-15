# aginfer runbook — Dynamo stack startup

> How to bring up the aginfer stack on **NVIDIA Dynamo + DeepSeek-V4-Flash** (the current
> platform). Replaces the old harbor / standalone-sglang reproducer (archived). For the
> platform internals see `../dynamo/README.md`; for the S2 experiment + the full operational
> diagnosis see `../dynamo/S2_RESULTS.md`; for *what* to run see `EXP_PLAN.md`.

## Topology
- **Container** `aginfer_dyn` (GPUs 5,6). Mounts: host `sglang`→`/workspace/sglang`,
  `dynamo`→`/workspace/dynamo`, HF cache, deep_gemm cache.
- **Worker** — `python -m dynamo.sglang` (V4-Flash, our fork; launch script in `/tmp` or
  the mounted `dev/dynamo/launch_v4_*.sh`). Exposes endpoint `dynamo.backend.generate`,
  system port `:8081`.
- **Bridge** `:9200` — `dev/dynamo/aginfer_bridge.py`: proxies the daemon's 4 `/aginfer/*`
  calls into the worker's tokenizer_manager (no Dynamo Rust change).
- **Router** — `dynamo.thunderagent_router` (the baseline; exposes
  `dynamo.thunderagent_router.generate`, routes to `dynamo.backend.generate`).
- **Frontend** `:8100` — `dynamo.frontend` (OpenAI HTTP; round-robin).
- **Daemon** `:9100` — `dev/aginfer/daemon/main.py`: pulls `/aginfer/state`, computes `V_u`,
  PUTs hints / migrates (ours arm only).
- **Python**: dynamo venv `/opt/dynamo/venv/bin/python` (has dynamo.runtime + httpx + orjson).
  agentreplay is docker-cp'd to `/tmp/agentreplay`.

## Launch scripts (in `dev/dynamo/`, mounted)
| script | eviction | tier | use |
|---|---|---|---|
| `launch_v4_fork_hbm.sh` | `priority` + `SGLANG_KV_POLICY_MODULE=aginfer:hint_v_u` | **HBM-only** | ours (S2: real re-prefill signal, deadlock-free) |
| `launch_v4_lru_hbm.sh`  | `lru` (no scorer) | **HBM-only** | LRU baseline |
| `launch_v4_fork.sh` / `launch_v4_lru.sh` | same, but `--enable-hierarchical-cache` (HiCache write_through) | 4-tier | the 4-tier story; **deadlock-prone under heavy oversubscription** (see S2_RESULTS) |

## Cold bring-up (copy/paste)
```bash
# 0) agentreplay into the container (only if /tmp/agentreplay is gone after a container restart)
docker cp /scratch/yuzhou/projects/agentreplay/agentreplay aginfer_dyn:/tmp/agentreplay_pkg
docker exec aginfer_dyn bash -lc 'mkdir -p /tmp/agentreplay && mv /tmp/agentreplay_pkg /tmp/agentreplay/agentreplay'

# 1) WORKER — pick a launch script. HBM-only fork shown. (deep_gemm JIT cache persists
#    in-container across reboots; first boot of a session pays the 10–20 min JIT.)
docker exec aginfer_dyn bash -lc "pkill -9 -f 'dynamo.sglang|launch_v4|daemon.main|thunderagent_router|dynamo.frontend'; sleep 4"
docker exec -d aginfer_dyn bash -lc "bash /workspace/sglang/dev/dynamo/launch_v4_fork_hbm.sh > /tmp/sglang_backend.log 2>&1"
# wait until ready (bridge probes the worker):  curl -s :9200/health == '{"ok": true}'

# 2) ROUTER + FRONTEND — MUST restart after EVERY worker reboot (else they stay pinned to the
#    dead worker instance → requests never route → worker idle, client hangs).
#    pause/soft = 1.0 DISABLES ThunderAgent pausing (needed for an engine-EVICTION experiment,
#    else it soft-demotes the fleet at occ≥0.80 and the engine never reaches the eviction regime).
docker exec -d aginfer_dyn bash -lc "python -m dynamo.thunderagent_router --endpoint dynamo.backend.generate --model-name deepseek-ai/DeepSeek-V4-Flash --router-block-size 64 --router-reset-states --pause-threshold 1.0 --soft-demote-threshold 1.0 > /tmp/router_ta.log 2>&1"
docker exec -d aginfer_dyn bash -lc "python -m dynamo.frontend --http-port 8100 --router-mode round-robin --router-reset-states > /tmp/frontend.log 2>&1"
sleep 30   # router discovery — DO NOT skip (a too-early request errors fast: n_error=1, ~0.2s)

# 3) DAEMON (ours arm only; evict-only for S2). LRU arm: leave the daemon OFF.
docker exec -d aginfer_dyn bash -lc 'cd /workspace/sglang/dev/aginfer && PYTHONPATH=/workspace/sglang/python:/workspace/sglang/dev/aginfer AGINFER_DISABLE_MIGRATE=1 AGINFER_DISABLE_PROMOTE=1 /opt/dynamo/venv/bin/python -m daemon.main --sglang-base-url=http://127.0.0.1:9200 --host=127.0.0.1 --port=9100 --kv-scheduler=enabled --admission-controller=disabled --theta-hi=0.85 --theta-lo=0.70 --theta-crit=0.90 --heartbeat-s=5.0 > /tmp/daemon.log 2>&1'

# 4) SMOKE — confirm routing + token-exact before any real run:
docker exec aginfer_dyn bash -lc "cd /tmp/agentreplay && PYTHONPATH=/tmp/agentreplay timeout 120 /opt/dynamo/venv/bin/python -m agentreplay replay-dynamo --trace /workspace/sglang/dev/dynamo/s2_block1.jsonl --model deepseek-ai/DeepSeek-V4-Flash --namespace dynamo --mode arrival --max-concurrency 1 --limit 1 --salt smk-\$(date +%s) --verify-exact"
#   expect: "n_ok": 1, "force_exact_rate": 1.0, wall_s ~0.5s
```

## Health checks
```bash
docker exec aginfer_dyn bash -lc 'curl -s :9200/health'   # bridge→worker: {"ok": true, "ranks": 1}
docker exec aginfer_dyn bash -lc 'curl -s :9100/health'   # daemon: {"status":"ok",...}
docker exec aginfer_dyn bash -lc 'pgrep -c -f dynamo.sglang'  # worker alive = 1 (0 = crashed; check /tmp/sglang_backend.log for "watchdog timeout")
docker exec aginfer_dyn bash -lc "grep -a state_fetched /tmp/daemon.log | tail -1"  # occ_hbm / n_holders
```

## Operational gotchas (all hit + fixed this session — the non-obvious ones)
1. **Restart router + frontend after EVERY worker reboot** (stale instance pins).
2. **Disable ThunderAgent pausing** (`--pause-threshold 1.0 --soft-demote-threshold 1.0`) for
   an engine-eviction experiment, else it starves the fleet and the engine never evicts.
3. **HBM-only for the re-prefill signal**: HiCache `write_through` mirrors HBM→DRAM, so HBM
   can't be evicted without deadlocking DRAM, and a DRAM reload is ~free (no signal). HBM-only
   ⇒ evict = DROP = recompute (the real saving) and drops are instant (no demote-stall).
4. **Fresh `--salt` per worker boot** (routing freshness; tokens are salt-independent).
5. **V4-Flash crashes under heavy oversubscription** (scheduler watchdog, occ≈0.98) — run a
   **moderate** regime. Full diagnosis + tuning: `../dynamo/S2_RESULTS.md`.

## Run an experiment
Orchestrators + per-scenario packages: see **`EXP_PLAN.md`** (what) and
**`../dynamo/s2_replay_ab.py`** / **`reproduce/RQ1/scenarios/`** (how).

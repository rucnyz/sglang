#!/usr/bin/env python3
"""S2 holder-count A/B on Dynamo via the TOKEN-EXACT agentreplay harness.

Replaces the live-chat fleet_ab/s2_ab path with agentreplay's dynamo-native replay
(token-exact teacher forcing through Dynamo's router → V4-Flash worker). The two
arms replay the SAME block trace files, so prompts are BYTE-IDENTICAL by construction
(the only per-arm difference is the routing salt, which never touches token_ids) —
this structurally fixes the token-identity confound the chat path had.

N = independent BLOCKS (build_s2_trace.py --blocks N): each block is a complete S2
scenario whose fleet-shared 24k prefix is a DIFFERENT real CC program → distinct
tokens → no cross-block radix reuse → genuinely independent, paired measurements.
No cache flush needed.

Per block we run one dynamo-native replay, bracket it with host wall epochs (host and
container share the kernel clock), then parse re-prefill (#new-token, lower=better)
from the backend log over that window (parse_window.py) and INJECT it into the replay's
--out json so `python -m agentreplay.report` can verdict it.

Run ONE arm per invocation (the eviction policy is launch-time, so I reboot the worker
between arms manually):
  ours: worker on /tmp/launch_v4_fork.sh (SGLANG_KV_POLICY_MODULE=aginfer:hint_v_u,
        --radix-eviction-policy priority) + daemon evict-only on :9100 (this script
        (re)starts it).
  lru : worker on /tmp/launch_v4_lru.sh (no aginfer scorer, --radix-eviction-policy lru)
        + daemon OFF (this script kills it).

  # ours arm (worker already on the fork boot):
  python s2_replay_ab.py --arm ours --salt s2r1 --blocks 3
  # reboot worker to LRU, then:
  python s2_replay_ab.py --arm lru  --salt s2r1 --blocks 3
  # verdict:
  python -m agentreplay.report --ours <out>/m_ours_b*.json --base <out>/m_lru_b*.json \
      --metrics reprefill_new e2e_ms.mean ttft_ms.mean
"""
import argparse
import json
import subprocess
import time
import statistics as st

C = "aginfer_dyn"
BK = "/tmp/sglang_backend.log"
MODEL = "deepseek-ai/DeepSeek-V4-Flash"
PYP = "/workspace/sglang/python:/workspace/sglang/dev/aginfer"
VENV = "/opt/dynamo/venv/bin/python"
AR = "/tmp/agentreplay"                       # agentreplay package copied into the container
BLOCKDIR = "/workspace/sglang/dev/dynamo"     # mounted; trace + out json live here
HOSTDIR = "/scratch/yuzhou/projects/sglang/dev/dynamo"   # same dir on the host
DAEMON_CMD = (
    f"cd /workspace/sglang/dev/aginfer && PYTHONPATH={PYP} {{env}} "
    f"{VENV} -m daemon.main --sglang-base-url=http://127.0.0.1:9200 "
    f"--host=127.0.0.1 --port=9100 --kv-scheduler=enabled "
    f"--admission-controller=disabled --theta-hi=0.85 --theta-lo=0.70 "
    f"--theta-crit=0.90 --heartbeat-s=5.0"
)


def dex(cmd, timeout=600):
    return subprocess.run(["docker", "exec", C, "bash", "-lc", cmd],
                          capture_output=True, text=True, timeout=timeout)


def dex_d(cmd):
    subprocess.run(["docker", "exec", "-d", C, "bash", "-lc", cmd], check=False)


def _daemon_up(timeout_s=25):
    for _ in range(timeout_s):
        r = dex("curl -s -m3 http://127.0.0.1:9100/health 2>/dev/null", timeout=15)
        if '"status":"ok"' in (r.stdout or ""):
            return True
        time.sleep(1)
    return False


def restart_router():
    """After ANY worker reboot the thunderagent_router + frontend hold state pinned to
    the DEAD worker instance → requests never route to the fresh worker (worker idle,
    client hangs). Restart both so they re-discover the new instance. MUST be called
    once after each worker reboot, before any replay."""
    dex("pkill -9 -f 'dynamo.thunderagent_router' 2>/dev/null; "
        "pkill -9 -f 'dynamo.frontend' 2>/dev/null; sleep 3", timeout=30)
    # pause/soft thresholds=1.0 DISABLE the ThunderAgent router's program pausing — otherwise
    # it soft-demotes the fleet at occ≥0.80 and the engine NEVER reaches the eviction regime
    # (the S2 lever). Same router for both arms → fair; isolates the ENGINE eviction lever.
    dex_d("python -m dynamo.thunderagent_router --endpoint dynamo.backend.generate "
          "--model-name deepseek-ai/DeepSeek-V4-Flash --router-block-size 64 "
          "--router-reset-states --pause-threshold 1.0 --soft-demote-threshold 1.0 "
          "> /tmp/router_ta.log 2>&1")
    dex_d("python -m dynamo.frontend --http-port 8100 --router-mode round-robin "
          "--router-reset-states > /tmp/frontend.log 2>&1")
    time.sleep(12)
    print("  [router+frontend] restarted (re-discover fresh worker)", flush=True)


def set_arm(arm):
    dex("pkill -9 -f 'daemon.main' 2>/dev/null; sleep 2", timeout=30)
    if arm == "lru":
        print("  [lru] daemon OFF (worker must be on /tmp/launch_v4_lru.sh)", flush=True)
        return True
    if arm == "ours":
        env = "AGINFER_DISABLE_MIGRATE=1 AGINFER_DISABLE_PROMOTE=1 HICACHE_WRITE_POLICY=write_through"
        dex(": > /tmp/daemon.log", timeout=10)
        dex_d(f"{DAEMON_CMD.format(env=env)} > /tmp/daemon.log 2>&1")
        ok = _daemon_up()
        print(f"  [ours] daemon EVICT-ONLY {'ready' if ok else 'NOT ready'}", flush=True)
        return ok
    raise ValueError(arm)


def run_block(a, arm, b):
    trace = f"{BLOCKDIR}/s2_block{b}.jsonl"
    outj = f"{BLOCKDIR}/m_{arm}_b{b}.json"
    salt = f"{a.salt}-{arm}"                 # fresh per arm (routing); tokens are identical
    # Faithful closed-loop replay: each program's turns run sequentially, waiting for the
    # response before the next (parent blocks on subagents, real tool gaps). This bounds
    # in-flight KV (≤ max_conc programs) so the scheduler always has evictable headroom and
    # never enters the evict-storm that trips the 300s watchdog.
    cmd = (f"cd {AR} && PYTHONPATH={AR} timeout 900 {VENV} -m agentreplay replay-dynamo "
           f"--trace {trace} --model {MODEL} --namespace dynamo "
           f"--stagger {a.stagger} --max-concurrency {a.max_conc} --salt {salt}-b{b} "
           f"--verify-exact --label {arm}_b{b} --out {outj}")
    t0 = time.time()
    r = dex(cmd, timeout=950)
    t1 = time.time()
    if r.returncode != 0 or not r.stdout.strip():
        print(f"    [{arm} b{b}] replay FAILED rc={r.returncode}: {r.stderr[-300:]}", flush=True)
        return None
    time.sleep(3)
    pw = dex(f"{VENV} {BLOCKDIR}/parse_window.py {BK} {t0:.3f} {t1 + a.pad:.3f}", timeout=60)
    try:
        prefill = json.loads(pw.stdout.strip().splitlines()[-1])
    except Exception:
        print(f"    [{arm} b{b}] parse_window FAILED: {pw.stdout[-200:]} {pw.stderr[-200:]}", flush=True)
        prefill = {"new": -1, "cached": -1, "cache_hit_pct": -1, "peak_util": -1}
    # inject the re-prefill metric into the replay's out json (host reads the mounted file)
    hostout = f"{HOSTDIR}/m_{arm}_b{b}.json"
    try:
        with open(hostout) as fh:
            m = json.load(fh)
    except Exception:
        m = {}
    m["reprefill_new"] = prefill["new"]
    m["reprefill_cached"] = prefill["cached"]
    m["cache_hit_pct"] = prefill["cache_hit_pct"]
    m["peak_util"] = prefill.get("peak_util")
    m["wall_window"] = [round(t0, 3), round(t1 + a.pad, 3)]
    with open(hostout, "w") as fh:
        json.dump(m, fh, indent=2)
    fx = m.get("force_exact_rate")
    print(f"    [{arm} b{b}] reprefill(new)={prefill['new']} cache-hit={prefill['cache_hit_pct']}% "
          f"peak_util={prefill.get('peak_util')} ttft={m.get('ttft_ms',{}).get('mean')} "
          f"force_exact={fx} fails={m.get('n_error')} ({t1-t0:.0f}s)", flush=True)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["ours", "lru"])
    ap.add_argument("--salt", default="s2r")
    ap.add_argument("--blocks", default="1,2,3",
                    help="comma list of measured block indices (each a distinct real shared prefix)")
    ap.add_argument("--warmup-block", dest="warmup_block", type=int, default=0,
                    help="block run+DISCARDED first (DeepGEMM JIT + baseline pressure); -1 to skip")
    ap.add_argument("--max-conc", dest="max_conc", type=int, default=4,
                    help="closed-loop concurrency: ≤ this many programs in flight → bounded in-flight KV")
    ap.add_argument("--stagger", type=float, default=0.5, help="seconds between program starts")
    ap.add_argument("--restart-router", action="store_true",
                    help="restart router+frontend first (REQUIRED once after a worker reboot)")
    ap.add_argument("--pad", type=float, default=4.0)
    a = ap.parse_args()
    measured = [int(x) for x in str(a.blocks).split(",") if x.strip() != ""]
    # PROTOCOL: reboot the worker FRESH per arm (empty cache) BEFORE running this — both
    # arms then start from an identical empty cache and replay the identical blocks, so the
    # cache state at each block's start is the arm's OWN deterministic behaviour on the same
    # trace (only the eviction policy differs). No warmup block is used (reusing a block would
    # pre-resident its shared prefix and contaminate that block's measurement asymmetrically);
    # each block's distinct real shared prefix is cold-for-the-arm by design. cuda-graph is
    # captured at boot, so the worker is compute-warm at "ready".
    print(f"=== S2 token-exact replay A/B  arm={a.arm}  measured={measured}  "
          f"warmup={a.warmup_block}  conc={a.max_conc}  salt={a.salt} ===", flush=True)
    if a.restart_router:
        restart_router()
    if not set_arm(a.arm):
        print("arm setup failed", flush=True); return
    time.sleep(3)
    if a.warmup_block >= 0:
        print(f"  warmup block {a.warmup_block} (DISCARDED: DeepGEMM JIT + baseline pressure)...", flush=True)
        run_block(a, a.arm, a.warmup_block)
    recs = [run_block(a, a.arm, b) for b in measured]
    recs = [r for r in recs if r]
    # co-temporal daemon evidence (n_holders) for ours
    if a.arm == "ours":
        ev = dex("grep -aE 'n_holders|hint push|S2' /tmp/daemon.log | tail -5", timeout=20)
        print("  [ours] daemon n_holders evidence:", flush=True)
        for ln in (ev.stdout or "").strip().splitlines():
            print("     " + ln, flush=True)
    if recs:
        news = [r["reprefill_new"] for r in recs if r["reprefill_new"] >= 0]
        m = st.mean(news); s = st.pstdev(news) if len(news) > 1 else 0.0
        print(f"=== [{a.arm}] N={len(news)} reprefill(new) = {m:.0f}±{s:.0f}  "
              f"per-block={news} ===", flush=True)


if __name__ == "__main__":
    main()

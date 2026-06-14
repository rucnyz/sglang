#!/usr/bin/env python3
"""A/B orchestrator (host-side): thunderagent_router vs aginfer_router on the 4-tier
HiCache sglang backend. Switches the router, runs the fleet driver N cycles per arm,
parses re-prefill / cache-hit from the backend log over each cycle's wall window, and
prints per-cycle + per-arm mean +/- std.

Backend (dynamo.sglang 4-tier, --max-total-tokens cap) + frontend (:8100) must already
be running in container `aginfer_dyn`. Only the router is swapped here.

Usage: python3 run_ab.py --cycles 3 --classA 24 --turnsA 8 --tokA 700 \
          --classB 10 --turnsB 2 --tokB 2200 --gap 0.3 --maxtok 100
"""
import argparse, json, subprocess, sys, time, statistics as st

C = "aginfer_dyn"
BK = "/tmp/sglang_backend.log"

def dex(cmd, timeout=200):
    return subprocess.run(["docker", "exec", C, "bash", "-lc", cmd],
                          capture_output=True, text=True, timeout=timeout)

def dex_d(cmd):
    subprocess.run(["docker", "exec", "-d", C, "bash", "-lc", cmd], check=False)

def switch_router(arm, model):
    mod = "dynamo.thunderagent_router" if arm == "ta" else "dynamo.aginfer_router"
    dex('pkill -9 -f "dynamo.thunderagent_router|dynamo.aginfer_router" 2>/dev/null; sleep 2')
    dex_d(f"python -m {mod} --endpoint dynamo.backend.generate --model-name {model} "
          f"--router-block-size 64 --router-reset-states > /tmp/router_{arm}.log 2>&1")
    for i in range(30):
        r = dex('curl -s -m8 http://localhost:8100/v1/chat/completions -H "Content-Type: application/json" '
                '-d \'{"model":"' + model + '","messages":[{"role":"user","content":"hi"}],"max_tokens":4,'
                '"nvext":{"agent_context":{"trajectory_id":"smoke","session_id":"s","session_type_id":"agent"}}}\' '
                '2>/dev/null | grep -c chatcmpl')
        if r.stdout.strip().endswith("1"):
            print(f"  [{arm}] router ready (try {i+1})", flush=True)
            return True
        time.sleep(2)
    print(f"  [{arm}] router NOT ready", flush=True)
    return False

def run_cycle(a, arm, cyc):
    salt = f"{arm}c{cyc}_{int(time.time())}"
    cmd = (f"cd /workspace/sglang/dev/dynamo && timeout 160 python fleet_ab.py "
           f"--base http://localhost:8100 --model {a.model}  --classA {a.classA} --turnsA {a.turnsA} --tokA {a.tokA} "
           f"--classB {a.classB} --turnsB {a.turnsB} --tokB {a.tokB} --gap {a.gap} "
           f"--max-tokens {a.maxtok} --req-timeout 90 --run-salt {salt} --tag {arm}_c{cyc} "
           f"2>&1 | grep '^RESULT' | sed 's/^RESULT //'")
    out = dex(cmd, timeout=200)
    line = out.stdout.strip().splitlines()[-1] if out.stdout.strip() else ""
    try:
        client = json.loads(line)
    except Exception:
        print(f"    [{arm} c{cyc}] driver FAILED: {out.stdout[-300:]}", flush=True)
        return None
    time.sleep(3)  # flush trailing prefill stats
    pw = dex(f"python3 /workspace/sglang/dev/dynamo/parse_window.py {BK} "
             f"{client['t_start_wall']} {client['t_end_wall']+4}")
    try:
        prefill = json.loads(pw.stdout.strip().splitlines()[-1])
    except Exception:
        prefill = {"new": -1, "cached": -1, "cache_hit_pct": -1, "peak_util": -1}
    rec = {"arm": arm, "cycle": cyc, "prefill": prefill, "client": client}
    print(f"    [{arm} c{cyc}] makespan={client['makespan_s']}s fails={client['fails']} "
          f"to={client['timeouts']} | re-prefill={prefill['new']} cache-hit={prefill['cache_hit_pct']}% "
          f"peak_util={prefill['peak_util']} | ttft_mean={client['ttft_mean']} "
          f"resume_p95={client['resume_ttft_p95']}", flush=True)
    return rec

def summarize(recs, arm):
    recs = [r for r in recs if r]
    if not recs:
        print(f"  [{arm}] NO valid cycles"); return
    def col(path):
        vals = []
        for r in recs:
            d = r
            for k in path: d = d[k]
            if d is not None: vals.append(d)
        return vals
    def ms(vals):
        if not vals: return "n/a"
        m = st.mean(vals); s = st.pstdev(vals) if len(vals) > 1 else 0.0
        return f"{m:.1f}±{s:.1f}"
    print(f"  [{arm}] N={len(recs)}  "
          f"re-prefill={ms(col(['prefill','new']))}  "
          f"cache-hit%={ms(col(['prefill','cache_hit_pct']))}  "
          f"makespan={ms(col(['client','makespan_s']))}  "
          f"ttft_mean={ms([v*1000 for v in col(['client','ttft_mean'])])}ms  "
          f"resume_p95={ms([v*1000 for v in col(['client','resume_ttft_p95'])])}ms  "
          f"fails={sum(col(['client','fails']))}", flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, default=3)
    ap.add_argument("--classA", type=int, default=24); ap.add_argument("--turnsA", type=int, default=8)
    ap.add_argument("--tokA", type=int, default=700)
    ap.add_argument("--classB", type=int, default=10); ap.add_argument("--turnsB", type=int, default=2)
    ap.add_argument("--tokB", type=int, default=2200)
    ap.add_argument("--gap", type=float, default=0.3); ap.add_argument("--maxtok", type=int, default=100)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--order", default="ta,ag")
    a = ap.parse_args()
    print(f"=== A/B  cycles={a.cycles}  A={a.classA}x{a.turnsA}@{a.tokA}  "
          f"B={a.classB}x{a.turnsB}@{a.tokB}  gap={a.gap} maxtok={a.maxtok} ===", flush=True)
    allrecs = {}
    for arm in a.order.split(","):
        if not switch_router(arm, a.model):
            continue
        recs = [run_cycle(a, arm, c + 1) for c in range(a.cycles)]
        allrecs[arm] = recs
    print("=== SUMMARY ===", flush=True)
    for arm in a.order.split(","):
        summarize(allrecs.get(arm, []), arm)

if __name__ == "__main__":
    main()

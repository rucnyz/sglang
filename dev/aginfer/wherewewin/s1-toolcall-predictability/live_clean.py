"""Clean LIVE autonomous win: sustained background load (full eviction + DENSE
event stream so the warm fires on time) while victims park-and-resume.

The 10-program synchronized-gap driver gave only ~18% because gaps overlapped →
PARTIAL eviction + sparse events → late warm.  Here a continuous background of
floods (a) keeps HBM pressure high so a parked victim's prefix FULLY evicts during
its gap, and (b) posts llm_prefill events every request (as a real proxy would) so
the action-timeline clock advances densely and the daemon's warm lands before the
resume.  Victims measure resume TTFT; arm=ours posts events (daemon warms),
arm=b does not (recompute).  Same background both arms.
"""
import requests, time, statistics, sys, threading, json as J
B = "http://127.0.0.1:30000"; D = "http://127.0.0.1:9100"; V = 129000
ARM = sys.argv[1] if len(sys.argv) > 1 else "ours"
import os
VICT = int(os.environ.get("LC_VICT", "30000")); ETA = float(os.environ.get("LC_ETA", "12.0"))
N_VICT = int(os.environ.get("LC_NVICT", "3")); BG_THREADS = int(os.environ.get("LC_BG", "1"))
BG_LEN = int(os.environ.get("LC_BGLEN", "30000")); RUN_S = float(os.environ.get("LC_RUN", "100"))

stop = threading.Event()


def seq(s, n): return [(s + i) % V for i in range(n)]


def gen(ids, pid, mx=2):
    try:
        requests.post(B + "/generate", json={"input_ids": ids, "sampling_params":
            {"temperature": 0, "max_new_tokens": mx, "ignore_eos": True},
            "program_id": pid}, timeout=300)
    except Exception:
        pass


def ev(kind, pid, payload=None):
    try:
        requests.post(D + "/aginfer/event", json={"kind": kind, "session": pid,
                      **(payload or {})}, timeout=10)
    except Exception:
        pass


def reg(pid, ids):
    try:
        requests.post(D + "/aginfer/session_prefix",
                      json={"program_id": pid, "input_ids": ids}, timeout=15)
    except Exception:
        pass


def ttft(ids, pid):
    body = {"input_ids": ids, "sampling_params": {"temperature": 0,
            "max_new_tokens": 6, "ignore_eos": True}, "program_id": pid, "stream": True}
    t0 = time.perf_counter(); tt = None; c = 0
    with requests.post(B + "/generate", json=body, timeout=300, stream=True) as r:
        for line in r.iter_lines():
            if not line:
                continue
            if tt is None:
                tt = (time.perf_counter() - t0) * 1000
            s = line.decode() if isinstance(line, bytes) else line
            if s.startswith("data:"):
                s = s[5:].strip()
            if s and s != "[DONE]":
                try:
                    c = int((J.loads(s).get("meta_info") or {}).get("cached_tokens") or c)
                except Exception:
                    pass
            if tt and c:
                break
    return tt, c


def background(tid):
    """Continuous distinct floods + DENSE llm_prefill events (as a real proxy)."""
    k = 0
    while not stop.is_set():
        pid = f"BG{tid}_{k}"
        ev("llm_prefill", pid)                       # dense event stream
        gen(seq((11000 + tid * 6007 + k * 311) % V, BG_LEN), pid, mx=2)
        k += 1


resume_rows = []
rlock = threading.Lock()


def victim(vid_i):
    inject = (ARM in ("ours", "ta"))
    pid = f"VIC{vid_i}"
    cycle = 0
    time.sleep(vid_i * 2.0)                           # stagger victims
    while not stop.is_set():
        vids = seq((vid_i * 7001 + 1 + cycle * 97) % V, VICT)
        gen(vids, pid, mx=4)                          # establish/refresh the prefix
        if inject:
            ev("session_arrival", pid); ev("llm_prefill", pid)
            reg(pid, vids)
            ev("tool_call_start", pid, {"tool_name": "bash",
               "tool_args": {"command": f"sleep {int(ETA)}"}, "tool_eta_s": ETA})
        gap_end = time.time() + ETA
        while time.time() < gap_end and not stop.is_set():
            time.sleep(0.3)                           # parked: background evicts the prefix
        to, co = ttft(vids, pid)                      # resume
        with rlock:
            resume_rows.append((to, co))
        cycle += 1
    if inject:
        ev("session_end", pid)


bg = [threading.Thread(target=background, args=(i,), daemon=True) for i in range(BG_THREADS)]
vt = [threading.Thread(target=victim, args=(i,), daemon=True) for i in range(N_VICT)]
for t in bg:
    t.start()
time.sleep(3.0)                                       # warm up background pressure
for t in vt:
    t.start()
time.sleep(RUN_S)
stop.set()
time.sleep(3.0)

rows = [r for r in resume_rows if r[0] is not None]
ttfts = sorted(r[0] for r in rows)
cached = [r[1] for r in rows]
n = len(ttfts)
print(f"=== LIVE-CLEAN arm={ARM}  victim resumes={n} (background={BG_THREADS} floods) ===")
if n:
    print(f"resume TTFT ms: mean={statistics.mean(ttfts):.0f}  p50={ttfts[n//2]:.0f}  "
          f"p90={ttfts[min(n-1,int(0.9*n))]:.0f}")
    print(f"resume cached: mean={statistics.mean(cached):.0f}/{VICT}  "
          f"(fully-evicted-then-hit fraction)")
    print(J.dumps({"arm": ARM, "n": n, "ttft_mean": statistics.mean(ttfts),
                   "cached_mean": statistics.mean(cached)}))

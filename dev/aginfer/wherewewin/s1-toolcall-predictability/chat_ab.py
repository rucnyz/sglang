"""S1 chat-interface 3-arm A/B: B vs ThunderAgent (TA) vs Ours, all on
/v1/chat/completions (so TA can proxy them).  Each program: establish a long
chat prefix -> park (tool gap) -> resume the SAME prefix, measuring resume TTFT +
cached prompt tokens.  Staggered establishes form a burst of memory pressure that
evicts the parked prefixes; the gap then leaves GPU idle.

Arms:
  b    : chat -> sglang :30000 (HiCache only).
  ta   : chat -> ThunderAgent :9000 (router-side pause; HiCache-blind, never promotes).
  ours : chat -> :30000, AND during the gap the predictive WARM (a chat call with
         max_tokens=0 = prefill-only) re-stages the prefix into HBM before resume.
         (Driver-executed warm = the production "proxy executes the daemon's
         action-timeline promote" model; the daemon-autonomous warm is proven
         separately via the token interface in auto_victim_warm.py.)

Usage: chat_ab.py <b|ta|ours>   (env: LC_NVICT, LC_CTX_REPEAT, LC_ETA, LC_STAGGER, LC_WARMLEAD)
"""
import requests, time, statistics, sys, threading, json as J, os
MODEL = "deepseek-ai/DeepSeek-V4-Flash"
ARM = sys.argv[1] if len(sys.argv) > 1 else "ours"
GEN = "http://127.0.0.1:9000" if ARM == "ta" else "http://127.0.0.1:30000"
SGLANG = "http://127.0.0.1:30000"   # the warm always goes to the backend directly
N_VICT = int(os.environ.get("LC_NVICT", "12"))
CTX_REPEAT = int(os.environ.get("LC_CTX_REPEAT", "3000"))   # ~10 tok/repeat -> ~30K-tok prefix
ETA = float(os.environ.get("LC_ETA", "16.0"))
STAGGER = float(os.environ.get("LC_STAGGER", "2.0"))
WARMLEAD = float(os.environ.get("LC_WARMLEAD", "3.0"))

stop = threading.Event()
rows = []; rlock = threading.Lock()


def prefix_msgs(vid):
    # distinct long prefix per program (so they evict each other, not share)
    ctx = (f"Session {vid} context. " +
           ("The quick brown fox jumps over the lazy dog. " * CTX_REPEAT))
    return [{"role": "system", "content": ctx}]


def chat_ttft(msgs, pid, url, max_tokens=6):
    body = {"model": MODEL, "messages": msgs, "max_tokens": max_tokens,
            "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}
    headers = {"X-Session-ID": pid}            # TA tracks per-program
    t0 = time.perf_counter(); tt = None; cached = None
    try:
        with requests.post(url + "/v1/chat/completions", json=body, headers=headers,
                           timeout=75, stream=True) as r:
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
                        u = (J.loads(s).get("usage") or {})
                        d = u.get("prompt_tokens_details") or {}
                        if d.get("cached_tokens") is not None:
                            cached = d["cached_tokens"]
                    except Exception:
                        pass
    except Exception as e:
        return None, None
    return tt, cached


def warm(msgs, pid):
    """Predictive promote: prefill-only chat (max_tokens=0) -> stage prefix into HBM."""
    try:
        requests.post(SGLANG + "/v1/chat/completions",
                      json={"model": MODEL, "messages": msgs, "max_tokens": 0, "temperature": 0},
                      headers={"X-Session-ID": pid}, timeout=300)
    except Exception:
        pass


def victim(vid_i):
    pid = f"VIC{vid_i}"
    time.sleep(vid_i * STAGGER)
    msgs = prefix_msgs(vid_i)
    chat_ttft(msgs, pid, GEN, max_tokens=4)            # establish the prefix
    gap_end = time.time() + ETA
    warmed = False
    while time.time() < gap_end and not stop.is_set():
        if ARM == "ours" and not warmed and time.time() >= gap_end - WARMLEAD:
            warm(msgs, pid)                            # predictive promote during the gap
            warmed = True
        time.sleep(0.3)
    to, co = chat_ttft(msgs + [{"role": "user", "content": "continue"}], pid, GEN, max_tokens=6)
    with rlock:
        rows.append((to, co))


T0 = time.perf_counter()
vt = [threading.Thread(target=victim, args=(i,), daemon=True) for i in range(N_VICT)]
for t in vt:
    t.start()
for t in vt:
    t.join()

MAKESPAN = time.perf_counter() - T0
ok = [r for r in rows if r[0] is not None]
ttfts = sorted(r[0] for r in ok)
cached = [r[1] for r in ok if r[1] is not None]
n = len(ttfts)
print(f"=== CHAT-AB arm={ARM}  resumes={n} (N_VICT={N_VICT}, ctx~{CTX_REPEAT*10//1000}K tok) ===")
if n:
    print(f"resume TTFT ms: mean={statistics.mean(ttfts):.0f}  p50={ttfts[n//2]:.0f}  p90={ttfts[min(n-1,int(0.9*n))]:.0f}")
    if cached:
        print(f"resume cached_tokens: mean={statistics.mean(cached):.0f}")
    thr = n / MAKESPAN if MAKESPAN>0 else 0
    print(f"makespan(wall) = {MAKESPAN:.1f}s   completed = {n}/{N_VICT}   throughput = {thr:.3f} prog/s")
    print(J.dumps({"arm": ARM, "n": n, "completed_frac": n/N_VICT, "makespan_s": MAKESPAN, "throughput_prog_s": thr, "ttft_mean": statistics.mean(ttfts), "cached_mean": (statistics.mean(cached) if cached else None)}))

"""S1 DISK-tier win, end-to-end & controlled (no daemon, no regime).

Forcing DISK residence via the migrate API is blocked (units aren't host-leaves;
state doesn't report DISK). So create the eviction by FLOODING: build a victim
prefix P, then flood K other large prefixes so the working set exceeds HBM+DRAM
(=pool*2.5 at hicache_ratio 1.5) -> P is evicted past DRAM onto mooncake/DISK
(or dropped). Then:
  B    = time a streaming resume of P directly (pays the deep load_back / recompute
         on the critical path).
  ours = ACCESS P once untimed first (the predictive promote: load happens during
         the idle gap, OFF the critical path), THEN time the resume (hits HBM).
The TTFT delta = the cost S1's pre-stage moves off the resume critical path.
"""
import requests, time, statistics, sys
B = "http://127.0.0.1:30000"; V = 129000
VICTIM = int(sys.argv[1]) if len(sys.argv) > 1 else 50000
FLOOD_N = int(sys.argv[2]) if len(sys.argv) > 2 else 8
FLOOD_LEN = int(sys.argv[3]) if len(sys.argv) > 3 else 50000
TRIALS = int(sys.argv[4]) if len(sys.argv) > 4 else 3


def seq(salt, n):
    return [(salt + i) % V for i in range(n)]


def gen(ids, pid, mx=4):
    requests.post(B + "/generate", json={"input_ids": ids, "sampling_params":
        {"temperature": 0, "max_new_tokens": mx, "ignore_eos": True}, "program_id": pid}, timeout=300).raise_for_status()


def gen_stream_ttft(ids, pid):
    body = {"input_ids": ids, "sampling_params": {"temperature": 0, "max_new_tokens": 6, "ignore_eos": True},
            "program_id": pid, "stream": True}
    t0 = time.perf_counter(); ttft = None; cached = 0
    with requests.post(B + "/generate", json=body, timeout=300, stream=True) as r:
        r.raise_for_status()
        import json as _j
        for line in r.iter_lines():
            if not line:
                continue
            if ttft is None:
                ttft = (time.perf_counter() - t0) * 1000.0
            s = line.decode() if isinstance(line, bytes) else line
            if s.startswith("data:"):
                s = s[5:].strip()
            if s and s != "[DONE]":
                try:
                    cached = int((_j.loads(s).get("meta_info") or {}).get("cached_tokens") or cached)
                except Exception:
                    pass
            if ttft is not None and cached:
                break
    return ttft, cached


def res_of(pid):
    s = requests.get(B + "/aginfer/state", timeout=30).json()
    out = []
    for rk in s.get("per_rank", [s]):
        for u in rk.get("units", []):
            if pid in (u.get("session_ids") or []):
                out.append(sorted(u.get("residence") or []))
    return out


def flood(tag):
    # distinct salts per flood unit; enough to exceed HBM+DRAM
    for k in range(FLOOD_N):
        gen(seq((90000 + k * 4001 + tag * 131) % V, FLOOD_LEN), f"FL{tag}_{k}", mx=2)


b_ttft, ours_ttft = [], []
for t in range(TRIALS):
    vid = f"VICT{t}"
    vids = seq((t * 7001 + 1) % V, VICTIM)
    gen(vids, vid, mx=4); time.sleep(0.5)               # cache victim P (HBM+DRAM, + mooncake write-through)
    # ---- B: flood to evict P deep, then time resume directly
    flood(2 * t)
    rb = res_of(vid)
    tb, cb = gen_stream_ttft(vids, vid)
    # ---- ours: re-evict P (the B resume re-homed it), then PRE-ACCESS (off-timer), then time resume
    flood(2 * t + 1)
    gen(vids, vid, mx=2)                                # predictive promote: load P back, UNTIMED
    to, co = gen_stream_ttft(vids, vid)
    b_ttft.append(tb); ours_ttft.append(to)
    bmode = "DISK-load" if cb >= VICTIM * 0.8 else ("recompute" if cb < VICTIM * 0.2 else "partial")
    print(f"trial {t}: B={tb:.0f}ms(cached={cb},{bmode})  ours={to:.0f}ms(cached={co})  saved={tb-to:.0f}ms  res_before_B={rb}")

print(f"\n=== S1 DISK-tier microbench (victim={VICTIM}, flood={FLOOD_N}x{FLOOD_LEN}, N={TRIALS}) ===")
print(f"B    (resume from deep tier): mean={statistics.mean(b_ttft):.0f}ms  {[round(x) for x in b_ttft]}")
print(f"ours (pre-staged to HBM):     mean={statistics.mean(ours_ttft):.0f}ms  {[round(x) for x in ours_ttft]}")
sv = statistics.mean(b_ttft) - statistics.mean(ours_ttft)
print(f"=> ours saves {sv:.0f}ms/resume ({100*sv/statistics.mean(b_ttft):.0f}% faster)")

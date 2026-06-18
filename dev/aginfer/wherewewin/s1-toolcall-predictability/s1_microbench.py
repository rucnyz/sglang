"""S1 win, isolated & deterministic (no regime, no pool tuning).

The S1 claim distilled: a resume whose prefix is PRE-STAGED in HBM (ours: the
predictive promote ran during the gap) beats one that must LOAD_BACK from DRAM
on access (B: HiCache on-access). This measures exactly that per-resume delta,
controlling the prefix's residence by hand (the demote/promote that the daemon
would do automatically; the mechanism is already proven by probe_mimic 5/5).

For each trial: build prefix P (cache it). Then alternately measure the resume
TTFT (streaming /generate of P) with P forced to DRAM-only [B: load_back] vs P
forced back to HBM [ours: pre-staged]. Prediction (pre-registered): ours < B,
gap = the 12K-token DRAM->HBM load_back latency.
"""
import requests, time, statistics, sys
B = "http://127.0.0.1:30000"; V = 129000
PREFIX = int(sys.argv[1]) if len(sys.argv) > 1 else 12000
TRIALS = int(sys.argv[2]) if len(sys.argv) > 2 else 5


def units(pid):
    s = requests.get(B + "/aginfer/state", timeout=30).json()
    out = []
    for rk in s.get("per_rank", [s]):
        for u in rk.get("units", []):
            if pid in (u.get("session_ids") or []):
                out.append({"hash": u["hash"], "res": sorted(u.get("residence") or []),
                            "dl": u.get("is_device_leaf"),
                            "nb": sum(sum(sp.values()) for sp in (u.get("n_bytes") or {}).values())})
    return out


def migrate(actions):
    return requests.post(B + "/aginfer/migrate", json={"actions": actions}, timeout=30).json()


def gen_stream_ttft(ids, pid):
    """Streaming /generate; return TTFT ms (time to first token)."""
    body = {"input_ids": ids, "sampling_params": {"temperature": 0, "max_new_tokens": 8, "ignore_eos": True},
            "program_id": pid, "stream": True}
    t0 = time.perf_counter(); ttft = None
    with requests.post(B + "/generate", json=body, timeout=120, stream=True) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if line and ttft is None:
                ttft = (time.perf_counter() - t0) * 1000.0
                break
    return ttft


def force_dram_only(pid, ids):
    """Demote P's whole chain to DRAM (leaf-inward, sglang sorts deepest-first)."""
    for _ in range(6):
        us = [u for u in units(pid) if "HBM" in u["res"]]
        if not us:
            return
        acts = []
        for u in us:
            if u["res"] == ["HBM"]:
                acts.append({"hash": u["hash"], "add_tiers": ["DRAM"], "remove_tiers": ["HBM"], "action_id": "d"})
            else:
                acts.append({"hash": u["hash"], "add_tiers": [], "remove_tiers": ["HBM"], "action_id": "d"})
        migrate(acts); time.sleep(0.4)


def force_hbm(pid):
    """Promote P's DRAM-only units back to HBM (pre-stage)."""
    us = [u for u in units(pid) if u["res"] == ["DRAM"]]
    if us:
        migrate([{"hash": u["hash"], "add_tiers": ["HBM"], "remove_tiers": [], "action_id": "p"} for u in us])
        time.sleep(0.6)


b_ttft, ours_ttft = [], []
for t in range(TRIALS):
    pid = f"MB{t}"
    salt = (t * 5003) % V
    ids = [(salt + i) % V for i in range(PREFIX)]
    # build + cache P (HBM+DRAM via write-through over a couple of touches)
    requests.post(B + "/generate", json={"input_ids": ids, "sampling_params": {"temperature": 0, "max_new_tokens": 4, "ignore_eos": True}, "program_id": pid}, timeout=120)
    time.sleep(1.0)
    # ---- B: P in DRAM only -> resume load_backs
    force_dram_only(pid, ids)
    res = [tuple(u["res"]) for u in units(pid)]
    tb = gen_stream_ttft(ids, pid)
    # ---- ours: P pre-staged to HBM -> resume hits HBM
    force_dram_only(pid, ids)   # the just-run resume re-homed it to HBM; demote again
    force_hbm(pid)
    to = gen_stream_ttft(ids, pid)
    b_ttft.append(tb); ours_ttft.append(to)
    print(f"trial {t}: B(load_back)={tb:.1f}ms  ours(pre-staged)={to:.1f}ms  saved={tb-to:.1f}ms  (P res before B={res})")

print("\n=== S1 microbench: resume TTFT (prefix=%d, N=%d) ===" % (PREFIX, TRIALS))
print(f"B   (DRAM load_back): mean={statistics.mean(b_ttft):.1f}ms  ({[round(x) for x in b_ttft]})")
print(f"ours(pre-staged HBM): mean={statistics.mean(ours_ttft):.1f}ms  ({[round(x) for x in ours_ttft]})")
print(f"=> ours saves {statistics.mean(b_ttft)-statistics.mean(ours_ttft):.1f}ms/resume "
      f"({100*(1-statistics.mean(ours_ttft)/statistics.mean(b_ttft)):.0f}% faster)")

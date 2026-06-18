"""Isolate the batch-peel fix against the daemon's exact batch shape.

Builds a 2-turn chain, then sends ONE /aginfer/migrate batch shaped like the
daemon's joint_decide plan: the HBM-only fresh leaf via (add DRAM, remove HBM)
and the [HBM,DRAM] internals via (remove HBM).  Predicts: with the batch-aware
device-leaf re-derivation, the whole chain peels (applied == len).  If it fails
with remove_hbm_not_device_leaf:dev_children, the batch-peel fix has a bug.
"""
import requests, time
from collections import Counter
B = "http://127.0.0.1:30000"; PID = "PROBE-MIMIC"; V = 129000


def gen(ids, forced, mx):
    sp = {"temperature": 0.0, "max_new_tokens": mx, "ignore_eos": True}
    if forced is not None:
        sp["custom_params"] = {"forced_output_ids": forced}
    requests.post(B + "/generate", json={"input_ids": ids, "sampling_params": sp,
                  "program_id": PID, "stream": False}, timeout=120).raise_for_status()


def units():
    s = requests.get(B + "/aginfer/state", timeout=30).json()
    out = []
    for rk in s.get("per_rank", [s]):
        for u in rk.get("units", []):
            if PID in (u.get("session_ids") or []):
                nb = sum(sum(sp.values()) for sp in (u.get("n_bytes") or {}).values())
                out.append({"hash": u["hash"], "bytes": nb,
                            "res": sorted(u.get("residence") or []),
                            "dl": u.get("is_device_leaf")})
    return out


pre = [(23 + i) % V for i in range(4000)]
gen(pre, [(930000 + i) % V for i in range(400)], 400)
gen(pre + [(930000 + i) % V for i in range(400)], [(980000 + i) % V for i in range(300)], 300)
time.sleep(1.5)
us = units()
print("chain:", [(u['bytes'], u['res'], 'L' if u['dl'] else 'i')
                 for u in sorted(us, key=lambda x: -x['bytes'])])
acts = []
for i, u in enumerate(us):
    if u['res'] == ['HBM']:
        acts.append({"hash": u["hash"], "add_tiers": ["DRAM"], "remove_tiers": ["HBM"], "action_id": f"m{i}"})
    elif 'HBM' in u['res']:
        acts.append({"hash": u["hash"], "add_tiers": [], "remove_tiers": ["HBM"], "action_id": f"m{i}"})
resp = requests.post(B + "/aginfer/migrate", json={"actions": acts}, timeout=30).json()
print(f"DAEMON-MIMIC batch: applied={resp['applied']}/{len(acts)} "
      f"skip={dict(Counter(s.get('reason') for s in resp.get('skipped', [])))}")
time.sleep(1)
print("after:", [(u['bytes'], u['res']) for u in sorted(units(), key=lambda x: -x['bytes'])])

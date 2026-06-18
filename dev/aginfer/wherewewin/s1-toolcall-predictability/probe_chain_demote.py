"""Probe: can a whole multi-turn chain be demoted HBM->DRAM leaf-inward?

Creates a fresh PROBE program (prefix -> forced output -> a second turn so the
prefix becomes a non-device-leaf internal node), then issues /aginfer/migrate
remove-HBM leaf-inward and checks whether the BULK prefix actually reaches DRAM
(residence loses HBM).  This de-risks the whole-chain-demote design before the
daemon change: if sglang peels the chain leaf-inward, the daemon just has to
propose the chain in that order.
"""
import requests, time, sys
B = "http://127.0.0.1:30000"
PID = "PROBE-CHAIN"
VOCAB = 129000


def gen(ids, forced, maxnew):
    sp = {"temperature": 0.0, "max_new_tokens": maxnew, "ignore_eos": True}
    if forced is not None:
        sp["custom_params"] = {"forced_output_ids": forced}
    r = requests.post(B + "/generate", json={"input_ids": ids, "sampling_params": sp,
                                             "program_id": PID, "stream": False}, timeout=120)
    r.raise_for_status()


def units():
    s = requests.get(B + "/aginfer/state", timeout=30).json()
    r = s.get("per_rank", [s])
    out = []
    for rk in r:
        for u in rk.get("units", []):
            if PID in (u.get("session_ids") or []):
                nb = sum(sum(sp.values()) for sp in (u.get("n_bytes") or {}).values())
                out.append({"hash": u["hash"], "bytes": nb, "res": sorted(u.get("residence") or []),
                            "devleaf": u.get("is_device_leaf", True), "ntok": u.get("n_tokens")})
    return out


def post_migrate(actions):
    r = requests.post(B + "/aginfer/migrate", json={"actions": actions}, timeout=30)
    r.raise_for_status()
    return r.json()


# Build chain: prefix(4000) -> out0(forced 500) ; then second turn so prefix is internal.
prefix = [(7 + i) % VOCAB for i in range(4000)]
out0 = [(900000 + i) % VOCAB for i in range(500)]
gen(prefix, out0, 500)
gen(prefix + out0, [(950000 + i) % VOCAB for i in range(300)], 300)
for _ in range(20):
    us = units()
    if us:
        break
    time.sleep(0.4)
print("=== PROBE units (the program's chain) ===")
for u in sorted(units(), key=lambda x: -x["bytes"]):
    print(f"  hash={u['hash']} bytes={u['bytes']} ntok={u['ntok']} res={u['res']} devleaf={u['devleaf']}")

# Demote leaf-inward: repeatedly find a device-leaf HBM unit of PID, remove HBM, re-check.
print("=== leaf-inward remove-HBM ===")
for step in range(8):
    us = [u for u in units() if "HBM" in u["res"]]
    if not us:
        print("  all HBM removed for PROBE chain"); break
    leaves = [u for u in us if u["devleaf"]]
    if not leaves:
        print(f"  step{step}: {len(us)} HBM units left but NONE is a device-leaf -> peel STUCK");
        for u in us: print(f"     stuck hash={u['hash']} bytes={u['bytes']} devleaf={u['devleaf']}")
        break
    tgt = min(leaves, key=lambda x: x["bytes"])  # peel smallest leaf first
    resp = post_migrate([{"hash": tgt["hash"], "add_tiers": [], "remove_tiers": ["HBM"],
                          "action_id": f"peel{step}"}])
    time.sleep(0.5)
    after = next((u for u in units() if u["hash"] == tgt["hash"]), None)
    print(f"  step{step}: remove HBM {tgt['hash']} ({tgt['bytes']}B) applied={resp['applied']} "
          f"skipped={[s.get('reason') for s in resp.get('skipped',[])]} -> res_now={after['res'] if after else 'GONE'}")
print("=== final PROBE residence ===")
for u in sorted(units(), key=lambda x: -x["bytes"]):
    print(f"  hash={u['hash']} bytes={u['bytes']} res={u['res']} devleaf={u['devleaf']}")

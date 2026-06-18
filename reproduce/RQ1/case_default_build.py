"""Build the CORRECTED RQ1 case2 + case3 traces for the DEFAULT-split methodology.

HARD CONSTRAINT (why this rebuild exists). The A/B comparison must NOT tune
--mamba-full-memory-ratio: both base and sys run at sglang's DEFAULT split
(mamba_full_memory_ratio=0.9 -> mamba ~21GB / max_running ~147, KV ~24GB / ~560k
tokens). The system's value is auto-reallocating FROM that default; hand-setting
the ratio does its job manually and is forbidden. The prior case2/case3 builders
forced a mamba-bound regime with RATIO=0.1; that is exactly what we may no longer
do. Regimes here are created by SHAPING THE WORKLOAD, not the ratio.

REGIME MATH AT THE DEFAULT SPLIT. Real CC is intrinsically KV-bound: a real
session carries a 16k+ system prefix, so a handful of long contexts fill the
~560k-token KV pool long before the mamba cap (~147 concurrent) binds. To create
a mamba-bound (concurrency-bound) regime at the DEFAULT split we must TRUNCATE
sessions into SHORT requests and run them at HIGH concurrency, so max_running
(~147) binds before KV fills:

    per-request KV footprint must be < KV_pool / max_running ~= 560000/147 ~= 3800
    tokens. A ~3500-token prompt + ~192 forced output stays under that, so ~256
    such requests in flight hit the mamba cap (queue builds) while KV sits ~idle.

TOKEN-EXACTNESS. Prompt = the first SWARM_PROMPT_LEN tokens of a real session's
input_ids (an exact PREFIX); forced output = the first SWARM_OUT_LEN tokens of a
real record's forced_output_ids (an exact PREFIX). Both come from ONE chosen real
record per source root (the root's richest turn, so >=128 forced tokens exist), so
each is an exact slice of real tokens. No token is fabricated. Truncation +
duplication of real-CC tokens (distinct program_ids) is explicitly sanctioned.

BUILD 1 -> data/traces/cc_qwen_case2_swarm.jsonl
    A burst of N_SWARM (~288) SHORT childless root requests, all arriving in a
    tight window, enough to exceed max_running ~147 and keep mamba slots busy with
    KV idle. Sourced by TRUNCATING + DUPLICATING the 73 real cc_qwen_t6 roots.
    Replayed with --stagger 0.02 (swarm: uniform tight stagger, not the trace
    timeline). At the default split: base hits max_running ~147 (queue builds, KV
    idle) -> mamba-bound; sys grows mamba from idle KV (k2m) so max_running rises
    and the swarm drains.

BUILD 2 -> data/traces/cc_qwen_case3_default.jsonl
    A workload-driven binding FLIP at the DEFAULT split, on ONE timeline:
      Phase A (KV-bound):  N_PHASE_A (~10) real LONG root-only sessions from
          cc_qwen3p5_9b, each max prompt-len in [60k, 200k), spawning over
          [0, T_A]. At the default split a few long contexts saturate the ~560k KV
          pool long before max_running 147 -> KV-bound, mamba idle. sys grows KV
          from idle mamba (m2k).
      Phase B (mamba-bound):  the SAME short swarm as BUILD 1, spawning over
          [PHASE_B_START, PHASE_B_START + W_B] (>= ~900-1200s, after A drains). At
          the default split this binds max_running ~147 with KV idle -> mamba-bound.
          sys grows mamba from idle KV (k2m).
    At ONE fixed default split the binding flips A(KV)->B(mamba) purely from the
    workload; base (static default) is wrong-sized for one phase. Replayed with
    --stagger - (absolute trace t) so the recorded A->B timeline drives the flip.

Re-timing: each session is an atomic unit. Every record's t / spawn_ts is shifted
by one delta = (new start - the unit's own min t), so internal turn gaps and any
child spawn offsets are preserved exactly; only absolute placement moves. step /
parent links / token id lists / tool_gap_after are copied verbatim.
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

# agentreplay holds the raw corpus (the long phase-A source we never co-locate,
# 2.3 GB); each case's own trace lives under reproduce/RQ1/caseN/data/.
AR = "/scratch/yuzhou/projects/agentreplay"
RQ1 = os.path.dirname(os.path.abspath(__file__))
SRC_LONG = os.path.join(AR, "data/traces/cc_qwen3p5_9b.jsonl")     # THE 2.3G corpus: every case selects from this.
SRC_SHORT = SRC_LONG                                               # swarm = short 3500-tok PREFIXES of the same corpus roots.
OUT_CASE2 = os.path.join(RQ1, "case2/data/cc_qwen_case2_swarm.jsonl")
OUT_CASE3 = os.path.join(RQ1, "case3/data/cc_qwen_case3_default.jsonl")
OUT_CASE1 = os.path.join(RQ1, "case1/data/cc_qwen_case1_longkv.jsonl")

# --- swarm (case2 + case3 phase B) knobs --------------------------------------
# Per-request KV footprint must stay < KV_pool/max_running ~= 560000/147 ~= 3800
# tokens so the mamba cap binds before KV fills. 3500 prompt + 192 output = 3692
# < 3800, with margin.
SWARM_PROMPT_LEN = 3500   # exact PREFIX of a real session's input_ids.
SWARM_OUT_LEN = 192       # exact PREFIX of a real record's forced_output_ids;
                          # long enough each request holds its mamba slot through
                          # ~192 decode steps so 147 concurrency is exceeded.
N_SWARM = 6000            # >> max_running 147, and SUSTAINED: at ~192 output/req
                          # and ~30 completions/s (147 cap), the swarm runs ~200s,
                          # long enough for the 1s-tick Budgeter to observe pressure,
                          # fire k2m, and grow max_running. A short burst (288 -> 9s)
                          # finishes before the Budgeter can react. 58 corpus roots
                          # DUPLICATED ~103x with distinct program_ids.
SWARM_WINDOW = 60.0       # roots arrive over [0, SWARM_WINDOW]; with conc 256 the
                          # in-flight set stays saturated and the queue sustains.

# --- phase-A (case3 KV-bound) knobs -------------------------------------------
N_PHASE_A = 10            # ~8-12 real LONG roots; a few saturate the ~560k KV pool.
PHASE_A_MIN_PROMPT = 60000    # floor: long enough to pressure KV; p50 must be >60k.
PHASE_A_MAX_PROMPT = 200000   # ceiling: strictly < server --context-length 262144.
T_A = 150.0              # phase-A spawn window [0, T_A]: long roots stagger in.
PHASE_B_START = 1000.0   # swarm begins long after phase A drains KV (>= ~900-1200s).

# --- case1 (KV-bound m2k WIN) knobs -------------------------------------------
# The dominant win regime: a burst of the LONGEST root sessions so a handful
# saturate the ~560k KV pool (only ~5-6 admitted) while the rest queue; every
# session is one O(1) mamba slot (mamba idle << 147); and the running set is small
# enough that prefill is admission-bound (HEADROOM) not compute-bound. sys grows KV
# from idle mamba (m2k) -> admits more -> prefill throughput up, queue drains. Take
# the LONGEST roots (not phase-A's even spread) to maximize KV pressure + headroom.
CASE1_MIN_PROMPT = 64000      # only genuinely-long roots (KV-heavy donors-of-pressure).
CASE1_MAX_PROMPT = 262144     # strictly <= server --context-length (drops the 276k root).
N_CASE1 = 52                  # all long roots in the band (corpus has ~52); deep queue.


def load_sessions(path: str) -> Tuple[Dict[str, List[dict]], Dict[str, str]]:
    """Group records by program_id -> step-sorted list; return (sessions, parent_of)."""
    by: Dict[str, List[dict]] = defaultdict(list)
    with open(path, errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            by[r["program_id"]].append(r)
    parent_of: Dict[str, str] = {}
    for pid, recs in by.items():
        recs.sort(key=lambda r: (int(r.get("step", 0)), float(r.get("t", 0.0))))
        parent_of[pid] = recs[0].get("parent_program_id")
    return by, parent_of


def roots(parent_of) -> List[str]:
    return [pid for pid, par in parent_of.items() if par is None]


def root_max_in(by, pid) -> int:
    return max(len(r["input_ids"]) for r in by[pid])


def richest_turn(by, pid) -> dict:
    """The root's turn with the most forced_output_ids (so a >=128 output slice
    exists). Both the prompt prefix and the output slice are sourced from this ONE
    real record, keeping each an exact prefix of a single real trace record."""
    return max(by[pid], key=lambda r: len(r.get("forced_output_ids") or []))


# --- BUILD 1: the short swarm --------------------------------------------------

def build_swarm(start_base: float, prefix: str) -> List[dict]:
    """N_SWARM childless short roots in a tight burst over
    [start_base, start_base + SWARM_WINDOW]. Real corpus roots TRUNCATED to a
    SWARM_PROMPT_LEN prompt prefix + SWARM_OUT_LEN forced-output prefix, DUPLICATED
    (distinct program_ids) up to N_SWARM. Used directly for case2 and as case3's
    phase B."""
    short_by, short_par = load_sessions(SRC_SHORT)
    src_roots = sorted(roots(short_par))  # deterministic order
    records: List[dict] = []
    for i in range(N_SWARM):
        src_pid = src_roots[i % len(src_roots)]
        rec = richest_turn(short_by, src_pid)
        in_ids = rec["input_ids"][:SWARM_PROMPT_LEN]               # exact prefix
        out_ids = (rec.get("forced_output_ids") or [])[:SWARM_OUT_LEN]  # exact prefix
        dup = i // len(src_roots)
        start = start_base + (SWARM_WINDOW * i / max(1, N_SWARM - 1))
        records.append({
            "t": round(start, 4),
            "program_id": f"{prefix}{src_pid}__d{dup}",
            "step": 1,
            "parent_program_id": None,
            "spawned_at_step": None,
            "spawn_ts": None,
            "input_ids": in_ids,
            "forced_output_ids": out_ids,
            "tool_gap_after": 0.0,
        })
    return records


# --- BUILD 2 phase A: long KV-bound roots -------------------------------------

def build_phase_a(prefix: str) -> List[dict]:
    """N_PHASE_A real LONG root-only sessions (max prompt in [60k,200k)), each kept
    VERBATIM (all turns, token-exact), staggered over [0, T_A]. Chosen to span the
    band (p50 > 60k) for a representative KV-saturating fill, all under the 262144
    context limit."""
    long_by, long_par = load_sessions(SRC_LONG)
    cand = [pid for pid in roots(long_par)
            if PHASE_A_MIN_PROMPT <= root_max_in(long_by, pid) < PHASE_A_MAX_PROMPT]
    # Span the band: sort by max prompt-len, take an even spread so p50 stays >60k
    # and the longest (KV-heaviest) roots are represented.
    cand.sort(key=lambda pid: root_max_in(long_by, pid))
    n = min(N_PHASE_A, len(cand))
    picks = [cand[int(round(j * (len(cand) - 1) / max(1, n - 1)))] for j in range(n)]
    picks = list(dict.fromkeys(picks))  # de-dup any rounding collisions
    records: List[dict] = []
    for i, pid in enumerate(picks):
        start = (T_A * i / max(1, len(picks) - 1)) if len(picks) > 1 else 0.0
        recs = sorted(long_by[pid], key=lambda r: (int(r.get("step", 0)), float(r.get("t", 0.0))))
        t0 = min(float(r.get("t", 0.0)) for r in recs)
        delta = start - t0
        remap = f"{prefix}{pid}"
        for r in recs:
            nr = dict(r)
            nr["program_id"] = remap
            nr["parent_program_id"] = None  # root-only: subtrees dropped
            nr["t"] = round(float(r["t"]) + delta, 4)
            nr["spawn_ts"] = None
            nr["spawned_at_step"] = None
            records.append(nr)
    return records


# --- BUILD 3: case1 long-KV burst ---------------------------------------------

def build_case1() -> List[dict]:
    """The N_CASE1 LONGEST real root sessions (max prompt in [CASE1_MIN, CASE1_MAX)),
    kept VERBATIM: every turn, full real input_ids AND full real forced_output_ids,
    token-exact by construction (no truncation, no decode cap). Root-only (subtrees
    dropped). The only curation is selecting the longest roots. The runner bursts
    them in; a handful saturate the ~560k KV pool (admission-bound, mamba idle = one
    O(1) slot per session) and the rest queue -> the m2k KV-grow win regime."""
    long_by, long_par = load_sessions(SRC_LONG)
    cand = [pid for pid in roots(long_par)
            if CASE1_MIN_PROMPT <= root_max_in(long_by, pid) < CASE1_MAX_PROMPT]
    cand.sort(key=lambda pid: root_max_in(long_by, pid), reverse=True)  # longest first
    records: List[dict] = []
    for pid in cand[:N_CASE1]:
        recs = sorted(long_by[pid],
                      key=lambda r: (int(r.get("step", 0)), float(r.get("t", 0.0))))
        for r in recs:
            nr = dict(r)                       # verbatim: input_ids + forced_output_ids intact
            nr["program_id"] = f"c1_{pid}"
            nr["parent_program_id"] = None     # root-only: subtrees dropped
            nr["spawn_ts"] = None
            nr["spawned_at_step"] = None
            records.append(nr)
    return records


# --- writers + token-exactness check ------------------------------------------

def write_trace(records: List[dict], path: str) -> None:
    records = sorted(records, key=lambda r: (float(r["t"]), r["program_id"], int(r.get("step", 0))))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    return records


def check_swarm_token_exact(records: List[dict]) -> int:
    """Each swarm record's input_ids / forced_output_ids must equal the
    corresponding PREFIX of its source root's richest turn."""
    short_by, short_par = load_sessions(SRC_SHORT)
    miss = 0
    for r in records:
        # program_id = "<prefix><src_pid>__d<dup>": strip the phase prefix (case3
        # phase B prepends "B_"; case2 has none) and the "__d<dup>" duplicate tag.
        pid = r["program_id"]
        body = pid.split("__d")[0]
        src_pid = body[2:] if body.startswith("B_") else body
        rec = richest_turn(short_by, src_pid)
        if r["input_ids"] != rec["input_ids"][:SWARM_PROMPT_LEN] or \
           r["forced_output_ids"] != (rec.get("forced_output_ids") or [])[:SWARM_OUT_LEN]:
            miss += 1
    return miss


def check_phase_a_token_exact(records: List[dict]) -> int:
    """Each phase-A record's token lists must equal the source root's record at the
    same step (verbatim, exact)."""
    long_by, _ = load_sessions(SRC_LONG)
    miss = 0
    for r in records:
        pid = r["program_id"]
        src_pid = pid[2:] if pid[:2] == "A_" else pid
        step = int(r.get("step", 0))
        src = next((s for s in long_by.get(src_pid, []) if int(s.get("step", 0)) == step), None)
        if src is None or src["input_ids"] != r["input_ids"] or \
           src.get("forced_output_ids") != r.get("forced_output_ids"):
            miss += 1
    return miss


# --- validation / summary (CPU only) ------------------------------------------

REQUIRED = ("t", "program_id", "step", "parent_program_id", "spawned_at_step",
            "spawn_ts", "input_ids", "forced_output_ids", "tool_gap_after")


def _read_valid(path: str) -> List[dict]:
    out = []
    prev_t = None
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)  # round-trips as valid JSON
            for k in REQUIRED:
                assert k in r, f"record missing required key {k!r}"
            assert isinstance(r["input_ids"], list) and r["input_ids"], \
                f"empty/invalid input_ids in {r.get('program_id')}"
            assert isinstance(r["forced_output_ids"], list), "forced_output_ids not list"
            t = float(r["t"])
            assert prev_t is None or t >= prev_t, "records not sorted by t"
            prev_t = t
            out.append(r)
    return out


def pct(vals, q):
    v = sorted(vals)
    return v[int(q * (len(v) - 1))] if v else 0


def validate_case2(path: str) -> None:
    recs = _read_valid(path)
    rootids = {r["program_id"] for r in recs if r["parent_program_id"] is None}
    assert len(rootids) == len(recs), "case2 must be childless roots only"
    starts = [float(r["t"]) for r in recs]
    plen = [len(r["input_ids"]) for r in recs]
    olen = [len(r["forced_output_ids"]) for r in recs]
    miss = check_swarm_token_exact(recs)
    print("=" * 72)
    print(f"BUILD 1  case2 short-swarm: {path}")
    print(f"  records={len(recs)}  roots={len(rootids)}  childless={len(rootids)==len(recs)}")
    print(f"  ROOT COUNT = {len(rootids)}  (>147 -> exceeds max_running)  "
          f"{'OK' if len(rootids) > 147 else 'FAIL'}")
    print(f"  prompt-len: p50={pct(plen,.5)} p90={pct(plen,.9)} max={max(plen)}  "
          f"(all < ~4000)  {'OK' if max(plen) < 4000 else 'FAIL'}")
    print(f"  output-len: p50={pct(olen,.5)} p90={pct(olen,.9)} min={min(olen)} max={max(olen)}")
    print(f"  spawn_ts window: [{min(starts):.2f}s, {max(starts):.2f}s]")
    print(f"  token-exact (prefix slices vs source): {miss} mismatches  "
          f"{'OK' if miss == 0 else 'FAIL'}")
    print("=" * 72)


def validate_case3(path: str) -> None:
    recs = _read_valid(path)
    by: Dict[str, List[dict]] = defaultdict(list)
    for r in recs:
        by[r["program_id"]].append(r)
    a_roots, b_roots = [], []
    for pid, rs in by.items():
        rs.sort(key=lambda r: (int(r.get("step", 0)), float(r.get("t", 0.0))))
        start = min(float(r["t"]) for r in rs)
        max_in = max(len(r["input_ids"]) for r in rs)
        info = dict(pid=pid, start=start, max_in=max_in, nturns=len(rs))
        (a_roots if pid[:2] == "A_" else b_roots).append(info)
    a_starts = sorted(r["start"] for r in a_roots)
    b_starts = sorted(r["start"] for r in b_roots)
    a_maxin = [r["max_in"] for r in a_roots]
    b_plen = [r["max_in"] for r in b_roots]
    a_recs = [r for r in recs if r["program_id"][:2] == "A_"]
    b_recs = [r for r in recs if r["program_id"][:2] == "B_"]
    miss_a = check_phase_a_token_exact(a_recs)
    miss_b = check_swarm_token_exact(b_recs)
    sep = a_starts[-1] < b_starts[0]
    print("=" * 72)
    print(f"BUILD 2  case3 dynamic (default split): {path}")
    print(f"  records={len(recs)}  programs={len(by)}")
    print(f"  PHASE A (KV-bound, long, root-only): roots={len(a_roots)}")
    print(f"    count={len(a_roots)} (target 8-12)  {'OK' if 8 <= len(a_roots) <= 12 else 'CHECK'}")
    print(f"    max prompt-len: p50={pct(a_maxin,.5)} max={max(a_maxin)}  "
          f"(<200000)  {'OK' if max(a_maxin) < 200000 else 'FAIL'}")
    print(f"    p50 prompt-len = {pct(a_maxin,.5)} > 60000  "
          f"{'OK' if pct(a_maxin,.5) > 60000 else 'FAIL'}")
    print(f"    spawn_ts range = [{a_starts[0]:.2f}s, {a_starts[-1]:.2f}s]")
    print(f"    token-exact (verbatim vs source): {miss_a} mismatches  "
          f"{'OK' if miss_a == 0 else 'FAIL'}")
    print(f"  PHASE B (mamba-bound, short swarm): roots={len(b_roots)}")
    print(f"    count={len(b_roots)} (>147)  {'OK' if len(b_roots) > 147 else 'FAIL'}")
    print(f"    prompt-len p50={pct(b_plen,.5)} max={max(b_plen)}  (<4000)  "
          f"{'OK' if max(b_plen) < 4000 else 'FAIL'}")
    print(f"    spawn_ts range = [{b_starts[0]:.2f}s, {b_starts[-1]:.2f}s]  "
          f"(B starts ~{PHASE_B_START:.0f}s)")
    print(f"    token-exact (prefix slices vs source): {miss_b} mismatches  "
          f"{'OK' if miss_b == 0 else 'FAIL'}")
    print("  FLIP / SEPARATION:")
    print(f"    A precedes B: A starts {a_starts[0]:.1f}s, B starts {b_starts[0]:.1f}s  "
          f"{'OK' if a_starts[0] <= b_starts[0] else 'FAIL'}")
    print(f"    phases SEPARATED (A last {a_starts[-1]:.1f}s < B first {b_starts[0]:.1f}s)  "
          f"{'OK' if sep else 'FAIL (overlap)'}")
    print(f"    gap A_last -> B_first = {b_starts[0] - a_starts[-1]:.1f}s")
    print("=" * 72)


def validate_case1(path: str) -> None:
    recs = _read_valid(path)
    by = defaultdict(list)
    for r in recs:
        by[r["program_id"]].append(r)
    maxin = sorted(max(len(r["input_ids"]) for r in rs) for rs in by.values())
    n = len(by)
    p50 = pct(maxin, 0.5)
    print("=" * 72)
    print(f"case1 (KV-bound m2k burst) -> {path}")
    print(f"  roots={n}  records={len(recs)}  (verbatim root-only from the corpus)")
    print(f"  max prompt-len: min={maxin[0]} p50={p50} max={maxin[-1]} "
          f"(band [{CASE1_MIN_PROMPT}, {CASE1_MAX_PROMPT}))")
    print(f"  KV: ~{560000 // max(1, p50)} of {n} roots fit the ~560k pool at once "
          f"(rest queue = prefill headroom); mamba slots = {n} << 147 (idle donor)")
    outs = sorted(len(r.get("forced_output_ids") or []) for r in recs)
    tot_out = sum(outs)
    turns = sorted(len(rs) for rs in by.values())
    print(f"  turns/root: p50={pct(turns,.5)} max={turns[-1]}  "
          f"output-len/req: p50={pct(outs,.5)} max={outs[-1]}  "
          f"TOTAL decode tokens={tot_out} (full real outputs, no cap)")
    print(f"  count>=20 and p50>=floor: {'OK' if n >= 20 and p50 >= CASE1_MIN_PROMPT else 'CHECK'}")
    print("=" * 72)


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "case1"):
        write_trace(build_case1(), OUT_CASE1)
        validate_case1(OUT_CASE1)
    if which in ("all", "case2"):
        write_trace(build_swarm(0.0, prefix=""), OUT_CASE2)
        validate_case2(OUT_CASE2)
    if which in ("all", "case3"):
        write_trace(build_phase_a("A_") + build_swarm(PHASE_B_START, "B_"), OUT_CASE3)
        validate_case3(OUT_CASE3)


if __name__ == "__main__":
    main()

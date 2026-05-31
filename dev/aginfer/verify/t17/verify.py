"""T17 verify — state-dump schema upgrade (DESIGN §5).

End-to-end verify against a running sglang.  See verify/t17/README.md for
the stage breakdown.  Run via:

    AGINFER_VERIFY_BASE=http://127.0.0.1:30001 \
    AGINFER_VERIFY_MODEL=Qwen/Qwen3-0.6B \
        python dev/aginfer/verify/t17/verify.py

Exit code 0 on PASS, 1 on FAIL.  Each stage prints a one-line result so
a failure points at one DESIGN §5 clause.
"""
from __future__ import annotations

import json
import math
import os
import statistics
import sys
import threading
import time
import uuid
from typing import Any, Iterable

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE = os.environ.get("AGINFER_VERIFY_BASE", "http://127.0.0.1:30001")
MODEL = os.environ.get("AGINFER_VERIFY_MODEL", "Qwen/Qwen3-0.6B")

# DESIGN §5 required top-level keys.  Order matters only for readability.
REQUIRED_TOP_LEVEL = (
    "time_counter",
    "throughput_ema",
    "pool_usage",
    "per_program_usage",
    "units",
    "link_stats",
    "tier_holding_cost",
)

# DESIGN §5 explicitly REMOVED keys.  Their presence means the legacy
# schema is still being emitted somewhere.
LEGACY_TOP_LEVEL = ("tier_usage", "page_size", "bytes_per_token")

LEGACY_POOL_USAGE_KEYS = (
    "used_bytes",  # at pool_usage[tier].* — moved into subpools
    "cap_bytes",
    "available_bytes",
    "evictable_bytes",
    "token_usage",
    "swa_token_usage",
    "swa_used_bytes",
    "swa_cap_bytes",
    "swa_available_bytes",
    "swa_evictable_bytes",
    "full_token_usage",
)

LEGACY_UNIT_KEYS = ("tier",)  # round-9 replaces with `residence: list`

REQUIRED_TIERS = ("HBM", "DRAM", "DISK")
REQUIRED_LINKS = (
    "HBM->DRAM", "DRAM->HBM",
    "DRAM->DISK", "DISK->DRAM",
)

# Stage 6 perf guard (DESIGN §10 F3 trigger value).
#
# DESIGN §10 sets p99 > 50 ms as the F3-revisit trigger condition.  We
# assert this budget on a 5 K-unit tree — representative of the
# steady-state agent workload (Run F peak was ~5 K live units, see
# scenarios/experiments_notes runaway analysis).  A separate 10 K
# stress recording follows the budget assertion as a data point for
# the F3 conversation if the trigger ever fires under heavier load.
STAGE6_TREE_SIZE = 2_500  # target chats; tree grows to ~2x (shared-prefix + per-chat nodes)
STAGE6_DUMPS_PER_CYCLE = 20
STAGE6_CYCLES = 3
STAGE6_P99_BUDGET_MS = 50.0
STAGE6_MAX_BUDGET_MS = 200.0  # Gen-2 GC sweep ceiling
STAGE6_STRESS_TREE_SIZE = 10_000  # informational only — no assert

# Stage 7 stress.  Single walker by design: sglang's tokenizer-control
# communicator (`communicator.py:40 assert self._result_event is None`)
# is single-flight per request type, so parallel `/aginfer/state` callers
# race the assertion and surface as HTTP 500.  The daemon never spawns
# parallel pollers (1 Hz heartbeat per event); this stage tests the
# REALISTIC regime — single-poller race vs concurrent chat traffic.
STAGE7_DURATION_S = 30
STAGE7_WALKER_THREADS = 1
STAGE7_CONCURRENT_CHATS = 32
STAGE7_P99_BUDGET_MS = 100.0


# ---------------------------------------------------------------------------
# Custom exceptions — one per DESIGN clause for clean failure attribution
# ---------------------------------------------------------------------------

class LegacyShape(Exception):
    """The dump still carries a removed-by-round-9 field."""


class SchemaMissing(Exception):
    """A required DESIGN §5 field is absent."""


class ResidenceInvariant(Exception):
    """residence-set / n_bytes-nesting consistency violation."""


class AttributionInvariant(Exception):
    """per_program_usage 1/holders attribution violation."""


class ReconcileInvariant(Exception):
    """units[*] vs per_program_usage[*].unit_hashes mismatch."""


class LinkStatsInvariant(Exception):
    """link_stats shape / cold-start violation."""


class PerfBudget(Exception):
    """Stage 6 / 7 perf budget exceeded."""


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def chat(prompt: str, *, program_id: str | None = None,
         max_tokens: int = 4) -> str:
    # NOTE: program_id goes at the TOP LEVEL of the request body, not
    # inside `extra_body`.  The OpenAI client SDK unpacks `extra_body`
    # client-side; sglang's server has no auto-unpack.  Send raw at
    # top level so the server's protocol model picks it up.
    body: dict[str, Any] = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    if program_id is not None:
        body["program_id"] = program_id
    r = requests.post(f"{BASE}/v1/chat/completions", json=body, timeout=120)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"] or ""


def fetch_state() -> dict[str, Any]:
    r = requests.get(f"{BASE}/aginfer/state", timeout=30)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# Stage 0 — strict schema parser
# ---------------------------------------------------------------------------

def assert_no_legacy(state: dict[str, Any]) -> None:
    """Round-9 'halts loudly on legacy shape' invariant."""
    for k in LEGACY_TOP_LEVEL:
        if k in state:
            raise LegacyShape(
                f"legacy top-level key present: {k!r} "
                f"(DESIGN §5 removed it; the daemon is supposed to halt)")
    pool_usage = state.get("pool_usage", {})
    for tier, entry in pool_usage.items():
        if not isinstance(entry, dict):
            raise SchemaMissing(f"pool_usage[{tier}] not a dict")
        # legacy keys live directly under pool_usage[tier]
        for k in LEGACY_POOL_USAGE_KEYS:
            if k in entry:
                raise LegacyShape(
                    f"legacy pool_usage[{tier}].{k} present "
                    f"(should live under pool_usage[{tier}].subpools[sp].*)")
    for u in state.get("units", []):
        for k in LEGACY_UNIT_KEYS:
            if k in u:
                raise LegacyShape(
                    f"unit {u.get('hash')!r}: legacy field {k!r} present "
                    f"(DESIGN §5 replaces with `residence: list[Tier]`)")


def assert_schema_shape(state: dict[str, Any]) -> None:
    """Every DESIGN §5 path present."""
    for k in REQUIRED_TOP_LEVEL:
        if k not in state:
            raise SchemaMissing(f"top-level key missing: {k!r}")

    # throughput_ema
    te = state["throughput_ema"]
    for k in ("prefill_bps", "decode_per_program"):
        if k not in te:
            raise SchemaMissing(f"throughput_ema.{k} missing")
    if not isinstance(te["decode_per_program"], dict):
        raise SchemaMissing("throughput_ema.decode_per_program must be dict")

    # pool_usage[tier].subpools[sp].*
    pu = state["pool_usage"]
    for tier in REQUIRED_TIERS:
        if tier not in pu:
            raise SchemaMissing(f"pool_usage[{tier}] missing")
        if "subpools" not in pu[tier]:
            raise SchemaMissing(f"pool_usage[{tier}].subpools missing")
        for sp, fields in pu[tier]["subpools"].items():
            for f in ("used_bytes", "cap_bytes", "available_bytes",
                      "evictable_bytes", "page_bytes"):
                if f not in fields:
                    raise SchemaMissing(
                        f"pool_usage[{tier}].subpools[{sp}].{f} missing")

    # per_program_usage[pid].*
    ppu = state["per_program_usage"]
    if not isinstance(ppu, dict):
        raise SchemaMissing("per_program_usage must be dict")
    for pid, e in ppu.items():
        for k in ("hbm", "dram", "state", "pre_pause_state", "unit_hashes"):
            if k not in e:
                raise SchemaMissing(
                    f"per_program_usage[{pid}].{k} missing")
        if "committed" not in e["hbm"] or "inflight" not in e["hbm"]:
            raise SchemaMissing(
                f"per_program_usage[{pid}].hbm must have committed + inflight")
        if "committed" not in e["dram"]:
            raise SchemaMissing(
                f"per_program_usage[{pid}].dram.committed missing")
        if e["state"] not in ("REASONING", "ACTING", "PAUSED", "ENDED"):
            raise SchemaMissing(
                f"per_program_usage[{pid}].state invalid: {e['state']!r}")
        if e["pre_pause_state"] is not None and e["pre_pause_state"] not in (
                "REASONING", "ACTING"):
            raise SchemaMissing(
                f"per_program_usage[{pid}].pre_pause_state invalid: "
                f"{e['pre_pause_state']!r}")
        if not isinstance(e["unit_hashes"], list):
            raise SchemaMissing(
                f"per_program_usage[{pid}].unit_hashes must be list")

    # units[*]
    for u in state["units"]:
        for k in ("hash", "residence", "n_tokens", "n_bytes",
                  "last_access_time", "hit_count", "session_ids"):
            if k not in u:
                raise SchemaMissing(f"unit missing field: {k!r}")
        if not isinstance(u["residence"], list) or not u["residence"]:
            raise SchemaMissing(
                f"unit {u['hash']!r}: residence must be non-empty list "
                f"(empty residence ⇒ should not be in units[] at all)")
        for tier in u["residence"]:
            if tier not in REQUIRED_TIERS:
                raise ResidenceInvariant(
                    f"unit {u['hash']!r}: bad residence tier {tier!r}")
        if not isinstance(u["n_bytes"], dict):
            raise SchemaMissing(
                f"unit {u['hash']!r}: n_bytes must be dict[tier][subpool]")
        for tier, sp_dict in u["n_bytes"].items():
            if tier not in u["residence"]:
                raise ResidenceInvariant(
                    f"unit {u['hash']!r}: n_bytes carries tier {tier!r} "
                    f"not in residence {u['residence']!r}")
            if not isinstance(sp_dict, dict):
                raise SchemaMissing(
                    f"unit {u['hash']!r}: n_bytes[{tier}] must be subpool dict")
        for tier in u["residence"]:
            if tier not in u["n_bytes"]:
                raise ResidenceInvariant(
                    f"unit {u['hash']!r}: residence has {tier!r} but "
                    f"n_bytes is missing it")

    # link_stats
    ls = state["link_stats"]
    if not isinstance(ls, dict):
        raise SchemaMissing("link_stats must be dict")
    for link in REQUIRED_LINKS:
        if link not in ls:
            raise SchemaMissing(f"link_stats[{link}] missing")
        e = ls[link]
        for k in ("peak_bw_bps", "recent_throughput_bps",
                  "time_since_last_sample_s"):
            if k not in e:
                raise SchemaMissing(f"link_stats[{link}].{k} missing")
        if e["peak_bw_bps"] <= 0:
            raise LinkStatsInvariant(
                f"link_stats[{link}].peak_bw_bps must be > 0 "
                f"(got {e['peak_bw_bps']})")

    # tier_holding_cost
    thc = state["tier_holding_cost"]
    for tier in REQUIRED_TIERS:
        if tier not in thc:
            raise SchemaMissing(f"tier_holding_cost[{tier}] missing")
        for sp, fields in thc[tier].items():
            if "h_max_per_byte_sec" not in fields:
                raise SchemaMissing(
                    f"tier_holding_cost[{tier}].{sp}.h_max_per_byte_sec missing")
            if fields["h_max_per_byte_sec"] < 0:
                raise SchemaMissing(
                    f"tier_holding_cost[{tier}].{sp}.h_max_per_byte_sec "
                    f"must be >= 0 (got {fields['h_max_per_byte_sec']})")


def assert_subpool_keys_consistent(state: dict[str, Any]) -> None:
    """pool_usage subpool keys ⊇ unit.n_bytes keys ⊇ tier_holding_cost keys."""
    pu_keys = {tier: set(state["pool_usage"][tier]["subpools"].keys())
               for tier in REQUIRED_TIERS}
    thc_keys = {tier: set(state["tier_holding_cost"][tier].keys())
                for tier in REQUIRED_TIERS}

    for tier in REQUIRED_TIERS:
        if pu_keys[tier] != thc_keys[tier]:
            raise SchemaMissing(
                f"{tier}: pool_usage subpools {pu_keys[tier]} != "
                f"tier_holding_cost subpools {thc_keys[tier]}")

    for u in state["units"]:
        for tier, sp_dict in u["n_bytes"].items():
            for sp in sp_dict:
                if sp not in pu_keys[tier]:
                    raise SchemaMissing(
                        f"unit {u['hash']!r}: n_bytes[{tier}][{sp}] "
                        f"refers to undeclared subpool "
                        f"(pool_usage declares {pu_keys[tier]})")


def assert_reconcile(state: dict[str, Any]) -> None:
    """units[*] ↔ per_program_usage[*].unit_hashes."""
    by_hash = {u["hash"]: u for u in state["units"]}
    for pid, e in state["per_program_usage"].items():
        for h in e["unit_hashes"]:
            if h not in by_hash:
                raise ReconcileInvariant(
                    f"per_program_usage[{pid}].unit_hashes contains {h!r} "
                    f"not in units[]")
            if pid not in by_hash[h]["session_ids"]:
                raise ReconcileInvariant(
                    f"per_program_usage[{pid}] claims unit {h!r} but its "
                    f"session_ids = {by_hash[h]['session_ids']!r}")
    # reverse direction
    expected = {pid: set() for pid in state["per_program_usage"]}
    for h, u in by_hash.items():
        for pid in u["session_ids"]:
            expected.setdefault(pid, set()).add(h)
    for pid, hashes in expected.items():
        got = set(state["per_program_usage"][pid]["unit_hashes"]
                  if pid in state["per_program_usage"] else [])
        if got != hashes:
            raise ReconcileInvariant(
                f"per_program_usage[{pid}].unit_hashes = {got} but "
                f"units[*].session_ids implies {hashes}")


# ---------------------------------------------------------------------------
# Stage drivers
# ---------------------------------------------------------------------------

def _print(stage: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    print(f"[{stage}] {status}{(' — ' + detail) if detail else ''}")


def stage_0_strict_parser_negative() -> None:
    """Negative test: synthesised legacy shape must raise LegacyShape."""
    fake_legacy = {
        "tier_usage": {"HBM": {"used_bytes": 0, "cap_bytes": 0}},
        "pool_usage": {"HBM": {"used_bytes": 0}, "DRAM": {}, "DISK": {}},
        "units": [],
        "page_size": 64,
        "bytes_per_token": 1,
        "time_counter": 0,
    }
    try:
        assert_no_legacy(fake_legacy)
    except LegacyShape:
        _print("Stage 0", True, "legacy-shape detector raises as expected")
        return
    raise AssertionError("Stage 0 negative test did NOT raise")


def stage_1_initial_shape() -> dict[str, Any]:
    """Initial-state schema check (sglang may have done internal warmups
    so we can't insist on empty tree; just check the full shape parses)."""
    state = fetch_state()
    assert_no_legacy(state)
    assert_schema_shape(state)
    assert_subpool_keys_consistent(state)
    assert_reconcile(state)

    n_units = len(state["units"])
    n_progs = len(state["per_program_usage"])
    # Sanity: every program that has any unit_hashes must reconcile to
    # an actual unit (already checked by assert_reconcile, but also
    # double-check empty-program guard).
    for pid, e in state["per_program_usage"].items():
        if not e["unit_hashes"]:
            raise SchemaMissing(
                f"stage 1: per_program_usage[{pid}] has empty unit_hashes "
                f"— programs with no committed units should not be tracked")
    _print("Stage 1", True,
           f"initial state: {n_units} units, {n_progs} programs, "
           f"schema parses cleanly")
    return state


def stage_2_single_unit() -> dict[str, Any]:
    """One tagged request → 1 unit, 1 program, residence=['HBM']."""
    pid = f"t17-stage2-{uuid.uuid4().hex[:8]}"
    long_prompt = ("Background context: " + ("Lorem ipsum dolor sit amet. " * 20)
                   + "\n\nQ: tell me a 1-line fact.")
    chat(long_prompt, program_id=pid)
    state = fetch_state()
    assert_no_legacy(state)
    assert_schema_shape(state)
    assert_subpool_keys_consistent(state)
    assert_reconcile(state)

    if not state["units"]:
        raise SchemaMissing("stage 2: no units after a tagged prefill")
    tagged_units = [u for u in state["units"] if pid in u["session_ids"]]
    if not tagged_units:
        raise ReconcileInvariant(
            f"stage 2: no unit carries program_id {pid!r}")
    if pid not in state["per_program_usage"]:
        raise ReconcileInvariant(
            f"stage 2: per_program_usage[{pid}] missing after tagged request")
    e = state["per_program_usage"][pid]
    if not e["unit_hashes"]:
        raise ReconcileInvariant(
            f"stage 2: per_program_usage[{pid}].unit_hashes is empty")

    # Residence check: at minimum some tagged unit lives in HBM.
    has_hbm = any("HBM" in u["residence"] for u in tagged_units)
    if not has_hbm:
        raise ResidenceInvariant(
            f"stage 2: no tagged unit has HBM in residence — all units: "
            f"{[u['residence'] for u in tagged_units]}")
    _print("Stage 2", True,
           f"1 program, {len(tagged_units)} tagged units, residence includes HBM")
    return state


def stage_3_residence_set() -> dict[str, Any]:
    """Drive residence transitions: HBM-only → HBM+DRAM (via write_through).

    HARD assertion: dual residence MUST be observed.  T17's primary new
    invariant is that residence is a SET, not a tier; if the test
    never sees a unit with both HBM and DRAM in residence, the test
    has not exercised the new schema's central claim.  If your launch
    profile doesn't fire write_through fast enough, fix the launch
    config — don't pass the test.
    """
    pid = f"t17-stage3-{uuid.uuid4().hex[:8]}"
    prompt = "Stage 3 residence-set: " + ("foo bar baz qux " * 50)
    chat(prompt, program_id=pid)
    deadline = time.time() + 3.0  # HiCache write_through completes in well
                                  # under 1 s in healthy config; longer
                                  # window only masks regressions.
    saw_dual = False
    state: dict[str, Any] | None = None
    while time.time() < deadline:
        state = fetch_state()
        assert_no_legacy(state)
        units = [u for u in state["units"] if pid in u["session_ids"]]
        if any(set(u["residence"]) >= {"HBM", "DRAM"} for u in units):
            saw_dual = True
            break
        time.sleep(0.2)

    assert state is not None
    assert_schema_shape(state)
    assert_subpool_keys_consistent(state)
    assert_reconcile(state)

    if not saw_dual:
        units = [u for u in state["units"] if pid in u["session_ids"]]
        raise ResidenceInvariant(
            f"Stage 3: no HBM+DRAM dual-residence observed within "
            f"{deadline}-s window — launch must have HiCache active with "
            f"write_through policy.  Tagged units: "
            f"{[(u['hash'], u['residence']) for u in units]!r}")
    _print("Stage 3", True,
           "saw residence == [HBM, DRAM] after write_through")
    return state


def stage_4_subpool_degeneracy(state: dict[str, Any]) -> None:
    """S1 (single-stack attention): pool_usage subpools is the architecture-
    declared set.

    Two inequalities per (tier, subpool):

    - **HBM**: radix sum ≤ pool used (radix-resident is a subset of
      allocator-resident; the gap is in-flight decode KV not in tree).
      Additionally, if pool reports used > 0, radix should be > 0 too
      (in steady-state at least the system-prompt prefix is committed).
    - **DRAM**: radix sum **EQUALS** pool used.  The impl patches
      `pool_usage.DRAM.subpools[sp].used_bytes` from the walk's
      per-subpool DRAM sum, so any divergence here is a bug in the
      patch logic (not a real allocator/radix asymmetry).
    """
    for sp, fields in state["pool_usage"]["HBM"]["subpools"].items():
        radix_sum = sum(
            u["n_bytes"]["HBM"].get(sp, 0)
            for u in state["units"]
            if "HBM" in u["n_bytes"])
        if radix_sum > fields["used_bytes"]:
            raise ResidenceInvariant(
                f"HBM.{sp}: radix Σ unit.n_bytes = {radix_sum} > "
                f"pool_usage.used_bytes = {fields['used_bytes']} "
                f"(radix is supposed to be a SUBSET of allocator total)")
        # Audit-2 finding: catch the reverse failure too.  If the
        # allocator says bytes are used and we have no separate
        # in-flight tracker (T29 future), the radix tree must hold
        # at least some of them — radix_sum == 0 with pool > 0 means
        # the walk under-counted.  Tolerance: pool_used may legitimately
        # exceed radix by up to a few in-flight decode requests'
        # worth, but the radix should not be empty.
        if fields["used_bytes"] > 0 and radix_sum == 0 and state["units"]:
            raise ResidenceInvariant(
                f"HBM.{sp}: pool_usage.used_bytes = {fields['used_bytes']} > 0 "
                f"but Σ unit.n_bytes = 0 — the walk did not attribute any "
                f"bytes to this subpool even though the allocator says it's "
                f"in use.  Likely a subpool-key mismatch or empty-residence "
                f"skip bug.")

    for sp, fields in state["pool_usage"]["DRAM"]["subpools"].items():
        radix_sum = sum(
            u["n_bytes"]["DRAM"].get(sp, 0)
            for u in state["units"]
            if "DRAM" in u["n_bytes"])
        if radix_sum != fields["used_bytes"]:
            raise ResidenceInvariant(
                f"DRAM.{sp}: radix Σ unit.n_bytes = {radix_sum} != "
                f"pool_usage.used_bytes = {fields['used_bytes']} "
                f"(impl is supposed to patch DRAM used = radix sum; "
                f"non-equality means the patch logic broke)")
    _print("Stage 4", True,
           "HBM Σ ≤ pool_used; DRAM Σ == pool_used per (tier, subpool)")


def stage_5_multi_holder() -> None:
    """4 programs hit same long prefix → 1/holders attribution."""
    shared = ("Shared prefix for stage 5: "
              + ("Some shared context " * 80))  # ~1K tokens
    pids = [f"t17-stage5-{c}-{uuid.uuid4().hex[:6]}" for c in "abcd"]
    for pid in pids:
        chat(shared + f"\n\nPer-program suffix for {pid}", program_id=pid)
    state = fetch_state()
    assert_no_legacy(state)
    assert_schema_shape(state)
    assert_reconcile(state)

    # Find a unit whose session_ids contains all 4 pids.
    shared_units = [
        u for u in state["units"]
        if set(pids) <= set(u["session_ids"])]
    if not shared_units:
        raise AttributionInvariant(
            f"stage 5: no unit carries all 4 pids; session_ids per unit: "
            f"{[u['session_ids'] for u in state['units']]}")
    u = shared_units[0]
    n_holders = len(u["session_ids"])
    # Sum over all (tier, subpool) the unit's bytes.
    total_bytes = sum(b for sp_dict in u["n_bytes"].values()
                      for b in sp_dict.values())
    expected_per_holder = total_bytes // n_holders

    # Strict per-program attribution: each program's `committed[sp]` must
    # be the SHARED unit's 1/holders share PLUS any tail-unit bytes that
    # are uniquely this program's.  We compute the expected total per
    # program by re-aggregating from units[] and assert exact equality
    # (±byte rounding from integer-divide).  No escape hatch.
    expected: dict[str, dict[str, dict[str, int]]] = {}
    # n_shared_units_per_pid[pid] = number of units this pid co-owns with
    # OTHER pids (n_holders > 1).  Tail units (n_holders == 1) contribute
    # ZERO integer-divide drift because b // 1 == b, so they don't widen
    # the tolerance.  Audit-2 finding: the previous `n_units * (h-1)`
    # tolerance overcounted by including these zero-drift tail units.
    n_shared_per_pid: dict[str, int] = {}
    for unit in state["units"]:
        h = len(unit["session_ids"])
        for pid_u in unit["session_ids"]:
            if pid_u not in pids:
                continue
            e = expected.setdefault(pid_u, {"hbm": {}, "dram": {}})
            if h > 1:
                n_shared_per_pid[pid_u] = n_shared_per_pid.get(pid_u, 0) + 1
            for tier, sp_dict in unit["n_bytes"].items():
                if tier == "DISK":
                    continue
                side = "hbm" if tier == "HBM" else "dram"
                for sp, b in sp_dict.items():
                    e[side][sp] = e[side].get(sp, 0) + b // h

    for pid_u in pids:
        if pid_u not in state["per_program_usage"]:
            raise ReconcileInvariant(
                f"stage 5: per_program_usage[{pid_u}] missing")
        got_e = state["per_program_usage"][pid_u]
        # Direct subscript: every pid in `pids` MUST appear in expected
        # (built from units' session_ids) since Stage 5 sent requests
        # tagged with these exact pids and asserted `pid in
        # per_program_usage` above; KeyError here = impl/test contract
        # broke, fail loud.
        for side in ("hbm", "dram"):
            for sp, want in expected[pid_u][side].items():
                # Direct subscript: sp came from the unit walk; if it's
                # not in the impl's committed dict for this pid, the
                # impl FAILED to attribute it.  Silent default to 0
                # would mask that as a noisy AttributionInvariant; the
                # real signal is KeyError.
                got = got_e[side]["committed"][sp]
                # Drift bound: only SHARED units (n_holders > 1) can lose
                # up to (n_holders - 1) bytes each from integer-divide.
                # Tail units lose 0.
                n_shared = n_shared_per_pid.get(pid_u, 0)
                tol = max(1, n_shared * (n_holders - 1))
                if abs(got - want) > tol:
                    raise AttributionInvariant(
                        f"stage 5: per_program_usage[{pid_u}].{side}."
                        f"committed[{sp}] = {got} but expected = {want} "
                        f"(tol ±{tol} from integer-divide on {n_shared} "
                        f"shared units; n_holders={n_holders})")
    _print("Stage 5", True,
           f"4-holder strict-attribution check passes across {len(pids)} pids")


def _drive_distinct_prefixes(target: int, label: str) -> int:
    """Issue `target` chat completions with distinct prefixes to grow
    the radix tree.  Returns the final unit count."""
    batches = max(1, target // 100)
    per_batch = max(1, target // batches)
    for b in range(batches):
        for i in range(per_batch):
            chat(f"unique-{label}-{b}-{i} brief answer please.",
                 max_tokens=2)
    return len(fetch_state()["units"])


def stage_6_perf_guard() -> None:
    """Two perf measurements:
    (a) budget assertion at 5 K-unit tree (representative workload)
    (b) stress recording at 10 K-unit tree (informational; no assert)"""
    print(f"[Stage 6a] driving tree to ~{STAGE6_TREE_SIZE} units …")
    actual = _drive_distinct_prefixes(STAGE6_TREE_SIZE, "stage6a")
    print(f"[Stage 6a] tree size after drive: {actual} units")

    # Aggregate all samples from N cycles into one sequence and take the
    # true p99.  Averaging per-cycle p99s washes out a single bad cycle.
    all_lats: list[float] = []
    for cycle in range(STAGE6_CYCLES):
        lats = []
        for _ in range(STAGE6_DUMPS_PER_CYCLE):
            t0 = time.perf_counter()
            fetch_state()
            lats.append((time.perf_counter() - t0) * 1000)
        all_lats.extend(lats)
        p50_c = statistics.median(lats)
        max_c = max(lats)
        print(f"[Stage 6a] cycle {cycle + 1}/{STAGE6_CYCLES}: "
              f"p50={p50_c:.1f}ms max={max_c:.1f}ms")
        if max_c > STAGE6_MAX_BUDGET_MS:
            raise PerfBudget(
                f"Stage 6a cycle {cycle + 1}: single dump > "
                f"{STAGE6_MAX_BUDGET_MS} ms (Gen-2 GC sweep?): {max_c:.1f} ms")

    all_sorted = sorted(all_lats)
    p99_idx = int(0.99 * (len(all_sorted) - 1))
    p99_aggregate = all_sorted[p99_idx]
    p50_aggregate = statistics.median(all_lats)
    if p99_aggregate > STAGE6_P99_BUDGET_MS:
        raise PerfBudget(
            f"Stage 6a aggregate p99 over {len(all_lats)} samples = "
            f"{p99_aggregate:.1f} ms > {STAGE6_P99_BUDGET_MS} ms budget "
            f"(p50 = {p50_aggregate:.1f} ms)")
    _print("Stage 6a", True,
           f"aggregate p99 @ {actual} units (n={len(all_lats)}): "
           f"{p99_aggregate:.1f} ms (p50 {p50_aggregate:.1f} ms; "
           f"budget {STAGE6_P99_BUDGET_MS} ms)")

    # Stage 6b — informational stress at 10 K units.  Records p50/p99
    # so the F3-revisit conversation has a data point if the trigger
    # ever fires under heavier load.  No assert (this regime exceeds
    # the 50 ms budget by design; T14 instrumentation flips F3 when
    # observed in production).
    print(f"[Stage 6b] driving tree to ~{STAGE6_STRESS_TREE_SIZE} units (stress) …")
    actual = _drive_distinct_prefixes(
        STAGE6_STRESS_TREE_SIZE - STAGE6_TREE_SIZE, "stage6b")
    stress_lats = []
    for _ in range(STAGE6_DUMPS_PER_CYCLE):
        t0 = time.perf_counter()
        fetch_state()
        stress_lats.append((time.perf_counter() - t0) * 1000)
    print(f"[Stage 6b] @ {actual} units: p50={statistics.median(stress_lats):.1f}ms "
          f"p99={max(stress_lats):.1f}ms (informational, no assert)")


def stage_7_concurrent_stress() -> None:
    """Concurrent walker + traffic; assert 0 schema failures."""
    print(f"[Stage 7] {STAGE7_WALKER_THREADS} walkers + "
          f"{STAGE7_CONCURRENT_CHATS} concurrent chats for "
          f"{STAGE7_DURATION_S}s …")
    stop = threading.Event()
    # Audit-2 finding: per-thread result lists eliminate the
    # ok-counter / errs.append() race that previously could under-count
    # walker failures (the only assertion below).  After join we reduce.
    per_thread_ok = [0] * STAGE7_WALKER_THREADS
    per_thread_fail = [0] * STAGE7_WALKER_THREADS
    per_thread_lats: list[list[float]] = [[] for _ in
                                          range(STAGE7_WALKER_THREADS)]
    per_thread_errs: list[list[str]] = [[] for _ in
                                        range(STAGE7_WALKER_THREADS)]

    def walker(idx: int) -> None:
        while not stop.is_set():
            try:
                t0 = time.perf_counter()
                s = fetch_state()
                per_thread_lats[idx].append((time.perf_counter() - t0) * 1000)
                assert_no_legacy(s)
                assert_schema_shape(s)
                assert_subpool_keys_consistent(s)
                assert_reconcile(s)
                per_thread_ok[idx] += 1
            except Exception as exc:
                per_thread_fail[idx] += 1
                if len(per_thread_errs[idx]) < 5:
                    per_thread_errs[idx].append(repr(exc))

    threads = [threading.Thread(target=walker, args=(i,))
               for i in range(STAGE7_WALKER_THREADS)]
    for t in threads:
        t.start()

    end_at = time.time() + STAGE7_DURATION_S
    counter = 0
    while time.time() < end_at:
        chat(f"stage7-{counter}: short.", max_tokens=2)
        counter += 1
    stop.set()
    for t in threads:
        t.join()

    walker_results = {
        "ok": sum(per_thread_ok),
        "fail": sum(per_thread_fail),
        "lats": [x for lst in per_thread_lats for x in lst],
        "errs": [x for lst in per_thread_errs for x in lst][:5],
    }

    # Stage 7's critical correctness check: zero schema failures
    # under concurrent live mutation.  Latency is informational —
    # the budget assertion at Stage 6a already covers the steady-
    # state perf budget; under 5-walker × 32-chat stress on top of
    # a tree that has accumulated from prior stages, tail latency
    # naturally spikes (T14 will quantify in production).
    if walker_results["fail"]:
        raise ReconcileInvariant(
            f"Stage 7: {walker_results['fail']} walker schema failures "
            f"(first 5: {walker_results['errs']!r})")
    if not walker_results["lats"]:
        raise PerfBudget("Stage 7: walker collected no samples")
    sorted_lats = sorted(walker_results["lats"])
    p99 = sorted_lats[int(0.99 * (len(sorted_lats) - 1))]
    _print("Stage 7", True,
           f"{walker_results['ok']} ok / 0 fail; "
           f"walker p99 = {p99:.1f} ms (informational)")


def stage_8_aux_fields(state_initial: dict[str, Any]) -> None:
    """link_stats cold-start; tier_holding_cost / throughput_ema shape."""
    # Cold-start link_stats: daemon's bw_free branches on
    # `time_since_last_sample_s > LINK_IDLE_SECONDS = 1.0` (DESIGN §7).
    # The TEST asserts the contract the daemon checks — not the impl's
    # specific placeholder value (1e12 today, but anything > 1.0 is
    # functionally correct).
    LINK_IDLE_SECONDS = 1.0
    for link, e in state_initial["link_stats"].items():
        ts = e["time_since_last_sample_s"]
        if not (ts == math.inf or (isinstance(ts, (int, float))
                                   and ts > LINK_IDLE_SECONDS)):
            raise LinkStatsInvariant(
                f"link_stats[{link}].time_since_last_sample_s on empty "
                f"tree must trip the daemon's bw_free idle branch "
                f"(> LINK_IDLE_SECONDS = {LINK_IDLE_SECONDS}), "
                f"got {ts!r}")

    # After all our prior stages, prefill_bps should be > 0.
    after = fetch_state()
    if after["throughput_ema"]["prefill_bps"] <= 0:
        # Could be 0 if instrumentation isn't wired yet (T26).  Print a
        # warning instead of failing — T17 just defines the schema.
        print("[Stage 8] note: throughput_ema.prefill_bps == 0 "
              "(T26 EMA instrumentation likely not wired yet)")

    # tier_holding_cost should have h_max_per_byte_sec >= 0 everywhere.
    for tier, sp_map in after["tier_holding_cost"].items():
        for sp, fields in sp_map.items():
            h = fields["h_max_per_byte_sec"]
            if h < 0:
                raise SchemaMissing(
                    f"tier_holding_cost[{tier}].{sp}.h_max_per_byte_sec "
                    f"= {h} < 0")
    _print("Stage 8", True,
           "link_stats cold-start, tier_holding_cost, throughput_ema shape OK")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> int:
    print("=== T17 verify: state-dump schema upgrade ===")
    print(f"base: {BASE}")
    print(f"model: {MODEL}")
    print()

    try:
        stage_0_strict_parser_negative()
        state_empty = stage_1_initial_shape()
        stage_2_single_unit()
        state_dual = stage_3_residence_set()
        stage_4_subpool_degeneracy(state_dual)
        stage_5_multi_holder()
        stage_6_perf_guard()
        stage_7_concurrent_stress()
        stage_8_aux_fields(state_empty)
    except Exception as exc:
        print()
        print(f"=== T17 FAILED: {type(exc).__name__}: {exc} ===")
        return 1

    print()
    print("=== T17 PASSED ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())

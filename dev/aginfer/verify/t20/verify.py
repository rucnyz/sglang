"""T20 verify — POST /aginfer/migrate residence-set payload (DESIGN §6).

End-to-end verify against a running sglang.  See verify/t20/README.md
for the stage breakdown.  Run via:

    AGINFER_VERIFY_BASE=http://127.0.0.1:30001 \
    AGINFER_VERIFY_MODEL=Qwen/Qwen3-0.6B \
        python dev/aginfer/verify/t20/verify.py

Exit 0 PASS, 1 FAIL.  Each stage prints a one-line result.
"""
from __future__ import annotations

import os
import sys
import uuid
from typing import Any

import requests

BASE = os.environ.get("AGINFER_VERIFY_BASE", "http://127.0.0.1:30001")
MODEL = os.environ.get("AGINFER_VERIFY_MODEL", "Qwen/Qwen3-0.6B")


class SchemaMissing(Exception):
    """Response envelope missing a required field."""


class WrongResidence(Exception):
    """post-migrate residence != claim."""


class SkipReasonMismatch(Exception):
    """skip-reason doesn't match the expected DESIGN §6 class."""


class ActionIdMismatch(Exception):
    """skip entry's action_id doesn't echo the request."""


def chat(prompt: str, *, program_id: str | None = None,
         max_tokens: int = 4) -> str:
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


def post_migrate(actions: list[dict[str, Any]]) -> dict[str, Any]:
    r = requests.post(f"{BASE}/aginfer/migrate",
                      json={"actions": actions}, timeout=30)
    r.raise_for_status()
    return r.json()


def find_unit(state: dict[str, Any], unit_hash: str) -> dict[str, Any] | None:
    for u in state["units"]:
        if u["hash"] == unit_hash:
            return u
    return None


def assert_envelope_shape(resp: dict[str, Any]) -> None:
    for k in ("applied", "applied_hashes", "skipped"):
        if k not in resp:
            raise SchemaMissing(f"migrate response missing {k!r}; got "
                                f"keys={sorted(resp)}")
    if not isinstance(resp["applied"], int):
        raise SchemaMissing(
            f"applied must be int, got {type(resp['applied']).__name__}")
    if not isinstance(resp["applied_hashes"], list):
        raise SchemaMissing("applied_hashes must be list")
    if not isinstance(resp["skipped"], list):
        raise SchemaMissing("skipped must be list")
    for s in resp["skipped"]:
        for f in ("hash", "action_id", "reason"):
            if f not in s:
                raise SchemaMissing(
                    f"skipped entry missing {f!r}; got keys={sorted(s)}")


def assert_skip_reason(resp: dict[str, Any], expected_prefix: str,
                       *, action_id: str | None = None) -> dict[str, Any]:
    """Find a skip entry whose reason starts with `expected_prefix`.
    If `action_id` given, also verify echo.  Returns the entry."""
    for s in resp["skipped"]:
        if s["reason"].startswith(expected_prefix):
            if action_id is not None and s["action_id"] != action_id:
                raise ActionIdMismatch(
                    f"skip reason {s['reason']!r} action_id "
                    f"{s['action_id']!r} != expected {action_id!r}")
            return s
    raise SkipReasonMismatch(
        f"no skip entry with reason prefix {expected_prefix!r}; "
        f"skipped={resp['skipped']!r}")


def _print(stage: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    print(f"[{stage}] {status}{(' — ' + detail) if detail else ''}")


def _action(hash_: str, add: list[str], remove: list[str],
            action_id: str | None = None) -> dict[str, Any]:
    return {
        "hash": hash_,
        "add_tiers": list(add),
        "remove_tiers": list(remove),
        "action_id": action_id or f"verify-{uuid.uuid4().hex[:8]}",
    }


def _seed_unit(label: str) -> str:
    """Drive one chat with a unique prefix; return the hash of a
    leaf unit owned exclusively by this prefix's program_id."""
    prompt = (f"Stage {label} unique prefix: "
              + ("alpha beta gamma delta epsilon. " * 30)
              + f"\n\nQ: tell me a fact about prime {label}.")
    pid = f"t20-{label}-{uuid.uuid4().hex[:6]}"
    chat(prompt, program_id=pid)
    state = fetch_state()
    candidates = [u for u in state["units"]
                  if u["session_ids"] == [pid]]
    if not candidates:
        raise AssertionError(
            f"_seed_unit({label}): no exclusive unit for pid={pid}; "
            f"matching units session_ids: "
            f"{[u['session_ids'] for u in state['units'] if pid in u['session_ids']]!r}")
    return candidates[-1]["hash"]


def stage_0_schema() -> None:
    """A request against a non-existent hash returns a well-formed
    envelope with the action_id echoed in the skip entry."""
    aid = "stage0-noop"
    resp = post_migrate([_action("nonexistent-hash-0", [], [], aid)])
    assert_envelope_shape(resp)
    assert_skip_reason(resp, "not_in_tree", action_id=aid)
    _print("Stage 0", True, "envelope shape + skip echoes action_id")


def stage_1_add_dram() -> str:
    """Drive HBM-only unit (or HBM+DRAM under write_through), POST
    add=[DRAM], verify residence post-state includes both tiers."""
    h = _seed_unit("1")
    state = fetch_state()
    u = find_unit(state, h)
    assert u is not None, f"stage 1: seeded unit {h} missing"
    # If write_through already gave DRAM, remove first to get a
    # clean HBM-only pre-condition.
    if "DRAM" in u["residence"]:
        post_migrate([_action(h, [], ["DRAM"])])
        state = fetch_state()
        u = find_unit(state, h)
        if u is None or "HBM" not in u["residence"]:
            raise AssertionError(
                f"stage 1: failed to bring unit {h} to HBM-only; "
                f"post-remove residence="
                f"{u['residence'] if u else None!r}")
    aid = "stage1-add-dram"
    resp = post_migrate([_action(h, ["DRAM"], [], aid)])
    assert_envelope_shape(resp)
    if resp["applied"] != 1:
        raise SchemaMissing(
            f"stage 1: expected applied=1, got {resp['applied']}; "
            f"skipped={resp['skipped']!r}")
    u = find_unit(fetch_state(), h)
    if u is None:
        raise WrongResidence(f"stage 1: unit {h} vanished after add=DRAM")
    if set(u["residence"]) != {"HBM", "DRAM"}:
        raise WrongResidence(
            f"stage 1: post-add residence = {u['residence']!r}, "
            f"expected [HBM, DRAM]")
    _print("Stage 1", True, f"add=[DRAM] residence={u['residence']!r}")
    return h


def stage_2_remove_hbm(unit_hash: str) -> str:
    """remove=[HBM] from {HBM, DRAM} → residence becomes [DRAM]."""
    aid = "stage2-remove-hbm"
    resp = post_migrate([_action(unit_hash, [], ["HBM"], aid)])
    assert_envelope_shape(resp)
    if resp["applied"] != 1:
        raise SchemaMissing(
            f"stage 2: expected applied=1, got {resp['applied']}; "
            f"skipped={resp['skipped']!r}")
    u = find_unit(fetch_state(), unit_hash)
    if u is None:
        raise WrongResidence(
            f"stage 2: unit {unit_hash} dropped on remove=[HBM]")
    if set(u["residence"]) != {"DRAM"}:
        raise WrongResidence(
            f"stage 2: post-remove residence = {u['residence']!r}, "
            f"expected [DRAM]")
    _print("Stage 2", True, "remove=[HBM] kept DRAM copy")
    return unit_hash


def stage_3_remove_drops_unit() -> None:
    """remove=[HBM, DRAM] on a leaf → unit DROPped from units[]."""
    h = _seed_unit("3")
    aid = "stage3-drop"
    resp = post_migrate([_action(h, [], ["HBM", "DRAM"], aid)])
    assert_envelope_shape(resp)
    if resp["applied"] != 1:
        raise SchemaMissing(
            f"stage 3: expected applied=1, got {resp['applied']}; "
            f"skipped={resp['skipped']!r}")
    if find_unit(fetch_state(), h) is not None:
        raise WrongResidence(
            f"stage 3: unit {h} still in units[] after full DROP")
    _print("Stage 3", True, f"remove=[HBM, DRAM] dropped unit")


def stage_4_load_back(unit_hash: str) -> None:
    """add=[HBM] on DRAM-only unit; either applies or graceful decline."""
    aid = "stage4-load-back"
    resp = post_migrate([_action(unit_hash, ["HBM"], [], aid)])
    assert_envelope_shape(resp)
    if resp["applied"] == 1:
        u = find_unit(fetch_state(), unit_hash)
        if u is None or "HBM" not in set(u["residence"]):
            raise WrongResidence(
                f"stage 4: applied=1 but post residence "
                f"{u['residence'] if u else None!r} doesn't include HBM")
        _print("Stage 4", True,
               f"add=[HBM] load_back succeeded; residence={u['residence']!r}")
    else:
        entry = assert_skip_reason(
            resp, "promote_load_back_declined", action_id=aid)
        _print("Stage 4", True,
               f"add=[HBM] gracefully declined: {entry['reason']!r}")


def stage_5_duplicate_in_batch() -> None:
    """Two actions on same hash in one batch → first applies,
    second is `already_acted_this_batch`."""
    h = _seed_unit("5")
    u = find_unit(fetch_state(), h)
    assert u is not None
    if "HBM" not in u["residence"]:
        raise AssertionError(
            f"stage 5: setup failed; residence={u['residence']!r}")
    aid1 = "stage5-first"
    aid2 = "stage5-second"
    resp = post_migrate([
        _action(h, [], ["HBM"], aid1),
        _action(h, [], ["HBM"], aid2),
    ])
    assert_envelope_shape(resp)
    if resp["applied"] != 1:
        raise SchemaMissing(
            f"stage 5: expected applied=1, got {resp['applied']}; "
            f"skipped={resp['skipped']!r}")
    assert_skip_reason(resp, "already_acted_this_batch", action_id=aid2)
    _print("Stage 5", True, "duplicate hash: 1 applied + 1 skipped")


def stage_6_idempotency(unit_hash: str) -> None:
    """Re-apply add=[DRAM] on a unit that already has DRAM."""
    u = find_unit(fetch_state(), unit_hash)
    if u is None:
        unit_hash = _seed_unit("6")
        u = find_unit(fetch_state(), unit_hash)
    assert u is not None
    if "DRAM" not in u["residence"]:
        post_migrate([_action(unit_hash, ["DRAM"], [])])
    aid = "stage6-replay"
    resp = post_migrate([_action(unit_hash, ["DRAM"], [], aid)])
    assert_envelope_shape(resp)
    if resp["applied"] != 0:
        raise SchemaMissing(
            f"stage 6: expected applied=0 (already present), got "
            f"{resp['applied']}; skipped={resp['skipped']!r}")
    assert_skip_reason(resp, "add_already_present", action_id=aid)
    _print("Stage 6", True, "re-applied add=[DRAM] → add_already_present")


def stage_7_unknown_hash() -> None:
    aid = "stage7-unknown"
    resp = post_migrate([_action("node-99999999", ["DRAM"], [], aid)])
    assert_envelope_shape(resp)
    if resp["applied"] != 0:
        raise SchemaMissing(f"stage 7: expected applied=0, got {resp['applied']}")
    assert_skip_reason(resp, "not_in_tree", action_id=aid)
    _print("Stage 7", True, "unknown hash → not_in_tree")


def stage_8_disk_in_add() -> None:
    """add=[DISK] → disk_tier_not_yet_wired."""
    h = _seed_unit("8")
    aid = "stage8-disk"
    resp = post_migrate([_action(h, ["DISK"], [], aid)])
    assert_envelope_shape(resp)
    if resp["applied"] != 0:
        raise SchemaMissing(
            f"stage 8: expected applied=0, got {resp['applied']}; "
            f"skipped={resp['skipped']!r}")
    assert_skip_reason(resp, "disk_tier_not_yet_wired", action_id=aid)
    _print("Stage 8", True, "add=[DISK] → disk_tier_not_yet_wired")


def stage_9_action_id_echo() -> None:
    """3 distinct action_ids; each skip echoes its origin."""
    h_known = _seed_unit("9")
    aids = ["stage9-a", "stage9-b", "stage9-c"]
    resp = post_migrate([
        _action("node-99999991", ["DRAM"], [], aids[0]),
        _action("node-99999992", ["HBM"],  [], aids[1]),
        _action(h_known, ["DISK"], [], aids[2]),
    ])
    assert_envelope_shape(resp)
    seen = {s["action_id"]: s["reason"] for s in resp["skipped"]}
    for aid in aids:
        if aid not in seen:
            raise ActionIdMismatch(
                f"stage 9: action_id {aid!r} not echoed in skip; "
                f"got action_ids={sorted(seen)!r}")
    _print("Stage 9", True,
           f"3 action_ids all echoed; reasons={list(seen.values())}")


def stage_10_malformed_action_fails_loud() -> None:
    """Audit D1/D2: missing required action fields must surface as a
    loud error (HTTP 5xx or 4xx), NOT a silent ``noop_action`` skip.

    Pre-fix: ``apply_aginfer_migrations`` used ``action.get("add_tiers",
    [])`` etc., so a malformed POST became an action with empty sets
    that triggered the ``noop_action`` skip branch — sglang returned
    200 with applied=0, and the daemon's observability would never
    surface the protocol break.

    Contract: every action MUST carry hash + add_tiers + remove_tiers
    + action_id (DESIGN §6 wire payload).  Any missing field is a
    daemon-side bug worth surfacing — either as a 400 from the HTTP
    envelope or a 500 from the scheduler-side KeyError.  Both are
    acceptable; the failure mode being tested is "200 OK with a
    silent noop skip".
    """
    # Two malformed variants — each should fail loud.  HTTP layer may
    # convert KeyError to 500 or validate at envelope and return 400.
    malformed_variants = [
        # Missing add_tiers
        {"hash": "node-9999", "remove_tiers": [], "action_id": "stage10-a"},
        # Missing remove_tiers
        {"hash": "node-9999", "add_tiers": [], "action_id": "stage10-b"},
        # Missing action_id
        {"hash": "node-9999", "add_tiers": [], "remove_tiers": []},
    ]
    for i, bad in enumerate(malformed_variants):
        r = requests.post(f"{BASE}/aginfer/migrate",
                          json={"actions": [bad]}, timeout=30)
        if 200 <= r.status_code < 300:
            # Inspect: if the body says noop_action, that's the silent
            # swallow we want to catch.  If the body somehow processed
            # the action correctly anyway (impossible with missing
            # required field), still a contract violation.
            body = r.json()
            skipped = body.get("skipped", [])
            reasons = [s.get("reason", "") for s in skipped]
            raise SchemaMissing(
                f"stage 10 variant {i} (missing {set(bad).symmetric_difference({'hash', 'add_tiers', 'remove_tiers', 'action_id'})!r}): "
                f"server returned HTTP {r.status_code} (silent swallow); "
                f"skip reasons={reasons!r}.  Expected 4xx/5xx so the "
                f"daemon's malformed POST is loud.")
    _print("Stage 10", True,
           "3 malformed payload variants all rejected loud (≥400)")


def main() -> int:
    print("=== T20 verify: POST /aginfer/migrate residence-set payload ===")
    print(f"base: {BASE}")
    print(f"model: {MODEL}\n")
    try:
        try:
            requests.post(f"{BASE}/flush_cache", timeout=10)
        except Exception:
            pass
        stage_0_schema()
        h1 = stage_1_add_dram()
        h2 = stage_2_remove_hbm(h1)
        stage_3_remove_drops_unit()
        stage_4_load_back(h2)
        stage_5_duplicate_in_batch()
        stage_6_idempotency(h1)
        stage_7_unknown_hash()
        stage_8_disk_in_add()
        stage_9_action_id_echo()
        stage_10_malformed_action_fails_loud()
    except Exception as exc:
        print()
        print(f"=== T20 FAILED: {type(exc).__name__}: {exc} ===")
        return 1
    print()
    print("=== T20 PASSED ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())

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
import time
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
    out = r.json()["choices"][0]["message"]["content"] or ""
    # /aginfer/state is an eventually-consistent snapshot: the tagged unit
    # becomes visible a fraction of a second AFTER the HTTP response (dump is
    # taken on the scheduler loop's snapshot).  _seed_unit reads state right
    # after this returns, so settle on program_id visibility first (bounded,
    # best-effort) to avoid racing the snapshot — exactly how the daemon polls.
    if program_id is not None:
        deadline = time.time() + 8.0
        while time.time() < deadline:
            try:
                st = fetch_state()
            except Exception:
                break
            ranks = st.get("per_rank", [st])
            if any(program_id in (u.get("session_ids") or [])
                   for rk in ranks for u in rk.get("units", [])):
                break
            time.sleep(0.3)
    return out


def fetch_state() -> dict[str, Any]:
    r = requests.get(f"{BASE}/aginfer/state", timeout=30)
    r.raise_for_status()
    return r.json()


def post_migrate(actions: list[dict[str, Any]]) -> dict[str, Any]:
    r = requests.post(f"{BASE}/aginfer/migrate",
                      json={"actions": actions}, timeout=30)
    r.raise_for_status()
    resp = r.json()
    # /aginfer/state reflects a migrate's residence change ASYNCHRONOUSLY: the
    # host write_backup (add) / device evict (remove) completes a fraction of a
    # second AFTER the migrate ACK.  Callers read residence right after this, so
    # settle on the applied actions taking effect before returning (bounded,
    # best-effort; the stage's own assertion is the source of truth).
    applied = set(resp.get("applied_hashes") or [])
    want = {a["hash"]: (set(a.get("add_tiers") or []), set(a.get("remove_tiers") or []))
            for a in actions if a["hash"] in applied
            and (a.get("add_tiers") or a.get("remove_tiers"))}
    if want:
        deadline = time.time() + 8.0
        while time.time() < deadline:
            try:
                st = fetch_state()
            except Exception:
                break
            byhash = {u["hash"]: set(u.get("residence") or [])
                      for rk in st.get("per_rank", [st])
                      for u in rk.get("units", [])}
            done = True
            for h, (add, rem) in want.items():
                res = byhash.get(h)
                if res is None:          # unit fully dropped -> nothing to wait on
                    continue
                if not add.issubset(res) or (rem & res):
                    done = False
                    break
            if done:
                break
            time.sleep(0.3)
    return resp


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
    """remove the unit's full residence on a leaf → unit DROPped from units[].

    A freshly-seeded unit is an HBM-only host-leaf under write_through_selective;
    removing its full current residence ({HBM}) is the valid full-drop.  (Removing
    DRAM from a built-up 2-tier unit is rejected by the host-leaf invariant —
    `remove_dram_not_host_leaf` — so we drop what the unit actually holds.)"""
    h = _seed_unit("3")
    u = find_unit(fetch_state(), h)
    assert u is not None, f"stage 3: seeded unit {h} missing"
    tiers = sorted(set(u["residence"]))
    aid = "stage3-drop"
    resp = post_migrate([_action(h, [], tiers, aid)])
    assert_envelope_shape(resp)
    if resp["applied"] != 1:
        raise SchemaMissing(
            f"stage 3: expected applied=1, got {resp['applied']}; "
            f"skipped={resp['skipped']!r}")
    if find_unit(fetch_state(), h) is not None:
        raise WrongResidence(
            f"stage 3: unit {h} still in units[] after full DROP")
    _print("Stage 3", True, f"remove={tiers!r} dropped unit")


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
    """add=[DISK] (P5 safe-subset, superseding the old blanket
    ``disk_tier_not_yet_wired``): with NO ``--hicache-storage-backend``
    configured (this stage's own launch recipe below has none), the guard
    declines with ``disk_add_declined:no_storage_backend`` — a real
    storage-backed run would instead either apply (host-backed unit) or
    decline with ``disk_add_declined:not_host_backed`` (see
    ``dev/aginfer/verify/disk_tier_migrate/`` for the offline unit tests
    covering all four add=[DISK] branches without needing a live
    storage backend)."""
    h = _seed_unit("8")
    aid = "stage8-disk"
    resp = post_migrate([_action(h, ["DISK"], [], aid)])
    assert_envelope_shape(resp)
    if resp["applied"] == 1:
        _print("Stage 8", True,
               "add=[DISK] applied (server launched with a storage backend)")
        return
    entry = assert_skip_reason(resp, "disk_add_declined:", action_id=aid)
    _print("Stage 8", True, f"add=[DISK] gracefully declined: {entry['reason']!r}")


def stage_12_disk_in_remove() -> None:
    """remove=[DISK] → disk_remove_unsupported_upstream (P5 safe-subset):
    sglang's storage backends expose no delete API, so aginfer must not
    report a fake success for a DISK removal request.  Applies regardless
    of whether a storage backend is configured — the rejection is
    unconditional (see cache_hooks.apply_aginfer_migrations)."""
    h = _seed_unit("12")
    aid = "stage12-disk-remove"
    resp = post_migrate([_action(h, [], ["DISK"], aid)])
    assert_envelope_shape(resp)
    if resp["applied"] != 0:
        raise SchemaMissing(
            f"stage 12: expected applied=0, got {resp['applied']}; "
            f"skipped={resp['skipped']!r}")
    assert_skip_reason(resp, "disk_remove_unsupported_upstream", action_id=aid)
    # The unit itself must be untouched (still present, same residence).
    u = find_unit(fetch_state(), h)
    if u is None:
        raise WrongResidence(f"stage 12: unit {h} vanished after rejected remove=[DISK]")
    _print("Stage 12", True,
           f"remove=[DISK] → disk_remove_unsupported_upstream; "
           f"residence unchanged={u['residence']!r}")


def stage_13_disk_add_conflicts() -> None:
    """add=[DRAM,DISK] and add=[DISK],remove=[DRAM] are both rejected up
    front (review PR #4, discussion_r3921269467): letting `write_backup`'s
    async device->host copy or a same-action host-buffer free race against
    `write_backup_storage`'s async host->storage read would risk reading a
    stale/freed host buffer, and nothing on this path (`writing_check` only
    drains device->host acks) would otherwise block it. Rejected purely on
    tier-set shape before hash resolution, so a nonexistent hash also hits
    this — same pattern as remove=[DISK] alone (stage 12); see
    dev/aginfer/verify/disk_tier_migrate/ for the offline unit tests."""
    aid_a = "stage13-add-dram-disk"
    resp_a = post_migrate([_action("node-99999993", ["DRAM", "DISK"], [], aid_a)])
    assert_envelope_shape(resp_a)
    if resp_a["applied"] != 0:
        raise SchemaMissing(
            f"stage 13a: expected applied=0, got {resp_a['applied']}; "
            f"skipped={resp_a['skipped']!r}")
    assert_skip_reason(resp_a, "disk_add_conflicts_with_dram_add", action_id=aid_a)

    aid_b = "stage13-add-disk-remove-dram"
    resp_b = post_migrate([_action("node-99999994", ["DISK"], ["DRAM"], aid_b)])
    assert_envelope_shape(resp_b)
    if resp_b["applied"] != 0:
        raise SchemaMissing(
            f"stage 13b: expected applied=0, got {resp_b['applied']}; "
            f"skipped={resp_b['skipped']!r}")
    assert_skip_reason(resp_b, "disk_add_conflicts_with_dram_remove", action_id=aid_b)
    _print("Stage 13", True,
           "add=[DRAM,DISK] and add=[DISK],remove=[DRAM] both rejected up front")


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


def stage_11_combined_add_remove_no_scheduler_crash() -> None:
    """Combined add+remove in ONE action (`{HBM} → {DRAM}` = the
    canonical migrate transition the policy emits routinely) must
    NOT crash the scheduler.

    The bug this stage was added to catch: write_backup is ASYNC
    (enqueues D→H copy on cache_controller's background thread,
    records pending lock in ongoing_write_through).  Pre-fix, T20
    immediately evicted the device while the copy was still in
    flight — freeing the buffer being read.  sglang's
    invariant_checker tripped on the categories-no-longer-disjoint
    state and crashed the scheduler.

    Discovered by the T33+T20 e2e smoke (see verify/e2e_smoke/);
    captured here as a single-stage assertion + verified by
    /health 200 post-action.
    """
    h = _seed_unit("11")
    state = fetch_state()
    u = find_unit(state, h)
    assert u is not None
    # Pre-condition: bring to HBM-only.
    if "DRAM" in u["residence"]:
        post_migrate([_action(h, [], ["DRAM"])])
        u = find_unit(fetch_state(), h)
        if u is None or "HBM" not in u["residence"]:
            raise AssertionError(
                f"stage 11 setup: failed to reach HBM-only; "
                f"residence={u['residence'] if u else None!r}")
    aid = "stage11-combined"
    resp = post_migrate([_action(h, ["DRAM"], ["HBM"], aid)])
    assert_envelope_shape(resp)
    if resp["applied"] != 1:
        raise SchemaMissing(
            f"stage 11: expected applied=1, got {resp['applied']}; "
            f"skipped={resp['skipped']!r}")
    # Verify scheduler still alive (not crashed by pool-leak invariant).
    health = requests.get(f"{BASE}/health", timeout=10)
    if health.status_code != 200:
        raise SchemaMissing(
            f"stage 11: scheduler unhealthy after combined add+remove "
            f"({health.status_code}); the async write_backup + sync "
            f"evict race tripped sglang's invariant_checker")
    u_post = find_unit(fetch_state(), h)
    if u_post is None:
        raise WrongResidence(
            f"stage 11: unit {h} dropped — combined action should "
            f"have kept DRAM copy")
    if set(u_post["residence"]) != {"DRAM"}:
        raise WrongResidence(
            f"stage 11: post-state residence = {u_post['residence']!r}, "
            f"expected [DRAM]")
    _print("Stage 11", True,
           "{HBM} → {DRAM} combined action: scheduler healthy, "
           "residence == [DRAM]")


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
        stage_11_combined_add_remove_no_scheduler_crash()
        stage_12_disk_in_remove()
        stage_13_disk_add_conflicts()
    except Exception as exc:
        print()
        print(f"=== T20 FAILED: {type(exc).__name__}: {exc} ===")
        return 1
    print()
    print("=== T20 PASSED ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())

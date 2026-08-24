"""Black-box acceptance test for the SGLang Dead-KV lifecycle.

The test deliberately uses the native ``/generate`` endpoint and only public
control/state surfaces.  It proves this sequence against a live server:

1. Program A completes a response but still owns KV (response != SESSION_END).
2. Program B reuses A's long prefix; state contains shared plus A/B-exclusive
   units.
3. Explicit SESSION_END(A) returns a real all-rank cleanup acknowledgement.
4. A-exclusive HBM/DRAM units disappear, A is removed from shared holders, and
   B's shared/exclusive units remain resident.
5. A second SESSION_END(A) is an idempotent no-op.
6. Repeating B hits the retained prefix cache; ending B cleans the remainder.
7. A fresh generation still succeeds after reclamation.

No third-party Python package is required.

Typical TP=4 invocation::

    python verify_dead_kv_e2e.py --base-url http://127.0.0.1:30001 \
      --artifact-dir /tmp/deadkv-e2e --confirm-dedicated-server

The server must have been launched with UnifiedRadixCache and the aginfer
endpoints enabled (for this branch, ``SGLANG_ENABLE_UNIFIED_RADIX_TREE=1``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


class CheckFailed(RuntimeError):
    """An acceptance invariant failed with an operator-facing explanation."""


def _short_json(value: Any, limit: int = 1200) -> str:
    try:
        rendered = json.dumps(value, sort_keys=True, ensure_ascii=False)
    except Exception:
        rendered = repr(value)
    if len(rendered) > limit:
        return rendered[:limit] + f"... <{len(rendered) - limit} chars omitted>"
    return rendered


class HttpClient:
    def __init__(self, base_url: str, timeout: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        hostname = urllib.parse.urlsplit(self.base_url).hostname
        # Loopback verification traffic must never leave the node through a
        # host-level forward proxy. Non-loopback targets honor the environment.
        self._opener = (
            urllib.request.build_opener(urllib.request.ProxyHandler({}))
            if hostname in {"127.0.0.1", "::1", "localhost"}
            else urllib.request.build_opener()
        )

    def request(
        self,
        method: str,
        path: str,
        *,
        payload: Any | None = None,
        expected: Sequence[int] = (200,),
        timeout: float | None = None,
    ) -> tuple[int, bytes, Mapping[str, str]]:
        url = self.base_url + path
        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with self._opener.open(req, timeout=timeout or self.timeout) as resp:
                status = int(resp.status)
                body = resp.read()
                response_headers = dict(resp.headers.items())
        except urllib.error.HTTPError as exc:
            body = exc.read()
            detail = body.decode("utf-8", "replace")
            raise CheckFailed(
                f"{method} {path} returned HTTP {exc.code}; body={detail[:2000]!r}"
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise CheckFailed(
                f"{method} {path} could not reach {url}: {type(exc).__name__}: {exc}"
            ) from exc
        if status not in expected:
            detail = body.decode("utf-8", "replace")
            raise CheckFailed(
                f"{method} {path} returned HTTP {status}, expected {tuple(expected)}; "
                f"body={detail[:2000]!r}"
            )
        return status, body, response_headers

    def json(
        self,
        method: str,
        path: str,
        *,
        payload: Any | None = None,
        expected: Sequence[int] = (200,),
        timeout: float | None = None,
    ) -> Any:
        _, body, _ = self.request(
            method, path, payload=payload, expected=expected, timeout=timeout
        )
        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise CheckFailed(
                f"{method} {path} returned non-JSON body: "
                f"{body.decode('utf-8', 'replace')[:2000]!r}"
            ) from exc


UnitRef = tuple[int, str]


def state_ranks(state: Any) -> list[dict[str, Any]]:
    if not isinstance(state, dict):
        raise CheckFailed(
            f"/aginfer/state must return an object; got {type(state).__name__}: "
            f"{_short_json(state)}"
        )
    raw_ranks: Any
    if "per_rank" in state:
        raw_ranks = state["per_rank"]
        if not isinstance(raw_ranks, list) or not raw_ranks:
            raise CheckFailed(
                "/aginfer/state per_rank must be a non-empty list; "
                f"got {_short_json(raw_ranks)}"
            )
    else:
        raw_ranks = [state]
    ranks: list[dict[str, Any]] = []
    required = ("units", "pool_usage", "per_program_usage")
    for rank_index, rank in enumerate(raw_ranks):
        if not isinstance(rank, dict):
            raise CheckFailed(
                f"/aginfer/state rank {rank_index} must be an object; "
                f"got {_short_json(rank)}"
            )
        if "unsupported_tree_cache" in rank:
            raise CheckFailed(
                "SGLang reports unsupported_tree_cache="
                f"{rank['unsupported_tree_cache']!r}; launch with "
                "SGLANG_ENABLE_UNIFIED_RADIX_TREE=1"
            )
        missing = [key for key in required if key not in rank]
        if missing:
            raise CheckFailed(
                f"/aginfer/state rank {rank_index} is missing {missing}; "
                f"keys={sorted(rank)}. This is not the expected aginfer schema."
            )
        if not isinstance(rank["units"], list):
            raise CheckFailed(f"rank {rank_index} units is not a list")
        if not isinstance(rank["pool_usage"], dict):
            raise CheckFailed(f"rank {rank_index} pool_usage is not an object")
        if not isinstance(rank["per_program_usage"], dict):
            raise CheckFailed(f"rank {rank_index} per_program_usage is not an object")
        ranks.append(rank)
    return ranks


def unit_index(state: Any) -> dict[UnitRef, dict[str, Any]]:
    result: dict[UnitRef, dict[str, Any]] = {}
    for rank_index, rank in enumerate(state_ranks(state)):
        for unit_index_in_rank, unit in enumerate(rank["units"]):
            if not isinstance(unit, dict):
                raise CheckFailed(
                    f"rank {rank_index} unit {unit_index_in_rank} is not an object"
                )
            unit_hash = unit.get("hash")
            if not isinstance(unit_hash, str) or not unit_hash:
                raise CheckFailed(
                    f"rank {rank_index} unit {unit_index_in_rank} has invalid hash: "
                    f"{unit_hash!r}"
                )
            ref = (rank_index, unit_hash)
            if ref in result:
                raise CheckFailed(f"duplicate unit hash within rank: {ref!r}")
            result[ref] = unit
    return result


def holders(unit: Mapping[str, Any]) -> set[str]:
    raw = unit.get("session_ids") or []
    if not isinstance(raw, list):
        raise CheckFailed(
            f"unit {unit.get('hash')!r} session_ids is not a list: {_short_json(raw)}"
        )
    return {str(value) for value in raw}


def program_refs(state: Any, program_id: str) -> set[UnitRef]:
    return {
        ref for ref, unit in unit_index(state).items() if program_id in holders(unit)
    }


def program_in_usage(state: Any, program_id: str) -> bool:
    return any(program_id in rank["per_program_usage"] for rank in state_ranks(state))


def pool_used_bytes(state: Any) -> dict[str, int]:
    totals = {"HBM": 0, "DRAM": 0, "DISK": 0}
    for rank_index, rank in enumerate(state_ranks(state)):
        for tier in totals:
            tier_data = rank["pool_usage"].get(tier, {})
            if not isinstance(tier_data, dict):
                raise CheckFailed(
                    f"rank {rank_index} pool_usage[{tier}] is not an object"
                )
            subpools = tier_data.get("subpools", {})
            if not isinstance(subpools, dict):
                raise CheckFailed(
                    f"rank {rank_index} pool_usage[{tier}].subpools is not an object"
                )
            for subpool, fields in subpools.items():
                if not isinstance(fields, dict) or "used_bytes" not in fields:
                    raise CheckFailed(
                        f"rank {rank_index} {tier}/{subpool} lacks used_bytes"
                    )
                totals[tier] += int(fields["used_bytes"])
    return totals


def refs_bytes_by_tier(state: Any, refs: Iterable[UnitRef]) -> dict[str, int]:
    index = unit_index(state)
    totals = {"HBM": 0, "DRAM": 0, "DISK": 0}
    for ref in refs:
        unit = index.get(ref)
        if unit is None:
            continue
        raw_n_bytes = unit.get("n_bytes") or {}
        if not isinstance(raw_n_bytes, dict):
            raise CheckFailed(f"unit {ref!r} n_bytes is not an object")
        for tier in totals:
            subpools = raw_n_bytes.get(tier, {}) or {}
            if not isinstance(subpools, dict):
                raise CheckFailed(f"unit {ref!r} n_bytes[{tier}] is not an object")
            totals[tier] += sum(int(value) for value in subpools.values())
    return totals


def state_summary(state: Any, programs: Sequence[str] = ()) -> dict[str, Any]:
    index = unit_index(state)
    return {
        "rank_count": len(state_ranks(state)),
        "unit_count": len(index),
        "used_bytes": pool_used_bytes(state),
        "program_unit_counts": {
            program_id: len(program_refs(state, program_id)) for program_id in programs
        },
        "programs_in_usage": {
            program_id: program_in_usage(state, program_id) for program_id in programs
        },
    }


@dataclass
class Artifacts:
    directory: Path | None
    written: list[str] = field(default_factory=list)

    def write(self, name: str, value: Any) -> None:
        if self.directory is None:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / name
        with path.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
        self.written.append(str(path))


def poll_state(
    client: HttpClient,
    *,
    description: str,
    timeout: float,
    interval: float,
    predicate: Callable[[Any], tuple[bool, str]],
) -> Any:
    deadline = time.monotonic() + timeout
    last_detail = "no snapshot fetched"
    attempts = 0
    last_state: Any = None
    while True:
        attempts += 1
        try:
            last_state = client.json("GET", "/aginfer/state")
            state_ranks(last_state)
            ok, last_detail = predicate(last_state)
            if ok:
                return last_state
        except CheckFailed as exc:
            last_detail = str(exc)
        if time.monotonic() >= deadline:
            summary = None
            if last_state is not None:
                try:
                    summary = state_summary(last_state)
                except Exception as exc:  # diagnostics must not hide root failure
                    summary = {"summary_error": str(exc)}
            raise CheckFailed(
                f"timed out after {timeout:.1f}s waiting for {description}; "
                f"attempts={attempts}; last_observation={last_detail}; "
                f"last_state_summary={_short_json(summary)}"
            )
        time.sleep(interval)


def generate(
    client: HttpClient,
    *,
    prompt: str,
    program_id: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    response = client.json(
        "POST",
        "/generate",
        payload={
            "text": prompt,
            "program_id": program_id,
            "stream": False,
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": max_new_tokens,
                "ignore_eos": True,
            },
        },
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if not isinstance(response, dict):
        raise CheckFailed(
            f"POST /generate for {program_id} returned {type(response).__name__}, "
            f"expected object: {_short_json(response)}"
        )
    meta = response.get("meta_info")
    if not isinstance(meta, dict):
        raise CheckFailed(
            f"POST /generate for {program_id} lacks object meta_info: "
            f"{_short_json(response)}"
        )
    return {
        "elapsed_ms": elapsed_ms,
        "prompt_tokens": meta.get("prompt_tokens"),
        "cached_tokens": meta.get("cached_tokens"),
        "completion_tokens": meta.get("completion_tokens"),
        "finish_reason": meta.get("finish_reason"),
        "response": response,
    }


def end_program(client: HttpClient, program_id: str) -> dict[str, Any]:
    response = client.json(
        "POST", "/aginfer/session_end", payload={"program_id": program_id}
    )
    if not isinstance(response, dict):
        raise CheckFailed(
            f"SESSION_END({program_id}) response is not an object: "
            f"{_short_json(response)}"
        )
    if response.get("ok") is not True:
        raise CheckFailed(
            f"SESSION_END({program_id}) did not acknowledge complete cleanup: "
            f"{_short_json(response)}"
        )
    if response.get("program_id") != program_id:
        raise CheckFailed(
            f"SESSION_END({program_id}) echoed wrong program_id: "
            f"{response.get('program_id')!r}"
        )
    per_rank = response.get("per_rank")
    if not isinstance(per_rank, list) or not per_rank:
        raise CheckFailed(
            f"SESSION_END({program_id}) per_rank must be non-empty: "
            f"{_short_json(per_rank)}"
        )
    for rank_index, ack in enumerate(per_rank):
        if not isinstance(ack, dict):
            raise CheckFailed(
                f"SESSION_END({program_id}) rank {rank_index} ack is not an object"
            )
        if ack.get("ok") is not True or bool(ack.get("deferred")):
            raise CheckFailed(
                f"SESSION_END({program_id}) rank {rank_index} is not complete: "
                f"{_short_json(ack)}"
            )
        if int(ack.get("remaining_nodes", 0) or 0) != 0:
            raise CheckFailed(
                f"SESSION_END({program_id}) rank {rank_index} still reports "
                f"remaining_nodes={ack.get('remaining_nodes')}: {_short_json(ack)}"
            )
    return response


def ack_totals(response: Mapping[str, Any]) -> dict[str, int]:
    fields = (
        "matched_nodes",
        "holders_removed",
        "released_nodes",
        "released_hbm_tokens",
        "released_dram_tokens",
        "remaining_nodes",
    )
    return {
        field: sum(
            int(rank.get(field, 0) or 0)
            for rank in response.get("per_rank", [])
            if isinstance(rank, dict)
        )
        for field in fields
    }


def validate_idempotent_ack(response: Mapping[str, Any], program_id: str) -> None:
    bad_statuses: list[str] = []
    for rank_index, ack in enumerate(response["per_rank"]):
        if ack.get("status") != "already_absent":
            bad_statuses.append(f"rank {rank_index}: {ack.get('status')!r}")
        for field in (
            "matched_nodes",
            "holders_removed",
            "released_nodes",
            "released_hbm_tokens",
            "released_dram_tokens",
            "remaining_nodes",
        ):
            if int(ack.get(field, 0) or 0) != 0:
                raise CheckFailed(
                    f"second SESSION_END({program_id}) is not a no-op: rank "
                    f"{rank_index} {field}={ack.get(field)}; ack={_short_json(ack)}"
                )
    if bad_statuses:
        raise CheckFailed(
            f"second SESSION_END({program_id}) did not report already_absent: "
            + ", ".join(bad_statuses)
        )


def _build_prompts(
    run_id: str, shared_repeats: int, tail_repeats: int
) -> tuple[str, str]:
    shared_sentence = (
        "Dead KV verification shared context: retain this exact prefix across "
        "two independent agent programs and answer only after reading it. "
    )
    common = shared_sentence * shared_repeats
    tail_a = (
        f" Program A exclusive branch {run_id}: alpha-only evidence must not "
        "be shared with beta. "
    ) * tail_repeats
    tail_b = (
        f" Program B exclusive branch {run_id}: beta-only evidence must not "
        "be shared with alpha. "
    ) * tail_repeats
    suffix = " Respond with exactly one short word."
    return common + tail_a + suffix, common + tail_b + suffix


def _format_refs(refs: Iterable[UnitRef], limit: int = 8) -> str:
    ordered = sorted(refs)
    shown = [f"r{rank}:{unit_hash[:14]}" for rank, unit_hash in ordered[:limit]]
    if len(ordered) > limit:
        shown.append(f"+{len(ordered) - limit} more")
    return "[" + ", ".join(shown) + "]"


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.shared_repeats < 1 or args.tail_repeats < 1:
        raise CheckFailed("--shared-repeats and --tail-repeats must be positive")
    if args.max_new_tokens < 1:
        raise CheckFailed("--max-new-tokens must be positive")
    if args.poll_interval <= 0 or args.settle_timeout <= 0:
        raise CheckFailed("--poll-interval and --settle-timeout must be positive")
    if not args.skip_flush and not args.confirm_dedicated_server:
        raise CheckFailed(
            "--confirm-dedicated-server is required because the verifier calls "
            "/flush_cache"
        )

    client = HttpClient(args.base_url, args.request_timeout)
    run_id = uuid.uuid4().hex[:12]
    program_a = f"deadkv-A-{run_id}"
    program_b = f"deadkv-B-{run_id}"
    program_c = f"deadkv-C-{run_id}"
    prompt_a, prompt_b = _build_prompts(run_id, args.shared_repeats, args.tail_repeats)
    artifacts = Artifacts(Path(args.artifact_dir) if args.artifact_dir else None)
    started = time.perf_counter()
    result: dict[str, Any] = {
        "run_id": run_id,
        "base_url": args.base_url.rstrip("/"),
        "programs": {"A": program_a, "B": program_b, "C": program_c},
        "parameters": {
            "shared_repeats": args.shared_repeats,
            "tail_repeats": args.tail_repeats,
            "max_new_tokens": args.max_new_tokens,
        },
    }

    print(f"Dead-KV E2E run={run_id} server={args.base_url.rstrip('/')}")
    print("[1/9] preflight: health and aginfer state schema")
    client.request("GET", "/health", expected=(200,))
    initial = client.json("GET", "/aginfer/state")
    state_ranks(initial)
    artifacts.write("00_preflight_state.json", initial)

    if not args.skip_flush:
        print("[2/9] flush cache and wait for an empty unit set")
        _, body, _ = client.request(
            "POST", f"/flush_cache?timeout={args.settle_timeout:g}", expected=(200,)
        )
        result["flush_response"] = body.decode("utf-8", "replace").strip()
        empty = poll_state(
            client,
            description="/flush_cache to become visible in /aginfer/state",
            timeout=args.settle_timeout,
            interval=args.poll_interval,
            predicate=lambda state: (
                len(unit_index(state)) == 0,
                f"unit_count={len(unit_index(state))}",
            ),
        )
        artifacts.write("01_after_flush_state.json", empty)
    else:
        print(
            "[2/9] cache flush skipped by request; unique program IDs still isolate tags"
        )

    print("[3/9] generate A, then prove completed response did not end its KV lifetime")
    gen_a = generate(
        client,
        prompt=prompt_a,
        program_id=program_a,
        max_new_tokens=args.max_new_tokens,
    )
    artifacts.write("02_generate_a_response.json", gen_a["response"])
    state_after_a = poll_state(
        client,
        description="program A to remain visible after its /generate response",
        timeout=args.settle_timeout,
        interval=args.poll_interval,
        predicate=lambda state: (
            bool(program_refs(state, program_a)) and program_in_usage(state, program_a),
            f"A_units={len(program_refs(state, program_a))}, "
            f"A_in_usage={program_in_usage(state, program_a)}",
        ),
    )
    artifacts.write("03_after_a_response_state.json", state_after_a)
    print(
        "      response complete; A still owns "
        f"{len(program_refs(state_after_a, program_a))} unit(s)"
    )

    print("[4/9] generate B and wait for shared, A-exclusive, and B-exclusive units")
    gen_b = generate(
        client,
        prompt=prompt_b,
        program_id=program_b,
        max_new_tokens=args.max_new_tokens,
    )
    artifacts.write("04_generate_b_response.json", gen_b["response"])

    topology: dict[str, set[UnitRef]] = {}

    def topology_ready(state: Any) -> tuple[bool, str]:
        refs_a = program_refs(state, program_a)
        refs_b = program_refs(state, program_b)
        shared = refs_a & refs_b
        exclusive_a = refs_a - refs_b
        exclusive_b = refs_b - refs_a
        ready = bool(shared and exclusive_a and exclusive_b)
        if ready:
            topology.update(
                {
                    "A": refs_a,
                    "B": refs_b,
                    "shared": shared,
                    "A_exclusive": exclusive_a,
                    "B_exclusive": exclusive_b,
                }
            )
        return (
            ready,
            f"A={len(refs_a)}, B={len(refs_b)}, shared={len(shared)}, "
            f"A_exclusive={len(exclusive_a)}, B_exclusive={len(exclusive_b)}",
        )

    before_end = poll_state(
        client,
        description="A/B shared-prefix topology",
        timeout=args.settle_timeout,
        interval=args.poll_interval,
        predicate=topology_ready,
    )
    artifacts.write("05_before_end_a_state.json", before_end)
    before_used = pool_used_bytes(before_end)
    exclusive_a_bytes = refs_bytes_by_tier(before_end, topology["A_exclusive"])
    print(
        "      topology: "
        f"shared={len(topology['shared'])} {_format_refs(topology['shared'])}; "
        f"A-only={len(topology['A_exclusive'])} "
        f"{_format_refs(topology['A_exclusive'])}; "
        f"B-only={len(topology['B_exclusive'])} "
        f"{_format_refs(topology['B_exclusive'])}"
    )
    print(f"      A-exclusive bytes by tier: {exclusive_a_bytes}")
    if exclusive_a_bytes["HBM"] <= 0 and exclusive_a_bytes["DRAM"] <= 0:
        raise CheckFailed(
            "A-exclusive units exist but report no HBM/DRAM bytes; cannot prove "
            "physical Dead-KV reclamation"
        )

    print("[5/9] SESSION_END(A): require all-rank ACK and poll complete reclamation")
    end_a = end_program(client, program_a)
    artifacts.write("06_end_a_response.json", end_a)
    end_a_totals = ack_totals(end_a)
    if end_a_totals["released_nodes"] <= 0:
        raise CheckFailed(
            "SESSION_END(A) acknowledged but released_nodes=0 despite observed "
            f"A-exclusive units; ack={_short_json(end_a)}"
        )
    if exclusive_a_bytes["HBM"] > 0 and end_a_totals["released_hbm_tokens"] <= 0:
        raise CheckFailed(
            "A-exclusive state contained HBM bytes, but SESSION_END(A) reported "
            f"released_hbm_tokens=0; ack={_short_json(end_a)}"
        )
    if exclusive_a_bytes["DRAM"] > 0 and end_a_totals["released_dram_tokens"] <= 0:
        raise CheckFailed(
            "A-exclusive state contained DRAM bytes, but SESSION_END(A) reported "
            f"released_dram_tokens=0; ack={_short_json(end_a)}"
        )

    def a_fully_reclaimed(state: Any) -> tuple[bool, str]:
        index = unit_index(state)
        current_a = program_refs(state, program_a)
        current_b = program_refs(state, program_b)
        a_exclusive_still_present = topology["A_exclusive"] & set(index)
        lost_shared = topology["shared"] - set(index)
        lost_b_exclusive = topology["B_exclusive"] - set(index)
        bad_shared = {
            ref
            for ref in topology["shared"] & set(index)
            if program_a in holders(index[ref]) or program_b not in holders(index[ref])
        }
        bad_b_exclusive = {
            ref
            for ref in topology["B_exclusive"] & set(index)
            if program_b not in holders(index[ref])
        }
        used = pool_used_bytes(state)
        no_drop_tiers = [
            tier
            for tier in ("HBM", "DRAM")
            if exclusive_a_bytes[tier] > 0 and used[tier] >= before_used[tier]
        ]
        ok = (
            not current_a
            and not program_in_usage(state, program_a)
            and not a_exclusive_still_present
            and not lost_shared
            and not lost_b_exclusive
            and not bad_shared
            and not bad_b_exclusive
            and bool(current_b)
            and program_in_usage(state, program_b)
            and not no_drop_tiers
        )
        detail = (
            f"A_units={len(current_a)}, A_in_usage={program_in_usage(state, program_a)}, "
            f"A_only_present={len(a_exclusive_still_present)}, "
            f"lost_shared={len(lost_shared)}, bad_shared={len(bad_shared)}, "
            f"lost_B_only={len(lost_b_exclusive)}, "
            f"bad_B_only={len(bad_b_exclusive)}, B_units={len(current_b)}, "
            f"used_before={before_used}, used_now={used}, "
            f"tiers_without_drop={no_drop_tiers}"
        )
        return ok, detail

    after_end_a = poll_state(
        client,
        description="A holder removal plus exclusive HBM/DRAM reclamation",
        timeout=args.settle_timeout,
        interval=args.poll_interval,
        predicate=a_fully_reclaimed,
    )
    artifacts.write("07_after_end_a_state.json", after_end_a)
    after_a_used = pool_used_bytes(after_end_a)
    print(f"      all-rank ACK totals: {end_a_totals}")
    print(
        "      pool used-byte delta: "
        + str({tier: before_used[tier] - after_a_used[tier] for tier in before_used})
    )

    print("[6/9] repeat SESSION_END(A) and require idempotent already_absent no-op")
    end_a_again = end_program(client, program_a)
    validate_idempotent_ack(end_a_again, program_a)
    artifacts.write("08_end_a_idempotent_response.json", end_a_again)

    print("[7/9] repeat B prompt: retained shared/B KV must produce a cache hit")
    gen_b_repeat = generate(
        client,
        prompt=prompt_b,
        program_id=program_b,
        max_new_tokens=args.max_new_tokens,
    )
    artifacts.write("09_generate_b_repeat_response.json", gen_b_repeat["response"])
    cached_tokens = gen_b_repeat.get("cached_tokens")
    if not isinstance(cached_tokens, (int, float)):
        raise CheckFailed(
            "repeated B /generate response lacks numeric meta_info.cached_tokens; "
            f"meta evidence={_short_json(gen_b_repeat['response'].get('meta_info'))}"
        )
    if int(cached_tokens) <= 0:
        raise CheckFailed(
            "repeated B prompt reported cached_tokens=0 after A cleanup; shared B KV "
            "was not reusable"
        )
    after_b_repeat = poll_state(
        client,
        description="B to remain live after repeated generation while A stays absent",
        timeout=args.settle_timeout,
        interval=args.poll_interval,
        predicate=lambda state: (
            bool(program_refs(state, program_b))
            and not program_refs(state, program_a)
            and not program_in_usage(state, program_a),
            f"B_units={len(program_refs(state, program_b))}, "
            f"A_units={len(program_refs(state, program_a))}, "
            f"A_in_usage={program_in_usage(state, program_a)}",
        ),
    )
    artifacts.write("10_after_b_repeat_state.json", after_b_repeat)
    print(f"      B cache hit: cached_tokens={int(cached_tokens)}")

    print("[8/9] SESSION_END(B) and verify the retained branch also fully cleans")
    end_b = end_program(client, program_b)
    artifacts.write("11_end_b_response.json", end_b)
    after_end_b = poll_state(
        client,
        description="program B to disappear after SESSION_END(B)",
        timeout=args.settle_timeout,
        interval=args.poll_interval,
        predicate=lambda state: (
            not program_refs(state, program_b)
            and not program_in_usage(state, program_b),
            f"B_units={len(program_refs(state, program_b))}, "
            f"B_in_usage={program_in_usage(state, program_b)}",
        ),
    )
    artifacts.write("12_after_end_b_state.json", after_end_b)

    print("[9/9] generate after full cleanup; server must remain healthy")
    health_prompt = (
        f"Post-reclamation health probe {run_id}. Reply with exactly the word healthy."
    )
    gen_c = generate(
        client,
        prompt=health_prompt,
        program_id=program_c,
        max_new_tokens=args.max_new_tokens,
    )
    artifacts.write("13_post_cleanup_generate_response.json", gen_c["response"])
    end_c = end_program(client, program_c)
    artifacts.write("14_end_c_response.json", end_c)
    client.request("GET", "/health", expected=(200,))

    result.update(
        {
            "status": "PASS",
            "elapsed_seconds": time.perf_counter() - started,
            "topology": {
                "A_units": len(topology["A"]),
                "B_units": len(topology["B"]),
                "shared_units": len(topology["shared"]),
                "A_exclusive_units": len(topology["A_exclusive"]),
                "B_exclusive_units": len(topology["B_exclusive"]),
                "A_exclusive_bytes": exclusive_a_bytes,
            },
            "pool_used_before_end_a": before_used,
            "pool_used_after_end_a": after_a_used,
            "pool_released_bytes": {
                tier: before_used[tier] - after_a_used[tier] for tier in before_used
            },
            "end_a_ack_totals": end_a_totals,
            "b_repeat_cached_tokens": int(cached_tokens),
            "generate_ms": {
                "A": gen_a["elapsed_ms"],
                "B": gen_b["elapsed_ms"],
                "B_repeat": gen_b_repeat["elapsed_ms"],
                "post_cleanup": gen_c["elapsed_ms"],
            },
            "artifact_files": artifacts.written,
        }
    )
    artifacts.write("result.json", result)
    if artifacts.directory is not None:
        result["artifact_files"] = artifacts.written
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a strict black-box Dead-KV full-pipeline acceptance test."
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("SGLANG_BASE_URL", "http://127.0.0.1:30001"),
        help="SGLang HTTP base URL (default: %(default)s)",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=300.0,
        help="per-HTTP-request timeout in seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--settle-timeout",
        type=float,
        default=30.0,
        help="eventual-state polling timeout in seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=0.2,
        help="state polling interval in seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1,
        help="tokens generated per request (default: %(default)s)",
    )
    parser.add_argument(
        "--shared-repeats",
        type=int,
        default=48,
        help="size of shared text prefix (default: %(default)s)",
    )
    parser.add_argument(
        "--tail-repeats",
        type=int,
        default=32,
        help="size of each distinct branch (default: %(default)s)",
    )
    parser.add_argument(
        "--skip-flush",
        action="store_true",
        help="do not call /flush_cache before the test",
    )
    parser.add_argument(
        "--confirm-dedicated-server",
        action="store_true",
        help="acknowledge that the default verification path flushes the KV cache",
    )
    parser.add_argument(
        "--artifact-dir",
        help="optional directory for full request/state/ACK JSON evidence",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="print a traceback in addition to the concise failure reason",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run(args)
    except CheckFailed as exc:
        print(f"\nFAIL: {exc}", file=sys.stderr)
        if args.debug:
            traceback.print_exc()
        return 1
    except KeyboardInterrupt:
        print("\nABORTED by user", file=sys.stderr)
        return 130
    except Exception as exc:
        print(
            f"\nERROR: unexpected {type(exc).__name__}: {exc}. "
            "Re-run with --debug for a traceback.",
            file=sys.stderr,
        )
        if args.debug:
            traceback.print_exc()
        return 2

    print("\nPASS: Dead-KV full pipeline reclaimed A without damaging B")
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

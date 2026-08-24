"""Paired Dead-KV benchmark: no-END control versus explicit SESSION_END.

The primary mode talks directly to one SGLang server so that the only changed
variable is the terminal lifecycle signal:

* ``baseline``: identical inference requests, but no ``SESSION_END``.
* ``ours``: sends ``SESSION_END`` after each short-lived session completes.

Both conditions use the same model process, cache capacity, prompts, request
order, concurrency, and sampling parameters.  Every condition starts after a
verified cache flush.  A fixed live working set is probed before and after
short-lived session churn.  By default every prompt has a unique page-aligned
prefix, avoiding holder-set bookkeeping as a confound; shared-prefix stress can
be enabled explicitly with ``--shared-pages``.

Only the Python standard library is required.  The program writes a durable
JSON artifact directory and a compact Markdown report even when a run fails.

This benchmark flushes the target server's KV cache.  Use it only against a
dedicated benchmark deployment and pass ``--confirm-dedicated-server``.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import datetime as dt
import hashlib
import json
import math
import os
import pathlib
import random
import statistics
import sys
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

TIERS = ("HBM", "DRAM", "DISK")
CONDITIONS = ("baseline", "ours")
PromptInput = str | list[int]


class BenchmarkError(RuntimeError):
    """The target was unreachable or an experimental invariant was violated."""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def short(value: Any, limit: int = 2000) -> str:
    try:
        rendered = json.dumps(value, sort_keys=True, ensure_ascii=False)
    except Exception:
        rendered = repr(value)
    if len(rendered) > limit:
        return rendered[:limit] + f"... <{len(rendered) - limit} chars omitted>"
    return rendered


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def atomic_write_json(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")
    temporary.replace(path)


@dataclasses.dataclass
class HttpResult:
    method: str
    url: str
    status: int | None
    elapsed_seconds: float
    headers: dict[str, str]
    body_text: str
    body_json: Any
    body_size_bytes: int
    body_sha256: str
    transport_error: str | None = None

    @property
    def ok(self) -> bool:
        return self.status is not None and 200 <= self.status < 300

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def http_opener(url: str) -> urllib.request.OpenerDirector:
    # Loopback benchmark traffic must never leave the node through a host-level
    # forward proxy. Non-loopback targets continue to honor the environment.
    hostname = urllib.parse.urlsplit(url).hostname
    if hostname in {"127.0.0.1", "::1", "localhost"}:
        return urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return urllib.request.build_opener()


def http_request(
    method: str,
    url: str,
    *,
    payload: Any = None,
    headers: Mapping[str, str] | None = None,
    timeout: float = 180.0,
) -> HttpResult:
    request_headers = {
        "Accept": "application/json",
        "User-Agent": "deadkv-ab/1",
    }
    if headers:
        request_headers.update(headers)
    data = None
    if payload is not None:
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        request_headers.setdefault("Content-Type", "application/json")
    request = urllib.request.Request(
        url=url,
        data=data,
        headers=request_headers,
        method=method.upper(),
    )
    started = time.perf_counter()
    status: int | None = None
    response_headers: dict[str, str] = {}
    body = b""
    transport_error: str | None = None
    try:
        with http_opener(url).open(request, timeout=timeout) as response:
            status = int(response.status)
            response_headers = dict(response.headers.items())
            body = response.read()
    except urllib.error.HTTPError as exc:
        status = int(exc.code)
        response_headers = dict(exc.headers.items()) if exc.headers else {}
        body = exc.read()
    except Exception as exc:
        transport_error = f"{type(exc).__name__}: {exc}"
    elapsed = time.perf_counter() - started
    body_text_full = body.decode("utf-8", errors="replace")
    body_json: Any = None
    if body_text_full.strip():
        try:
            body_json = json.loads(body_text_full)
        except json.JSONDecodeError:
            pass
    return HttpResult(
        method=method.upper(),
        url=url,
        status=status,
        elapsed_seconds=elapsed,
        headers=response_headers,
        body_text=short(body_text_full, 16000),
        body_json=body_json,
        body_size_bytes=len(body),
        body_sha256=hashlib.sha256(body).hexdigest(),
        transport_error=transport_error,
    )


def require_ok(result: HttpResult, label: str) -> None:
    if not result.ok:
        raise BenchmarkError(
            f"{label} failed: status={result.status}, "
            f"transport_error={result.transport_error!r}, body={result.body_text!r}"
        )


@dataclasses.dataclass
class RequestMetric:
    label: str
    role: str
    program_id: str
    status: int | None
    ok: bool
    elapsed_seconds: float
    ttft_seconds: float | None
    ttft_source: str | None
    first_byte_seconds: float | None
    first_event_seconds: float | None
    prompt_tokens: int | None
    cached_tokens: int | None
    completion_tokens: int | None
    response_size_bytes: int
    response_sha256: str
    event_count: int
    error: str | None
    started_at: str

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _numeric_int(value: Any) -> int | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(value)
    return None


def _cached_from_usage(usage: Any) -> int | None:
    if not isinstance(usage, Mapping):
        return None
    details = usage.get("prompt_tokens_details")
    if isinstance(details, Mapping):
        cached = _numeric_int(details.get("cached_tokens"))
        if cached is not None:
            return cached
    return _numeric_int(usage.get("cached_tokens"))


def _stream_event_fields(
    event: Mapping[str, Any], backend: str
) -> tuple[bool, str | None, Mapping[str, Any] | None]:
    """Return (contains_output_token, error, usage/meta mapping)."""
    if "error" in event:
        return False, short(event.get("error")), None
    if backend == "direct":
        text_value = event.get("text")
        output_ids = event.get("output_ids")
        has_token = (isinstance(text_value, str) and bool(text_value)) or (
            isinstance(output_ids, list) and bool(output_ids)
        )
        meta = event.get("meta_info")
        return has_token, None, meta if isinstance(meta, Mapping) else None
    choices = event.get("choices")
    has_token = False
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, Mapping):
                continue
            delta = choice.get("delta")
            if isinstance(delta, Mapping):
                content = delta.get("content")
                if isinstance(content, str) and content:
                    has_token = True
            text_value = choice.get("text")
            if isinstance(text_value, str) and text_value:
                has_token = True
    usage = event.get("usage")
    return has_token, None, usage if isinstance(usage, Mapping) else None


def stream_json_events(
    *,
    backend: str,
    label: str,
    role: str,
    program_id: str,
    url: str,
    payload: Mapping[str, Any],
    headers: Mapping[str, str],
    timeout: float,
) -> RequestMetric:
    data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    request_headers = {
        "Accept": "text/event-stream",
        "Content-Type": "application/json",
        "User-Agent": "deadkv-ab/1",
        **dict(headers),
    }
    request = urllib.request.Request(
        url=url, data=data, headers=request_headers, method="POST"
    )
    started_wall = utc_now()
    started = time.perf_counter()
    first_byte: float | None = None
    first_event: float | None = None
    first_token: float | None = None
    status: int | None = None
    event_count = 0
    body_size = 0
    digest = hashlib.sha256()
    prompt_tokens: int | None = None
    cached_tokens: int | None = None
    completion_tokens: int | None = None
    error: str | None = None
    try:
        with http_opener(url).open(request, timeout=timeout) as response:
            status = int(response.status)
            for raw_line in response:
                now_delta = time.perf_counter() - started
                if raw_line and first_byte is None:
                    first_byte = now_delta
                body_size += len(raw_line)
                digest.update(raw_line)
                stripped = raw_line.strip()
                if not stripped:
                    continue
                if stripped.startswith(b"data:"):
                    stripped = stripped[5:].strip()
                if stripped in (b"[DONE]", b"DONE"):
                    continue
                try:
                    event = json.loads(stripped)
                except json.JSONDecodeError:
                    # Some gateways may emit SSE comments or event labels.
                    continue
                if not isinstance(event, Mapping):
                    continue
                event_count += 1
                if first_event is None:
                    first_event = now_delta
                has_token, event_error, usage = _stream_event_fields(event, backend)
                if event_error is not None:
                    error = event_error
                if has_token and first_token is None:
                    first_token = now_delta
                if isinstance(usage, Mapping):
                    prompt_tokens = (
                        _numeric_int(usage.get("prompt_tokens"))
                        if _numeric_int(usage.get("prompt_tokens")) is not None
                        else prompt_tokens
                    )
                    cached_tokens = (
                        _cached_from_usage(usage)
                        if _cached_from_usage(usage) is not None
                        else cached_tokens
                    )
                    completion_tokens = (
                        _numeric_int(usage.get("completion_tokens"))
                        if _numeric_int(usage.get("completion_tokens")) is not None
                        else completion_tokens
                    )
    except urllib.error.HTTPError as exc:
        status = int(exc.code)
        body = exc.read()
        body_size += len(body)
        digest.update(body)
        error = body.decode("utf-8", errors="replace")[:4000]
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    elapsed = time.perf_counter() - started
    if status is not None and not 200 <= status < 300 and error is None:
        error = f"HTTP {status}"
    if first_token is None and error is None:
        error = "stream completed without an observable output-token event"
    return RequestMetric(
        label=label,
        role=role,
        program_id=program_id,
        status=status,
        ok=error is None and status is not None and 200 <= status < 300,
        elapsed_seconds=elapsed,
        ttft_seconds=first_token,
        ttft_source="first_output_sse_event" if first_token is not None else None,
        first_byte_seconds=first_byte,
        first_event_seconds=first_event,
        prompt_tokens=prompt_tokens,
        cached_tokens=cached_tokens,
        completion_tokens=completion_tokens,
        response_size_bytes=body_size,
        response_sha256=digest.hexdigest(),
        event_count=event_count,
        error=error,
        started_at=started_wall,
    )


def nonstream_metric(
    *,
    backend: str,
    label: str,
    role: str,
    program_id: str,
    result: HttpResult,
) -> RequestMetric:
    payload = result.body_json
    prompt_tokens: int | None = None
    cached_tokens: int | None = None
    completion_tokens: int | None = None
    error = result.transport_error
    if isinstance(payload, Mapping):
        if "error" in payload:
            error = short(payload.get("error"))
        metrics: Any
        if backend == "direct":
            metrics = payload.get("meta_info")
        else:
            metrics = payload.get("usage")
        if isinstance(metrics, Mapping):
            prompt_tokens = _numeric_int(metrics.get("prompt_tokens"))
            cached_tokens = _cached_from_usage(metrics)
            completion_tokens = _numeric_int(metrics.get("completion_tokens"))
    if not result.ok and error is None:
        error = f"HTTP {result.status}: {result.body_text}"
    return RequestMetric(
        label=label,
        role=role,
        program_id=program_id,
        status=result.status,
        ok=result.ok and error is None,
        elapsed_seconds=result.elapsed_seconds,
        ttft_seconds=None,
        ttft_source="unavailable_non_stream",
        first_byte_seconds=None,
        first_event_seconds=None,
        prompt_tokens=prompt_tokens,
        cached_tokens=cached_tokens,
        completion_tokens=completion_tokens,
        response_size_bytes=result.body_size_bytes,
        response_sha256=result.body_sha256,
        event_count=0,
        error=error,
        started_at=utc_now(),
    )


def normalize_states(payload: Any, backend: str) -> list[dict[str, Any]]:
    if backend == "dynamo":
        if not isinstance(payload, Mapping):
            raise BenchmarkError(
                f"Dynamo state response is not an object: {short(payload)}"
            )
        items = payload.get("result")
        if isinstance(items, Mapping):
            items = [items]
        if not isinstance(items, list) or not items:
            raise BenchmarkError(
                f"Dynamo state response lacks result[]: {short(payload)}"
            )
        states: list[dict[str, Any]] = []
        for rank, item in enumerate(items):
            state: Any = None
            if isinstance(item, Mapping):
                raw = item.get("state_bytes")
                if isinstance(raw, str) and raw.strip():
                    try:
                        state = json.loads(raw)
                    except json.JSONDecodeError as exc:
                        raise BenchmarkError(
                            f"rank {rank} state_bytes is invalid JSON: {exc}"
                        ) from exc
                elif isinstance(item.get("state"), Mapping):
                    state = item["state"]
            elif isinstance(item, str):
                try:
                    state = json.loads(item)
                except json.JSONDecodeError as exc:
                    raise BenchmarkError(
                        f"rank {rank} state string is invalid JSON: {exc}"
                    ) from exc
            if not isinstance(state, dict):
                raise BenchmarkError(f"rank {rank} has no usable state: {short(item)}")
            states.append(state)
    else:
        if not isinstance(payload, Mapping):
            raise BenchmarkError(
                f"SGLang state response is not an object: {short(payload)}"
            )
        per_rank = payload.get("per_rank")
        if per_rank is None:
            states = [dict(payload)]
        elif isinstance(per_rank, list) and per_rank:
            states = []
            for rank, state in enumerate(per_rank):
                if not isinstance(state, Mapping):
                    raise BenchmarkError(
                        f"direct state per_rank[{rank}] is not an object: {short(state)}"
                    )
                states.append(dict(state))
        else:
            raise BenchmarkError(
                f"direct state has invalid per_rank: {short(per_rank)}"
            )
    for rank, state in enumerate(states):
        if not isinstance(state.get("units"), list):
            raise BenchmarkError(
                f"rank {rank} lacks units[]; expected aginfer-enabled state schema"
            )
        if not isinstance(state.get("pool_usage"), Mapping):
            raise BenchmarkError(f"rank {rank} lacks pool_usage object")
        if not isinstance(state.get("per_program_usage"), Mapping):
            raise BenchmarkError(f"rank {rank} lacks per_program_usage object")
    return states


def unit_holders(unit: Mapping[str, Any]) -> set[str]:
    raw = unit.get("session_ids")
    if not isinstance(raw, list):
        return set()
    return {str(value) for value in raw}


def unit_tier_bytes(unit: Mapping[str, Any]) -> dict[str, int]:
    totals = {tier: 0 for tier in TIERS}
    raw = unit.get("n_bytes")
    if not isinstance(raw, Mapping):
        return totals
    for tier in TIERS:
        subpools = raw.get(tier)
        if not isinstance(subpools, Mapping):
            continue
        totals[tier] = sum(
            int(amount)
            for amount in subpools.values()
            if isinstance(amount, (int, float)) and not isinstance(amount, bool)
        )
    return totals


def add_tiers(target: dict[str, int], source: Mapping[str, int]) -> None:
    for tier in TIERS:
        target[tier] = target.get(tier, 0) + int(source.get(tier, 0))


def analyze_states(
    states: Sequence[Mapping[str, Any]],
    tracked_programs: set[str],
    ended_programs: set[str],
) -> dict[str, Any]:
    pool_used = {tier: 0 for tier in TIERS}
    pool_cap = {tier: 0 for tier in TIERS}
    radix_physical = {tier: 0 for tier in TIERS}
    tracked_physical = {tier: 0 for tier in TIERS}
    dead_physical = {tier: 0 for tier in TIERS}
    programs_present: set[str] = set()
    programs_in_usage: set[str] = set()
    external_holders: set[str] = set()
    capacity_signature: list[dict[str, Any]] = []
    rank_summaries: list[dict[str, Any]] = []
    tracked_unit_count = 0
    dead_unit_count = 0
    tracked_holder_references = 0
    tracked_hit_count = 0
    for rank_index, state in enumerate(states):
        rank_pool_used = {tier: 0 for tier in TIERS}
        rank_pool_cap = {tier: 0 for tier in TIERS}
        rank_radix = {tier: 0 for tier in TIERS}
        rank_tracked = {tier: 0 for tier in TIERS}
        rank_dead = {tier: 0 for tier in TIERS}
        rank_capacity: dict[str, Any] = {"rank": rank_index, "tiers": {}}
        pool_usage = state.get("pool_usage", {})
        for tier in TIERS:
            tier_entry = (
                pool_usage.get(tier, {}) if isinstance(pool_usage, Mapping) else {}
            )
            subpools = (
                tier_entry.get("subpools", {})
                if isinstance(tier_entry, Mapping)
                else {}
            )
            rank_capacity["tiers"][tier] = {}
            if isinstance(subpools, Mapping):
                for name, fields in sorted(
                    subpools.items(), key=lambda pair: str(pair[0])
                ):
                    if not isinstance(fields, Mapping):
                        continue
                    used = _numeric_int(fields.get("used_bytes")) or 0
                    cap = _numeric_int(fields.get("cap_bytes")) or 0
                    rank_pool_used[tier] += used
                    rank_pool_cap[tier] += cap
                    rank_capacity["tiers"][tier][str(name)] = {
                        "cap_bytes": cap,
                        "page_bytes": _numeric_int(fields.get("page_bytes")) or 0,
                        "decode_bytes_per_token": _numeric_int(
                            fields.get("decode_bytes_per_token")
                        )
                        or 0,
                    }
        usage = state.get("per_program_usage", {})
        if isinstance(usage, Mapping):
            programs_in_usage.update(
                str(pid) for pid in usage if str(pid) in tracked_programs
            )
        rank_programs: set[str] = set()
        rank_tracked_units = 0
        rank_dead_units = 0
        for unit in state.get("units", []):
            if not isinstance(unit, Mapping):
                continue
            holders = unit_holders(unit)
            amounts = unit_tier_bytes(unit)
            add_tiers(rank_radix, amounts)
            tracked_holders = holders & tracked_programs
            if not tracked_holders:
                continue
            rank_tracked_units += 1
            tracked_unit_count += 1
            tracked_holder_references += len(tracked_holders)
            rank_programs.update(tracked_holders)
            programs_present.update(tracked_holders)
            external_holders.update(holders - tracked_programs)
            add_tiers(rank_tracked, amounts)
            hit_count = _numeric_int(unit.get("hit_count"))
            if hit_count is not None:
                tracked_hit_count += hit_count
            # A unit is dead only when every holder is a client-declared ended
            # program.  The live anchor therefore protects shared-prefix units.
            if holders and holders.issubset(ended_programs):
                rank_dead_units += 1
                dead_unit_count += 1
                add_tiers(rank_dead, amounts)
        add_tiers(pool_used, rank_pool_used)
        add_tiers(pool_cap, rank_pool_cap)
        add_tiers(radix_physical, rank_radix)
        add_tiers(tracked_physical, rank_tracked)
        add_tiers(dead_physical, rank_dead)
        capacity_signature.append(rank_capacity)
        rank_summaries.append(
            {
                "rank": rank_index,
                "unit_count": len(state.get("units", [])),
                "tracked_unit_count": rank_tracked_units,
                "dead_unit_count": rank_dead_units,
                "programs_present": sorted(rank_programs),
                "pool_used_bytes": rank_pool_used,
                "pool_cap_bytes": rank_pool_cap,
                "radix_physical_bytes": rank_radix,
                "tracked_physical_bytes": rank_tracked,
                "dead_physical_bytes": rank_dead,
            }
        )
    utilization = {
        tier: (pool_used[tier] / pool_cap[tier] if pool_cap[tier] > 0 else None)
        for tier in TIERS
    }
    return {
        "rank_count": len(states),
        "unit_count": sum(rank["unit_count"] for rank in rank_summaries),
        "tracked_unit_count": tracked_unit_count,
        "dead_unit_count": dead_unit_count,
        "tracked_holder_references": tracked_holder_references,
        "tracked_hit_count": tracked_hit_count,
        "programs_present": sorted(programs_present),
        "programs_in_usage": sorted(programs_in_usage),
        "ended_programs_present": sorted(programs_present & ended_programs),
        "external_holders": sorted(external_holders),
        "pool_used_bytes": pool_used,
        "pool_cap_bytes": pool_cap,
        "pool_utilization": utilization,
        "radix_physical_bytes": radix_physical,
        "tracked_physical_bytes": tracked_physical,
        "dead_physical_bytes": dead_physical,
        "tracked_physical_bytes_total": sum(tracked_physical.values()),
        "dead_physical_bytes_total": sum(dead_physical.values()),
        "capacity_signature": capacity_signature,
        "ranks": rank_summaries,
    }


def infer_page_sizes(states: Sequence[Mapping[str, Any]]) -> list[int]:
    sizes: set[int] = set()
    for state in states:
        pool = state.get("pool_usage")
        hbm = pool.get("HBM") if isinstance(pool, Mapping) else None
        subpools = hbm.get("subpools") if isinstance(hbm, Mapping) else None
        if not isinstance(subpools, Mapping):
            continue
        for fields in subpools.values():
            if not isinstance(fields, Mapping):
                continue
            page_bytes = _numeric_int(fields.get("page_bytes")) or 0
            decode_bpt = _numeric_int(fields.get("decode_bytes_per_token")) or 0
            if page_bytes > 0 and decode_bpt > 0 and page_bytes % decode_bpt == 0:
                sizes.add(page_bytes // decode_bpt)
    return sorted(sizes)


class Artifacts:
    def __init__(self, root: pathlib.Path, run_id: str) -> None:
        self.run_dir = root / run_id
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.summary: dict[str, Any] = {
            "schema_version": 1,
            "run_id": run_id,
            "started_at": utc_now(),
            "status": "running",
            "warnings": [],
            "trials": [],
        }
        self._lock = threading.Lock()
        self.save_summary()

    def path(self, name: str) -> pathlib.Path:
        return self.run_dir / name

    def write(self, name: str, value: Any) -> pathlib.Path:
        path = self.path(name)
        atomic_write_json(path, value)
        return path

    def save_summary(self) -> None:
        with self._lock:
            atomic_write_json(self.path("summary.json"), self.summary)

    def finish(self, status: str, **details: Any) -> None:
        self.summary.update(details)
        self.summary["status"] = status
        self.summary["finished_at"] = utc_now()
        self.save_summary()


class Backend:
    """Small transport adapter shared by direct SGLang and Dynamo modes."""

    kind: str

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args

    def preflight(self) -> dict[str, Any]:
        raise NotImplementedError

    def infer(
        self, *, label: str, role: str, program_id: str, prompt: PromptInput
    ) -> RequestMetric:
        raise NotImplementedError

    def end_program(self, program_id: str) -> dict[str, Any]:
        raise NotImplementedError

    def fetch_states(self) -> list[dict[str, Any]]:
        raise NotImplementedError

    def flush(self) -> dict[str, Any]:
        raise NotImplementedError


class DirectBackend(Backend):
    kind = "direct"

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.base_url = args.server_url.rstrip("/")

    def preflight(self) -> dict[str, Any]:
        health = http_request(
            "GET", self.base_url + "/health", timeout=min(self.args.request_timeout, 30)
        )
        require_ok(health, "direct SGLang health")
        states = self.fetch_states()
        analysis = analyze_states(states, set(), set())
        return {
            "backend": self.kind,
            "server_url": self.base_url,
            "health": health.as_dict(),
            "rank_count": len(states),
            "inferred_page_sizes_tokens": infer_page_sizes(states),
            "pool_cap_bytes": analysis["pool_cap_bytes"],
        }

    def _payload(self, prompt: PromptInput, program_id: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "program_id": program_id,
            "stream": bool(self.args.stream),
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": self.args.max_tokens,
                "ignore_eos": True,
            },
        }
        if isinstance(prompt, str):
            payload["text"] = prompt
        else:
            payload["input_ids"] = prompt
        return payload

    def infer(
        self, *, label: str, role: str, program_id: str, prompt: PromptInput
    ) -> RequestMetric:
        payload = self._payload(prompt, program_id)
        if self.args.stream:
            return stream_json_events(
                backend=self.kind,
                label=label,
                role=role,
                program_id=program_id,
                url=self.base_url + "/generate",
                payload=payload,
                headers={},
                timeout=self.args.request_timeout,
            )
        result = http_request(
            "POST",
            self.base_url + "/generate",
            payload=payload,
            timeout=self.args.request_timeout,
        )
        return nonstream_metric(
            backend=self.kind,
            label=label,
            role=role,
            program_id=program_id,
            result=result,
        )

    def end_program(self, program_id: str) -> dict[str, Any]:
        result = http_request(
            "POST",
            self.base_url + "/aginfer/session_end",
            payload={"program_id": program_id},
            timeout=self.args.request_timeout,
        )
        require_ok(result, f"SESSION_END({program_id})")
        payload = result.body_json
        if not isinstance(payload, Mapping) or payload.get("ok") is not True:
            raise BenchmarkError(
                f"SESSION_END({program_id}) returned incomplete ACK: {short(payload)}"
            )
        per_rank = payload.get("per_rank")
        if not isinstance(per_rank, list) or not per_rank:
            raise BenchmarkError(
                f"SESSION_END({program_id}) ACK lacks per_rank[]: {short(payload)}"
            )
        for rank, ack in enumerate(per_rank):
            if not isinstance(ack, Mapping):
                raise BenchmarkError(f"SESSION_END rank {rank} ACK is not an object")
            if ack.get("ok") is not True or bool(ack.get("deferred")):
                raise BenchmarkError(
                    f"SESSION_END rank {rank} did not complete: {short(ack)}"
                )
            if (_numeric_int(ack.get("remaining_nodes")) or 0) != 0:
                raise BenchmarkError(
                    f"SESSION_END rank {rank} retained nodes: {short(ack)}"
                )
        return {
            "http": result.as_dict(),
            "ack": payload,
        }

    def fetch_states(self) -> list[dict[str, Any]]:
        result = http_request(
            "GET", self.base_url + "/aginfer/state", timeout=self.args.request_timeout
        )
        require_ok(result, "GET /aginfer/state")
        return normalize_states(result.body_json, self.kind)

    def flush(self) -> dict[str, Any]:
        query = urllib.parse.urlencode({"timeout": self.args.poll_timeout})
        result = http_request(
            "POST",
            self.base_url + "/flush_cache?" + query,
            timeout=max(self.args.request_timeout, self.args.poll_timeout + 5),
        )
        require_ok(result, "direct flush_cache")
        return result.as_dict()


class DynamoBackend(Backend):
    kind = "dynamo"

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.frontend_url = args.frontend_url.rstrip("/")
        self.worker_url = args.worker_url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {args.api_key}"}
        self.model = args.model

    def _discover_model(self) -> str:
        result = http_request(
            "GET",
            self.frontend_url + "/v1/models",
            headers=self.headers,
            timeout=self.args.request_timeout,
        )
        require_ok(result, "Dynamo model discovery")
        payload = result.body_json
        candidates = payload.get("data") if isinstance(payload, Mapping) else None
        if isinstance(candidates, list):
            for item in candidates:
                if isinstance(item, Mapping) and isinstance(item.get("id"), str):
                    return str(item["id"])
        raise BenchmarkError(f"/v1/models returned no model: {short(payload)}")

    def preflight(self) -> dict[str, Any]:
        frontend_health = http_request(
            "GET",
            self.frontend_url + "/health",
            headers=self.headers,
            timeout=min(self.args.request_timeout, 30),
        )
        require_ok(frontend_health, "Dynamo frontend health")
        worker_health = http_request(
            "GET",
            self.worker_url + "/health",
            timeout=min(self.args.request_timeout, 30),
        )
        require_ok(worker_health, "Dynamo worker health")
        if not self.model:
            self.model = self._discover_model()
        states = self.fetch_states()
        analysis = analyze_states(states, set(), set())
        return {
            "backend": self.kind,
            "frontend_url": self.frontend_url,
            "worker_url": self.worker_url,
            "model": self.model,
            "frontend_health": frontend_health.as_dict(),
            "worker_health": worker_health.as_dict(),
            "rank_count": len(states),
            "inferred_page_sizes_tokens": infer_page_sizes(states),
            "pool_cap_bytes": analysis["pool_cap_bytes"],
            "confound_warning": (
                "Dynamo mode validates the end-to-end seam, but router program "
                "tracking/admission can respond to baseline accumulation. Use "
                "direct mode for the isolated performance comparison."
            ),
        }

    def _payload(self, prompt: PromptInput) -> dict[str, Any]:
        if not self.model:
            raise BenchmarkError("Dynamo model was not discovered")
        if not isinstance(prompt, str):
            raise BenchmarkError(
                "Dynamo OpenAI mode cannot carry the direct input_ids workload; "
                "use --prompt-mode text"
            )
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": self.args.max_tokens,
            "stream": bool(self.args.stream),
        }
        if self.args.stream:
            payload["stream_options"] = {"include_usage": True}
        return payload

    def infer(
        self, *, label: str, role: str, program_id: str, prompt: PromptInput
    ) -> RequestMetric:
        headers = {**self.headers, "x-dynamo-session-id": program_id}
        payload = self._payload(prompt)
        if self.args.stream:
            return stream_json_events(
                backend=self.kind,
                label=label,
                role=role,
                program_id=program_id,
                url=self.frontend_url + "/v1/chat/completions",
                payload=payload,
                headers=headers,
                timeout=self.args.request_timeout,
            )
        result = http_request(
            "POST",
            self.frontend_url + "/v1/chat/completions",
            payload=payload,
            headers=headers,
            timeout=self.args.request_timeout,
        )
        return nonstream_metric(
            backend=self.kind,
            label=label,
            role=role,
            program_id=program_id,
            result=result,
        )

    def end_program(self, program_id: str) -> dict[str, Any]:
        headers = {
            **self.headers,
            "x-dynamo-session-id": program_id,
            "x-dynamo-session-final": "true",
        }
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": "Finalize this session without retaining KV cache.",
                }
            ],
            "temperature": 0,
            "max_tokens": 1,
            "stream": False,
        }
        result = http_request(
            "POST",
            self.frontend_url + "/v1/chat/completions",
            payload=payload,
            headers=headers,
            timeout=self.args.request_timeout,
        )
        require_ok(result, f"Dynamo final({program_id})")
        return {"http": result.as_dict(), "ack": result.body_json}

    def _call_manager(self, method: str) -> HttpResult:
        return http_request(
            "POST",
            self.worker_url + "/engine/call_tokenizer_manager",
            payload={"method": method},
            timeout=self.args.request_timeout,
        )

    def fetch_states(self) -> list[dict[str, Any]]:
        result = self._call_manager("get_aginfer_state")
        require_ok(result, "Dynamo get_aginfer_state")
        return normalize_states(result.body_json, self.kind)

    def flush(self) -> dict[str, Any]:
        result = self._call_manager("flush_cache")
        require_ok(result, "Dynamo flush_cache")
        payload = result.body_json
        if isinstance(payload, Mapping):
            if payload.get("success") is False or payload.get("status") == "error":
                raise BenchmarkError(f"Dynamo flush rejected: {short(payload)}")
        return result.as_dict()


def make_text_prompts(
    *, shared_chunks: int, tail_chunks: int, live_sessions: int, sessions: int
) -> tuple[list[PromptInput], list[PromptInput]]:
    shared = "\n".join(
        f"Shared benchmark fact {index:04d}: cedar river cobalt lantern "
        "is immutable common context for every agent in this trial."
        for index in range(shared_chunks)
    )
    suffix = "\nAnswer with exactly eight short lowercase words."
    anchor_prompts: list[PromptInput] = []
    for live_index in range(live_sessions):
        anchor_tail = "\n".join(
            f"Anchor {live_index:04d} private memory {index:04d}: "
            f"alder-osprey-silver-{live_index:04d} remains live."
            for index in range(tail_chunks)
        )
        anchor_prompts.append(shared + "\n\n" + anchor_tail + suffix)
    victim_prompts: list[PromptInput] = []
    for session_index in range(sessions):
        tail = "\n".join(
            f"Ephemeral branch {session_index:04d}/{chunk:04d}: "
            f"marker-{session_index:04d}-amber-falcon-maple-quartz is private."
            for chunk in range(tail_chunks)
        )
        victim_prompts.append(shared + "\n\n" + tail + suffix)
    return anchor_prompts, victim_prompts


def _repeat_to_length(pattern: Sequence[int], length: int) -> list[int]:
    if not pattern or length < 0:
        raise BenchmarkError("token pattern must be non-empty and length non-negative")
    return [int(pattern[index % len(pattern)]) for index in range(length)]


def make_aligned_input_ids(
    *,
    page_size_tokens: int,
    shared_pages: int,
    tail_pages: int,
    live_sessions: int,
    sessions: int,
    token_base: int,
) -> tuple[list[PromptInput], list[PromptInput]]:
    shared_len = page_size_tokens * shared_pages
    tail_len = page_size_tokens * tail_pages
    common = _repeat_to_length(
        [token_base, token_base + 1, token_base + 2, token_base + 3], shared_len
    )
    anchor_prompts: list[PromptInput] = []
    for live_index in range(live_sessions):
        marker = token_base + 100 + live_index * 2
        tail = _repeat_to_length([marker, marker + 1], tail_len)
        anchor_prompts.append(common + tail)
    victim_prompts: list[PromptInput] = []
    for session_index in range(sessions):
        marker = token_base + 1000 + session_index * 2
        tail = _repeat_to_length([marker, marker + 1], tail_len)
        victim_prompts.append(common + tail)
    return anchor_prompts, victim_prompts


def prompt_manifest(
    anchors: Sequence[PromptInput], victims: Sequence[PromptInput]
) -> dict[str, Any]:
    def one(value: PromptInput) -> dict[str, Any]:
        if isinstance(value, str):
            encoded = value.encode("utf-8")
            return {
                "kind": "text",
                "characters": len(value),
                "utf8_bytes": len(encoded),
                "sha256": hashlib.sha256(encoded).hexdigest(),
            }
        encoded = json.dumps(value, separators=(",", ":")).encode("ascii")
        return {
            "kind": "input_ids",
            "tokens": len(value),
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "first_ids": value[:8],
            "last_ids": value[-8:],
        }

    return {
        "anchors": [one(prompt) for prompt in anchors],
        "victims": [one(prompt) for prompt in victims],
        "sequence_sha256": sha256_text(
            json.dumps(
                [
                    *[one(prompt) for prompt in anchors],
                    *[one(prompt) for prompt in victims],
                ],
                sort_keys=True,
            )
        ),
    }


def percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def numeric_stats(values: Iterable[int | float | None]) -> dict[str, Any]:
    cleaned = [float(value) for value in values if value is not None]
    if not cleaned:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "stdev": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
        }
    return {
        "count": len(cleaned),
        "mean": statistics.fmean(cleaned),
        "median": statistics.median(cleaned),
        "min": min(cleaned),
        "max": max(cleaned),
        "stdev": statistics.stdev(cleaned) if len(cleaned) > 1 else 0.0,
        "p50": percentile(cleaned, 0.50),
        "p90": percentile(cleaned, 0.90),
        "p95": percentile(cleaned, 0.95),
        "p99": percentile(cleaned, 0.99),
    }


def metric_summary(metrics: Sequence[RequestMetric]) -> dict[str, Any]:
    successful = [metric for metric in metrics if metric.ok]
    prompt_tokens = [metric.prompt_tokens for metric in successful]
    cached_tokens = [metric.cached_tokens for metric in successful]
    completion_tokens = [metric.completion_tokens for metric in successful]
    sum_prompt = sum(value for value in prompt_tokens if value is not None)
    sum_cached = sum(value for value in cached_tokens if value is not None)
    sum_completion = sum(value for value in completion_tokens if value is not None)
    return {
        "count": len(metrics),
        "success_count": len(successful),
        "failure_count": len(metrics) - len(successful),
        "latency_seconds": numeric_stats(
            metric.elapsed_seconds for metric in successful
        ),
        "ttft_seconds": numeric_stats(metric.ttft_seconds for metric in successful),
        "prompt_tokens": {
            "reported_count": sum(value is not None for value in prompt_tokens),
            "sum": sum_prompt,
        },
        "cached_tokens": {
            "reported_count": sum(value is not None for value in cached_tokens),
            "sum": sum_cached,
            "per_request": numeric_stats(cached_tokens),
            "fraction_of_prompt": (sum_cached / sum_prompt if sum_prompt > 0 else None),
        },
        "completion_tokens": {
            "reported_count": sum(value is not None for value in completion_tokens),
            "sum": sum_completion,
        },
    }


def run_concurrent(
    items: Sequence[Any],
    max_workers: int,
    function: Callable[[Any], Any],
) -> tuple[list[Any], float]:
    if not items:
        return [], 0.0
    start_event = threading.Event()

    def wrapped(item: Any) -> Any:
        start_event.wait()
        return function(item)

    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(max_workers, len(items)), thread_name_prefix="deadkv-ab"
    ) as executor:
        futures = [executor.submit(wrapped, item) for item in items]
        start_event.set()
        results = [future.result() for future in futures]
    return results, time.perf_counter() - started


def capacity_hash(analysis: Mapping[str, Any]) -> str:
    return sha256_text(
        json.dumps(
            analysis.get("capacity_signature"), sort_keys=True, separators=(",", ":")
        )
    )


def total_bytes(by_tier: Mapping[str, Any]) -> int:
    return sum(int(by_tier.get(tier, 0) or 0) for tier in TIERS)


def poll_analysis(
    backend: Backend,
    *,
    tracked_programs: set[str],
    ended_programs: set[str],
    timeout: float,
    interval: float,
    predicate: Callable[[Mapping[str, Any]], bool],
    description: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    deadline = time.perf_counter() + timeout
    observations: list[dict[str, Any]] = []
    last_states: list[dict[str, Any]] = []
    last_analysis: dict[str, Any] = {}
    while True:
        last_states = backend.fetch_states()
        last_analysis = analyze_states(last_states, tracked_programs, ended_programs)
        matched = bool(predicate(last_analysis))
        observations.append(
            {
                "at": utc_now(),
                "matched": matched,
                "programs_present": last_analysis["programs_present"],
                "ended_programs_present": last_analysis["ended_programs_present"],
                "dead_physical_bytes": last_analysis["dead_physical_bytes"],
                "pool_used_bytes": last_analysis["pool_used_bytes"],
            }
        )
        if matched:
            return last_states, last_analysis, observations
        if time.perf_counter() >= deadline:
            raise BenchmarkError(
                f"timed out after {timeout:.1f}s waiting for {description}; "
                f"last={short(last_analysis)}"
            )
        time.sleep(interval)


def flush_and_wait(
    backend: Backend,
    *,
    timeout: float,
    interval: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    flush_result = backend.flush()
    states, analysis, _ = poll_analysis(
        backend,
        tracked_programs=set(),
        ended_programs=set(),
        timeout=timeout,
        interval=interval,
        predicate=lambda observed: observed["unit_count"] == 0,
        description="flush_cache to remove all radix units",
    )
    return flush_result, states, analysis


def make_backend(args: argparse.Namespace) -> Backend:
    if args.backend == "direct":
        return DirectBackend(args)
    return DynamoBackend(args)


def _check_request_metrics(metrics: Sequence[RequestMetric], context: str) -> None:
    failed = [metric for metric in metrics if not metric.ok]
    if failed:
        raise BenchmarkError(
            f"{context}: {len(failed)} request(s) failed: "
            + "; ".join(
                f"{metric.label}: status={metric.status}, error={metric.error}"
                for metric in failed[:8]
            )
        )


def _program_ids(
    run_id: str,
    pair: int,
    condition: str,
    live_sessions: int,
    sessions: int,
) -> tuple[list[str], list[str]]:
    prefix = f"deadkv-ab-{run_id}-p{pair:02d}-{condition}"
    anchors = [prefix + f"-live{index:02d}" for index in range(live_sessions)]
    victims = [prefix + f"-dead{index:04d}" for index in range(sessions)]
    return anchors, victims


def _write_state_if_requested(
    artifacts: Artifacts,
    filename: str,
    states: Sequence[Mapping[str, Any]],
    enabled: bool,
) -> str | None:
    if not enabled:
        return None
    artifacts.write(filename, list(states))
    return filename


def tier_auc(
    checkpoints: Sequence[Mapping[str, Any]], field: str
) -> tuple[dict[str, float], float]:
    """Piecewise-constant byte-seconds over timestamped state checkpoints.

    State is treated as holding its last observed value until the next sample.
    This avoids inventing a gradual ramp before a discrete SESSION_END event.
    """
    ordered = sorted(
        (
            (float(point["elapsed_seconds"]), point["analysis"])
            for point in checkpoints
            if isinstance(point.get("elapsed_seconds"), (int, float))
            and isinstance(point.get("analysis"), Mapping)
        ),
        key=lambda pair: pair[0],
    )
    auc = {tier: 0.0 for tier in TIERS}
    if len(ordered) < 2:
        return auc, 0.0
    for (left_t, left), (right_t, right) in zip(ordered, ordered[1:]):
        width = max(0.0, right_t - left_t)
        left_values = left.get(field, {})
        for tier in TIERS:
            left_y = (
                float(left_values.get(tier, 0))
                if isinstance(left_values, Mapping)
                else 0.0
            )
            auc[tier] += width * left_y
    return auc, ordered[-1][0] - ordered[0][0]


def run_trial(
    *,
    backend: Backend,
    args: argparse.Namespace,
    artifacts: Artifacts,
    run_id: str,
    pair_index: int,
    condition: str,
    trial_index: int,
    victim_order: Sequence[int],
    anchor_prompts: Sequence[PromptInput],
    victim_prompts: Sequence[PromptInput],
) -> dict[str, Any]:
    if condition not in CONDITIONS:
        raise BenchmarkError(f"unknown condition {condition!r}")
    trial_clock_started = time.perf_counter()
    anchor_ids, victim_ids = _program_ids(
        run_id,
        pair_index,
        condition,
        len(anchor_prompts),
        len(victim_prompts),
    )
    tracked = {*anchor_ids, *victim_ids}
    ended: set[str] = set()
    prefix = f"trial_{trial_index:03d}_p{pair_index:02d}_{condition}"
    trial: dict[str, Any] = {
        "trial_index": trial_index,
        "pair_index": pair_index,
        "condition": condition,
        "started_at": utc_now(),
        "programs": {"live": anchor_ids, "terminal": victim_ids},
        "victim_order": list(victim_order),
        "requests": [],
        "final_calls": [],
        "reclaim_events": [],
        "waves": [],
        "checkpoints": [],
        "status": "running",
    }
    artifacts.write(prefix + ".json", trial)

    def save_trial() -> None:
        artifacts.write(prefix + ".json", trial)

    def checkpoint(label: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        states = backend.fetch_states()
        analysis = analyze_states(states, tracked, ended)
        if analysis["external_holders"]:
            raise BenchmarkError(
                f"{prefix}/{label}: tracked units have external holders; dedicated "
                f"cache isolation was lost: {analysis['external_holders']}"
            )
        raw_name = _write_state_if_requested(
            artifacts,
            f"{prefix}_{len(trial['checkpoints']):03d}_{label}_state.json",
            states,
            args.save_raw_states,
        )
        trial["checkpoints"].append(
            {
                "label": label,
                "at": utc_now(),
                "elapsed_seconds": time.perf_counter() - trial_clock_started,
                "analysis": analysis,
                "raw_state_file": raw_name,
            }
        )
        save_trial()
        return states, analysis

    flush_result, initial_states, initial = flush_and_wait(
        backend, timeout=args.poll_timeout, interval=args.poll_interval
    )
    trial["flush"] = flush_result
    initial_raw = _write_state_if_requested(
        artifacts,
        prefix + "_000_after_flush_state.json",
        initial_states,
        args.save_raw_states,
    )
    trial["checkpoints"].append(
        {
            "label": "after_flush",
            "at": utc_now(),
            "elapsed_seconds": time.perf_counter() - trial_clock_started,
            "analysis": initial,
            "raw_state_file": initial_raw,
        }
    )
    trial["capacity_hash"] = capacity_hash(initial)
    save_trial()

    inference_active_seconds = 0.0
    control_active_seconds = 0.0

    def infer_one(index: int, repetition: int) -> RequestMetric:
        role = "victim_first" if repetition == 0 else "victim_repeat"
        return backend.infer(
            label=f"victim_{index:04d}_request_{repetition:02d}",
            role=role,
            program_id=victim_ids[index],
            prompt=victim_prompts[index],
        )

    # Establish one live entry first, then add the remaining live working set.
    # This keeps cold-start ordering identical in every arm.
    anchor_started = time.perf_counter()
    first_anchor = backend.infer(
        label="live_000_seed",
        role="live_seed",
        program_id=anchor_ids[0],
        prompt=anchor_prompts[0],
    )
    inference_active_seconds += time.perf_counter() - anchor_started
    trial["requests"].append(first_anchor.as_dict())
    _check_request_metrics([first_anchor], f"{prefix} first live seed")
    if len(anchor_ids) > 1:
        other_live = list(range(1, len(anchor_ids)))
        live_metrics, live_elapsed = run_concurrent(
            other_live,
            args.concurrency,
            lambda index: backend.infer(
                label=f"live_{index:03d}_seed",
                role="live_seed",
                program_id=anchor_ids[index],
                prompt=anchor_prompts[index],
            ),
        )
        _check_request_metrics(live_metrics, f"{prefix} remaining live seeds")
        inference_active_seconds += live_elapsed
        trial["requests"].extend(metric.as_dict() for metric in live_metrics)
    _, after_anchor, anchor_poll = poll_analysis(
        backend,
        tracked_programs=tracked,
        ended_programs=ended,
        timeout=args.poll_timeout,
        interval=args.poll_interval,
        predicate=lambda observed: set(anchor_ids).issubset(
            set(observed["programs_present"])
        ),
        description="all live programs to appear in state",
    )
    trial["checkpoints"].append(
        {
            "label": "after_live_seeds",
            "at": utc_now(),
            "elapsed_seconds": time.perf_counter() - trial_clock_started,
            "analysis": after_anchor,
            "poll": anchor_poll,
            "raw_state_file": None,
        }
    )
    save_trial()

    ordered = list(victim_order)
    waves = [
        ordered[offset : offset + args.dead_per_epoch]
        for offset in range(0, len(ordered), args.dead_per_epoch)
    ]
    for wave_index, wave_indices in enumerate(waves):
        wave: dict[str, Any] = {
            "wave_index": wave_index,
            "victim_indices": list(wave_indices),
            "batches": [],
            "inference_batch_seconds": [],
            "final_batch_seconds": 0.0,
        }
        for offset in range(0, len(wave_indices), args.concurrency):
            batch_indices = wave_indices[offset : offset + args.concurrency]
            batch_number = offset // args.concurrency
            batch: dict[str, Any] = {
                "batch_index": batch_number,
                "victim_indices": list(batch_indices),
                "inference_batch_seconds": [],
                "logical_end_at": None,
                "logical_end_to_signal_seconds": None,
                "final_batch_seconds": 0.0,
                "reclaim_observed_seconds": None,
            }
            for repetition in range(args.requests_per_session):
                metrics, batch_elapsed = run_concurrent(
                    batch_indices,
                    args.concurrency,
                    lambda index, repetition=repetition: infer_one(index, repetition),
                )
                _check_request_metrics(
                    metrics, f"{prefix} wave {wave_index} repetition {repetition}"
                )
                trial["requests"].extend(metric.as_dict() for metric in metrics)
                wave["inference_batch_seconds"].append(batch_elapsed)
                batch["inference_batch_seconds"].append(batch_elapsed)
                inference_active_seconds += batch_elapsed

            batch_programs = [victim_ids[index] for index in batch_indices]
            ended.update(batch_programs)
            logical_end_started = time.perf_counter()
            batch["logical_end_at"] = utc_now()
            before_states, before_end, before_poll = poll_analysis(
                backend,
                tracked_programs=tracked,
                ended_programs=ended,
                timeout=args.poll_timeout,
                interval=args.poll_interval,
                predicate=lambda observed, batch_set=set(
                    batch_programs
                ): batch_set.issubset(set(observed["programs_present"])),
                description=(
                    f"epoch {wave_index} batch {batch_number} completed programs "
                    "to appear in state"
                ),
            )
            before_raw = _write_state_if_requested(
                artifacts,
                f"{prefix}_{len(trial['checkpoints']):03d}_epoch_{wave_index:03d}_batch_{batch_number:03d}_before_end_state.json",
                before_states,
                args.save_raw_states,
            )
            trial["checkpoints"].append(
                {
                    "label": (
                        f"epoch_{wave_index:03d}_batch_{batch_number:03d}_before_end_signal"
                    ),
                    "at": utc_now(),
                    "elapsed_seconds": time.perf_counter() - trial_clock_started,
                    "analysis": before_end,
                    "poll": before_poll,
                    "raw_state_file": before_raw,
                }
            )
            batch["before_end_signal"] = {
                "dead_physical_bytes": before_end["dead_physical_bytes"],
                "pool_used_bytes": before_end["pool_used_bytes"],
                "ended_programs_present": before_end["ended_programs_present"],
            }
            save_trial()

            if condition == "ours":

                def finish_program(program_id: str) -> dict[str, Any]:
                    started_at = utc_now()
                    started = time.perf_counter()
                    response = backend.end_program(program_id)
                    return {
                        "program_id": program_id,
                        "started_at": started_at,
                        "elapsed_seconds": time.perf_counter() - started,
                        "response": response,
                    }

                end_signal_started = time.perf_counter()
                batch["logical_end_to_signal_seconds"] = (
                    end_signal_started - logical_end_started
                )
                finals, final_batch_elapsed = run_concurrent(
                    batch_programs, args.final_concurrency, finish_program
                )
                control_active_seconds += final_batch_elapsed
                batch["final_batch_seconds"] = final_batch_elapsed
                wave["final_batch_seconds"] += final_batch_elapsed
                trial["final_calls"].extend(finals)
                states_after, after_end, reclaim_poll = poll_analysis(
                    backend,
                    tracked_programs=tracked,
                    ended_programs=ended,
                    timeout=args.poll_timeout,
                    interval=args.poll_interval,
                    predicate=lambda observed, batch_set=set(batch_programs): not (
                        set(observed["programs_present"]) & batch_set
                    ),
                    description=(
                        f"epoch {wave_index} batch {batch_number} ended programs "
                        "to disappear"
                    ),
                )
                batch["reclaim_observed_seconds"] = (
                    time.perf_counter() - end_signal_started
                )
                trial["reclaim_events"].append(
                    {
                        "epoch": wave_index,
                        "batch": batch_number,
                        "programs": batch_programs,
                        "elapsed_seconds": batch["reclaim_observed_seconds"],
                    }
                )
                raw_name = _write_state_if_requested(
                    artifacts,
                    f"{prefix}_{len(trial['checkpoints']):03d}_epoch_{wave_index:03d}_batch_{batch_number:03d}_after_end_state.json",
                    states_after,
                    args.save_raw_states,
                )
                trial["checkpoints"].append(
                    {
                        "label": (
                            f"epoch_{wave_index:03d}_batch_{batch_number:03d}_after_end_signal"
                        ),
                        "at": utc_now(),
                        "elapsed_seconds": time.perf_counter() - trial_clock_started,
                        "analysis": after_end,
                        "poll": reclaim_poll,
                        "raw_state_file": raw_name,
                    }
                )
                batch["after_end_signal"] = {
                    "dead_physical_bytes": after_end["dead_physical_bytes"],
                    "pool_used_bytes": after_end["pool_used_bytes"],
                    "ended_programs_present": after_end["ended_programs_present"],
                }
            else:
                _, after_end = checkpoint(
                    f"epoch_{wave_index:03d}_batch_{batch_number:03d}_no_end_signal"
                )
                batch["after_end_signal"] = {
                    "dead_physical_bytes": after_end["dead_physical_bytes"],
                    "pool_used_bytes": after_end["pool_used_bytes"],
                    "ended_programs_present": after_end["ended_programs_present"],
                }
            wave["batches"].append(batch)
            save_trial()
        trial["waves"].append(wave)
        save_trial()

    # The live set is deliberately probed only after all terminal-session
    # churn. A cache miss is a valid baseline outcome and is measured rather
    # than treated as a harness failure.
    anchor_probes, anchor_probe_elapsed = run_concurrent(
        list(range(len(anchor_ids))),
        args.concurrency,
        lambda index: backend.infer(
            label=f"live_{index:03d}_probe_after_churn",
            role="live_probe",
            program_id=anchor_ids[index],
            prompt=anchor_prompts[index],
        ),
    )
    inference_active_seconds += anchor_probe_elapsed
    trial["requests"].extend(metric.as_dict() for metric in anchor_probes)
    _check_request_metrics(anchor_probes, f"{prefix} live probes")
    _, after_anchor_probe = checkpoint("after_anchor_probe")
    ended.update(anchor_ids)
    logical_end_all_started = time.perf_counter()
    trial["all_programs_logically_ended_at"] = utc_now()

    if condition == "ours":

        def finish_live(program_id: str) -> dict[str, Any]:
            started_at = utc_now()
            started = time.perf_counter()
            response = backend.end_program(program_id)
            return {
                "program_id": program_id,
                "started_at": started_at,
                "elapsed_seconds": time.perf_counter() - started,
                "response": response,
            }

        live_finals, live_final_elapsed = run_concurrent(
            anchor_ids, args.final_concurrency, finish_live
        )
        control_active_seconds += live_final_elapsed
        trial["final_calls"].extend(live_finals)
        end_states, end_analysis, end_poll = poll_analysis(
            backend,
            tracked_programs=tracked,
            ended_programs=ended,
            timeout=args.poll_timeout,
            interval=args.poll_interval,
            predicate=lambda observed: not observed["programs_present"],
            description="all ended programs to disappear",
        )
        trial["all_reclaim_observed_seconds"] = (
            time.perf_counter() - logical_end_all_started
        )
    else:
        end_states = backend.fetch_states()
        end_analysis = analyze_states(end_states, tracked, ended)
        end_poll = []
        trial["all_reclaim_observed_seconds"] = None
    end_raw = _write_state_if_requested(
        artifacts, prefix + "_end_state.json", end_states, args.save_raw_states
    )
    trial["checkpoints"].append(
        {
            "label": "all_programs_logically_ended",
            "at": utc_now(),
            "elapsed_seconds": time.perf_counter() - trial_clock_started,
            "analysis": end_analysis,
            "poll": end_poll,
            "raw_state_file": end_raw,
        }
    )

    # Observe both conditions for the same duration.  Baseline latency is
    # right-censored when its dead units are still resident at the deadline.
    retention_samples: list[dict[str, Any]] = []
    retention_started = time.perf_counter()
    retention_deadline = retention_started + args.retention_seconds
    while True:
        states = backend.fetch_states()
        observed = analyze_states(states, tracked, ended)
        retention_samples.append(
            {
                "elapsed_seconds": time.perf_counter() - retention_started,
                "trial_elapsed_seconds": time.perf_counter() - trial_clock_started,
                "analysis": observed,
            }
        )
        if time.perf_counter() >= retention_deadline:
            break
        time.sleep(min(args.poll_interval, retention_deadline - time.perf_counter()))
    final_observation = retention_samples[-1]["analysis"]
    final_raw = _write_state_if_requested(
        artifacts,
        prefix + "_final_observation_state.json",
        states,
        args.save_raw_states,
    )
    trial["checkpoints"].append(
        {
            "label": "retention_window_end",
            "at": utc_now(),
            "elapsed_seconds": time.perf_counter() - trial_clock_started,
            "analysis": final_observation,
            "raw_state_file": final_raw,
        }
    )
    trial["retention_observation"] = {
        "requested_seconds": args.retention_seconds,
        "actual_seconds": time.perf_counter() - retention_started,
        "sample_count": len(retention_samples),
        "samples": retention_samples,
        "reclaim_latency_censored": bool(final_observation["programs_present"]),
        "censored_lower_bound_seconds": (
            time.perf_counter() - logical_end_all_started
            if final_observation["programs_present"]
            else None
        ),
    }

    requests = [RequestMetric(**metric) for metric in trial["requests"]]
    by_role = {
        role: metric_summary([metric for metric in requests if metric.role == role])
        for role in sorted({metric.role for metric in requests})
    }
    request_summary = metric_summary(requests)
    completion_token_sum = request_summary["completion_tokens"]["sum"]
    pipeline_active_seconds = inference_active_seconds + control_active_seconds
    checkpoint_analyses = [
        checkpoint_entry["analysis"] for checkpoint_entry in trial["checkpoints"]
    ]
    dead_auc, auc_duration = tier_auc(trial["checkpoints"], "dead_physical_bytes")
    pool_auc, _ = tier_auc(trial["checkpoints"], "pool_used_bytes")
    peak_pool = {
        tier: max(
            int(analysis["pool_used_bytes"].get(tier, 0))
            for analysis in checkpoint_analyses
        )
        for tier in TIERS
    }
    peak_dead = {
        tier: max(
            int(analysis["dead_physical_bytes"].get(tier, 0))
            for analysis in checkpoint_analyses
        )
        for tier in TIERS
    }
    trial["metrics"] = {
        "requests": request_summary,
        "requests_by_role": by_role,
        "inference_active_wall_seconds": inference_active_seconds,
        "control_active_wall_seconds": control_active_seconds,
        "pipeline_active_wall_seconds": pipeline_active_seconds,
        "inference_request_throughput_rps": (
            len(requests) / inference_active_seconds
            if inference_active_seconds > 0
            else None
        ),
        "pipeline_request_throughput_rps": (
            len(requests) / pipeline_active_seconds
            if pipeline_active_seconds > 0
            else None
        ),
        "inference_output_tokens_per_second": (
            completion_token_sum / inference_active_seconds
            if inference_active_seconds > 0 and completion_token_sum > 0
            else None
        ),
        "pipeline_output_tokens_per_second": (
            completion_token_sum / pipeline_active_seconds
            if pipeline_active_seconds > 0 and completion_token_sum > 0
            else None
        ),
        "final_http_latency_seconds": numeric_stats(
            final_call.get("elapsed_seconds") for final_call in trial["final_calls"]
        ),
        "wave_reclaim_latency_seconds": numeric_stats(
            event.get("elapsed_seconds") for event in trial["reclaim_events"]
        ),
        "live_probes": metric_summary(anchor_probes),
        "live_probe_requests": [metric.as_dict() for metric in anchor_probes],
        "anchor_hit_count_after_probe": after_anchor_probe["tracked_hit_count"],
        "peak_pool_used_bytes": peak_pool,
        "peak_dead_physical_bytes": peak_dead,
        "dead_byte_seconds_auc": dead_auc,
        "pool_used_byte_seconds_auc": pool_auc,
        "auc_duration_seconds": auc_duration,
        "mean_dead_bytes_over_observation": {
            tier: dead_auc[tier] / auc_duration if auc_duration > 0 else None
            for tier in TIERS
        },
        "mean_pool_used_bytes_over_observation": {
            tier: pool_auc[tier] / auc_duration if auc_duration > 0 else None
            for tier in TIERS
        },
        "end_pool_used_bytes": final_observation["pool_used_bytes"],
        "end_pool_delta_from_flush_bytes": {
            tier: int(final_observation["pool_used_bytes"].get(tier, 0))
            - int(initial["pool_used_bytes"].get(tier, 0))
            for tier in TIERS
        },
        "end_dead_physical_bytes": final_observation["dead_physical_bytes"],
        "end_dead_physical_bytes_total": final_observation["dead_physical_bytes_total"],
        "end_ended_programs_present": final_observation["ended_programs_present"],
        "capacity_hash": trial["capacity_hash"],
        "pool_cap_bytes": initial["pool_cap_bytes"],
    }
    if condition == "ours" and final_observation["programs_present"]:
        raise BenchmarkError(
            f"{prefix}: ours retained ended programs after explicit final: "
            f"{final_observation['programs_present']}"
        )

    # Cleanup happens strictly after the retained-state observation and is not
    # included in any performance metric.  It is essential in Dynamo mode:
    # flush_cache clears SGLang KV but cannot remove the router's program map.
    if condition == "baseline":
        cleanup_started = time.perf_counter()
        cleanup_calls: list[dict[str, Any]] = []
        for program_id in [*victim_ids, *anchor_ids]:
            started = time.perf_counter()
            response = backend.end_program(program_id)
            cleanup_calls.append(
                {
                    "program_id": program_id,
                    "elapsed_seconds": time.perf_counter() - started,
                    "response": response,
                }
            )
        _, cleanup_analysis, cleanup_poll = poll_analysis(
            backend,
            tracked_programs=tracked,
            ended_programs=ended,
            timeout=args.poll_timeout,
            interval=args.poll_interval,
            predicate=lambda observed: not observed["programs_present"],
            description="post-measurement baseline cleanup",
        )
        trial["post_measurement_cleanup"] = {
            "elapsed_seconds": time.perf_counter() - cleanup_started,
            "calls": cleanup_calls,
            "poll": cleanup_poll,
            "final_analysis": cleanup_analysis,
            "excluded_from_metrics": True,
        }
    trial["status"] = "passed"
    trial["finished_at"] = utc_now()
    save_trial()
    return trial


def trial_projection(trial: Mapping[str, Any]) -> dict[str, Any]:
    metrics = trial["metrics"]
    requests = metrics["requests"]
    live = metrics["live_probes"]
    return {
        "condition": trial["condition"],
        "pair_index": trial["pair_index"],
        "end_dead_bytes_total": metrics["end_dead_physical_bytes_total"],
        "end_dead_bytes": metrics["end_dead_physical_bytes"],
        "end_pool_used_bytes": metrics["end_pool_used_bytes"],
        "end_pool_delta_bytes": metrics["end_pool_delta_from_flush_bytes"],
        "peak_pool_used_bytes": metrics["peak_pool_used_bytes"],
        "peak_dead_bytes": metrics["peak_dead_physical_bytes"],
        "dead_byte_seconds_auc": metrics["dead_byte_seconds_auc"],
        "mean_dead_bytes": metrics["mean_dead_bytes_over_observation"],
        "ttft_p50_seconds": requests["ttft_seconds"]["p50"],
        "ttft_p95_seconds": requests["ttft_seconds"]["p95"],
        "request_latency_p50_seconds": requests["latency_seconds"]["p50"],
        "cached_tokens_sum": requests["cached_tokens"]["sum"],
        "cached_fraction": requests["cached_tokens"]["fraction_of_prompt"],
        "anchor_cached_tokens": live["cached_tokens"]["per_request"]["median"],
        "anchor_ttft_seconds": live["ttft_seconds"]["median"],
        "inference_request_throughput_rps": metrics["inference_request_throughput_rps"],
        "pipeline_request_throughput_rps": metrics["pipeline_request_throughput_rps"],
        "inference_output_tokens_per_second": metrics[
            "inference_output_tokens_per_second"
        ],
        "pipeline_output_tokens_per_second": metrics[
            "pipeline_output_tokens_per_second"
        ],
        "final_http_latency_p50_seconds": metrics["final_http_latency_seconds"]["p50"],
        "reclaim_latency_p50_seconds": metrics["wave_reclaim_latency_seconds"]["p50"],
        "reclaim_latency_censored": trial["retention_observation"][
            "reclaim_latency_censored"
        ],
        "reclaim_censored_lower_bound_seconds": trial["retention_observation"][
            "censored_lower_bound_seconds"
        ],
        "capacity_hash": trial["capacity_hash"],
        "pool_cap_bytes": metrics["pool_cap_bytes"],
    }


def _nested_numbers(
    rows: Sequence[Mapping[str, Any]], field: str, tier: str | None = None
) -> list[float | None]:
    values: list[float | None] = []
    for row in rows:
        value: Any = row.get(field)
        if tier is not None:
            value = value.get(tier) if isinstance(value, Mapping) else None
        values.append(float(value) if isinstance(value, (int, float)) else None)
    return values


def summarize_condition(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    scalar_fields = (
        "end_dead_bytes_total",
        "ttft_p50_seconds",
        "ttft_p95_seconds",
        "request_latency_p50_seconds",
        "cached_tokens_sum",
        "cached_fraction",
        "anchor_cached_tokens",
        "anchor_ttft_seconds",
        "inference_request_throughput_rps",
        "pipeline_request_throughput_rps",
        "inference_output_tokens_per_second",
        "pipeline_output_tokens_per_second",
        "final_http_latency_p50_seconds",
        "reclaim_latency_p50_seconds",
        "reclaim_censored_lower_bound_seconds",
    )
    summary = {
        field: numeric_stats(_nested_numbers(rows, field)) for field in scalar_fields
    }
    for field in (
        "end_dead_bytes",
        "end_pool_used_bytes",
        "end_pool_delta_bytes",
        "peak_pool_used_bytes",
        "peak_dead_bytes",
        "dead_byte_seconds_auc",
        "mean_dead_bytes",
        "pool_cap_bytes",
    ):
        summary[field] = {
            tier: numeric_stats(_nested_numbers(rows, field, tier)) for tier in TIERS
        }
    summary["trial_count"] = len(rows)
    summary["reclaim_censored_count"] = sum(
        bool(row.get("reclaim_latency_censored")) for row in rows
    )
    return summary


def relative_change(ours: Any, baseline: Any, *, lower_is_better: bool) -> float | None:
    if not isinstance(ours, (int, float)) or not isinstance(baseline, (int, float)):
        return None
    if baseline == 0:
        return None
    if lower_is_better:
        return (baseline - ours) / baseline
    return (ours - baseline) / baseline


def paired_summary(trials: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    projections = [trial_projection(trial) for trial in trials]
    by_condition = {
        condition: [row for row in projections if row["condition"] == condition]
        for condition in CONDITIONS
    }
    pairs: list[dict[str, Any]] = []
    pair_indices = sorted({int(row["pair_index"]) for row in projections})
    for pair_index in pair_indices:
        rows = {
            row["condition"]: row
            for row in projections
            if int(row["pair_index"]) == pair_index
        }
        if set(rows) != set(CONDITIONS):
            continue
        baseline = rows["baseline"]
        ours = rows["ours"]
        pairs.append(
            {
                "pair_index": pair_index,
                "baseline": baseline,
                "ours": ours,
                "dead_kv_bytes_reclaimed_advantage": baseline["end_dead_bytes_total"]
                - ours["end_dead_bytes_total"],
                "dead_kv_reduction_fraction": relative_change(
                    ours["end_dead_bytes_total"],
                    baseline["end_dead_bytes_total"],
                    lower_is_better=True,
                ),
                "end_hbm_pool_reduction_bytes": baseline["end_pool_used_bytes"]["HBM"]
                - ours["end_pool_used_bytes"]["HBM"],
                "end_dram_pool_reduction_bytes": baseline["end_pool_used_bytes"]["DRAM"]
                - ours["end_pool_used_bytes"]["DRAM"],
                "hbm_dead_byte_seconds_reduction": baseline["dead_byte_seconds_auc"][
                    "HBM"
                ]
                - ours["dead_byte_seconds_auc"]["HBM"],
                "dram_dead_byte_seconds_reduction": baseline["dead_byte_seconds_auc"][
                    "DRAM"
                ]
                - ours["dead_byte_seconds_auc"]["DRAM"],
                "anchor_cached_tokens_delta": (
                    ours["anchor_cached_tokens"] - baseline["anchor_cached_tokens"]
                    if isinstance(ours["anchor_cached_tokens"], (int, float))
                    and isinstance(baseline["anchor_cached_tokens"], (int, float))
                    else None
                ),
                "anchor_ttft_improvement_fraction": relative_change(
                    ours["anchor_ttft_seconds"],
                    baseline["anchor_ttft_seconds"],
                    lower_is_better=True,
                ),
                "all_request_ttft_p50_improvement_fraction": relative_change(
                    ours["ttft_p50_seconds"],
                    baseline["ttft_p50_seconds"],
                    lower_is_better=True,
                ),
                "inference_throughput_change_fraction": relative_change(
                    ours["inference_request_throughput_rps"],
                    baseline["inference_request_throughput_rps"],
                    lower_is_better=False,
                ),
                "pipeline_throughput_change_fraction": relative_change(
                    ours["pipeline_request_throughput_rps"],
                    baseline["pipeline_request_throughput_rps"],
                    lower_is_better=False,
                ),
            }
        )
    paired_fields = (
        "dead_kv_bytes_reclaimed_advantage",
        "dead_kv_reduction_fraction",
        "end_hbm_pool_reduction_bytes",
        "end_dram_pool_reduction_bytes",
        "hbm_dead_byte_seconds_reduction",
        "dram_dead_byte_seconds_reduction",
        "anchor_cached_tokens_delta",
        "anchor_ttft_improvement_fraction",
        "all_request_ttft_p50_improvement_fraction",
        "inference_throughput_change_fraction",
        "pipeline_throughput_change_fraction",
    )
    return {
        "conditions": {
            condition: summarize_condition(rows)
            for condition, rows in by_condition.items()
        },
        "pairs": pairs,
        "paired_aggregate": {
            field: numeric_stats(
                pair.get(field) if isinstance(pair.get(field), (int, float)) else None
                for pair in pairs
            )
            for field in paired_fields
        },
    }


def _fmt_number(value: Any, digits: int = 4) -> str:
    if not isinstance(value, (int, float)):
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return f"{value:,}"


def _fmt_bytes(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "n/a"
    amount = float(value)
    for suffix in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(amount) < 1024 or suffix == "TiB":
            return f"{amount:.2f} {suffix}"
        amount /= 1024
    return f"{amount:.2f} TiB"


def render_report(summary: Mapping[str, Any]) -> str:
    aggregate = summary.get("aggregate", {})
    conditions = (
        aggregate.get("conditions", {}) if isinstance(aggregate, Mapping) else {}
    )
    baseline = conditions.get("baseline", {}) if isinstance(conditions, Mapping) else {}
    ours = conditions.get("ours", {}) if isinstance(conditions, Mapping) else {}

    def median(
        condition: Mapping[str, Any], field: str, tier: str | None = None
    ) -> Any:
        value: Any = condition.get(field)
        if tier is not None:
            value = value.get(tier) if isinstance(value, Mapping) else None
        return value.get("median") if isinstance(value, Mapping) else None

    lines = [
        "# Dead-KV paired A/B result",
        "",
        f"- Status: `{summary.get('status')}`",
        f"- Backend: `{summary.get('configuration', {}).get('backend')}`",
        f"- Run ID: `{summary.get('run_id')}`",
        f"- Paired repetitions: `{summary.get('configuration', {}).get('repeats')}`",
        "",
        "The baseline is the same server/build with no terminal lifecycle signal. "
        "Ours sends `SESSION_END`; all other workload inputs are paired.",
        "",
        "| Median metric | Baseline: no END | Ours: SESSION_END |",
        "|---|---:|---:|",
        f"| Dead KV at observation end | {_fmt_bytes(median(baseline, 'end_dead_bytes_total'))} | {_fmt_bytes(median(ours, 'end_dead_bytes_total'))} |",
        f"| HBM pool used at end | {_fmt_bytes(median(baseline, 'end_pool_used_bytes', 'HBM'))} | {_fmt_bytes(median(ours, 'end_pool_used_bytes', 'HBM'))} |",
        f"| DRAM pool used at end | {_fmt_bytes(median(baseline, 'end_pool_used_bytes', 'DRAM'))} | {_fmt_bytes(median(ours, 'end_pool_used_bytes', 'DRAM'))} |",
        f"| Peak HBM pool used | {_fmt_bytes(median(baseline, 'peak_pool_used_bytes', 'HBM'))} | {_fmt_bytes(median(ours, 'peak_pool_used_bytes', 'HBM'))} |",
        f"| HBM dead-byte AUC (byte·s) | {_fmt_number(median(baseline, 'dead_byte_seconds_auc', 'HBM'), 1)} | {_fmt_number(median(ours, 'dead_byte_seconds_auc', 'HBM'), 1)} |",
        f"| DRAM dead-byte AUC (byte·s) | {_fmt_number(median(baseline, 'dead_byte_seconds_auc', 'DRAM'), 1)} | {_fmt_number(median(ours, 'dead_byte_seconds_auc', 'DRAM'), 1)} |",
        f"| Live-probe cached tokens | {_fmt_number(median(baseline, 'anchor_cached_tokens'))} | {_fmt_number(median(ours, 'anchor_cached_tokens'))} |",
        f"| Live-probe TTFT (s) | {_fmt_number(median(baseline, 'anchor_ttft_seconds'))} | {_fmt_number(median(ours, 'anchor_ttft_seconds'))} |",
        f"| All-request TTFT p50 (s) | {_fmt_number(median(baseline, 'ttft_p50_seconds'))} | {_fmt_number(median(ours, 'ttft_p50_seconds'))} |",
        f"| Inference throughput (req/s) | {_fmt_number(median(baseline, 'inference_request_throughput_rps'))} | {_fmt_number(median(ours, 'inference_request_throughput_rps'))} |",
        f"| Pipeline throughput incl. END (req/s) | {_fmt_number(median(baseline, 'pipeline_request_throughput_rps'))} | {_fmt_number(median(ours, 'pipeline_request_throughput_rps'))} |",
        "",
        "## Interpretation guardrails",
        "",
        "- Direct mode is the isolated experiment. Dynamo mode is useful only as an end-to-end cross-check because router state can react to accumulated baseline programs.",
        "- The default `shared_pages=0` gives every live/terminal program a unique page-aligned prefix. Set it above zero only for a separate shared-holder safety experiment.",
        "- Baseline reclaim latency is right-censored when dead KV remains at the end of the configured observation window.",
        "- A memory-retention win does not imply a statistically significant TTFT or throughput win. Use the paired JSON values and multiple repetitions.",
        "- Capacity fingerprints must match across every trial; otherwise the run fails.",
        "",
    ]
    warnings = summary.get("warnings")
    if isinstance(warnings, list) and warnings:
        lines.extend(["## Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)
        lines.append("")
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=("direct", "dynamo"),
        default="direct",
        help="direct is the isolated A/B; dynamo is an end-to-end cross-check",
    )
    parser.add_argument(
        "--server-url",
        default="http://127.0.0.1:30001",
        help="direct SGLang URL",
    )
    parser.add_argument("--frontend-url", default="http://127.0.0.1:8000")
    parser.add_argument("--worker-url", default="http://127.0.0.1:8081")
    parser.add_argument(
        "--model",
        default=os.environ.get("DYNAMO_MODEL"),
        help="Dynamo model id; defaults to first /v1/models entry",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("DYNAMO_API_KEY", "dummy"),
        help="Dynamo bearer token; never written to artifacts",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=2,
        help="number of paired comparisons; default 2 produces four ABBA arms",
    )
    parser.add_argument(
        "--order-mode",
        choices=("abba", "random"),
        default="abba",
        help="ABBA alternates pair order; random independently shuffles each pair",
    )
    parser.add_argument("--live-sessions", type=int, default=4)
    parser.add_argument("--dead-per-epoch", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--final-concurrency",
        type=int,
        default=1,
        help="terminal control concurrency; 1 is safest for TP collectives",
    )
    parser.add_argument("--requests-per-session", type=int, default=1)
    parser.add_argument(
        "--prompt-mode",
        choices=("auto", "aligned-input-ids", "text"),
        default="auto",
        help="auto selects exact page-aligned input_ids for direct mode",
    )
    parser.add_argument(
        "--page-size-tokens",
        type=int,
        default=None,
        help="direct cache page size; default derives it from /aginfer/state",
    )
    parser.add_argument(
        "--shared-pages",
        type=int,
        default=0,
        help="common pages shared by all programs; default 0 isolates Dead-KV",
    )
    parser.add_argument("--tail-pages", type=int, default=12)
    parser.add_argument(
        "--token-base",
        type=int,
        default=1000,
        help="base vocabulary id for synthetic direct-mode input_ids",
    )
    parser.add_argument("--shared-chunks", type=int, default=0)
    parser.add_argument("--tail-chunks", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--warmup-requests", type=int, default=3)
    parser.add_argument("--retention-seconds", type=float, default=3.0)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--poll-timeout", type=float, default=45.0)
    parser.add_argument("--poll-interval", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument(
        "--artifact-dir",
        type=pathlib.Path,
        default=pathlib.Path("/tmp/deadkv-ab/artifacts"),
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--save-raw-states",
        action="store_true",
        help="also retain full unit-level state at checkpoints (can be large)",
    )
    parser.add_argument(
        "--no-stream",
        dest="stream",
        action="store_false",
        help="disable SSE; true TTFT will then be unavailable",
    )
    parser.set_defaults(stream=True)
    parser.add_argument(
        "--expected-hbm-cap-bytes",
        type=int,
        default=None,
        help="optional expected aggregate HBM KV-pool capacity across all ranks",
    )
    parser.add_argument(
        "--min-baseline-dead-bytes",
        type=int,
        default=0,
        help="fail unless every baseline trial retains at least this many bytes",
    )
    parser.add_argument(
        "--confirm-dedicated-server",
        action="store_true",
        help="required acknowledgement that this benchmark flushes the KV cache",
    )
    args = parser.parse_args(argv)
    if not args.confirm_dedicated_server:
        parser.error(
            "--confirm-dedicated-server is required because the benchmark calls flush_cache"
        )
    positive_ints = {
        "repeats": args.repeats,
        "live-sessions": args.live_sessions,
        "dead-per-epoch": args.dead_per_epoch,
        "epochs": args.epochs,
        "concurrency": args.concurrency,
        "final-concurrency": args.final_concurrency,
        "requests-per-session": args.requests_per_session,
        "tail-chunks": args.tail_chunks,
        "tail-pages": args.tail_pages,
        "max-tokens": args.max_tokens,
    }
    for name, value in positive_ints.items():
        if value < 1:
            parser.error(f"--{name} must be positive")
    if args.page_size_tokens is not None and args.page_size_tokens < 1:
        parser.error("--page-size-tokens must be positive")
    if args.shared_pages < 0 or args.shared_chunks < 0:
        parser.error("--shared-pages and --shared-chunks cannot be negative")
    if args.warmup_requests < 0:
        parser.error("--warmup-requests cannot be negative")
    for name in (
        "retention_seconds",
        "request_timeout",
        "poll_timeout",
        "poll_interval",
    ):
        if getattr(args, name) <= 0:
            parser.error("--" + name.replace("_", "-") + " must be positive")
    if args.expected_hbm_cap_bytes is not None and args.expected_hbm_cap_bytes <= 0:
        parser.error("--expected-hbm-cap-bytes must be positive")
    if args.min_baseline_dead_bytes < 0:
        parser.error("--min-baseline-dead-bytes cannot be negative")
    if args.token_base < 0:
        parser.error("--token-base cannot be negative")
    if args.backend == "dynamo" and args.prompt_mode == "aligned-input-ids":
        parser.error("Dynamo mode does not support --prompt-mode aligned-input-ids")
    args.sessions = args.dead_per_epoch * args.epochs
    return args


def warmup(
    backend: Backend,
    args: argparse.Namespace,
    run_id: str,
    prompt: PromptInput,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index in range(args.warmup_requests):
        program_id = f"deadkv-ab-{run_id}-warmup-{index:03d}"
        metric = backend.infer(
            label=f"warmup_{index:03d}",
            role="warmup",
            program_id=program_id,
            prompt=prompt,
        )
        _check_request_metrics([metric], "warmup")
        final = backend.end_program(program_id)
        records.append({"request": metric.as_dict(), "final": final})
    flush_and_wait(backend, timeout=args.poll_timeout, interval=args.poll_interval)
    return records


def run_benchmark(args: argparse.Namespace, artifacts: Artifacts) -> None:
    backend = make_backend(args)
    preflight = backend.preflight()
    run_id = artifacts.summary["run_id"]
    prompt_mode = args.prompt_mode
    if prompt_mode == "auto":
        prompt_mode = "aligned-input-ids" if args.backend == "direct" else "text"
    resolved_page_size = args.page_size_tokens
    if prompt_mode == "aligned-input-ids" and resolved_page_size is None:
        inferred = preflight.get("inferred_page_sizes_tokens")
        if not isinstance(inferred, list) or len(inferred) != 1:
            raise BenchmarkError(
                "could not infer one cache page size from /aginfer/state; pass "
                f"--page-size-tokens explicitly (observed {inferred!r})"
            )
        resolved_page_size = int(inferred[0])
    if prompt_mode == "aligned-input-ids":
        anchor_prompts, victim_prompts = make_aligned_input_ids(
            page_size_tokens=int(resolved_page_size),
            shared_pages=args.shared_pages,
            tail_pages=args.tail_pages,
            live_sessions=args.live_sessions,
            sessions=args.sessions,
            token_base=args.token_base,
        )
    else:
        anchor_prompts, victim_prompts = make_text_prompts(
            shared_chunks=args.shared_chunks,
            tail_chunks=args.tail_chunks,
            live_sessions=args.live_sessions,
            sessions=args.sessions,
        )
    rng = random.Random(args.seed)
    plan: list[dict[str, Any]] = []
    for pair_index in range(args.repeats):
        if args.order_mode == "abba":
            condition_order = (
                ["baseline", "ours"] if pair_index % 2 == 0 else ["ours", "baseline"]
            )
        else:
            condition_order = list(CONDITIONS)
            rng.shuffle(condition_order)
        victim_order = []
        for epoch_index in range(args.epochs):
            epoch_order = list(
                range(
                    epoch_index * args.dead_per_epoch,
                    (epoch_index + 1) * args.dead_per_epoch,
                )
            )
            rng.shuffle(epoch_order)
            victim_order.extend(epoch_order)
        plan.append(
            {
                "pair_index": pair_index,
                "condition_order": condition_order,
                "victim_order": victim_order,
            }
        )
    public_configuration = {
        "backend": args.backend,
        "server_url": args.server_url if args.backend == "direct" else None,
        "frontend_url": args.frontend_url if args.backend == "dynamo" else None,
        "worker_url": args.worker_url if args.backend == "dynamo" else None,
        "model": getattr(backend, "model", None),
        "repeats": args.repeats,
        "order_mode": args.order_mode,
        "arms": args.repeats * 2,
        "live_sessions": args.live_sessions,
        "terminal_sessions": args.sessions,
        "dead_per_epoch": args.dead_per_epoch,
        "epochs": args.epochs,
        "concurrency": args.concurrency,
        "final_concurrency": args.final_concurrency,
        "requests_per_session": args.requests_per_session,
        "prompt_mode": prompt_mode,
        "page_size_tokens": (
            resolved_page_size if prompt_mode == "aligned-input-ids" else None
        ),
        "shared_pages": (
            args.shared_pages if prompt_mode == "aligned-input-ids" else None
        ),
        "tail_pages": args.tail_pages if prompt_mode == "aligned-input-ids" else None,
        "shared_input_tokens": (
            int(resolved_page_size) * args.shared_pages
            if prompt_mode == "aligned-input-ids"
            else None
        ),
        "tail_input_tokens": (
            int(resolved_page_size) * args.tail_pages
            if prompt_mode == "aligned-input-ids"
            else None
        ),
        "total_input_tokens": (
            int(resolved_page_size) * (args.shared_pages + args.tail_pages)
            if prompt_mode == "aligned-input-ids"
            else None
        ),
        "token_base": args.token_base if prompt_mode == "aligned-input-ids" else None,
        "shared_chunks": args.shared_chunks,
        "tail_chunks": args.tail_chunks,
        "max_tokens": args.max_tokens,
        "warmup_requests": args.warmup_requests,
        "retention_seconds": args.retention_seconds,
        "stream": args.stream,
        "seed": args.seed,
        "expected_hbm_cap_bytes": args.expected_hbm_cap_bytes,
        "min_baseline_dead_bytes": args.min_baseline_dead_bytes,
    }
    artifacts.summary["configuration"] = public_configuration
    artifacts.summary["preflight"] = preflight
    artifacts.summary["experimental_design"] = {
        "isolation": (
            "same live server/build and capacity; cache flushed before each trial; "
            "paired prompts/order; only SESSION_END differs"
        ),
        "baseline": "no terminal signal; native pressure/LRU behavior remains active",
        "ours": "explicit terminal signal after every short-lived session",
        "live_set": (
            f"{args.live_sessions} live sessions remain resident during terminal-session "
            "churn; their final probes measure cache preservation and TTFT"
        ),
        "order_control": (
            "ABBA across four default arms"
            if args.order_mode == "abba"
            else "condition order randomized independently for every pair"
        ),
        "input_alignment": (
            f"direct input_ids: {args.shared_pages} shared + {args.tail_pages} tail "
            f"pages at {resolved_page_size} tokens/page; shared_pages=0 gives unique "
            "prefixes and removes growing holder-set cost from the baseline"
            if prompt_mode == "aligned-input-ids"
            else "text prompts; exact token-page alignment unavailable"
        ),
        "throughput_definitions": {
            "inference": "only inference-batch wall time",
            "pipeline": "inference plus SESSION_END control wall time; state polling excluded",
        },
    }
    artifacts.summary["prompt_manifest"] = prompt_manifest(
        anchor_prompts, victim_prompts
    )
    artifacts.summary["plan"] = plan
    artifacts.save_summary()
    artifacts.write("plan.json", {"configuration": public_configuration, "plan": plan})

    print(
        f"Dead-KV paired A/B run={run_id} backend={args.backend} "
        f"arms={args.repeats * 2} live={args.live_sessions} "
        f"terminal={args.sessions} ({args.epochs}x{args.dead_per_epoch})"
    )
    if args.warmup_requests:
        print(f"Warmup: {args.warmup_requests} request(s), then verified flush")
        artifacts.write("warmup.json", warmup(backend, args, run_id, anchor_prompts[0]))

    trials: list[dict[str, Any]] = []
    trial_index = 0
    for pair in plan:
        pair_index = int(pair["pair_index"])
        for condition in pair["condition_order"]:
            print(
                f"Pair {pair_index + 1}/{args.repeats}: {condition} "
                f"(trial {trial_index + 1}/{args.repeats * 2})"
            )
            trial = run_trial(
                backend=backend,
                args=args,
                artifacts=artifacts,
                run_id=run_id,
                pair_index=pair_index,
                condition=str(condition),
                trial_index=trial_index,
                victim_order=pair["victim_order"],
                anchor_prompts=anchor_prompts,
                victim_prompts=victim_prompts,
            )
            trials.append(trial)
            projection = trial_projection(trial)
            artifacts.summary["trials"].append(
                {
                    "trial_index": trial_index,
                    "pair_index": pair_index,
                    "condition": condition,
                    "artifact": f"trial_{trial_index:03d}_p{pair_index:02d}_{condition}.json",
                    "projection": projection,
                }
            )
            artifacts.save_summary()
            print(
                f"  end dead={_fmt_bytes(projection['end_dead_bytes_total'])}, "
                f"anchor cached={_fmt_number(projection['anchor_cached_tokens'], 0)}, "
                f"TTFT={_fmt_number(projection['anchor_ttft_seconds'])}s"
            )
            trial_index += 1

    capacity_hashes = {trial["capacity_hash"] for trial in trials}
    if len(capacity_hashes) != 1:
        raise BenchmarkError(
            "KV-pool capacity changed between paired trials; results are not comparable: "
            f"{sorted(capacity_hashes)}"
        )
    capacities = [trial["metrics"]["pool_cap_bytes"] for trial in trials]
    hbm_cap = int(capacities[0].get("HBM", 0))
    if (
        args.expected_hbm_cap_bytes is not None
        and hbm_cap != args.expected_hbm_cap_bytes
    ):
        raise BenchmarkError(
            f"aggregate HBM KV-pool capacity is {hbm_cap}, expected "
            f"{args.expected_hbm_cap_bytes}"
        )
    baseline_trials = [trial for trial in trials if trial["condition"] == "baseline"]
    for trial in baseline_trials:
        observed = int(trial["metrics"]["end_dead_physical_bytes_total"])
        if observed < args.min_baseline_dead_bytes:
            raise BenchmarkError(
                f"baseline pair {trial['pair_index']} retained only {observed} dead bytes, "
                f"below --min-baseline-dead-bytes={args.min_baseline_dead_bytes}"
            )
    aggregate = paired_summary(trials)
    artifacts.summary["capacity_evidence"] = {
        "capacity_hash": next(iter(capacity_hashes)),
        "pool_cap_bytes": capacities[0],
        "all_trials_match": True,
    }
    artifacts.summary["aggregate"] = aggregate
    if args.backend == "dynamo":
        artifacts.summary["warnings"].append(preflight["confound_warning"])
    if args.repeats < 3:
        artifacts.summary["warnings"].append(
            "Fewer than three pairs were run; treat TTFT/throughput differences as smoke data."
        )
    baseline_dead = [
        int(trial["metrics"]["end_dead_physical_bytes_total"])
        for trial in baseline_trials
    ]
    if baseline_dead and min(baseline_dead) == 0:
        artifacts.summary["warnings"].append(
            "At least one no-END baseline retained zero measured dead bytes; LRU/pressure "
            "may already have evicted the workload, so increase cache-safe workload size "
            "carefully or lengthen the observation window before claiming a retention win."
        )
    anchor_missing = [
        trial
        for trial in trials
        if trial["metrics"]["live_probes"]["cached_tokens"]["reported_count"]
        < args.live_sessions
    ]
    if anchor_missing:
        artifacts.summary["warnings"].append(
            "Some responses did not expose cached_tokens; state hit_count remains in trial "
            "artifacts, but cached-token comparison is incomplete."
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_id = args.run_id or (
        dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-"
        + uuid.uuid4().hex[:8]
    )
    artifacts = Artifacts(args.artifact_dir, run_id)
    try:
        run_benchmark(args, artifacts)
    except Exception as exc:
        artifacts.finish(
            "failed",
            error={
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        artifacts.path("report.md").write_text(
            render_report(artifacts.summary), encoding="utf-8"
        )
        print(f"FAIL: {exc}", file=sys.stderr)
        print(f"Artifacts: {artifacts.run_dir}", file=sys.stderr)
        return 1
    artifacts.finish("passed")
    artifacts.path("report.md").write_text(
        render_report(artifacts.summary), encoding="utf-8"
    )
    paired = artifacts.summary["aggregate"]["paired_aggregate"]
    print("PASS: paired Dead-KV benchmark completed")
    print(
        "Median paired dead-KV advantage: "
        + _fmt_bytes(paired["dead_kv_bytes_reclaimed_advantage"]["median"])
    )
    print(f"Artifacts: {artifacts.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

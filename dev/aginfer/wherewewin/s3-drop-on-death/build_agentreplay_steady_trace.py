#!/usr/bin/env python3
"""Build a deterministic open-loop AgentReplay trace for steady-state tests.

The generated trace has a fixed root-session arrival schedule.  A fraction of
the sessions stay live across several delayed turns, while the remaining
sessions finish after a short prefix and become dead-KV churn.  Baseline and
treatment must replay the exact same generated trace and manifest.

The source and generated JSONL files contain token IDs and remain private.  The
script itself and its tests contain no model-, host-, or trace-specific data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import random
import tempfile
from collections.abc import Mapping, Sequence
from typing import Any


SCHEMA_VERSION = 1


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=path.name + ".",
        delete=False,
    ) as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = pathlib.Path(stream.name)
    os.replace(temporary, path)


def load_trace(path: pathlib.Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number}: {exc}") from exc
            if not isinstance(row, Mapping):
                raise ValueError(f"trace line {line_number} is not an object")
            rows.append(dict(row))
    if not rows:
        raise ValueError("source trace is empty")
    return rows


def program_index(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    programs: dict[str, list[dict[str, Any]]] = {}
    for line_number, row in enumerate(rows, 1):
        program_id = row.get("program_id")
        if not isinstance(program_id, str) or not program_id:
            raise ValueError(f"trace row {line_number} has no string program_id")
        if not isinstance(row.get("input_ids"), list) or not row["input_ids"]:
            raise ValueError(f"trace row {line_number} has invalid input_ids")
        if not isinstance(row.get("forced_output_ids"), list):
            raise ValueError(f"trace row {line_number} has invalid forced_output_ids")
        programs.setdefault(program_id, []).append(dict(row))
    for program_id, records in programs.items():
        records.sort(key=lambda row: int(row.get("step") or 0))
        steps = [int(row.get("step") or 0) for row in records]
        if steps != list(range(1, len(records) + 1)):
            raise ValueError(
                f"program {program_id!r} does not have consecutive steps from 1"
            )
        parents = {row.get("parent_program_id") for row in records}
        spawns = {row.get("spawned_at_step") for row in records}
        if len(parents) != 1 or len(spawns) != 1:
            raise ValueError(f"program {program_id!r} changes parent metadata")
        for previous, current in zip(records, records[1:]):
            if current.get("context_reset"):
                continue
            prefix = list(previous["input_ids"]) + list(
                previous["forced_output_ids"]
            )
            if list(current["input_ids"][: len(prefix)]) != prefix:
                raise ValueError(
                    f"program {program_id!r} step {current['step']} does not "
                    "extend the preceding request"
                )
    return programs


def program_graph(
    programs: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[dict[str, str | None], dict[str, set[str]]]:
    parents: dict[str, str | None] = {}
    children: dict[str, set[str]] = {}
    for program_id, rows in programs.items():
        parent = rows[0].get("parent_program_id")
        if parent is not None and not isinstance(parent, str):
            raise ValueError(f"program {program_id!r} has an invalid parent")
        if parent is not None and parent not in programs:
            raise ValueError(f"program {program_id!r} has a missing parent")
        parents[program_id] = parent
        if parent is not None:
            spawn = rows[0].get("spawned_at_step")
            parent_steps = {int(row["step"]) for row in programs[parent]}
            if not isinstance(spawn, int) or spawn not in parent_steps:
                raise ValueError(f"program {program_id!r} has an invalid spawn step")
            children.setdefault(parent, set()).add(program_id)

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(program_id: str) -> None:
        if program_id in visiting:
            raise ValueError(f"program graph contains a cycle at {program_id!r}")
        if program_id in visited:
            return
        visiting.add(program_id)
        for child in children.get(program_id, set()):
            visit(child)
        visiting.remove(program_id)
        visited.add(program_id)

    for program_id in programs:
        visit(program_id)
    return parents, children


def root_closure(root: str, children: Mapping[str, set[str]]) -> list[str]:
    result = []
    pending = [root]
    while pending:
        program_id = pending.pop()
        result.append(program_id)
        pending.extend(sorted(children.get(program_id, set()), reverse=True))
    return result


def longest_common_prefix_length(values: Sequence[Sequence[int]]) -> int:
    if not values:
        return 0
    limit = min(len(value) for value in values)
    for index in range(limit):
        token = values[0][index]
        if any(value[index] != token for value in values[1:]):
            return index
    return limit


def identity_alphabet(rows: Sequence[Mapping[str, Any]]) -> tuple[int, int]:
    first: int | None = None
    for row in rows:
        for field in ("input_ids", "forced_output_ids"):
            for raw in row.get(field) or []:
                token = int(raw)
                if first is None:
                    first = token
                elif token != first:
                    return first, token
    raise ValueError("source trace does not contain two distinct token IDs")


def identity_tokens(index: int, width: int, alphabet: tuple[int, int]) -> list[int]:
    if index < 0 or index >= 2**width:
        raise ValueError(
            f"session index {index} does not fit in {width} identity tokens"
        )
    zero, one = alphabet
    return [one if index & (1 << bit) else zero for bit in reversed(range(width))]


def clone_bundle(
    programs: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    source_program_ids: Sequence[str],
    source_root_id: str,
    replica_index: int,
    arrival_seconds: float,
    role: str,
    live_steps: int,
    live_revisit_seconds: float,
    churn_gap_seconds: float,
    insertion_offset: int,
    identity: Sequence[int],
) -> list[dict[str, Any]]:
    if source_root_id not in source_program_ids:
        raise ValueError("source root is not part of its bundle")
    ordered_ids = list(source_program_ids)
    id_map = {
        source_id: f"steady-{replica_index:06d}-{position:02d}"
        for position, source_id in enumerate(ordered_ids)
    }
    new_root_id = id_map[source_root_id]
    source_bundle_start = float(programs[source_root_id][0].get("t") or 0.0)
    result: list[dict[str, Any]] = []
    for source_id in ordered_ids:
        source = programs[source_id]
        selected = source[:live_steps] if role == "live" else source
        new_program_id = id_map[source_id]
        for position, original in enumerate(selected):
            original_input = list(original["input_ids"])
            if insertion_offset > len(original_input):
                raise ValueError(
                    "identity insertion offset exceeds an input length in "
                    f"source program {source_id!r}"
                )
            row = dict(original)
            original_parent = original.get("parent_program_id")
            row["program_id"] = new_program_id
            row["parent_program_id"] = (
                id_map.get(str(original_parent))
                if original_parent is not None
                else None
            )
            spawn_ts = original.get("spawn_ts")
            row["spawn_ts"] = (
                arrival_seconds
                + max(0.0, float(spawn_ts) - source_bundle_start)
                if isinstance(spawn_ts, (int, float))
                else None
            )
            row["input_ids"] = [
                *original_input[:insertion_offset],
                *identity,
                *original_input[insertion_offset:],
            ]
            row["forced_output_ids"] = list(original["forced_output_ids"])
            row["t"] = arrival_seconds + max(
                0.0,
                float(original.get("t") or source_bundle_start)
                - source_bundle_start,
            )
            row["steady_role"] = role
            row["steady_root_id"] = new_root_id
            row["steady_is_root"] = source_id == source_root_id
            row["scheduled_session_arrival_s"] = arrival_seconds
            if position + 1 < len(selected):
                row["tool_gap_after"] = (
                    live_revisit_seconds if role == "live" else churn_gap_seconds
                )
            else:
                row["tool_gap_after"] = None
            result.append(row)

        cloned = [row for row in result if row["program_id"] == new_program_id]
        for previous, current in zip(cloned, cloned[1:]):
            if current.get("context_reset"):
                continue
            prefix = previous["input_ids"] + previous["forced_output_ids"]
            if current["input_ids"][: len(prefix)] != prefix:
                raise ValueError(
                    "identity insertion broke prefix continuity for "
                    f"{new_program_id}"
                )
    return result


def build_schedule(
    programs: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    arrival_rate: float,
    total_seconds: float,
    live_fraction: float,
    live_steps: int,
    revisit_seconds: float,
    churn_gap_seconds: float,
    seed: int,
    identity_width: int,
    identity_insertion_offset: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if arrival_rate <= 0 or total_seconds <= 0:
        raise ValueError("arrival rate and total duration must be positive")
    if not 0 < live_fraction < 1:
        raise ValueError("live fraction must be strictly between zero and one")
    if live_steps < 2:
        raise ValueError("live steps must be >= 2")
    if revisit_seconds <= 0 or churn_gap_seconds < 0:
        raise ValueError("live revisit must be positive and churn gap non-negative")
    parents, children = program_graph(programs)
    roots = sorted(
        program_id for program_id, parent in parents.items() if parent is None
    )
    live_templates = sorted(
        program_id
        for program_id in roots
        if not children.get(program_id)
        and len(programs[program_id]) >= live_steps
        and not any(
            row.get("context_reset") for row in programs[program_id][:live_steps]
        )
    )
    churn_templates = roots
    if not live_templates:
        raise ValueError(
            f"no childless source root has {live_steps} reusable live steps"
        )
    if not churn_templates:
        raise ValueError("source trace has no root bundles")

    first_inputs = [list(rows[0]["input_ids"]) for rows in programs.values()]
    insertion_offset = (
        longest_common_prefix_length(first_inputs)
        if identity_insertion_offset is None
        else identity_insertion_offset
    )
    if insertion_offset < 0:
        raise ValueError("identity insertion offset cannot be negative")
    minimum_input = min(
        len(row["input_ids"])
        for program_rows in programs.values()
        for row in program_rows
    )
    if insertion_offset > minimum_input:
        raise ValueError(
            f"identity insertion offset {insertion_offset} exceeds the minimum "
            f"input length {minimum_input}"
        )
    all_rows = [row for rows in programs.values() for row in rows]
    alphabet = identity_alphabet(all_rows)
    session_count = int(math.ceil(total_seconds * arrival_rate))
    if session_count > 2**identity_width:
        raise ValueError(
            f"{session_count} sessions need more than {identity_width} identity tokens"
        )

    rng = random.Random(seed)
    records: list[dict[str, Any]] = []
    role_counts = {"live": 0, "churn": 0}
    role_program_counts = {"live": 0, "churn": 0}
    template_uses: dict[str, int] = {}
    bundle_sizes: list[int] = []
    interval = 1.0 / arrival_rate
    for index in range(session_count):
        arrival = index * interval
        if arrival >= total_seconds:
            break
        role = "live" if rng.random() < live_fraction else "churn"
        templates = live_templates if role == "live" else churn_templates
        source_root_id = templates[rng.randrange(len(templates))]
        source_ids = (
            [source_root_id]
            if role == "live"
            else root_closure(source_root_id, children)
        )
        cloned = clone_bundle(
            programs,
            source_program_ids=source_ids,
            source_root_id=source_root_id,
            replica_index=index,
            arrival_seconds=arrival,
            role=role,
            live_steps=live_steps,
            live_revisit_seconds=revisit_seconds,
            churn_gap_seconds=churn_gap_seconds,
            insertion_offset=insertion_offset,
            identity=identity_tokens(index, identity_width, alphabet),
        )
        records.extend(cloned)
        role_counts[role] += 1
        role_program_counts[role] += len(source_ids)
        bundle_sizes.append(len(source_ids))
        template_uses[source_root_id] = template_uses.get(source_root_id, 0) + 1

    records.sort(key=lambda row: (float(row["t"]), str(row["program_id"]), row["step"]))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "arrival_rate_sessions_per_second": arrival_rate,
        "arrival_interval_seconds": interval,
        "arrival_duration_seconds": total_seconds,
        "live_fraction_requested": live_fraction,
        "live_revisit_seconds": revisit_seconds,
        "live_steps": live_steps,
        "churn_gap_seconds": churn_gap_seconds,
        "random_seed": seed,
        "identity": {
            "width_tokens": identity_width,
            "insertion_offset_tokens": insertion_offset,
            "purpose": (
                "make cloned sessions physically distinct after the real common prefix"
            ),
        },
        "session_count": sum(role_counts.values()),
        "program_count": sum(role_program_counts.values()),
        "request_count": len(records),
        "scheduled_request_rate_per_second": len(records) / total_seconds,
        "scheduled_forced_output_tokens_per_second": (
            sum(len(row["forced_output_ids"]) for row in records) / total_seconds
        ),
        "max_request_tokens": max(
            len(row["input_ids"]) + len(row["forced_output_ids"])
            for row in records
        ),
        "role_session_counts": role_counts,
        "role_program_counts": role_program_counts,
        "max_bundle_programs": max(bundle_sizes, default=0),
        "source_template_count": len(roots),
        "live_template_count": len(live_templates),
        "churn_template_count": len(churn_templates),
        "max_template_reuse": max(template_uses.values(), default=0),
    }
    return records, manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-trace", type=pathlib.Path, required=True)
    parser.add_argument("--out-trace", type=pathlib.Path, required=True)
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--arrival-rate", type=float, required=True)
    parser.add_argument("--warmup-seconds", type=float, required=True)
    parser.add_argument("--measurement-seconds", type=float, required=True)
    parser.add_argument("--cooldown-seconds", type=float, required=True)
    parser.add_argument("--live-fraction", type=float, default=0.25)
    parser.add_argument("--live-steps", type=int, default=4)
    parser.add_argument("--live-revisit-seconds", type=float, default=60.0)
    parser.add_argument("--churn-gap-seconds", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--identity-width", type=int, default=16)
    parser.add_argument("--max-request-tokens", type=int, default=131072)
    parser.add_argument(
        "--identity-insert-offset",
        type=int,
        default=None,
        help=(
            "insert the per-replica marker after this many prompt tokens; "
            "default is the source programs' exact common-prefix length"
        ),
    )
    args = parser.parse_args(argv)
    for name in (
        "arrival_rate",
        "warmup_seconds",
        "measurement_seconds",
        "cooldown_seconds",
        "live_revisit_seconds",
    ):
        if getattr(args, name) <= 0:
            parser.error("--" + name.replace("_", "-") + " must be positive")
    if args.identity_width <= 0 or args.max_request_tokens <= 0:
        parser.error("identity width and max request tokens must be positive")
    if args.churn_gap_seconds < 0:
        parser.error("--churn-gap-seconds cannot be negative")
    if args.identity_insert_offset is not None and args.identity_insert_offset < 0:
        parser.error("--identity-insert-offset cannot be negative")
    if args.out_trace.resolve() == args.source_trace.resolve():
        parser.error("--out-trace must not overwrite --source-trace")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    source = args.source_trace.expanduser().resolve()
    if not source.is_file():
        raise SystemExit(f"source trace does not exist: {source}")
    out_trace = args.out_trace.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    for path in (out_trace, manifest_path):
        if path.exists():
            raise SystemExit(f"refusing to overwrite existing artifact: {path}")

    source_rows = load_trace(source)
    programs = program_index(source_rows)
    total_seconds = (
        args.warmup_seconds + args.measurement_seconds + args.cooldown_seconds
    )
    records, manifest = build_schedule(
        programs,
        arrival_rate=args.arrival_rate,
        total_seconds=total_seconds,
        live_fraction=args.live_fraction,
        live_steps=args.live_steps,
        revisit_seconds=args.live_revisit_seconds,
        churn_gap_seconds=args.churn_gap_seconds,
        seed=args.seed,
        identity_width=args.identity_width,
        identity_insertion_offset=args.identity_insert_offset,
    )
    if manifest["max_request_tokens"] > args.max_request_tokens:
        raise SystemExit(
            "generated request length "
            f"{manifest['max_request_tokens']} exceeds --max-request-tokens "
            f"{args.max_request_tokens}"
        )
    out_trace.parent.mkdir(parents=True, exist_ok=True)
    with out_trace.open("x", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")

    manifest.update(
        {
            "source_trace": {
                "basename": source.name,
                "sha256": sha256_file(source),
            },
            "steady_trace": {
                "basename": out_trace.name,
                "sha256": sha256_file(out_trace),
            },
            "windows": {
                "warmup_seconds": args.warmup_seconds,
                "measurement_seconds": args.measurement_seconds,
                "cooldown_seconds": args.cooldown_seconds,
                "measurement_start_seconds": args.warmup_seconds,
                "measurement_end_seconds": (
                    args.warmup_seconds + args.measurement_seconds
                ),
            },
        }
    )
    atomic_write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

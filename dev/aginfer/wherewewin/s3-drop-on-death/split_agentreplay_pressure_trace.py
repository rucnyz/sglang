#!/usr/bin/env python3
"""Split a token trace into live-seed, terminal-churn, and live-probe phases.

A configurable number of childless root programs are selected deterministically,
preferring the largest step-4 input. Their first three turns form
``live-seed.jsonl`` and their fourth/final turn forms ``live-probe.jsonl``. Every
other root and its complete descendant closure forms terminal churn. With
``--terminal-waves N``, distinct root closures are deterministically balanced
across N mutually exclusive ``terminal-churn-wave-NNN.jsonl`` files. No trace is
duplicated between waves.

The phase traces necessarily contain private program and token IDs. Keep them in
a private artifact directory. ``manifest.json`` contains only counts, lengths,
and hashes; it never contains program IDs or token IDs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

SCHEMA_VERSION = 1
TRACE_NAMES = {
    "live_seed": "live-seed.jsonl",
    "live_probe": "live-probe.jsonl",
}


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=path.name + ".",
        delete=False,
    ) as stream:
        stream.write(text)
        temporary = pathlib.Path(stream.name)
    os.replace(temporary, path)


def load_trace(path: pathlib.Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8", errors="ignore") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at line {line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"trace line {line_number} is not an object")
            validate_record(record, line_number)
            records.append(record)
    if not records:
        raise ValueError("input trace is empty")
    return records


def validate_record(record: Mapping[str, Any], line_number: int) -> None:
    for field in ("t", "program_id", "step", "input_ids", "forced_output_ids"):
        if field not in record:
            raise ValueError(f"trace line {line_number} is missing {field!r}")
    if not isinstance(record["program_id"], str) or not record["program_id"]:
        raise ValueError(f"trace line {line_number} has invalid program_id")
    if not isinstance(record["step"], int) or record["step"] <= 0:
        raise ValueError(f"trace line {line_number} has invalid step")
    for field in ("input_ids", "forced_output_ids"):
        values = record[field]
        if not isinstance(values, list) or not values:
            raise ValueError(f"trace line {line_number} has empty/non-list {field}")
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in values
        ):
            raise ValueError(f"trace line {line_number} has invalid token in {field}")


def program_index(
    records: Sequence[dict[str, Any]], *, require_start_at_one: bool = True
) -> dict[str, list[dict[str, Any]]]:
    programs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        programs[record["program_id"]].append(record)
    for program_id, rows in programs.items():
        rows.sort(key=lambda row: int(row["step"]))
        steps = [int(row["step"]) for row in rows]
        first_step = 1 if require_start_at_one else steps[0]
        if steps != list(range(first_step, first_step + len(rows))):
            raise ValueError(
                f"program {program_id!r} has non-contiguous/duplicate steps {steps}"
            )
        parents = {row.get("parent_program_id") for row in rows}
        spawns = {row.get("spawned_at_step") for row in rows}
        if len(parents) != 1 or len(spawns) != 1:
            raise ValueError(f"program {program_id!r} changes parent/spawn metadata")
        for previous, current in zip(rows, rows[1:]):
            if current.get("context_reset"):
                continue
            expected = previous["input_ids"] + previous["forced_output_ids"]
            if current["input_ids"][: len(expected)] != expected:
                raise ValueError(
                    f"program {program_id!r} breaks prefix continuity at step {current['step']}"
                )
    return dict(programs)


def graph(
    programs: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[dict[str, str | None], dict[str, set[str]]]:
    parents: dict[str, str | None] = {}
    children: dict[str, set[str]] = defaultdict(set)
    for program_id, rows in programs.items():
        parent = rows[0].get("parent_program_id")
        if parent is not None and not isinstance(parent, str):
            raise ValueError(f"program {program_id!r} has non-string parent")
        parents[program_id] = parent
        if parent is not None:
            if parent not in programs:
                raise ValueError(
                    f"program {program_id!r} has missing parent {parent!r}"
                )
            spawn = rows[0].get("spawned_at_step")
            parent_steps = {int(row["step"]) for row in programs[parent]}
            if not isinstance(spawn, int) or spawn not in parent_steps:
                raise ValueError(
                    f"program {program_id!r} has invalid spawn step {spawn!r} in {parent!r}"
                )
            children[parent].add(program_id)

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
    return parents, dict(children)


def topology_hash(records: Sequence[Mapping[str, Any]]) -> str:
    topology = sorted(
        (
            str(record["program_id"]),
            int(record["step"]),
            record.get("parent_program_id"),
            record.get("spawned_at_step"),
            bool(record.get("background_spawn")),
            bool(record.get("context_reset")),
        )
        for record in records
    )
    encoded = json.dumps(topology, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def phase_stats(
    records: Sequence[Mapping[str, Any]], path: pathlib.Path
) -> dict[str, Any]:
    prompt_lengths = [len(record["input_ids"]) for record in records]
    output_lengths = [len(record["forced_output_ids"]) for record in records]
    programs = {str(record["program_id"]) for record in records}
    roots = {
        str(record["program_id"])
        for record in records
        if record.get("parent_program_id") is None
    }
    return {
        "file": path.name,
        "sha256": sha256_file(path),
        "requests": len(records),
        "programs": len(programs),
        "roots": len(roots),
        "prompt_tokens_sum": sum(prompt_lengths),
        "prompt_tokens_max": max(prompt_lengths, default=0),
        "output_tokens_sum": sum(output_lengths),
        "output_tokens_max": max(output_lengths, default=0),
        "max_request_tokens": max(
            (prompt + output for prompt, output in zip(prompt_lengths, output_lengths)),
            default=0,
        ),
        "topology_sha256": topology_hash(records),
    }


def write_trace(path: pathlib.Path, records: Sequence[Mapping[str, Any]]) -> None:
    atomic_write_text(
        path,
        "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records),
    )


def terminal_wave_path(
    out_dir: pathlib.Path, wave_number: int, wave_count: int
) -> pathlib.Path:
    if wave_count == 1:
        return out_dir / "terminal-churn.jsonl"
    return out_dir / f"terminal-churn-wave-{wave_number:03d}.jsonl"


def terminal_program_weight(rows: Sequence[Mapping[str, Any]]) -> int:
    """Approximate resident KV using the largest request in one program."""
    return max(len(row["input_ids"]) + len(row["forced_output_ids"]) for row in rows)


def split_terminal_waves(
    records: Sequence[dict[str, Any]], wave_count: int
) -> tuple[list[list[dict[str, Any]]], list[dict[str, int]]]:
    """Partition distinct root closures without splitting a program DAG."""
    programs = program_index(records)
    parents, children = graph(programs)
    roots = sorted(
        program_id for program_id, parent in parents.items() if parent is None
    )
    if wave_count <= 0:
        raise ValueError("terminal wave count must be positive")
    if wave_count > len(roots):
        raise ValueError(
            f"need at least {wave_count} terminal roots for distinct waves, found {len(roots)}"
        )

    def closure(root: str) -> set[str]:
        result: set[str] = set()
        pending = [root]
        while pending:
            program_id = pending.pop()
            if program_id in result:
                continue
            result.add(program_id)
            pending.extend(sorted(children.get(program_id, set()), reverse=True))
        return result

    components = []
    for root in roots:
        program_ids = closure(root)
        weight = sum(terminal_program_weight(programs[pid]) for pid in program_ids)
        requests = sum(len(programs[pid]) for pid in program_ids)
        components.append((weight, root, program_ids, requests))
    components.sort(key=lambda item: (-item[0], item[1]))

    bins = [
        {"weight": 0, "program_ids": set(), "root_count": 0, "requests": 0}
        for _ in range(wave_count)
    ]
    for weight, _root, program_ids, requests in components:
        index = min(range(wave_count), key=lambda item: (bins[item]["weight"], item))
        bins[index]["weight"] += weight
        bins[index]["program_ids"].update(program_ids)
        bins[index]["root_count"] += 1
        bins[index]["requests"] += requests

    waves = [
        [record for record in records if record["program_id"] in bucket["program_ids"]]
        for bucket in bins
    ]
    metadata = [
        {
            "wave_number": index,
            "root_count": int(bucket["root_count"]),
            "program_count": len(bucket["program_ids"]),
            "requests": int(bucket["requests"]),
            "estimated_resident_tokens": int(bucket["weight"]),
        }
        for index, bucket in enumerate(bins, 1)
    ]
    return waves, metadata


def split_trace(
    records: Sequence[dict[str, Any]], live_root_count: int
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    programs = program_index(records)
    parents, children = graph(programs)
    roots = sorted(
        program_id for program_id, parent in parents.items() if parent is None
    )
    candidates = []
    for program_id in roots:
        rows = programs[program_id]
        if children.get(program_id) or len(rows) != 4:
            continue
        if rows[3].get("context_reset"):
            continue
        candidates.append((len(rows[3]["input_ids"]), program_id))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    if len(candidates) < live_root_count:
        raise ValueError(
            f"need {live_root_count} childless four-step roots, found {len(candidates)}"
        )
    live_programs = {program_id for _, program_id in candidates[:live_root_count]}
    terminal_programs = set(programs) - live_programs

    for child, parent in parents.items():
        if (
            child in terminal_programs
            and parent is not None
            and parent not in terminal_programs
        ):
            raise ValueError(f"terminal trace would orphan {child!r} from {parent!r}")

    phases = {
        "live_seed": [
            record
            for record in records
            if record["program_id"] in live_programs and int(record["step"]) <= 3
        ],
        "terminal_churn": [
            record for record in records if record["program_id"] in terminal_programs
        ],
        "live_probe": [
            record
            for record in records
            if record["program_id"] in live_programs and int(record["step"]) == 4
        ],
    }
    if len(phases["live_seed"]) != live_root_count * 3:
        raise AssertionError(
            "live-seed does not contain exactly steps 1-3 per live root"
        )
    if len(phases["live_probe"]) != live_root_count:
        raise AssertionError("live-probe does not contain exactly step 4 per live root")
    for program_id in live_programs:
        seed_last = programs[program_id][2]
        probe = programs[program_id][3]
        prefix = seed_last["input_ids"] + seed_last["forced_output_ids"]
        if probe["input_ids"][: len(prefix)] != prefix:
            raise AssertionError(
                f"live probe for {program_id!r} does not reuse seed prefix"
            )

    selected = [
        {
            "selection_rank": rank,
            "final_input_tokens": final_input_tokens,
            "seed_steps": 3,
            "probe_step": 4,
        }
        for rank, (final_input_tokens, _program_id) in enumerate(
            candidates[:live_root_count], 1
        )
    ]
    metadata = {
        "live_programs": selected,
        "terminal_program_count": len(terminal_programs),
        "source_programs": len(programs),
        "source_roots": len(roots),
        "selection": (
            "childless roots with exactly four steps; descending step-4 input "
            "length, then program_id"
        ),
    }
    return phases, metadata


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=pathlib.Path, required=True)
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    parser.add_argument("--live-roots", type=int, default=2)
    parser.add_argument("--terminal-waves", type=int, default=1)
    args = parser.parse_args(argv)
    if args.live_roots <= 0:
        parser.error("--live-roots must be positive")
    if args.terminal_waves <= 0:
        parser.error("--terminal-waves must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    trace = args.trace.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    records = load_trace(trace)
    phases, metadata = split_trace(records, args.live_roots)
    terminal_waves, terminal_wave_metadata = split_terminal_waves(
        phases["terminal_churn"], args.terminal_waves
    )
    outputs = {name: out_dir / filename for name, filename in TRACE_NAMES.items()}
    wave_paths = [
        terminal_wave_path(out_dir, index, args.terminal_waves)
        for index in range(1, args.terminal_waves + 1)
    ]
    manifest_path = out_dir / "manifest.json"
    existing = [
        path
        for path in (*outputs.values(), *wave_paths, manifest_path)
        if path.exists()
    ]
    if existing:
        raise SystemExit(
            "refusing to overwrite output files: " + ", ".join(map(str, existing))
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("live_seed", "live_probe"):
        write_trace(outputs[name], phases[name])
    for path, wave_records in zip(wave_paths, terminal_waves):
        write_trace(path, wave_records)
    wave_stats = []
    for metadata_row, path, wave_records in zip(
        terminal_wave_metadata, wave_paths, terminal_waves
    ):
        stats = phase_stats(wave_records, path)
        wave_stats.append({**metadata_row, **stats})
    terminal_hash = (
        wave_stats[0]["sha256"]
        if len(wave_stats) == 1
        else hashlib.sha256(
            json.dumps(
                [stats["sha256"] for stats in wave_stats], separators=(",", ":")
            ).encode()
        ).hexdigest()
    )
    terminal_stats = {
        "wave_count": args.terminal_waves,
        "sha256": terminal_hash,
        "requests": len(phases["terminal_churn"]),
        "programs": len({record["program_id"] for record in phases["terminal_churn"]}),
        "roots": sum(stats["roots"] for stats in wave_stats),
        "prompt_tokens_sum": sum(stats["prompt_tokens_sum"] for stats in wave_stats),
        "prompt_tokens_max": max(stats["prompt_tokens_max"] for stats in wave_stats),
        "output_tokens_sum": sum(stats["output_tokens_sum"] for stats in wave_stats),
        "output_tokens_max": max(stats["output_tokens_max"] for stats in wave_stats),
        "max_request_tokens": max(stats["max_request_tokens"] for stats in wave_stats),
        "topology_sha256": topology_hash(phases["terminal_churn"]),
        "waves": wave_stats,
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "basename": trace.name,
            "sha256": sha256_file(trace),
            "requests": len(records),
            "topology_sha256": topology_hash(records),
        },
        **metadata,
        "phases": {
            "live_seed": phase_stats(phases["live_seed"], outputs["live_seed"]),
            "terminal_churn": terminal_stats,
            "live_probe": phase_stats(phases["live_probe"], outputs["live_probe"]),
        },
    }
    serialized = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if "input_ids" in serialized or "forced_output_ids" in serialized:
        raise AssertionError("manifest unexpectedly contains token-array field names")
    atomic_write_text(manifest_path, serialized)
    print(
        f"selected {args.live_roots} live roots; wrote {args.terminal_waves} "
        "distinct terminal waves with "
        f"{len(phases['live_seed'])}/{len(phases['terminal_churn'])}/"
        f"{len(phases['live_probe'])} seed/churn/probe requests to {out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

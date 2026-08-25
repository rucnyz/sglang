#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import pathlib
import tempfile
import unittest

SCRIPT = pathlib.Path(__file__).with_name("split_agentreplay_pressure_trace.py")
SPEC = importlib.util.spec_from_file_location(
    "split_agentreplay_pressure_trace", SCRIPT
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def program(
    program_id: str,
    final_extra: int,
    *,
    parent: str | None = None,
    spawned_at_step: int | None = None,
) -> list[dict]:
    base = 100 + sum(ord(character) for character in program_id)
    rows = []
    prompt = [base]
    for step in range(1, 5):
        if step == 4:
            prompt = [*prompt, *range(1000, 1000 + final_extra)]
        output = [base + step]
        rows.append(
            {
                "t": float(step),
                "program_id": program_id,
                "step": step,
                "input_ids": list(prompt),
                "forced_output_ids": output,
                "parent_program_id": parent,
                "spawned_at_step": spawned_at_step,
            }
        )
        prompt = [*prompt, *output, base + step + 10]
    return rows


class PressureSplitTests(unittest.TestCase):
    def test_selects_longest_childless_roots_and_preserves_dag(self):
        records = [
            *program("root-a", 1),
            *program("root-b", 4),
            *program("root-c", 7),
            *program("root-parent", 20),
            *program("child", 30, parent="root-parent", spawned_at_step=2),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            trace = root / "source.jsonl"
            trace.write_text(
                "".join(json.dumps(row) + "\n" for row in records), encoding="utf-8"
            )
            out_dir = root / "split"
            self.assertEqual(
                MODULE.main(["--trace", str(trace), "--out-dir", str(out_dir)]), 0
            )

            seed = MODULE.load_trace(out_dir / "live-seed.jsonl")
            probe = MODULE.load_trace(out_dir / "live-probe.jsonl")
            terminal = MODULE.load_trace(out_dir / "terminal-churn.jsonl")
            self.assertEqual({row["program_id"] for row in seed}, {"root-b", "root-c"})
            self.assertEqual([row["step"] for row in seed], [1, 2, 3, 1, 2, 3])
            self.assertEqual({row["program_id"] for row in probe}, {"root-b", "root-c"})
            self.assertEqual({row["step"] for row in probe}, {4})
            self.assertEqual(
                {row["program_id"] for row in terminal},
                {"root-a", "root-parent", "child"},
            )
            terminal_programs = MODULE.program_index(terminal)
            MODULE.graph(terminal_programs)
            self.assertEqual(
                terminal_programs["child"][0]["parent_program_id"], "root-parent"
            )

            manifest_text = (out_dir / "manifest.json").read_text(encoding="utf-8")
            manifest = json.loads(manifest_text)
            self.assertEqual(
                manifest["live_programs"],
                [
                    {
                        "final_input_tokens": len(
                            program("root-c", 7)[-1]["input_ids"]
                        ),
                        "probe_step": 4,
                        "seed_steps": 3,
                        "selection_rank": 1,
                    },
                    {
                        "final_input_tokens": len(
                            program("root-b", 4)[-1]["input_ids"]
                        ),
                        "probe_step": 4,
                        "seed_steps": 3,
                        "selection_rank": 2,
                    },
                ],
            )
            self.assertEqual(manifest["terminal_program_count"], 3)
            self.assertNotIn("input_ids", manifest_text)
            self.assertNotIn("forced_output_ids", manifest_text)
            for program_id in ("root-a", "root-b", "root-c", "root-parent", "child"):
                self.assertNotIn(json.dumps(program_id), manifest_text)

    def test_rejects_broken_prefix(self):
        records = program("broken", 2)
        records[2]["input_ids"] = [999]
        with self.assertRaisesRegex(ValueError, "prefix continuity"):
            MODULE.split_trace(records, 1)

    def test_multiple_terminal_waves_are_disjoint_and_preserve_root_closures(self):
        records = [
            *program("live-a", 100),
            *program("live-b", 90),
            *program("terminal-a", 8),
            *program("terminal-b", 6),
            *program("terminal-c", 4),
            *program("parent", 2),
            *program("child", 3, parent="parent", spawned_at_step=2),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            trace = root / "source.jsonl"
            trace.write_text(
                "".join(json.dumps(row) + "\n" for row in records), encoding="utf-8"
            )
            out_dir = root / "split"
            self.assertEqual(
                MODULE.main(
                    [
                        "--trace",
                        str(trace),
                        "--out-dir",
                        str(out_dir),
                        "--terminal-waves",
                        "3",
                    ]
                ),
                0,
            )
            waves = [
                MODULE.load_trace(out_dir / f"terminal-churn-wave-{index:03d}.jsonl")
                for index in range(1, 4)
            ]
            program_sets = [{row["program_id"] for row in wave} for wave in waves]
            self.assertTrue(all(program_sets))
            self.assertFalse(program_sets[0] & program_sets[1])
            self.assertFalse(program_sets[0] & program_sets[2])
            self.assertFalse(program_sets[1] & program_sets[2])
            self.assertEqual(
                set().union(*program_sets),
                {"terminal-a", "terminal-b", "terminal-c", "parent", "child"},
            )
            parent_wave = next(group for group in program_sets if "parent" in group)
            self.assertIn("child", parent_wave)
            manifest_text = (out_dir / "manifest.json").read_text(encoding="utf-8")
            manifest = json.loads(manifest_text)
            terminal = manifest["phases"]["terminal_churn"]
            self.assertEqual(terminal["wave_count"], 3)
            self.assertEqual(len(terminal["waves"]), 3)
            self.assertEqual(sum(row["requests"] for row in terminal["waves"]), 20)
            self.assertNotIn("terminal-a", manifest_text)
            self.assertNotIn("input_ids", manifest_text)

    def test_rejects_more_waves_than_distinct_terminal_roots(self):
        records = [
            *program("live-a", 100),
            *program("terminal-a", 1),
        ]
        phases, _metadata = MODULE.split_trace(records, 1)
        with self.assertRaisesRegex(ValueError, "distinct waves"):
            MODULE.split_terminal_waves(phases["terminal_churn"], 2)

    def test_terminal_weight_sums_context_epoch_high_water_marks(self):
        records = program("reset", 0)
        records[2]["context_reset"] = True
        records[2]["input_ids"] = [900, 901]
        records[3]["input_ids"] = [900, 901, records[2]["forced_output_ids"][0]]
        first_epoch = max(
            len(row["input_ids"]) + len(row["forced_output_ids"]) for row in records[:2]
        )
        second_epoch = max(
            len(row["input_ids"]) + len(row["forced_output_ids"]) for row in records[2:]
        )
        self.assertEqual(
            MODULE.terminal_program_weight(records), first_epoch + second_epoch
        )


if __name__ == "__main__":
    unittest.main()

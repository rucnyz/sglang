"""Server-free tests for the aginfer P2b/P2c event-plane wiring:

  agentreplay's lifecycle events (P2a) -> extra_args.aginfer_events -> Dynamo's
  update_aginfer_events RPC (P2b) -> AginferDriver.apply_events -> ProgramTracker
  belief transitions (P2c).

See EXP_PLAN.md P2a/P2b/P2c and scheduler_driver.AginferDriver.apply_events's
docstring for the kind -> transition mapping this exercises.
"""

from __future__ import annotations

import unittest

from sglang.srt.managers.io_struct import (
    UpdateAginferEventsReq,
    UpdateAginferEventsReqOutput,
)
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.tokenizer_control_mixin import _COMMUNICATOR_SPECS
from sglang.srt.mem_cache.aginfer.events import EventKind
from sglang.srt.mem_cache.aginfer.program_tracker import State
from sglang.srt.mem_cache.aginfer.scheduler_driver import AginferDriver
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class AginferDriverApplyEventsTest(unittest.TestCase):
    def test_tool_call_end_then_start_drives_reasoning_then_acting(self):
        driver = AginferDriver()
        result = driver.apply_events([
            {"kind": "tool_call_end", "session": "p1", "payload": {}},
        ])
        self.assertEqual(result, {"applied": 1, "skipped": 0})
        self.assertEqual(driver._tracker.state("p1"), State.REASONING)

        result = driver.apply_events([
            {"kind": "tool_call_start", "session": "p1", "payload": {}},
        ])
        self.assertEqual(result, {"applied": 1, "skipped": 0})
        self.assertEqual(driver._tracker.state("p1"), State.ACTING)

    def test_sub_dispatch_arrives_the_child_not_the_parent(self):
        driver = AginferDriver()
        driver.apply_events([{
            "kind": "sub_dispatch_blocking", "session": "child1",
            "payload": {"parent_session_id": "main", "fanout": 1},
        }])
        self.assertEqual(driver._tracker.state("child1"), State.REASONING)
        self.assertIsNone(driver._tracker.state("main"))

    def test_sub_return_arrives_the_parent(self):
        driver = AginferDriver()
        driver.apply_events([
            {"kind": "tool_call_end", "session": "main", "payload": {}},
            {"kind": "tool_call_start", "session": "main", "payload": {}},
        ])
        self.assertEqual(driver._tracker.state("main"), State.ACTING)
        driver.apply_events([{
            "kind": "sub_return", "session": "main",
            "payload": {"child_session_id": "child1"},
        }])
        self.assertEqual(driver._tracker.state("main"), State.REASONING)

    def test_sub_dispatch_async_also_arrives_the_child(self):
        driver = AginferDriver()
        driver.apply_events([{
            "kind": "sub_dispatch_async", "session": "child1",
            "payload": {"parent_session_id": "main", "fanout": 2},
        }])
        self.assertEqual(driver._tracker.state("child1"), State.REASONING)

    def test_unmapped_and_malformed_events_are_skipped_not_raised(self):
        driver = AginferDriver()
        result = driver.apply_events([
            {"kind": "memory_pressure", "session": None, "payload": {}},
            {"kind": "session_end", "session": "p1", "payload": {}},  # own end_program path
            {"session": "p1"},          # no kind
            {"kind": "tool_call_end"},   # no session
            "not-a-dict",
        ])
        self.assertEqual(result, {"applied": 0, "skipped": 5})

    def test_apply_events_before_any_tick_lazily_builds_the_tracker(self):
        driver = AginferDriver()
        self.assertIsNone(driver._tracker)
        driver.apply_events([{"kind": "tool_call_end", "session": "p1"}])
        self.assertIsNotNone(driver._tracker)


class SchedulerUpdateAginferEventsTest(unittest.TestCase):
    def test_rejected_when_in_engine_driver_not_armed(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler._aginfer_driver = None
        out = scheduler.update_aginfer_events(
            UpdateAginferEventsReq(events=[{"kind": "tool_call_end", "session": "p1"}])
        )
        self.assertIsInstance(out, UpdateAginferEventsReqOutput)
        self.assertFalse(out.ok)
        self.assertEqual(out.applied, 0)

    def test_applies_through_to_the_driver_when_armed(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler._aginfer_driver = AginferDriver()
        out = scheduler.update_aginfer_events(
            UpdateAginferEventsReq(events=[
                {"kind": "tool_call_end", "session": "p1"},
                {"kind": "unmapped_kind", "session": "p1"},
            ])
        )
        self.assertTrue(out.ok)
        self.assertEqual(out.applied, 1)
        self.assertEqual(out.skipped, 1)
        self.assertEqual(
            scheduler._aginfer_driver._tracker.state("p1"), State.REASONING
        )


class AginferEventsWireContractTest(unittest.TestCase):
    def test_sub_return_event_kind_exists(self):
        self.assertEqual(EventKind.SUB_RETURN.value, "sub_return")

    def test_tokenizer_control_spec_registered(self):
        self.assertIn(
            ("update_aginfer_events", UpdateAginferEventsReqOutput),
            _COMMUNICATOR_SPECS,
        )


if __name__ == "__main__":
    unittest.main()

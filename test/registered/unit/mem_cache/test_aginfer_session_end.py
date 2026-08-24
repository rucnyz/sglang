"""Server-free tests for the aginfer SESSION_END / Dead-KV path."""

from __future__ import annotations

import asyncio
import time
import unittest
from types import SimpleNamespace

from sglang.srt.managers.io_struct import (
    AginferSessionEndReq,
    AginferSessionEndReqOutput,
)
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.tokenizer_control_mixin import (
    _COMMUNICATOR_SPECS,
    TokenizerControlMixin,
)
from sglang.srt.mem_cache.aginfer.dead_kv import (
    aginfer_program_busy,
    end_aginfer_program,
)
from sglang.srt.mem_cache.aginfer.http_validators import validate_session_end_body
from sglang.srt.mem_cache.aginfer.program_tracker import State
from sglang.srt.mem_cache.aginfer.scheduler_driver import AginferDriver
from sglang.srt.mem_cache.unified_cache_components import EvictLayer
from sglang.srt.session.session_controller import Session, SessionController
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _ComponentData:
    def __init__(self, device_tokens=0, host_tokens=0):
        self.value = list(range(device_tokens)) if device_tokens else None
        self.host_value = list(range(host_tokens)) if host_tokens else None
        self.lock_ref = 0
        self.host_lock_ref = 0


class _Node:
    _next_id = 1

    def __init__(self, *, holders=(), device_tokens=1, host_tokens=0):
        self.id = _Node._next_id
        _Node._next_id += 1
        self.parent = None
        self.children = {}
        self.session_ids = set(holders)
        self.component_data = [_ComponentData(device_tokens, host_tokens)]
        self.write_through_pending_id = None


class _Component:
    def evict_component(self, node, target=EvictLayer.DEVICE):
        data = node.component_data[0]
        device_freed = 0
        host_freed = 0
        if EvictLayer.DEVICE in target and data.value is not None:
            device_freed = len(data.value)
            data.value = None
        if EvictLayer.HOST in target and data.host_value is not None:
            host_freed = len(data.host_value)
            data.host_value = None
        return device_freed, host_freed


class _Cache:
    def __init__(self):
        self.root_node = _Node(device_tokens=0)
        self._components_tuple = (_Component(),)
        self.evictable_device_leaves = set()
        self.evictable_host_leaves = set()
        self.ongoing_write_through = {}
        self.ongoing_load_back = {}
        self.ongoing_backup = {}
        self.ongoing_prefetch = {}
        self._aginfer_program_states = {}
        self.released_sessions = []
        self.eviction_order = []

    def attach(self, parent, child, key):
        child.parent = parent
        parent.children[key] = child

    def set_aginfer_program_state(self, *, pid, state, pre_pause_state):
        value = {"state": state, "pre_pause_state": pre_pause_state}
        changed = int(self._aginfer_program_states.get(pid) != value)
        self._aginfer_program_states[pid] = value
        return True, "ok", changed

    @staticmethod
    def _aginfer_unit_hash(node):
        return f"node-{node.id}"

    def _evict_component_and_detach_lru(self, node, component, target, tracker=None):
        self.eviction_order.append(node.id)
        return component.evict_component(node, target=target)

    @staticmethod
    def _update_evictable_leaf_sets(node):
        return None

    @staticmethod
    def _remove_leaf_from_parent(node):
        for key, child in list(node.parent.children.items()):
            if child is node:
                del node.parent.children[key]
                return
        raise AssertionError("node not attached")

    def end_aginfer_program(self, program_id):
        return end_aginfer_program(self, program_id)

    def release_session(self, session_id):
        self.released_sessions.append(session_id)


class AginferSessionEndCacheTest(unittest.TestCase):
    def test_exclusive_chain_is_released_leaf_to_root_and_idempotent(self):
        cache = _Cache()
        parent = _Node(holders={"p"}, device_tokens=2, host_tokens=2)
        leaf = _Node(holders={"p"}, device_tokens=3, host_tokens=3)
        cache.attach(cache.root_node, parent, "parent")
        cache.attach(parent, leaf, "leaf")

        first = end_aginfer_program(cache, "p")
        self.assertTrue(first["ok"])
        self.assertEqual(first["released_nodes"], 2)
        self.assertEqual(first["released_hbm_tokens"], 5)
        self.assertEqual(first["released_dram_tokens"], 5)
        self.assertEqual(first["skipped"], [])
        self.assertEqual(cache.eviction_order, [leaf.id, parent.id])
        self.assertEqual(cache.root_node.children, {})
        self.assertEqual(parent.session_ids, set())
        self.assertEqual(leaf.session_ids, set())
        self.assertEqual(cache._aginfer_program_states["p"]["state"], "ENDED")

        second = end_aginfer_program(cache, "p")
        self.assertTrue(second["ok"])
        self.assertEqual(second["status"], "already_absent")
        self.assertEqual(second["released_nodes"], 0)

    def test_shared_prefix_survives_until_last_holder_ends(self):
        cache = _Cache()
        shared = _Node(holders={"p", "q"}, device_tokens=2)
        p_leaf = _Node(holders={"p"}, device_tokens=3)
        cache.attach(cache.root_node, shared, "shared")
        cache.attach(shared, p_leaf, "p-leaf")

        ended_p = end_aginfer_program(cache, "p")
        self.assertEqual(ended_p["released_nodes"], 1)
        self.assertEqual(shared.session_ids, {"q"})
        self.assertIs(cache.root_node.children["shared"], shared)

        ended_q = end_aginfer_program(cache, "q")
        self.assertEqual(ended_q["released_nodes"], 1)
        self.assertEqual(cache.root_node.children, {})

    def test_locked_chain_is_skipped_then_retried(self):
        cache = _Cache()
        parent = _Node(holders={"p"}, device_tokens=2)
        leaf = _Node(holders={"p"}, device_tokens=3)
        leaf.component_data[0].lock_ref = 1
        cache.attach(cache.root_node, parent, "parent")
        cache.attach(parent, leaf, "leaf")

        first = end_aginfer_program(cache, "p")
        self.assertEqual(first["released_nodes"], 0)
        self.assertEqual(
            {entry["reason"] for entry in first["skipped"]},
            {"locked", "child_pending"},
        )
        self.assertEqual(first["remaining_nodes"], 2)
        self.assertEqual(leaf.session_ids, {"p"})

        leaf.component_data[0].lock_ref = 0
        second = end_aginfer_program(cache, "p")
        self.assertEqual(second["released_nodes"], 2)
        self.assertEqual(second["skipped"], [])
        self.assertEqual(cache.root_node.children, {})

    def test_busy_write_through_is_retryable(self):
        cache = _Cache()
        leaf = _Node(holders={"p"}, device_tokens=3, host_tokens=3)
        leaf.write_through_pending_id = leaf.id
        cache.attach(cache.root_node, leaf, "leaf")

        first = end_aginfer_program(cache, "p")
        self.assertEqual(first["skipped"][0]["reason"], "busy_write_through")
        self.assertEqual(leaf.session_ids, {"p"})

        leaf.write_through_pending_id = None
        second = end_aginfer_program(cache, "p")
        self.assertEqual(second["released_nodes"], 1)

    def test_busy_probe_is_non_mutating(self):
        cache = _Cache()
        leaf = _Node(holders={"p"}, device_tokens=3)
        leaf.component_data[0].lock_ref = 1
        cache.attach(cache.root_node, leaf, "leaf")

        blockers = aginfer_program_busy(cache, "p")
        self.assertEqual([entry["reason"] for entry in blockers], ["locked"])
        self.assertEqual(leaf.session_ids, {"p"})
        self.assertIn("leaf", cache.root_node.children)


class AginferSessionEndControlPlaneTest(unittest.TestCase):
    def test_driver_can_end_before_first_pressure_tick(self):
        driver = AginferDriver()
        driver.end_program("p")
        self.assertEqual(driver._tracker.state("p"), State.ENDED)
        self.assertIsNone(driver._policy)

    def test_validator_accepts_session_id_alias(self):
        self.assertEqual(
            validate_session_end_body({"session_id": " session-1 "}),
            ("session-1", "session-1"),
        )
        self.assertEqual(
            validate_session_end_body({"program_id": " program-1 "}),
            ("program-1", "program-1"),
        )
        self.assertEqual(
            validate_session_end_body(
                {"program_id": "program-1", "session_id": "sg-session-1"}
            ),
            ("program-1", "sg-session-1"),
        )
        with self.assertRaises(ValueError):
            validate_session_end_body({"program_id": " "})

    def test_tokenizer_control_returns_all_rank_acks(self):
        self.assertIn(
            ("aginfer_session_end", AginferSessionEndReqOutput),
            _COMMUNICATOR_SPECS,
        )
        outputs = [
            AginferSessionEndReqOutput(
                ok=True,
                program_id="p",
                session_id=None,
                dp_rank=rank,
                status="applied",
            )
            for rank in range(2)
        ]

        class _Communicator:
            async def __call__(self, obj, timeout=None):
                self.obj = obj
                self.timeout = timeout
                return outputs

        manager = SimpleNamespace(
            auto_create_handle_loop=lambda: None,
            aginfer_session_end_communicator=_Communicator(),
        )
        request = AginferSessionEndReq(program_id="p")
        result = asyncio.run(
            TokenizerControlMixin.end_aginfer_session(manager, request)
        )
        self.assertEqual([r.dp_rank for r in result], [0, 1])
        self.assertIsNotNone(manager.aginfer_session_end_communicator.obj.rid)

    def test_tokenizer_control_polls_deferred_until_complete(self):
        deferred = AginferSessionEndReqOutput(
            ok=True,
            program_id="p",
            session_id="p",
            dp_rank=0,
            status="deferred",
            deferred=True,
        )
        complete = AginferSessionEndReqOutput(
            ok=True,
            program_id="p",
            session_id="p",
            dp_rank=0,
            status="applied",
        )

        class _Communicator:
            def __init__(self):
                self.calls = 0

            async def __call__(self, obj, timeout=None):
                self.calls += 1
                self.timeout = timeout
                return [deferred] if self.calls == 1 else [complete]

        communicator = _Communicator()
        manager = SimpleNamespace(
            auto_create_handle_loop=lambda: None,
            aginfer_session_end_communicator=communicator,
        )
        result = asyncio.run(
            TokenizerControlMixin.end_aginfer_session(
                manager, AginferSessionEndReq(program_id="p")
            )
        )
        self.assertEqual(result, [complete])
        self.assertEqual(communicator.calls, 2)
        self.assertEqual(communicator.timeout, 5.0)

    def test_tokenizer_control_retries_timed_out_attempt_with_new_rid(self):
        complete = AginferSessionEndReqOutput(
            ok=True,
            program_id="p",
            session_id="p",
            dp_rank=0,
            status="applied",
        )

        class _Communicator:
            def __init__(self):
                self.rids = []

            async def __call__(self, obj, timeout=None):
                self.rids.append(obj.rid)
                if len(self.rids) == 1:
                    raise asyncio.TimeoutError
                return [complete]

        communicator = _Communicator()
        manager = SimpleNamespace(
            auto_create_handle_loop=lambda: None,
            aginfer_session_end_communicator=communicator,
        )
        result = asyncio.run(
            TokenizerControlMixin.end_aginfer_session(
                manager, AginferSessionEndReq(program_id="p")
            )
        )
        self.assertEqual(result, [complete])
        self.assertEqual(len(communicator.rids), 2)
        self.assertNotEqual(communicator.rids[0], communicator.rids[1])

    def test_scheduler_defers_for_inflight_sglang_session(self):
        cache = _Cache()
        leaf = _Node(holders={"p"}, device_tokens=3)
        cache.attach(cache.root_node, leaf, "leaf")
        sessions = SessionController(cache)
        session = Session(128, "s", streaming=True)
        session._inflight = True
        sessions.sessions["s"] = session

        scheduler = Scheduler.__new__(Scheduler)
        scheduler.tree_cache = cache
        scheduler.session_controller = sessions
        scheduler._aginfer_pending_session_ends = {}
        scheduler._aginfer_driver = None
        scheduler.ps = SimpleNamespace(dp_rank=0)

        ack = Scheduler.end_aginfer_session(
            scheduler,
            AginferSessionEndReq(program_id="p", session_id="s"),
        )
        self.assertTrue(ack.ok)
        self.assertTrue(ack.deferred)
        self.assertEqual(ack.status, "deferred")
        self.assertEqual(ack.reason, "inflight_session")
        self.assertIn("p", scheduler._aginfer_pending_session_ends)
        self.assertIn("leaf", cache.root_node.children)

        session._inflight = False
        sessions.maybe_reap(2.0, interval=0.0)
        Scheduler._aginfer_retry_pending_session_ends(scheduler)
        self.assertNotIn("p", scheduler._aginfer_pending_session_ends)
        self.assertEqual(cache.root_node.children, {})
        self.assertEqual(cache.released_sessions, ["s"])

    def test_session_end_defers_non_streaming_session_without_changing_close(self):
        cache = _Cache()
        sessions = SessionController(cache)
        session = Session(128, "s", streaming=False)
        request = SimpleNamespace(
            finished=lambda: False,
            multimodal_inputs=None,
        )
        session.req_nodes["r"] = SimpleNamespace(req=request)
        sessions.sessions["s"] = session

        scheduler = Scheduler.__new__(Scheduler)
        scheduler.session_controller = sessions
        deferred = Scheduler._aginfer_close_session_for_end(scheduler, "s")
        self.assertTrue(deferred)
        self.assertTrue(session.close_on_finish)
        self.assertIn("s", sessions.sessions)

    def test_scheduler_defers_for_program_request_without_sglang_session(self):
        cache = _Cache()
        leaf = _Node(holders={"p"}, device_tokens=3)
        cache.attach(cache.root_node, leaf, "leaf")
        request = SimpleNamespace(program_id="p", finished=lambda: False)

        scheduler = Scheduler.__new__(Scheduler)
        scheduler.tree_cache = cache
        scheduler.session_controller = SessionController(cache)
        scheduler._aginfer_pending_session_ends = {}
        scheduler._aginfer_driver = None
        scheduler.ps = SimpleNamespace(dp_rank=0)
        scheduler.waiting_queue = [request]

        ack = Scheduler.end_aginfer_session(
            scheduler, AginferSessionEndReq(program_id="p")
        )
        self.assertTrue(ack.deferred)
        self.assertEqual(ack.reason, "inflight_program")
        self.assertIn("leaf", cache.root_node.children)

        request.finished = lambda: True
        Scheduler._aginfer_retry_pending_session_ends(scheduler)
        self.assertEqual(cache.root_node.children, {})

    def test_post_drop_peer_incomplete_keeps_all_ranks_pending(self):
        cache = _Cache()
        leaf = _Node(holders={"p"}, device_tokens=3)
        cache.attach(cache.root_node, leaf, "leaf")

        scheduler = Scheduler.__new__(Scheduler)
        scheduler.tree_cache = cache
        scheduler.session_controller = SessionController(cache)
        scheduler._aginfer_pending_session_ends = {}
        scheduler._aginfer_driver = None
        scheduler.ps = SimpleNamespace(dp_rank=0, pp_size=1)

        # preflight blockers=False, post-drop error=False, but a peer rank
        # reports an incomplete DROP.  Even though this rank removed its
        # local slice, it must retain the pending marker and return deferred.
        global_values = iter((False, True, False, False, True))
        scheduler._aginfer_any_rank = lambda _value: next(global_values)

        ack = Scheduler.end_aginfer_session(
            scheduler, AginferSessionEndReq(program_id="p")
        )
        self.assertTrue(ack.ok)
        self.assertTrue(ack.deferred)
        self.assertEqual(ack.reason, "peer_rank_drop_incomplete")
        self.assertIn("p", scheduler._aginfer_pending_session_ends)

    def test_decode_kvcache_offload_fails_closed_before_mutation(self):
        cache = _Cache()
        leaf = _Node(holders={"p"}, device_tokens=3, host_tokens=2)
        cache.attach(cache.root_node, leaf, "leaf")

        scheduler = Scheduler.__new__(Scheduler)
        scheduler.tree_cache = cache
        scheduler.session_controller = SessionController(cache)
        scheduler._aginfer_pending_session_ends = {}
        scheduler._aginfer_driver = None
        scheduler.decode_offload_manager = object()
        scheduler.ps = SimpleNamespace(dp_rank=0, pp_size=1)

        ack = Scheduler.end_aginfer_session(
            scheduler, AginferSessionEndReq(program_id="p")
        )
        self.assertFalse(ack.ok)
        self.assertEqual(ack.status, "unsupported")
        self.assertEqual(
            ack.reason,
            "decode_kvcache_offload_session_end_not_supported",
        )
        self.assertIn("leaf", cache.root_node.children)

    def test_retry_waits_for_post_drop_all_rank_completion(self):
        cache = _Cache()
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.tree_cache = cache
        scheduler.session_controller = SessionController(cache)
        scheduler._aginfer_pending_session_ends = {"p": "p"}

        global_values = iter((False, False, True))
        scheduler._aginfer_any_rank = lambda _value: next(global_values)
        scheduler._aginfer_drop_program = lambda _program_id: {
            "ok": True,
            "skipped": [],
            "remaining_nodes": 0,
        }

        Scheduler._aginfer_retry_pending_session_ends(scheduler)
        self.assertIn("p", scheduler._aginfer_pending_session_ends)

    def test_background_completion_is_returned_to_next_poll_with_counters(self):
        cache = _Cache()
        leaf = _Node(holders={"p"}, device_tokens=3, host_tokens=2)
        cache.attach(cache.root_node, leaf, "leaf")

        scheduler = Scheduler.__new__(Scheduler)
        scheduler.tree_cache = cache
        scheduler.session_controller = SessionController(cache)
        scheduler._aginfer_pending_session_ends = {"p": "p"}
        scheduler._aginfer_completed_session_ends = {}
        scheduler._aginfer_driver = None
        scheduler.ps = SimpleNamespace(dp_rank=0, pp_size=1)
        scheduler._aginfer_any_rank = lambda value: value

        Scheduler._aginfer_retry_pending_session_ends(scheduler)
        self.assertNotIn("p", scheduler._aginfer_pending_session_ends)
        self.assertIn("p", scheduler._aginfer_completed_session_ends)

        ack = Scheduler.end_aginfer_session(
            scheduler, AginferSessionEndReq(program_id="p")
        )
        self.assertEqual(ack.status, "applied")
        self.assertEqual(ack.released_nodes, 1)
        self.assertEqual(ack.released_hbm_tokens, 3)
        self.assertEqual(ack.released_dram_tokens, 2)
        self.assertNotIn("p", scheduler._aginfer_completed_session_ends)

        duplicate = Scheduler.end_aginfer_session(
            scheduler, AginferSessionEndReq(program_id="p")
        )
        self.assertEqual(duplicate.status, "already_absent")

    def test_partial_completed_cache_hit_converges_through_drop_path(self):
        cache = _Cache()
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.tree_cache = cache
        scheduler.session_controller = SessionController(cache)
        scheduler._aginfer_pending_session_ends = {}
        scheduler._aginfer_completed_session_ends = {
            "p": (
                time.monotonic(),
                {
                    "ok": True,
                    "status": "applied",
                    "released_nodes": 1,
                },
            )
        }
        scheduler._aginfer_driver = None
        scheduler.ps = SimpleNamespace(dp_rank=0, pp_size=1)

        # This rank has a cached completion, but at least one peer does not.
        # All ranks must continue through the same idempotent DROP collectives.
        global_values = iter((True, True, False, False, False))
        scheduler._aginfer_any_rank = lambda _value: next(global_values)

        ack = Scheduler.end_aginfer_session(
            scheduler, AginferSessionEndReq(program_id="p")
        )
        self.assertTrue(ack.ok)
        self.assertEqual(ack.status, "already_absent")
        self.assertNotIn("p", scheduler._aginfer_completed_session_ends)


if __name__ == "__main__":
    unittest.main()

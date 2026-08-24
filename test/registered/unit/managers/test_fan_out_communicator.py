"""Unit tests for the generic scheduler fan-out communicator."""

from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace

from sglang.srt.managers.communicator import FanOutCommunicator
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Sender:
    def __init__(self):
        self.sent = []

    def send_pyobj(self, obj):
        self.sent.append(obj)


class FanOutCommunicatorTest(unittest.IsolatedAsyncioTestCase):
    async def test_queueing_calls_are_serialized(self):
        sender = _Sender()
        communicator = FanOutCommunicator(sender, fan_out=1)

        first_call = asyncio.create_task(communicator("first"))
        await asyncio.sleep(0)
        second_call = asyncio.create_task(communicator("second"))
        await asyncio.sleep(0)

        self.assertEqual(sender.sent, ["first"])
        communicator.handle_recv("first-response")
        self.assertEqual(await first_call, ["first-response"])
        await asyncio.sleep(0)

        self.assertEqual(sender.sent, ["first", "second"])
        communicator.handle_recv("second-response")
        self.assertEqual(await second_call, ["second-response"])

    async def test_cancelled_waiter_does_not_block_following_call(self):
        sender = _Sender()
        communicator = FanOutCommunicator(sender, fan_out=1)

        first_call = asyncio.create_task(communicator("first"))
        await asyncio.sleep(0)
        cancelled_waiter = asyncio.create_task(communicator("cancelled"))
        await asyncio.sleep(0)
        cancelled_waiter.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await cancelled_waiter

        third_call = asyncio.create_task(communicator("third"))
        communicator.handle_recv("first-response")
        self.assertEqual(await first_call, ["first-response"])
        await asyncio.sleep(0)

        self.assertEqual(sender.sent, ["first", "third"])
        communicator.handle_recv("third-response")
        self.assertEqual(await third_call, ["third-response"])

    async def test_correlated_timeout_drops_late_response_and_recovers(self):
        sender = _Sender()
        communicator = FanOutCommunicator(sender, fan_out=1, correlate_by_rid=True)

        with self.assertRaises(asyncio.TimeoutError):
            await communicator(SimpleNamespace(rid="old"), timeout=0.001)

        next_call = asyncio.create_task(
            communicator(SimpleNamespace(rid="new"), timeout=1.0)
        )
        await asyncio.sleep(0)
        communicator.handle_recv(SimpleNamespace(rid="old", value="late"))
        await asyncio.sleep(0)
        self.assertFalse(next_call.done())

        expected = SimpleNamespace(rid="new", value="current")
        communicator.handle_recv(expected)
        self.assertEqual(await next_call, [expected])

    async def test_correlated_cancellation_drops_late_response_and_recovers(self):
        sender = _Sender()
        communicator = FanOutCommunicator(sender, fan_out=1, correlate_by_rid=True)

        cancelled_call = asyncio.create_task(
            communicator(SimpleNamespace(rid="cancelled"))
        )
        await asyncio.sleep(0)
        cancelled_call.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await cancelled_call

        next_call = asyncio.create_task(
            communicator(SimpleNamespace(rid="next"), timeout=1.0)
        )
        await asyncio.sleep(0)
        communicator.handle_recv(SimpleNamespace(rid="cancelled", value="late"))
        self.assertFalse(next_call.done())

        expected = SimpleNamespace(rid="next", value="current")
        communicator.handle_recv(expected)
        self.assertEqual(await next_call, [expected])

    async def test_uncorrelated_timeout_fails_closed(self):
        sender = _Sender()
        communicator = FanOutCommunicator(sender, fan_out=1)

        with self.assertRaises(asyncio.TimeoutError):
            await communicator("first", timeout=0.001)
        communicator.handle_recv("late-response")

        with self.assertRaisesRegex(RuntimeError, "requires restart"):
            await communicator("second")
        self.assertEqual(sender.sent, ["first"])

    async def test_uncorrelated_cancellation_fails_closed(self):
        sender = _Sender()
        communicator = FanOutCommunicator(sender, fan_out=1)

        call = asyncio.create_task(communicator("first"))
        await asyncio.sleep(0)
        call.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await call

        with self.assertRaisesRegex(RuntimeError, "requires restart"):
            await communicator("second")
        self.assertEqual(sender.sent, ["first"])

    async def test_correlated_request_requires_rid_without_poisoning_state(self):
        sender = _Sender()
        communicator = FanOutCommunicator(sender, fan_out=1, correlate_by_rid=True)

        with self.assertRaisesRegex(ValueError, "must have a rid"):
            await communicator(SimpleNamespace())
        self.assertEqual(sender.sent, [])

        valid_call = asyncio.create_task(
            communicator(SimpleNamespace(rid="valid"), timeout=1.0)
        )
        await asyncio.sleep(0)
        response = SimpleNamespace(rid="valid")
        communicator.handle_recv(response)
        self.assertEqual(await valid_call, [response])


if __name__ == "__main__":
    unittest.main()

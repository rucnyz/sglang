from __future__ import annotations

import asyncio
import copy
from typing import Generic, List, Optional, TypeVar

import zmq

T = TypeVar("T")


class FanOutCommunicator(Generic[T]):
    """Fan-out request + collect response primitive over zmq.

    One send is fanned out to `fan_out` recipients; the caller awaits until
    all `fan_out` responses are collected. Supports two modes:
    - "queueing": requests are serialized; concurrent callers wait in a FIFO queue.
    - "watching": concurrent callers share a single in-flight request and all
      receive the same result when it completes.

    Only one request is in-flight at any time in either mode.
    """

    def __init__(
        self,
        sender: zmq.Socket,
        fan_out: int,
        mode="queueing",
        *,
        correlate_by_rid: bool = False,
    ):
        self._sender = sender
        self._fan_out = fan_out
        self._mode = mode
        self._result_event: Optional[asyncio.Event] = None
        self._result_values: Optional[List[T]] = None
        self._queue_lock = asyncio.Lock()
        self._queueing_broken_reason: Optional[str] = None
        self._correlate_by_rid = correlate_by_rid
        self._result_rid = None

        assert mode in ["queueing", "watching"]

    async def queueing_call(self, obj: T, timeout: Optional[float] = None):
        """Serialize a fan-out call and leave cancellation in a safe state.

        Uncorrelated control messages fail closed after timeout/cancellation,
        because a late response could otherwise be mistaken for the next
        request.  Callers that opt into ``correlate_by_rid`` can safely recover:
        late responses are discarded unless their ``rid`` matches the current
        request.
        """

        async with self._queue_lock:
            if self._queueing_broken_reason is not None:
                raise RuntimeError(self._queueing_broken_reason)
            assert self._result_event is None
            assert self._result_values is None

            request_rid = getattr(obj, "rid", None)
            if self._correlate_by_rid and request_rid is None:
                raise ValueError("correlated fan-out request must have a rid")

            event = asyncio.Event()
            self._result_event = event
            self._result_values = []
            self._result_rid = request_rid
            try:
                if obj is not None:
                    self._sender.send_pyobj(obj)
                if timeout is None:
                    await event.wait()
                else:
                    await asyncio.wait_for(event.wait(), timeout=timeout)
                return self._result_values
            except asyncio.TimeoutError:
                if not self._correlate_by_rid:
                    self._queueing_broken_reason = (
                        "fan-out request timed out; communicator requires restart"
                    )
                raise
            except asyncio.CancelledError:
                if not self._correlate_by_rid:
                    self._queueing_broken_reason = (
                        "fan-out request was cancelled; communicator requires restart"
                    )
                raise
            finally:
                self._result_event = self._result_values = None
                self._result_rid = None

    async def watching_call(self, obj):
        if self._result_event is None:
            assert self._result_values is None
            self._result_values = []
            self._result_event = asyncio.Event()

            if obj is not None:
                self._sender.send_pyobj(obj)

        # Capture local refs before await -- after event fires, the first
        # awakened coroutine clears shared state; later awaiters use local refs.
        values = self._result_values
        event = self._result_event
        await event.wait()

        result_values = copy.deepcopy(values)
        if self._result_event is event:
            self._result_event = self._result_values = None
        return result_values

    async def __call__(self, obj, timeout: Optional[float] = None):
        if self._mode == "queueing":
            return await self.queueing_call(obj, timeout=timeout)
        else:
            if timeout is not None:
                raise ValueError("timeout is only supported in queueing mode")
            return await self.watching_call(obj)

    def handle_recv(self, recv_obj: T):
        # A timed-out/cancelled queueing call is intentionally fail-closed.
        # Its uncorrelated late responses cannot be reused by another call.
        if self._result_values is None or self._result_event is None:
            return
        if (
            self._correlate_by_rid
            and getattr(recv_obj, "rid", None) != self._result_rid
        ):
            return
        self._result_values.append(recv_obj)
        if len(self._result_values) == self._fan_out:
            self._result_event.set()

    @staticmethod
    def merge_results(results):
        all_success = all([r.success for r in results])
        all_message = [r.message for r in results]
        all_message = " | ".join(all_message)
        return all_success, all_message

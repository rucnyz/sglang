"""Unit test for the overlap-compatible token-forcing core logic.

``forced_override_positions(reqs)`` is the pure bookkeeping half:
given the batch's reqs, it returns ``(batch_index, forced_token)`` for every req
that commits a token THIS batch and is still teacher-forcing, advancing each
req's ``forced_dispatched`` counter. The scheduler's
``_apply_forced_tokens`` then scatters those onto the GPU ``next_token_ids``
before the future-buffer stash.

This pins the bookkeeping (which reqs, which token, counter advance) without a
model: the commit-eligibility filter mirrors ``process_batch_result_*``
(skip finished / retracted / the batch's still-chunking req), the counter indexes
``forced_output_ids`` and advances once per dispatched token, and it stops at the
end of the forced sequence.

Run: python dev/aginfer/verify/forced_tokens/verify.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../python"))

from sglang.srt.managers.forced_tokens import forced_override_positions


class _SP:
    def __init__(self, forced):
        self.custom_params = {"forced_output_ids": forced} if forced is not None else None


class FakeReq:
    def __init__(self, forced=None, inflight_middle_chunks=0, finished=False,
                 retracted=False, dispatched=0):
        self.sampling_params = _SP(forced)
        # Kept only to prove the override no longer consults this lagging counter.
        self.inflight_middle_chunks = inflight_middle_chunks
        self._finished = finished
        self.is_retracted = retracted
        self.forced_dispatched = dispatched

    def finished(self):
        return self._finished


def test_basic_advance_and_index():
    r = FakeReq(forced=[10, 11, 12])
    assert forced_override_positions([r]) == [(0, 10)] and r.forced_dispatched == 1
    assert forced_override_positions([r]) == [(0, 11)] and r.forced_dispatched == 2
    assert forced_override_positions([r]) == [(0, 12)] and r.forced_dispatched == 3
    assert forced_override_positions([r]) == [] and r.forced_dispatched == 3
    print("  PASS  basic: advances, indexes forced[pos], stops at end")


def test_skips_ineligible_and_keeps_counter():
    chunked = FakeReq(forced=[10])
    fin = FakeReq(forced=[10], finished=True)
    retr = FakeReq(forced=[10], retracted=True)
    noforce = FakeReq(forced=None)
    reqs = [chunked, fin, retr, noforce]
    assert forced_override_positions(reqs, chunked) == []
    assert chunked.forced_dispatched == 0 and fin.forced_dispatched == 0
    assert retr.forced_dispatched == 0 and noforce.forced_dispatched == 0
    print("  PASS  skips chunked / finished / retracted / non-forced, no advance")


def test_mixed_batch_indices():
    a = FakeReq(forced=[100, 101])
    b = FakeReq(forced=[200])
    c = FakeReq(forced=[300, 301], dispatched=1)
    out = forced_override_positions([a, b, c], b)
    assert out == [(0, 100), (2, 301)], out
    assert a.forced_dispatched == 1 and b.forced_dispatched == 0 and c.forced_dispatched == 2
    print("  PASS  mixed batch: correct (index, token), only eligible advance")


def test_last_chunk_is_forced_despite_stale_counter():
    """Regression: the last prefill chunk commits a token, so it must be forced.

    Under overlap the batch-RESULT processor has not yet decremented
    ``inflight_middle_chunks`` when the last chunk is dispatched, so filtering on
    that counter dropped forced[0] and leaked the model's own first token.
    ``batch.chunked_req`` is None for the batch that finishes the prefill.
    """
    r = FakeReq(forced=[10, 11], inflight_middle_chunks=1)
    assert forced_override_positions([r], None) == [(0, 10)]
    assert r.forced_dispatched == 1
    print("  PASS  last chunk forced even while inflight_middle_chunks is stale")


def test_decode_batch_ignores_stale_chunked_req():
    """Regression: ``chunked_req`` is meaningless once the batch is decoding.

    When the running batch is empty the scheduler adopts the prefill batch object
    itself (``running_batch = last_batch``) and nothing clears ``chunked_req``, so
    the req that was mid-chunk in that batch kept matching ``req is chunked_req``
    on every later decode step — it was silently skipped and free-ran its entire
    output while the trace expected 2400 forced tokens.
    """
    r = FakeReq(forced=[10, 11, 12], dispatched=1)
    assert forced_override_positions([r], r, is_extend=False) == [(0, 11)]
    assert r.forced_dispatched == 2
    # still honoured while the batch really is prefilling that req
    assert forced_override_positions([r], r, is_extend=True) == []
    assert r.forced_dispatched == 2
    print("  PASS  decode batch ignores a stale chunked_req")


def test_retract_resync_semantics():
    r = FakeReq(forced=[10, 11, 12, 13], dispatched=3)
    assert forced_override_positions([r]) == [(0, 13)] and r.forced_dispatched == 4
    print("  PASS  resumes from a resynced counter at the right position")


def test_empty_forced_list():
    r = FakeReq(forced=[])
    assert forced_override_positions([r]) == []
    assert r.forced_dispatched == 0
    print("  PASS  empty forced list is a no-op")


def test_no_custom_params():
    class BareReq:
        def __init__(self):
            self.sampling_params = type("SP", (), {"custom_params": None})()
            self.inflight_middle_chunks = 0
            self.is_retracted = False
            self.forced_dispatched = 0
        def finished(self): return False
    assert forced_override_positions([BareReq()]) == []
    print("  PASS  no custom_params is a no-op")


def main():
    tests = [
        test_basic_advance_and_index,
        test_skips_ineligible_and_keeps_counter,
        test_mixed_batch_indices,
        test_last_chunk_is_forced_despite_stale_counter,
        test_decode_batch_ignores_stale_chunked_req,
        test_retract_resync_semantics,
        test_empty_forced_list,
        test_no_custom_params,
    ]
    print(f"\nforced_override_positions unit (n={len(tests)}):")
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            import traceback
            traceback.print_exc()
    ok = passed == len(tests)
    print(f"\nforced_tokens: {passed}/{len(tests)} {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

"""Unit test for the overlap-compatible token-forcing core logic (task #334).

`forced_override_positions(reqs)` is the pure bookkeeping half of candidate A:
given the batch's reqs, it returns `(batch_index, forced_token)` for every req
that commits a token THIS batch and is still teacher-forcing, advancing each
req's `forced_dispatched` counter. The scheduler's
`_apply_forced_tokens` then scatters those onto the GPU `next_token_ids`
before the future-buffer stash.

This pins the bookkeeping (which reqs, which token, counter advance) without a
model: the commit-eligibility filter mirrors `process_batch_result_*`
(skip finished / retracted / chunked), the counter indexes
`forced_output_ids` and advances once per dispatched token, and it stops at the
end of the forced sequence. The GPU scatter + true byte-identical behaviour are
gated separately by the e2e (overlap-on forced output == overlap-off).

Run: .venv/bin/python dev/interlayer/4_overlap_forcing/test_forced_override.py
"""
import sys

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")

from sglang.srt.managers.forced_tokens import (  # noqa: E402
    forced_override_positions,
)


class _SP:
    def __init__(self, forced):
        self.custom_params = {"forced_output_ids": forced} if forced is not None else None


class FakeReq:
    """Minimal stand-in exposing exactly what forced_override_positions reads."""
    def __init__(self, forced=None, inflight_middle_chunks=0, finished=False,
                 retracted=False, dispatched=0):
        self.sampling_params = _SP(forced)
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
    # past the end of forced -> no override, counter frozen
    assert forced_override_positions([r]) == [] and r.forced_dispatched == 3
    print("  PASS  basic: advances, indexes forced[pos], stops at end")


def test_skips_ineligible_and_keeps_counter():
    chunked = FakeReq(forced=[10], inflight_middle_chunks=2)
    fin = FakeReq(forced=[10], finished=True)
    retr = FakeReq(forced=[10], retracted=True)
    noforce = FakeReq(forced=None)
    assert forced_override_positions([chunked, fin, retr, noforce]) == []
    # none advanced
    assert chunked.forced_dispatched == 0 and fin.forced_dispatched == 0
    assert retr.forced_dispatched == 0 and noforce.forced_dispatched == 0
    print("  PASS  skips chunked / finished / retracted / non-forced, no advance")


def test_mixed_batch_indices():
    a = FakeReq(forced=[100, 101])           # eligible, idx 0
    b = FakeReq(forced=[200], inflight_middle_chunks=1)  # chunked, skipped
    c = FakeReq(forced=[300, 301], dispatched=1)  # eligible at pos 1, idx 2
    out = forced_override_positions([a, b, c])
    assert out == [(0, 100), (2, 301)], out
    assert a.forced_dispatched == 1 and b.forced_dispatched == 0 and c.forced_dispatched == 2
    print("  PASS  mixed batch: correct (index, token), only eligible advance")


def test_retract_resync_semantics():
    """The counter is reset to len(output_ids) on retract (done in Req); here we
    just confirm that resuming from a resynced counter indexes correctly."""
    r = FakeReq(forced=[10, 11, 12, 13], dispatched=3)  # as if 3 committed, resynced
    assert forced_override_positions([r]) == [(0, 13)] and r.forced_dispatched == 4
    print("  PASS  resumes from a resynced counter at the right position")


def main():
    tests = [
        test_basic_advance_and_index,
        test_skips_ineligible_and_keeps_counter,
        test_mixed_batch_indices,
        test_retract_resync_semantics,
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
    print(f"\nforced override: {passed}/{len(tests)} passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())

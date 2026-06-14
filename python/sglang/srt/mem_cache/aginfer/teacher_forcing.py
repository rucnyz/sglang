"""aginfer teacher-forcing (wherewewin/harness/teacher_forcing) — self-contained.

The scheduler's output-commit hot path carries only thin hooks (``force_token`` /
``validate_no_forced_output_with_spec``); the forced-replay logic lives here
(#251: thin hook in core, fat self-contained module).
"""
from __future__ import annotations


def force_token(req, sampled):
    """If ``req`` carries ``sampling_params.custom_params["forced_output_ids"]``,
    return the forced token for the current step instead of the sampled one, so a
    replay reproduces a captured decode EXACTLY (faithful multi-turn KV
    continuation).

    Done at the authoritative output-commit point (not via a logit processor,
    which adds ~30% under the overlap scheduler) so it is a true no-op: the next
    forward's ``input_ids`` is built from ``req.output_ids[-1]``, so overriding the
    committed token propagates with no extra tensor write. The per-req step counter
    is overlap-safe (one commit per token, in order). Returns ``sampled`` unchanged
    when no forcing is configured (single cheap branch off the hot path).
    """
    sp = getattr(req, "sampling_params", None)
    cp = getattr(sp, "custom_params", None) if sp is not None else None
    if not cp:
        return sampled
    forced = cp.get("forced_output_ids")
    if not forced:
        return sampled
    step = getattr(req, "_tf_step", 0)
    if step >= len(forced):
        return sampled
    req._tf_step = step + 1
    return int(forced[step])


def validate_no_forced_output_with_spec(req):
    """Teacher-forcing is NOT applied on the speculative-decode path (multi-token
    accept). Forced replay must not silently degrade to length-only there, so
    reject the combination loudly."""
    sp = getattr(req, "sampling_params", None)
    cp = getattr(sp, "custom_params", None) if sp is not None else None
    if cp and cp.get("forced_output_ids"):
        raise NotImplementedError(
            "aginfer forced_output_ids (teacher-forcing) is not supported with "
            "speculative decoding; disable spec for faithful replay."
        )

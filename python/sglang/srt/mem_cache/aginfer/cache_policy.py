"""aginfer cache-policy framework (refactor #251 Stage A.2).

Pluggable eviction-scorer + write-through-trigger env-symbol loaders, the
LRU-equivalent default scorer (byte-for-byte stock sglang when unset), and
the birth-seed constants. Extracted from unified_radix_cache.py so the
upstream cache file carries only a thin re-import hook. Identity of
`_default_eviction_score` is preserved across the re-import."""
import logging

logger = logging.getLogger("sglang.srt.mem_cache.unified_radix_cache")

# --- aginfer: pluggable eviction scorer -------------------------------------
# When set, replaces the default LRU heap key (= node.last_access_time) used by
# component drive_eviction / drive_host_eviction.  Lower score -> evict first.
#
# Wired via the env var SGLANG_KV_POLICY_MODULE="pkg.module:callable".  The
# callable signature is (node: UnifiedTreeNode, layer: EvictLayer) -> float.
# The default fallback below preserves stock sglang LRU behaviour.
import os
import importlib


# #177 (T38 follow-on, DESIGN §3 "one code path"): the in-process
# eviction DEFAULT is the LRU-equivalent V_u — bare last_access_time
# ("last_access as p_hat surrogate", DESIGN §3).  This is byte-for-byte
# stock sglang LRU AND literally the same function as the daemon-side
# default policy module
# (dev/aginfer/baselines/sglang_adapter.py:default_policy_score), so
# "aginfer disabled" and "aginfer default policy" are one code path —
# baseline-vs-ours ablations flip a policy parameter, not a path.
#
# hit_count is deliberately NOT used here: DESIGN §3 places hit_count in
# the WRITE-THROUGH trigger (_default_should_write_through / #178), not
# in eviction ordering.  (An earlier attempt at a `+ hit_count·2^-50`
# eviction tie-break was both non-functional — the bonus is below the
# float64 ULP at any realistic last_access_time — and near-pointless:
# the match path stamps ancestor nodes at cur_time, cur_time-1e-5, …
# (see update path), so exact last_access_time ties are effectively
# absent for realistic counter values.  At extreme counter magnitudes
# (≳2^40 cumulative accesses, where the 1e-5 spacing itself falls below
# the ULP) ancestor nodes DO tie — but there stock-sglang bare LRU ties
# arbitrarily too, so this is no regression vs stock.  See #177.)
# verify/t28 stage A3 is the cross-tree drift guard that pins this ==
# the adapter's default_policy_score.
def _default_eviction_score(node, layer) -> float:
    return float(node.last_access_time)


def _default_should_write_through(node, threshold) -> bool:
    """#178 (T28, DESIGN §3 write-through plugin) DEFAULT: the
    historical ``hit_count >= write_through_threshold`` trigger.
    Preserves stock sglang behaviour exactly when no daemon-/env-
    supplied policy is registered."""
    return node.hit_count >= threshold


def _load_eviction_scorer():
    """Resolve the inline scorer per SGLANG_KV_POLICY_MODULE.

    Always emits a single canonical startup line ``kv_policy_loaded=<spec>``
    so the T9 startup-invariant grep can detect (a) which scorer is
    active and (b) failed loads (which silently fall back to LRU).

    Accepted ``kv_policy_loaded`` values for T9 to match:
      * ``default_lru`` — env var unset (stock sglang behavior)
      * ``<module>:<callable>`` — env var set + loaded successfully
      * ``default_lru (load_failed:<reason>)`` — env var set but load
        failed; T9 should HALT the run on this value
    """
    spec = os.environ.get("SGLANG_KV_POLICY_MODULE", "").strip()
    if not spec:
        logger.info("[aginfer] kv_policy_loaded=default_lru")
        return _default_eviction_score
    if ":" not in spec:
        logger.warning(
            "[aginfer] kv_policy_loaded=default_lru "
            "(load_failed:malformed_spec=%r; expected module:callable)",
            spec,
        )
        return _default_eviction_score
    mod_name, attr = spec.split(":", 1)
    try:
        mod = importlib.import_module(mod_name)
        fn = getattr(mod, attr)
        logger.info("[aginfer] kv_policy_loaded=%s", spec)
        return fn
    except Exception as e:
        logger.warning(
            "[aginfer] kv_policy_loaded=default_lru "
            "(load_failed:%r exception=%s)",
            spec,
            e,
        )
        return _default_eviction_score


def _load_write_through_policy():
    """#178 (T28, DESIGN §3): resolve the write-through trigger per
    ``SGLANG_WRITE_THROUGH_MODULE="pkg.module:callable"``.  The callable
    signature is ``(node: UnifiedTreeNode, threshold: int) -> bool`` —
    True = create the host backup now.  Mirrors ``_load_eviction_scorer``
    exactly: emits one canonical ``write_through_loaded=<spec>`` startup
    line (T9 grep), and any failure falls back to the historical
    ``hit_count >= threshold`` default (logged as a load_failed so T9
    can HALT a misconfigured run rather than run baseline silently).

    Aginfer registers a V_u-aware version (fires when
    ``V_u(residence ∪ {DRAM}) > V_u(residence)``, DESIGN §3) once the
    in-process hint-table consumer exists; until then the default is
    the only policy and behaviour is identical to stock sglang."""
    spec = os.environ.get("SGLANG_WRITE_THROUGH_MODULE", "").strip()
    if not spec:
        logger.info("[aginfer] write_through_loaded=default_hitcount")
        return _default_should_write_through
    if ":" not in spec:
        logger.warning(
            "[aginfer] write_through_loaded=default_hitcount "
            "(load_failed:malformed_spec=%r; expected module:callable)",
            spec,
        )
        return _default_should_write_through
    mod_name, attr = spec.split(":", 1)
    try:
        mod = importlib.import_module(mod_name)
        fn = getattr(mod, attr)
        logger.info("[aginfer] write_through_loaded=%s", spec)
        return fn
    except Exception as e:
        logger.warning(
            "[aginfer] write_through_loaded=default_hitcount "
            "(load_failed:%r exception=%s)",
            spec,
            e,
        )
        return _default_should_write_through


# T27 (#188): the sentinel SGLANG_KV_POLICY_MODULE value that selects the
# aginfer hint-AWARE eviction scorer (a cache-bound method reading
# _aginfer_hints, not a free module:callable).  Distinct from
# `baselines.sglang_adapter:ours_greedy_score` (hint-UNAWARE, derives
# p_hat from hits/age) and from default_lru.
_AGINFER_HINT_SCORER_SPEC = "aginfer:hint_v_u"
# Write-through twin of the eviction sentinel (SGLANG_WRITE_THROUGH_MODULE):
# selects the cache-bound hint-aware write-through trigger (#178).
_AGINFER_WRITE_THROUGH_SPEC = "aginfer:hint_write_through"
# Birth-seed lambda for a newborn unit ("near-term expected-use", DESIGN
# §6): the unit was just created so a reuse is plausibly imminent.  Same
# order as the daemon's ACTING lambda default.
_AGINFER_BIRTH_LAMBDA = 0.2
# Birth-seed p_hat for a newborn unit: a NEUTRAL-LOW reuse prior, NOT 1.0.  An
# unproven newborn (zero demonstrated reuse) must not tie a heavily-reused unit
# at p_hat=1.0 — that seed let a fresh one-shot flood prefix out-rank a hot
# reused prefix during the pre-hint window (V_u collapses to size).  A low prior
# means a proven reused unit (daemon hint ->1.0, or the inline reuse proxy)
# out-ranks it; the unit's own generating tail is protected by the inflight gate,
# not by this seed.  Overwrite-by-stamp lets the daemon's real estimate replace
# it as soon as it lands (the seed only governs the pre-hint window).
_AGINFER_BIRTH_PHAT = 0.1
# Birth-seed STAMP — strictly below any real daemon stamp (the daemon's
# stamp is int(time_counter) >= 1, the counter starts at 1.0).  The
# birth seed is a FLOOR; overwrite-by-stamp in set_aginfer_hints skips
# on `stamp <= existing`, so a real-clock stamp here would let the seed
# SHADOW the daemon's first refinement if the counter hadn't advanced
# between birth and that unit's first dump (#188 audit C7).  -1 makes
# every real daemon push strictly win.
_AGINFER_BIRTH_STAMP = -1
# ---------------------------------------------------------------------------

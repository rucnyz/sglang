"""Tier-1 controlled eviction simulator (#230).

Deterministic discrete-event replay of the radix-cache eviction layer, to
characterize — without a GPU — how eviction quality depends on policy
(LRU vs hint-steered vs const-V_u), hint freshness (the latency budget),
imperative migrate, pressure, and reuse structure.

Models, honestly, the SCHEDULING LOGIC + TIMING, not GPU kernels:
  * L1 inline eviction: when an allocation needs space and the pool is
    full, evict lowest-score UNLOCKED units until it fits (zero latency).
  * L2 hint steering: the scorer reads a hint table whose values lag the
    ground truth by ``hint_delay_steps`` (the daemon push latency).  The
    crux: a parked program's true reuse probability RISES as its tool-
    return approaches; a stale hint lags that rise, so eviction in the lag
    window uses the old low score and evicts an about-to-return prefix.
  * L3 imperative migrate: a periodic daemon pass that proactively demotes
    the lowest-true-value resident units when occupancy exceeds a
    threshold, at ``migrate_latency_steps`` latency (off by default per arm).
  * lock protection: a unit actively decoding/prefilling this step is
    locked and cannot be evicted (sglang lock_ref).

The freshness knee in re-prefill-vs-delay is the hint-latency budget; its
physical meaning is the lead time between "reuse becomes predictable" and
"reuse happens".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# --------------------------------------------------------------- model


@dataclass
class Unit:
    """One program's resident context prefix in the HBM pool."""
    pid: str
    n_tokens: int
    last_access: int           # step of last access (LRU key)
    locked: bool = False       # actively decoding/prefilling this step
    resident: bool = True


@dataclass
class Program:
    """A synthetic agent: rounds of prefill → decode → tool-gap → return."""
    pid: str
    n_tokens: int              # context size (grows little; treat as fixed)
    arrive: int
    decode_steps: int          # decode length per round
    tool_gap: int              # steps parked in a tool between rounds
    rounds: int                # how many tool round-trips (reuse events)
    # p_hat rises over the last ``phat_lead`` steps before a return from
    # low_phat to 1.0 — the predictable-reuse lead time.
    phat_lead: int = 8
    low_phat: float = 0.05


# ------------------------------------------------------- workload gen


def make_workload(
    *,
    n_programs: int,
    n_tokens: int,
    decode_steps: int,
    tool_gap: int,
    rounds: int,
    arrival_spread: int,
    phat_lead: int = 8,
    seed_phase: int = 0,
) -> List[Program]:
    """Phased agents so that at any time some contexts are parked
    (evictable) and some are about-to-return (reuse-imminent)."""
    progs: List[Program] = []
    for i in range(n_programs):
        # Deterministic phase offset (no RNG — reproducible).
        arrive = (i * arrival_spread + seed_phase * 3) % max(1, arrival_spread * n_programs)
        progs.append(Program(
            pid=f"p{i}", n_tokens=n_tokens, arrive=arrive,
            decode_steps=decode_steps, tool_gap=tool_gap, rounds=rounds,
            phat_lead=phat_lead,
        ))
    return progs


# ------------------------------------------------------------- engine


@dataclass
class Metrics:
    reprefill_tokens: int = 0
    reprefill_events: int = 0
    reuse_events: int = 0           # total tool-returns (hit or miss)
    evict_imminent: int = 0         # evicted a unit returning within phat_lead
    inline_evict_tokens: int = 0    # tokens freed by L1 inline eviction
    migrate_demote_tokens: int = 0  # tokens proactively demoted by L3
    cache_hits: int = 0             # tool-returns that found prefix resident

    def hit_rate(self) -> float:
        return self.cache_hits / self.reuse_events if self.reuse_events else 1.0


# Per-program schedule: list of (step, kind) where kind in
# {"prefill","decode_end","tool_return","end"}.  Built by the engine.


def simulate(
    programs: List[Program],
    *,
    pool_cap_tokens: int,
    policy: str,                    # "lru" | "ours" | "const"
    hint_delay_steps: int = 0,
    migrate: bool = False,
    migrate_period: int = 4,
    migrate_latency_steps: int = 10,
    migrate_occ_threshold: float = 0.85,
    horizon: Optional[int] = None,
) -> Metrics:
    """Replay the workload under one arm; return outcome metrics.

    Deterministic: no RNG.  ``policy`` selects the L1 eviction score;
    ``hint_delay_steps`` lags the L2 hint table behind ground truth.
    """
    m = Metrics()
    units: Dict[str, Unit] = {}
    used = 0

    # ---- build the per-program event schedule + return-time index -------
    # return_step[pid] = sorted list of steps at which pid returns from a
    # tool (re-accesses its prefix).  Used both to drive events and to
    # compute the ground-truth rising p_hat.
    return_steps: Dict[str, List[int]] = {}
    end_step: Dict[str, int] = {}
    prefill_step: Dict[str, int] = {}
    for p in programs:
        t = p.arrive
        prefill_step[p.pid] = t
        rs: List[int] = []
        for _ in range(p.rounds):
            t += p.decode_steps          # decode this round
            t += p.tool_gap              # parked in tool
            rs.append(t)                 # returns here (re-access)
        return_steps[p.pid] = rs
        end_step[p.pid] = t + p.decode_steps
    if horizon is None:
        horizon = max(end_step.values()) + 2

    by_pid = {p.pid: p for p in programs}

    # ---- ground-truth p_hat: rises over phat_lead before next return ----
    def true_phat(pid: str, t: int) -> float:
        p = by_pid[pid]
        nxt = None
        for r in return_steps[pid]:
            if r >= t:
                nxt = r
                break
        if nxt is None:
            return 0.0               # no more reuse → safe to evict
        lead = nxt - t
        if lead <= 0:
            return 1.0
        if lead >= p.phat_lead:
            return p.low_phat
        # linear rise low_phat → 1.0 over the last phat_lead steps
        return p.low_phat + (1.0 - p.low_phat) * (1 - lead / p.phat_lead)

    # ---- hint table: value seen by the scorer lags truth by the delay ---
    # hint_table[pid] = p_hat the daemon last *delivered*.  We deliver the
    # truth-at-(t-delay) at step t (so a delay of D means the scorer at t
    # sees the ground truth as of t-D).
    hint_table: Dict[str, float] = {}

    def deliver_hints(t: int) -> None:
        src_t = t - hint_delay_steps
        if src_t < 0:
            return
        for pid in units:
            hint_table[pid] = true_phat(pid, src_t)

    # ---- eviction score (lower = evicted first) -------------------------
    def evict_score(u: Unit, t: int) -> float:
        if policy == "lru":
            return float(u.last_access)            # oldest first
        if policy == "const":
            return 0.0                             # tie-break by last_access
        # "ours": evict lowest hinted reuse prob; tie-break by recency.
        return hint_table.get(u.pid, 0.0)

    def free_space(need: int, t: int) -> bool:
        nonlocal used
        if used + need <= pool_cap_tokens:
            return True
        # evict unlocked units by ascending (score, last_access) until fit.
        cands = [u for u in units.values() if u.resident and not u.locked]
        cands.sort(key=lambda u: (evict_score(u, t), u.last_access))
        for u in cands:
            if used + need <= pool_cap_tokens:
                break
            # is this unit reuse-imminent? (returns within phat_lead)
            if true_phat(u.pid, t) >= by_pid[u.pid].low_phat + 0.5:
                m.evict_imminent += 1
            u.resident = False
            used -= u.n_tokens
            m.inline_evict_tokens += u.n_tokens
            del units[u.pid]
        return used + need <= pool_cap_tokens

    def allocate(pid: str, n: int, t: int) -> None:
        nonlocal used
        free_space(n, t)             # best-effort (oversub just over-evicts)
        units[pid] = Unit(pid=pid, n_tokens=n, last_access=t, locked=True)
        used += n

    # imperative migrate book-keeping
    migrate_pending: List[Tuple[int, str]] = []   # (apply_step, pid)

    # ---- main loop ------------------------------------------------------
    for t in range(horizon):
        # lock state: a program is locked while prefilling or decoding.
        for u in units.values():
            u.locked = False

        # 1. prefills (arrival): allocate + lock.
        for p in programs:
            if prefill_step[p.pid] == t:
                allocate(p.pid, p.n_tokens, t)

        # 2. tool-returns: re-access prefix (hit or re-prefill miss).
        for pid, rs in return_steps.items():
            if t in rs:
                m.reuse_events += 1
                u = units.get(pid)
                if u is not None and u.resident:
                    m.cache_hits += 1
                    u.last_access = t
                    u.locked = True
                else:
                    # evicted → re-prefill the whole context.
                    m.reprefill_tokens += by_pid[pid].n_tokens
                    m.reprefill_events += 1
                    allocate(pid, by_pid[pid].n_tokens, t)

        # 3. decode: programs in a decode window are locked + touch LRU.
        for p in programs:
            # in a decode window if between a (prefill|return) and the next
            # tool-gap.  Approximate: locked if last_access within
            # decode_steps and not parked.  Cheap proxy: refresh last_access
            # for residents that aren't currently parked.
            u = units.get(p.pid)
            if u is None or not u.resident:
                continue
            # parked if the nearest upcoming return is > 0 and we are inside
            # a tool gap (between last decode end and the return).
            nxt = next((r for r in return_steps[p.pid] if r >= t), None)
            parked = nxt is not None and (nxt - t) < p.tool_gap and (nxt - t) > 0
            if not parked:
                u.locked = True
                u.last_access = t

        # 4. L2 hint delivery (after this step's state settles).
        if policy == "ours":
            deliver_hints(t)

        # 5. L3 imperative migrate (proactive demote of low-value residents).
        if migrate:
            # apply matured demotes
            still: List[Tuple[int, str]] = []
            for ap, pid in migrate_pending:
                if ap == t:
                    u = units.get(pid)
                    if u is not None and u.resident and not u.locked \
                            and true_phat(pid, t) < 0.2:
                        u.resident = False
                        used -= u.n_tokens
                        m.migrate_demote_tokens += u.n_tokens
                        del units[pid]
                elif ap > t:
                    still.append((ap, pid))
            migrate_pending = still
            if t % migrate_period == 0 and used > migrate_occ_threshold * pool_cap_tokens:
                # schedule a demote of the lowest-true-value unlocked unit.
                cands = [u for u in units.values()
                         if u.resident and not u.locked]
                cands.sort(key=lambda u: true_phat(u.pid, t))
                if cands and true_phat(cands[0].pid, t) < 0.2:
                    migrate_pending.append((t + migrate_latency_steps, cands[0].pid))

    return m

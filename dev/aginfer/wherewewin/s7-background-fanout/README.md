# S7 — Background fan-out: proactive room-making for a predictable spike

**Claim:** a background sub-agent fan-out (`run_in_background:true`) launches K new
concurrent active programs at a **known instant**. We see the K
`SUB_DISPATCH_ASYNC` events and **make HBM room before the spike lands** (demote
idle tails, value-pause lowest-value programs), avoiding the thrash that a reactive
cache hits when the spike arrives.

## The situation (workload)
A parent spawns K sub-agents in the background and **keeps working** — so the
parent stays active AND K fresh children become active concurrently: a sharp,
**predictable** concurrency/HBM-demand step. (Contrast S6: blocking → parent idle;
background → parent active + K more.)

Construct with Claude Code: a parent that fires a background `Agent(...)` fan-out
of K and continues. Knob: K (spike size), child footprints, base load, pool
headroom.

## What our framework does
On the burst of `SUB_DISPATCH_ASYNC` events, forecast the imminent demand step and
**proactively** free HBM ahead of it — demote currently-idle tails and/or
value-pause the lowest-value programs — so the K children are admitted onto
already-available HBM instead of triggering reactive eviction/preemption mid-spike.

## Why we win
**Forecast + proactivity.** We know the spike is coming (the dispatch events
precede the children's first prefills) and prepare; the baseline only reacts once
HBM is already over. Avoiding a reactive eviction storm at the moment of the spike
reduces the latency hit. Metric: **p99 / TTFT of requests around the spike** and
whether a thrash (preempt/evict storm) occurs.

## Why vanilla sglang+HiCache cannot
It has no signal that K programs are about to start — it discovers the demand only
when their prefills arrive and HBM is already pressured, then evicts reactively
(possibly mid-decode of others).

## Why ThunderAgent cannot (fully)
TA's scheduler is periodic and reacts to its token-usage estimate after the fact;
it has no forecast of an imminent dispatch step and no proactive eviction/migration
to prepare HBM — it can only start withholding prefills once its estimate crosses,
i.e. after the spike is already biting.

## Measurement plan
Arms: B / TA / ours (all HiCache). Metric: TTFT/p99 of requests in the window
around each fan-out; count of reactive evictions/preemptions triggered by the
spike. Expected: ours pre-clears room → smaller latency bump, fewer evictions.

## 4-tier sizing (required)
**Live-pressure** mode, DISK + DRAM still enabled (full 4-tier; the binding
constraint here is the spike's live set, and the freed/demoted bytes cascade down
the real tiers). Size the pool so the fan-out's K new active children push the
running set **over HBM**, so without preparation B reaches the 4th state =
**preemption/eviction storm** at the spike. Our proactive room-making aims to keep
the spike below that frontier. Tune so the spike *does* cause a measurable
preempt/evict storm on B (else proactivity buys nothing) but the run doesn't
collapse. Confirm via B's logs that the spike triggers preemption/eviction.

## Honest status & falsification
- **Status:** DESIGN now adds the missing piece — `forecast(state, event)` carries
  an `event_carried_demand(event)` term (§8) so a `SUB_DISPATCH_ASYNC`'s K
  not-yet-in-flight children are *visible* to the forecast (previously it summed
  only over `inflight>0` programs and was blind to the spike). So the
  anticipation is now design-supported; it needs implementing: estimate per-child
  demand from the dispatch payload + wire the proactive demote/pause on the
  forecast crossing.
- **Falsifies the win if:** there's enough HBM headroom that the spike never
  causes a reactive storm (then proactivity buys nothing), or the children are
  small. Must show B actually suffers a measurable latency bump / eviction storm
  at the spike that proactivity removes.

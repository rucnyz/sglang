# integration_stress (#147, verify/README.md row 10)

The ONLY verify that launches a real sglang + daemon stack and
pushes traffic through it.  Per-task verifies (T7/T8/T36/T42/T164/…)
exercise components in isolation against stubs; this one tests
cross-component interaction on the same wire path Run K uses.

## SCOPE

Seven flavors A–E + G + F.  A–E and G run sequentially against a
single shared sglang + daemon launched at the start; F uses its own
daemon against a dead sglang port (see below).

| flavor | exercises | pass criteria |
|---|---|---|
| **A** proxy hot-path under load | daemon proxy → sglang | 24 concurrent /v1/chat/completions × 60s; ≥95% success; p99 < 5s; daemon alive |
| **B** /aginfer/state cost under traffic | sglang state-dump + daemon polling | 16 concurrent chats × 90s, /aginfer/state polled at 2 Hz; state_dump_metrics p99_max < 200ms (loose; PLAN F3-revisit hard ceiling = 50ms) |
| **C** event-router fan-in throughput | /aginfer/event + event_router + kv_scheduler.handle | 200 webhook events fired; ≥99% accepted by daemon |
| **D** migrate under traffic | sglang /aginfer/migrate + outbound batches | 12 concurrent chats + 200 migrate batches; both ≥95% OK |
| **E** threshold PUT atomicity | sglang PUT + daemon GET /aginfer/thresholds + traffic | PUTs accepted ≥95%; daemon GETs always show `lo < hi < crit` (no torn read); daemon ≥10 successful GETs |
| **G** SESSION_END migrate e2e (#191) | webhook → daemon composed handler (#185 F5 + #187 migrate) → sglang | tag exclusive units with a pid (direct chats, unique prompts) → fire `POST /aginfer/event {session_end}` → `per_program_usage[pid].state` becomes ENDED within ~10s (the F5 PUT round-trip), OR the units fully drop + the entry GC's (#186). Demote is OBSERVED + logged, not gated — cold-start `h_max≈0` ⇒ V_u(keep) ≥ 0 so the policy may decline (the per-action migrate WIRE is already e2e-covered by T20/T33; G covers the SESSION_END trigger→ENDED-propagation seam #187-audit G3 flagged untested) |
| **F** dead-sglang resilience | low-traffic dead-sglang + #164 escalate threshold | separate daemon vs dead sglang port; daemon STAYS alive (no false-positive escalate-to-fatal) for 8 events at 1 Hz pacing with `--sustained-escalate-fails=5 --sustained-escalate-age-s=5` |

**F is the complement of T164 stages B0/C0.**  T164 covers the
positive case (true high-volume sustained-fail → daemon
self-kills); F covers the negative ("low-traffic dead-sglang stays
alive") — the failure mode that #164's both-axis trigger was
designed to avoid.  The positive integration case (queue-populating
dead-sglang → fatal under real subprocess) is genuinely harder to
construct via the event router (state-fetch fails before queue
populates), so it lives in T164's in-process tests.

## STATE THE HARNESS REQUIRES

Caught during wire-up and now hard-coded in `harness.py`:

| flag / env | reason |
|---|---|
| `SGLANG_ENABLE_UNIFIED_RADIX_TREE=1` | sglang's default tree cache returns `{"unsupported_tree_cache": ...}` for /aginfer/state; only UnifiedRadixCache emits the aginfer schema |
| `--aginfer-notify-url http://daemon/aginfer/event` | without this, sglang's `update_aginfer_thresholds` returns `ok=False reason="sglang launched without --aginfer-notify-url; no webhook firer to update"` — flavor E PUTs all 400.  Also enables the actual webhook firing path so D's APPLY_FAILED + admission events become real on the wire |
| Daemon launched BEFORE sglang | sglang's T22 #165 `bootstrap_thresholds_into_server_args` GETs `daemon/aginfer/thresholds` at startup and halts on failure ("deployment-ordering bug").  Daemon tolerates a dead sglang at startup (cold_start_probe logs + continues) |

## REPRODUCING

```bash
source /scratch/yuzhou/miniconda3/etc/profile.d/conda.sh && conda activate agsched
cd /scratch/yuzhou/projects/sglang
python dev/aginfer/verify/integration_stress/verify.py
```

Total wall ~ 5–7 min:
* sglang+daemon cold start: ~160s (model load + hicache + CUDA graph)
* A: 60s
* B: 90s
* C: ~1s
* D: 45s
* E: 30s
* F: ~17s (separate daemon launch + 8s pacing + 8s observation window)

GPUs 5,6 by default (override via `T_INT_GPUS`).

## RESULTS

**PASSED** — all 7 flavors green (2026-06-03, with the #191 G flavor).

The G run confirmed the full SESSION_END seam live: a pid with 6
tagged units (ppu state REASONING) → `POST /aginfer/event
{session_end}` → `ENDED=True` AND the migrate actually fired
(`demoted 6→5 HBM units` — under the real HiCache cost model the
policy demoted, so both the F5 PUT and the #187 migrate half landed
end-to-end).

Earlier: all 6 flavors A–F green on a clean run (2026-06-01),
`results/20260601_integration_stress_run4.log`.  Earlier runs:
* `run1` — exposed missing `--aginfer-notify-url` and bad F event-trigger model
* `run2` — fixed body for sglang PUT, demonstrated F isn't trivially triggerable from events
* `run3` — added `--aginfer-notify-url` BUT had wrong launch order (sglang first); sglang halted on missing daemon
* `run4` — daemon-first launch order, all 6 green

| Flavor | Result | Headline number |
|---|---|---|
| A proxy hot-path under load | PASS | 10,488 req / 60s, 100% success, p99 272 ms on Qwen3-0.6B |
| B state-dump under traffic | PASS | 159 samples, p99_max 33.9 ms (under 50 ms PLAN F3-revisit hard ceiling — #160 may be closable) |
| C event-router fan-in | PASS | 200/200 events accepted |
| D migrate under traffic | PASS | 200/200 migrate batches OK |
| E threshold PUT atomicity | PASS | 268/268 PUTs ok, 0 torn reads in 553 daemon GETs |
| F dead-sglang resilience | PASS | daemon stays alive, /health responsive, no false-positive fatal |

## SIDE-FINDINGS

* **B comes in under #160's 50 ms F3-revisit threshold.**  Peak
  state-dump p99 max 34 ms with 7,426 units in the radix tree
  (Qwen3-0.6B, 16 concurrent chats sustaining ~1.2k-token unique
  prompts).  PLAN §2 T14 trigger condition is `p99 > 50 ms`; this
  run does not meet that condition.  #160 (F3-revisit) was opened
  on a different earlier run; this current run does not reproduce
  the trigger.  Worth re-examining whether #160 is still load-
  bearing or closable.

* **F's positive integration test is structurally hard** — events
  trigger state-fetch which fails first (before any migrate gets
  enqueued), so the outbound queue stays empty and the
  sustained-escalate counters never trip.  To force the positive
  case in an integration test, would need a fake-sglang that
  responds OK on /aginfer/state but fails on /aginfer/migrate; not
  in scope for this verify.  T164 stages B0/C0 already prove the
  positive case at the OutboundQueue + fatal() integration layer.

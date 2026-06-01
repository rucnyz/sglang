# T22 — `GET /aginfer/thresholds` + `PUT /aginfer/thresholds`

PLAN §3 T22 + DESIGN §6 round-6 H3 + DESIGN §10 "Threshold parity".
Closes G9 (theta mismatch between sglang webhook fire and daemon
admission gate) permanently — the two sides can no longer drift
because there's only one canonical source.

## WHAT WE PROMISED

**Source of truth.**  The daemon owns `theta_hi`, `theta_lo`,
`theta_crit`, `heartbeat_s`.  Sglang has NO local cache (round-14
dropped it).

**Daemon side** (`dev/aginfer/daemon/`):
- `EventRouter.__init__` takes all four threshold knobs.
- `attach_event_routes` mounts `GET /aginfer/thresholds` that
  returns `{theta_hi, theta_lo, theta_crit, heartbeat_s}`.
- `main.py` adds CLI flags `--theta-crit` and `--heartbeat-s`
  (existing `--theta-hi`/`--theta-lo` already covered admission).

**Sglang side** (`python/sglang/srt/managers/aginfer_webhook.py`):
- `AginferWebhookFirer.apply_thresholds(*, theta_hi, theta_lo,
  theta_crit, heartbeat_s)` — single tuple rebind (GIL-atomic on
  CPython); concurrent `maybe_fire` sees either all-old or
  all-new, never torn.
- The four public attributes `theta_hi` / `theta_lo` / `theta_crit`
  / `heartbeat_s` are now `@property` read-throughs to
  `self._theta_tuple`; legacy attribute-style writes still work but
  re-bind the whole tuple to preserve atomicity.
- `apply_thresholds_payload(firer, body) -> (ok, reason)` —
  validation: required keys present, all four numeric, ranges
  ([0,1] for theta, > 0 for heartbeat), hysteresis
  `theta_lo < theta_hi <= theta_crit`.  Mutates only on success.
- `fetch_bootstrap_thresholds(daemon_base_url, *, timeout_s)` —
  sync helper sglang launch calls at startup.  Raises `httpx.HTTPError`
  family on unreachable daemon; caller (sglang launch) MUST halt.

**Sglang HTTP route** (`python/sglang/srt/entrypoints/http_server.py`):
- `PUT /aginfer/thresholds` parses body, dispatches via tokenizer-
  manager → scheduler IPC, applies on each rank's firer.  Returns
  `{ok: True, ranks: N}` or HTTP 400 with structured reason.

**Sglang scheduler** (`python/sglang/srt/managers/scheduler.py`):
- `update_aginfer_thresholds(recv_req)` handler calls
  `apply_thresholds_payload(self.aginfer_webhook, body)`.  Returns
  `(ok, reason)` to the tokenizer-manager and out to HTTP.

**Bootstrap fetch + daemon-push (sustained broadcast)**.  This
commit lands the GET/PUT plumbing.  Wiring sglang's launch path to
call `fetch_bootstrap_thresholds(daemon_base_url)` and halt on
failure is a small follow-up that belongs in a sglang-side launch
patch — Phase B0/B1 of the verify exercise it once that wires.
Daemon-side broadcast on runtime threshold change is also a
follow-up (operator-signal handler + outbound enqueue of
`PUT /aginfer/thresholds`).  Both are out of scope for this PR.

## WORST CASE

| Failure mode | How to force | Floor | Assertion |
|---|---|---|---|
| Daemon GET schema drift | grep response keys | exactly {theta_hi, theta_lo, theta_crit, heartbeat_s} | A0 |
| Torn write to firer thresholds | 50 concurrent applies + 4 reader threads | reader never sees a hybrid pair | A1 |
| Missing field in PUT body | strip a key | 400 + "missing required field(s)" | A2 |
| Non-numeric value | "theta_hi": "high" | 400 + reason names type | A2 |
| Negative threshold | -0.1 | 400 + reason names range | A2 |
| Hysteresis violation | theta_lo > theta_hi | 400 + reason names hysteresis | A2 |
| Failed validation does NOT mutate state | check firer.theta_hi unchanged after bad PUT | preserved | A2 |
| Bootstrap fetch happy path | daemon up | dict of 4 floats returned | A3 |
| Bootstrap fetch with daemon down | unreachable port | raises httpx.HTTPError subclass (caller halts) | A3 |
| Shape mismatch (daemon returns garbage) | malformed body | ValueError (NOT a network error class) | A3 |

## HOW WE VERIFY

`verify/t22/verify.py` runs in-process, no live sglang needed:

```
A0  daemon GET /aginfer/thresholds returns the four canonical
    floats from EventRouter (theta_lo + heartbeat_s newly added
    to the router this commit)
A1  firer.apply_thresholds is atomic (50 concurrent writes alter-
    nating between two valid (hi,lo,crit,hb) tuples; 4 reader
    threads do 20k reads each; assert the reader NEVER sees a
    hybrid pair — anything outside the two ground-truth tuples
    proves torn write)
A2  apply_thresholds_payload(firer, body) rejects every malformed
    shape (missing key / non-numeric / negative / theta_lo>=
    theta_hi) AND does NOT mutate firer state on rejection AND
    DOES mutate on a happy-path body
A3  fetch_bootstrap_thresholds — happy returns the dict; against
    a dead port raises httpx.HTTPError subclass (not ValueError,
    which we reserve for shape mismatch from a malformed daemon)
```

Phase B live integration (launch daemon + sglang, sglang bootstrap
fetches from daemon, launch-without-daemon halts) is out of scope
here — it lands when the sglang launch path is wired to call
`fetch_bootstrap_thresholds` at startup.

## REPRODUCING

```bash
source /scratch/yuzhou/miniconda3/etc/profile.d/conda.sh && conda activate agsched
cd /scratch/yuzhou/projects/sglang
python dev/aginfer/verify/t22/verify.py
```

No GPU / no sglang launch.  Runs in < 1 s.

## RESULTS

**PASSED** — all 4 stages.

* date: 2026-06-01
* lines: ~95 sglang (`aginfer_webhook.py` properties + apply +
  validation helper + bootstrap fetch; `io_struct.py` two new req
  classes; `scheduler.py` handler; `tokenizer_control_mixin.py`
  communicator + wrapper; `http_server.py` PUT route); ~25 daemon
  (`event_router.py` theta_lo+heartbeat_s + GET endpoint;
  `proxy.py` + `main.py` thread through new params).

| Stage | Result |
|---|---|
| A0 daemon GET returns canonical shape | PASS |
| A1 firer.apply_thresholds is atomic | PASS — 20k reads × 4 threads × 50 concurrent writes, zero torn pairs |
| A2 PUT validation rejects malformed | PASS — 6 malformed cases all rejected with structured reasons; firer state preserved; happy path mutates |
| A3 bootstrap_fetch happy + unreachable | PASS — happy returns dict; dead port raises httpx.HTTPError subclass |

* raw log: `results/20260601_t22_initial_pass.log`

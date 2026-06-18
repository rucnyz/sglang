#!/usr/bin/env bash
# Unified calibration (#265): produce ONE source-able cost profile for a
# (model, GPU), folding all three cost constants the Admitter / fire-planner
# price against:
#
#   κ_i   (c_i recompute curve)  — OFFLINE, bench_one_batch sweep + fit
#                                  → SGLANG_CSIGMA_*   (persistent truth)
#   c^xfer (cross-pool fire wall) — engine BOOT PROBE, self-reversing fire
#                                  → SGLANG_XPOOL_NB_CHUNK_COST_INIT_US
#                                    (cold-start SEED; runtime EWMA drifts
#                                     on top — design intent preserved)
#   c_m   (mamba per-slot copy)   — engine BOOT PROBE, fixed-HW constant
#                                  → SGLANG_CM_MAMBA_PER_SLOT_US (truth)
#
# κ_i can't be probed in-engine (a hybrid forward can't be split into the
# KV-L² and mamba-L stacks), so it stays offline. c^xfer/c_m need the real
# VMM arena + actuator + mamba pool, so they're measured by booting the
# engine once with SGLANG_HIMA_BOOT_PROBE=1 and dumping the result via
# SGLANG_HIMA_PROBE_DUMP.
#
# Usage:
#   bash dev/eval/cost_model/calibrate_profile.sh Qwen/Qwen3.5-9B H200 [gpu]
# Then deploy with:
#   source dev/eval/cost_model/profiles/Qwen_Qwen3.5-9B_H200.sh
#
# Env overrides: REPEATS, MEM_FRACTION (κ_i sweep); PORT (default 31900),
# PROBE_TIMEOUT_S (default 900, engine boot + first budgeter tick).
set -euo pipefail

MODEL="${1:?usage: calibrate_profile.sh <model_path> <device_label> [gpu]}"
DEVICE="${2:?usage: calibrate_profile.sh <model_path> <device_label> [gpu]}"
GPU="${3:-0}"
PORT="${PORT:-31900}"
PROBE_TIMEOUT_S="${PROBE_TIMEOUT_S:-900}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
PY="$REPO/.venv/bin/python"
SLUG="$(echo "${MODEL}_${DEVICE}" | tr '/ ' '__')"
PROFILE_DIR="$HERE/profiles"
mkdir -p "$PROFILE_DIR"
PROFILE_SH="$PROFILE_DIR/${SLUG}.sh"
PROFILE_JSON="$PROFILE_DIR/${SLUG}.json"

# ---------------------------------------------------------------------------
# Stage 1 — κ_i (offline). calibrate.sh prints `export SGLANG_CSIGMA_*` on
# stdout; capture, eval into THIS shell (so the boot-probe engine in stage 2
# uses the freshly-calibrated curve), and keep the lines for the profile.
# ---------------------------------------------------------------------------
echo ">> [1/3] κ_i offline calibration (bench_one_batch sweep + fit) ..." >&2
CUDA_VISIBLE_DEVICES="$GPU" \
CSIGMA_EXPORTS="$(bash "$HERE/calibrate.sh" "$MODEL" "$DEVICE")"
eval "$CSIGMA_EXPORTS"
echo "$CSIGMA_EXPORTS" | sed 's/^/   /' >&2

# ---------------------------------------------------------------------------
# Stage 2 — c^xfer + c_m (engine boot probe). Boot once with the budgeter +
# boot probe on, dumping to a temp JSON; poll for the dump, then tear down.
# ---------------------------------------------------------------------------
echo ">> [2/3] c^xfer + c_m boot probe (engine boot, port $PORT, gpu $GPU) ..." >&2
DUMP="$(mktemp)"
SERVER_LOG="$(mktemp)"
rm -f "$DUMP"   # so polling waits for the probe to (re)create it

# Tear down by killing the server's OWN process group (the setsid leader, all
# TP-worker subprocesses, and any side processes it spawned) — never a broad
# `pkill -f --port`, which on a shared box would reap a co-tenant's server on
# the same port and can MISS the TP workers (GPU-memory orphans).
cleanup() {
    if [ -n "${SV_PGID:-}" ]; then
        kill -- "-$SV_PGID" 2>/dev/null || true
        sleep 1
        kill -9 -- "-$SV_PGID" 2>/dev/null || true
    fi
    rm -f "$DUMP" "$SERVER_LOG"
}
trap cleanup EXIT

EXTRA=""
case "$MODEL" in
    */Qwen3*|*Qwen3.5*) EXTRA="--reasoning-parser qwen3 --enforce-piecewise-cuda-graph" ;;
    *Kimi*)             EXTRA="--trust-remote-code --enforce-piecewise-cuda-graph" ;;
esac

# `setsid` puts the server in a NEW session/process group whose PGID == its PID,
# so the whole tree (TP workers included) can be signalled atomically via the
# negative-PID group kill above.
CUDA_VISIBLE_DEVICES="$GPU" \
SGLANG_HIMA=1 \
SGLANG_HIMA_TICK_S=1.0 \
SGLANG_HIMA_BOOT_PROBE=1 \
SGLANG_HIMA_PROBE_DUMP="$DUMP" \
setsid "$PY" -m sglang.launch_server \
    --model-path "$MODEL" --host 127.0.0.1 --port "$PORT" \
    --mem-fraction-static "${MEM_FRACTION:-0.7}" --log-level info \
    $EXTRA > "$SERVER_LOG" 2>&1 &
SV_PID=$!
SV_PGID="$SV_PID"   # setsid leader: PGID == PID
echo "   server pid=$SV_PID (pgid=$SV_PGID); waiting up to ${PROBE_TIMEOUT_S}s for probe dump ..." >&2

waited=0
while [ $waited -lt "$PROBE_TIMEOUT_S" ]; do
    sleep 5; waited=$((waited + 5))
    if [ -s "$DUMP" ]; then
        echo "   probe dump appeared after ${waited}s" >&2
        break
    fi
    if ! kill -0 "$SV_PID" 2>/dev/null; then
        echo "   ERROR: server died before dumping — last 25 log lines:" >&2
        tail -25 "$SERVER_LOG" >&2
        exit 1
    fi
done
if [ ! -s "$DUMP" ]; then
    echo "   ERROR: no probe dump after ${PROBE_TIMEOUT_S}s — log tail:" >&2
    tail -25 "$SERVER_LOG" >&2
    exit 1
fi
PROBE_JSON="$(cat "$DUMP")"
# Tear down the whole server process group (TP workers included), then disarm
# the EXIT trap's group-kill so it's a no-op for an already-reaped tree.
kill -- "-$SV_PGID" 2>/dev/null || true
sleep 1
kill -9 -- "-$SV_PGID" 2>/dev/null || true
SV_PID=""
SV_PGID=""

# ---------------------------------------------------------------------------
# Stage 3 — merge κ_i + boot-probe dump into the profile.
#
# Reproducibility: the .json `kappa_i` block is derived from the SAME captured
# `$CSIGMA_EXPORTS` that produced the .sh profile (NOT re-read from the
# gitignored on-disk kappa_fit.json, which could be stale or from a different
# run), so a single invocation's .sh and .json are guaranteed consistent.
# ---------------------------------------------------------------------------
echo ">> [3/3] writing profile → $PROFILE_SH" >&2
PROBE_JSON="$PROBE_JSON" CSIGMA_EXPORTS="$CSIGMA_EXPORTS" \
MODEL="$MODEL" DEVICE="$DEVICE" \
PROFILE_SH="$PROFILE_SH" PROFILE_JSON="$PROFILE_JSON" \
"$PY" - <<'PYEOF'
import json, os, datetime
probe = json.loads(os.environ["PROBE_JSON"])
csigma = os.environ["CSIGMA_EXPORTS"]
model, device = os.environ["MODEL"], os.environ["DEVICE"]
c_xfer = probe.get("c_xfer_us_per_page")
c_xfer_calibrated = bool(probe.get("c_xfer_calibrated"))
c_m = probe.get("c_m_us_per_slot")
stamp = datetime.datetime.now().isoformat(timespec="seconds")

# Derive κ_i straight from the captured export lines so the .json mirrors the
# exact same run as the .sh (no disk read of kappa_fit.json).
csigma_kv = {}
for line in csigma.splitlines():
    if not line.startswith("export "):
        continue
    key, _, val = line[len("export "):].partition("=")
    csigma_kv[key.strip()] = val.strip()


def _kv_float(name):
    raw = csigma_kv.get(name)
    return float(raw) if raw is not None else None


kappa_i = {
    "c_kv": {
        "alpha_ms_per_tok2": _kv_float("SGLANG_CSIGMA_KV_ALPHA"),
        "beta_ms_per_tok": _kv_float("SGLANG_CSIGMA_KV_BETA"),
        "gamma_ms": _kv_float("SGLANG_CSIGMA_KV_GAMMA"),
    },
    "c_m": {
        "alpha_ms_per_tok": _kv_float("SGLANG_CSIGMA_M_ALPHA"),
        "beta_ms": _kv_float("SGLANG_CSIGMA_M_BETA"),
    },
    "crossover_L_star": _kv_float("SGLANG_CSIGMA_LSTAR"),
}

lines = [f"# cost profile for {model} on {device} — generated {stamp}",
         "# κ_i recompute curve (persistent truth, offline bench fit):"]
lines += [l for l in csigma.splitlines() if l.startswith("export")]
lines.append("# c^xfer cross-pool fire wall — cold-start SEED "
             "(runtime EWMA drifts on top):")
if c_xfer is not None:
    lines.append(f"export SGLANG_XPOOL_NB_CHUNK_COST_INIT_US={c_xfer:.6g}")
else:
    # c^xfer probe did NOT seed a measured wall (it failed and left the
    # conservative default — the dump sets c_xfer_us_per_page=null via
    # is_boot_seeded, NOT via is_calibrated which is always False at boot).
    # Omit the seed so the runtime EWMA starts from the engine default
    # rather than baking in an unmeasured value.
    lines.append("# (c^xfer probe did not seed — omitted; "
                 "runtime EWMA starts from engine default)")
lines.append("# c_m mamba per-slot copy — fixed-HW constant "
             "(env-precedence; boot probe skips):")
if c_m is not None:
    lines.append(f"export SGLANG_CM_MAMBA_PER_SLOT_US={c_m:.6g}")
open(os.environ["PROFILE_SH"], "w").write("\n".join(lines) + "\n")

json.dump({
    "model": model, "device": device, "generated": stamp,
    "kappa_i": kappa_i,
    "c_xfer_us_per_page": c_xfer, "c_xfer_calibrated": c_xfer_calibrated,
    "c_m_us_per_slot": c_m, "c_m_calibrated": probe.get("c_m_calibrated"),
}, open(os.environ["PROFILE_JSON"], "w"), indent=2)
print(f"   c^xfer={c_xfer} µs/page (calibrated={c_xfer_calibrated})  "
      f"c_m={c_m} µs/slot", file=__import__("sys").stderr)
PYEOF

echo ">> done. deploy with:  source $PROFILE_SH" >&2
cat "$PROFILE_SH"

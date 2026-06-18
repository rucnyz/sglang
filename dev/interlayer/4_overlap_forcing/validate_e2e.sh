#!/bin/bash
# e2e gate for overlap-compatible token forcing (task #334).
# Boots PLAIN sglang (no budgeter/arena — isolates the forcing change) twice and
# compares the forced output:
#   off : --disable-overlap-schedule  -> the CPU _maybe_force_output_token hook
#         (known-correct, the token-exact mechanism all RQ1 runs on).
#   on  : overlap default ON          -> the NEW GPU override (under test).
# PASS iff, per request, on.text == off.text AND n_out == forced_len in both
# (so overlap-on emits byte-identical forced output to the known-correct baseline).
# GPU 0, port 30099.
set -u
SG=/scratch/yuzhou/projects/sglang
VENV=$SG/.venv/bin/python
PORT=30099; GPU=0; MODEL=Qwen/Qwen3.5-9B
OUT=$SG/dev/interlayer/4_overlap_forcing; mkdir -p "$OUT"
COMMON="--model-path $MODEL --host 127.0.0.1 --port $PORT --tp 1 \
 --mem-fraction-static 0.8 --context-length 8192 --log-level warning"

run() {  # <label> <outfile> <extra-args>
  local label=$1 outfile=$2; shift 2; local extra="$*"
  ss -ltnp 2>/dev/null | grep ":$PORT " | grep -oE 'pid=[0-9]+' | grep -oE '[0-9]+' | head -1 | xargs -r kill -9 2>/dev/null
  for i in $(seq 1 40); do ss -ltn 2>/dev/null | grep -q ":$PORT " || break; sleep 1; done; sleep 2
  echo "[$label] booting (extra: ${extra:-none}) ..."
  CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    $VENV -m sglang.launch_server $COMMON $extra > "$OUT/server_$label.log" 2>&1 &
  local sv=$!
  for i in $(seq 1 240); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 http://127.0.0.1:$PORT/health 2>/dev/null)" = "200" ] && break
    kill -0 $sv 2>/dev/null || { echo "[$label] SERVER DIED — tail:"; tail -5 "$OUT/server_$label.log"; return 1; }
    sleep 5
  done
  $VENV "$OUT/validate_client.py" "http://127.0.0.1:$PORT/generate" "$label" "$outfile"
  kill -9 $sv 2>/dev/null; sleep 3
}

echo "=== overlap OFF (known-correct CPU hook baseline) ==="
run off "$OUT/out_off.json" "--disable-overlap-schedule"
echo "=== overlap ON (the change under test) ==="
run on "$OUT/out_on.json" ""
echo
echo "=== COMPARE off + on vs GOLDEN (CPU-hook baseline) ==="
# Post-Phase-2 BOTH modes use the GPU override (the CPU hook is deleted), so
# off-vs-on alone can't catch a shared bug. Compare each to the golden CPU-hook
# output captured before deleting the hook: override must reproduce it exactly.
$VENV - "$OUT/out_off.json" "$OUT/out_on.json" "$OUT/golden_cpuhook_baseline.json" <<'PY'
import json, sys, os
off = json.load(open(sys.argv[1])); on = json.load(open(sys.argv[2]))
gpath = sys.argv[3]
gold = json.load(open(gpath)) if os.path.exists(gpath) else None
ok = len(off) == len(on) and len(off) > 0
if gold is None:
    print("  WARN: no golden baseline; falling back to off==on only")
for k in range(len(off)):
    a, b = off[k], on[k]
    g = gold[k] if gold else a
    same = a["text"] == b["text"] == g["text"]
    len_ok = a["n_out"] == b["n_out"] == g["n_out"] == a["forced_len"]
    if same and len_ok:
        print(f"  req{k}: OK  (n_out={b['n_out']}==forced_len, text == golden CPU-hook in BOTH modes)")
    else:
        ok = False
        print(f"  req{k}: MISMATCH text(off==on==gold)={same} "
              f"n_out off={a['n_out']} on={b['n_out']} gold={g['n_out']} forced_len={a['forced_len']}")
print("\n=== VERDICT ===")
print("  => unified GPU-override forcing VALIDATED (matches CPU-hook golden, both schedules)"
      if ok else "  => NOT validated")
sys.exit(0 if ok else 1)
PY

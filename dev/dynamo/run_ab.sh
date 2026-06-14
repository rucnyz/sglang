#!/usr/bin/env bash
# A/B orchestrator: thunderagent_router (baseline) vs aginfer_router (ours) on the
# 4-tier HiCache sglang backend. Switches the router, runs the fleet driver N cycles
# per arm, parses re-prefill (#new-token) / cache-hit (#cached-token) from the backend
# log over each cycle's window, and prints a per-cycle + per-arm summary.
#
# Backend (dynamo.sglang, --max-total-tokens 32768, 4-tier) and frontend (:8100) must
# already be running. This script only swaps the router and drives load.
#
# Usage: run_ab.sh <cycles> <classA> <turnsA> <tokA> <classB> <turnsB> <tokB> <gap> <maxtok>
set -u
C=aginfer_dyn
BK=/tmp/sglang_backend.log
CYCLES=${1:-3}; CA=${2:-20}; TA=${3:-12}; KA=${4:-800}
CB=${5:-10}; TB=${6:-2}; KB=${7:-2400}; GAP=${8:-0.3}; MT=${9:-120}
OUT=/tmp/ab_results.jsonl
: > $OUT

switch_router() {  # $1 = ta|ag
  local mod label
  if [ "$1" = ta ]; then mod=dynamo.thunderagent_router; label=THUNDERAGENT
  else mod=dynamo.aginfer_router; label=AGINFER; fi
  docker exec $C bash -lc 'pkill -9 -f "dynamo.thunderagent_router|dynamo.aginfer_router" 2>/dev/null; sleep 2'
  docker exec -d $C bash -lc "
    python -m $mod --endpoint dynamo.backend.generate --model-name Qwen/Qwen3-0.6B \
      --router-block-size 64 --router-reset-states > /tmp/router_$1.log 2>&1"
  # wait until a tagged smoke request returns content
  echo \"  switching to $label ...\"
  for i in $(seq 1 30); do
    R=$(docker exec $C bash -lc 'curl -s -m8 http://localhost:8100/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"Qwen/Qwen3-0.6B\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":4,\"nvext\":{\"agent_context\":{\"trajectory_id\":\"smoke\",\"session_id\":\"s\",\"session_type_id\":\"agent\"}}}" 2>/dev/null | grep -c chatcmpl')
    [ "$R" = "1" ] && { echo "    $label ready (try $i)"; return 0; }
    sleep 2
  done
  echo "    !! $label did not become ready"; return 1
}

parse_window() {  # $1=start_line  -> echo "new cached peakutil npause"
  docker exec $C bash -lc "
    awk 'NR> $1' $BK 2>/dev/null | sed -E 's/\x1b\[[0-9;]*m//g' | \
    grep -a report_prefill_stats | grep -oE '#new-token: [0-9]+, #cached-token: [0-9]+, token usage: [0-9.]+' | \
    awk -F'[ ,]+' '{n+=\$3; c+=\$6; if(\$9>pu)pu=\$9} END{printf \"%d %d %.3f\", n, c, pu}'
  "
}

run_arm() {  # $1=ta|ag  $2=armlabel
  switch_router "$1" || return 1
  for cyc in $(seq 1 $CYCLES); do
    local start np
    start=$(docker exec $C bash -lc "wc -l < $BK")
    local npause0=$(docker exec $C bash -lc "grep -ac _pause_until_safe /tmp/router_$1.log 2>/dev/null || echo 0")
    RES=$(docker exec $C bash -lc "
      cd /workspace/sglang/dev/dynamo
      python fleet_ab.py --base http://localhost:8100 --model Qwen/Qwen3-0.6B \
        --classA $CA --turnsA $TA --tokA $KA --classB $CB --turnsB $TB --tokB $KB \
        --gap $GAP --max-tokens $MT --req-timeout 90 \
        --run-salt ${1}c${cyc}_$RANDOM --tag ${2}_c$cyc 2>&1 | grep '^RESULT' | sed 's/^RESULT //'")
    sleep 3   # let trailing prefill stats flush to the log
    PW=$(parse_window "$start")
    local npause1=$(docker exec $C bash -lc "grep -ac _pause_until_safe /tmp/router_$1.log 2>/dev/null || echo 0")
    local pauses=$(( npause1 - npause0 ))
    echo "{\"arm\":\"$2\",\"cycle\":$cyc,\"pause_ticks\":$pauses,\"prefill\":{\"new\":$(echo $PW|awk '{print $1}'),\"cached\":$(echo $PW|awk '{print $2}'),\"peak_util\":$(echo $PW|awk '{print $3}')},\"client\":$RES}" | tee -a $OUT
  done
}

echo "=== A/B: CYCLES=$CYCLES  A=${CA}x${TA}@${KA}  B=${CB}x${TB}@${KB}  gap=$GAP maxtok=$MT ==="
run_arm ta THUNDERAGENT
run_arm ag AGINFER
echo "=== done -> $OUT ==="

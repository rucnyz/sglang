#!/bin/bash
# S1 milestone-1b: restart the daemon (fresh fire code, sglang stays up) and run
# the deterministic promote smoke test.
set -uo pipefail
cd /scratch/yuzhou/projects/sglang/dev/aginfer || exit 1
HERE="$PWD/wherewewin/s1-toolcall-predictability"
DLOG="logs/s1_daemon_smoke.log"

curl -sf http://127.0.0.1:30000/health >/dev/null 2>&1 || { echo "sglang not up"; exit 1; }

OLD=$(pgrep -f 'daemon.main' | head -1)
echo "[smoke] killing daemon pid=$OLD"
[ -n "$OLD" ] && kill "$OLD" 2>/dev/null
sleep 3
for i in $(seq 1 10); do ss -ltn 2>/dev/null | grep -q ':9100 ' || break; sleep 1; done
echo "[smoke] relaunching daemon (fresh fire code) ..."
PYTHONPATH="$PWD:${PYTHONPATH:-}" setsid nohup python -m daemon.main \
  --sglang-base-url=http://127.0.0.1:30000 --port=9100 \
  --kv-scheduler=enabled --admission-controller=enabled > "$DLOG" 2>&1 &
for i in $(seq 1 30); do
  grep -q "Uvicorn running on http://0.0.0.0:9100" "$DLOG" 2>/dev/null && break; sleep 1
done
curl -sf http://127.0.0.1:9100/health >/dev/null 2>&1 || { echo "[smoke] daemon failed"; tail "$DLOG"; exit 1; }
echo "[smoke] daemon UP"

OFFSET=$(wc -l < "$DLOG")
echo "[smoke] running promote smoke ..."
python "$HERE/s1_promote_smoke.py" --prefix-tokens 6000 --filler-tokens 28000 --eta 15 2>&1 | tail -20

echo "[smoke] === daemon promote evidence ==="
tail -n +"$((OFFSET+1))" "$DLOG" | grep -aE 'promote_scheduled|promote_dispatched|promote_skipped|migrate_enqueued|event=memory_pressure' | tail -20
echo "[smoke] sglang alive? $(pgrep -f sglang.launch_server >/dev/null && echo YES || echo NO)"
echo "[smoke] done"

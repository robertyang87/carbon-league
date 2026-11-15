#!/bin/bash
# h23 step B: stop ONLY the m3s2r1pre vs E0 pre-cal bigeval tree, verify frozen.
MARK="H23SWPB3K"
echo "MARKER $MARK START"
date '+%F %T %Z'
# --- locate target: cmdline must carry all three markers; main = whose ppid is not itself a match
MATCHES=""
for p in $(pgrep -f 'bigeval.py'); do
  cl=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null)
  case "$cl" in *m3s2r1pre.py*echo_e0.py*m3s2r1pre_e0*) MATCHES="$MATCHES $p" ;;
esac
done
echo "MATCHES:$MATCHES"
TARGET_PID=""
for p in $MATCHES; do
  pp=$(ps -o ppid= -p $p | tr -d ' ')
  in=0; for q in $MATCHES; do [ "$pp" = "$q" ] && in=1; done
  [ $in -eq 0 ] && TARGET_PID=$p
done
echo "TARGET_PID=$TARGET_PID"
if [ -z "$TARGET_PID" ]; then echo "MARKER $MARK ABORT-NO-TARGET"; echo "MARKER $MARK END"; exit 0; fi
echo "--- TREE BEFORE ---"
ps -eo pid,ppid,etime,args | grep bigeval | grep -v grep
echo "--- SNAPSHOT STATS (at-stop) ---"
PY=/root/carbon/.venv/bin/python; [ -x "$PY" ] || PY=python3
"$PY" - <<'PYEOF'
import json
f = '/root/carbon/data/bigeval_live/m3s2r1pre_e0.jsonl'
lines = open(f).read().strip().splitlines()
ms = []; errs = 0; w = 0
for l in lines:
    d = json.loads(l)
    if d.get('err'): errs += 1
    m = d.get('margin')
    if m is not None:
        ms.append(m)
        if m > 0: w += 1
print('AT-STOP lines=%d mean_margin_a=%.1f wins_a=%d err_lines=%d' % (len(lines), sum(ms)/len(ms) if ms else 0.0, w, errs))
PYEOF
# --- collect full descendant set of TARGET_PID
get_kids() { for k in $(pgrep -P $1 2>/dev/null); do echo $k; get_kids $k; done; }
ALL=$(echo $TARGET_PID; get_kids $TARGET_PID)
echo "KILL SET: $ALL"
kill $ALL 2>/dev/null
sleep 4
SURV=""
for p in $ALL; do
  if [ -d /proc/$p ]; then
    cl=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null)
    case "$cl" in *bigeval*) SURV="$SURV $p" ;;
    esac
  fi
done
if [ -n "$SURV" ]; then echo "TERM survivors:$SURV -> kill -9"; kill -9 $SURV 2>/dev/null; sleep 2; fi
echo "--- AFTER: bigeval procs on h23 ---"
ps -eo pid,ppid,etime,args | grep bigeval | grep -v grep || echo "NO BIGEVAL PROC"
echo "--- PARENT CHAIN BASH STATE ---"
ps -o pid,ppid,etime,args -p 115105 2>/dev/null || echo "parent bash gone"
C1=$(wc -l < /root/carbon/data/bigeval_live/m3s2r1pre_e0.jsonl)
echo "lines after kill: $C1"
sleep 30
C2=$(wc -l < /root/carbon/data/bigeval_live/m3s2r1pre_e0.jsonl)
echo "lines after +30s: $C2"
if [ "$C1" = "$C2" ]; then echo "STOPPED-CONFIRMED"; else echo "STILL-GROWING-ANOMALY"; fi
echo "MARKER $MARK END"

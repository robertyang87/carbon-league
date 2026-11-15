#!/bin/bash
# h23 step D: verify e0_m2_base baseline field is running and producing games.
MARK="H23SWPD8R"
echo "MARKER $MARK START"
date '+%F %T %Z'
echo "--- PROCS ---"
ps -eo pid,ppid,etime,args | grep 'e0_m2_base' | grep -v grep || echo "NO e0_m2_base PROC"
F=/root/carbon/data/bigeval_live/e0_m2_base.jsonl
if [ -f "$F" ]; then
  C1=$(wc -l < "$F"); echo "lines now: $C1"
  sleep 20
  C2=$(wc -l < "$F"); echo "lines +20s: $C2"
  if [ "$C2" -gt "$C1" ]; then echo "GROWING"; elif [ "$C2" = "$C1" ]; then echo "STATIC-IN-WINDOW"; else echo "SHRANK-ANOMALY"; fi
  PY=/root/carbon/.venv/bin/python; [ -x "$PY" ] || PY=python3
  "$PY" - <<'PYEOF'
import json
f = '/root/carbon/data/bigeval_live/e0_m2_base.jsonl'
lines = open(f).read().strip().splitlines()
print('total lines:', len(lines))
for l in lines[:3]:
    d = json.loads(l)
    print('GAME seed=%s side=%s margin=%s err=%s t=%s' % (d.get('seed'), d.get('side'), d.get('margin'), d.get('err'), d.get('t')))
ms = [json.loads(l).get('margin') for l in lines]
ms = [m for m in ms if m is not None]
errs = sum(1 for l in lines if json.loads(l).get('err'))
if ms:
    print('SO-FAR mean_margin_a=%.1f wins_a=%d err_lines=%d' % (sum(ms)/len(ms), sum(1 for m in ms if m > 0), errs))
PYEOF
else
  echo "JSONL NOT YET PRESENT"
fi
echo "--- OUT TAIL ---"
tail -5 /root/carbon/data/e0_m2_base.out 2>/dev/null
echo "MARKER $MARK END"

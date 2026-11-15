#!/bin/bash
# h23 step A: read-only probe before stopping the pre-cal E0 field.
MARK="H23SWPA7Q"
echo "MARKER $MARK START"
date '+%F %T %Z'
echo "--- BIGEVAL PROCS ---"
ps -eo pid,ppid,etime,args | grep -E 'bigeval' | grep -v grep || echo "NO BIGEVAL PROC"
echo "--- LIVE DIR (newest last) ---"
ls -ltr /root/carbon/data/bigeval_live/ | tail -8
echo "--- JSONL DETAIL ---"
PY=/root/carbon/.venv/bin/python
[ -x "$PY" ] || PY=python3
"$PY" - <<'PYEOF'
import json, glob, os
files = sorted(glob.glob('/root/carbon/data/bigeval_live/*.jsonl'), key=os.path.getmtime)[-4:]
for f in files:
    lines = open(f).read().strip().splitlines()
    print('== FILE', f, 'lines', len(lines))
    if not lines:
        continue
    print('FIRST:', lines[0][:600])
    try:
        d = json.loads(lines[0])
        print('KEYS:', list(d.keys()))
        if isinstance(d.get('scores'), dict):
            print('SCORES KEYS:', list(d['scores'].keys()))
    except Exception as e:
        print('parse err', e)
    ms = []
    errs = 0
    for l in lines:
        try:
            d = json.loads(l)
            if d.get('err'):
                errs += 1
            s = d.get('scores') or {}
            if 'a' in s and 'b' in s:
                ms.append(s['a'] - s['b'])
        except Exception:
            pass
    if ms:
        w = sum(1 for m in ms if m > 0)
        print('MARGIN a-b: n=%d mean=%.1f wins_a=%d err_lines=%d' % (len(ms), sum(ms)/len(ms), w, errs))
PYEOF
echo "--- EXP CHECK ---"
ls -la /root/carbon/code/exp/echo_e0.py /root/carbon/code/exp/opp_master2r.py /root/carbon/code/exp/m3s2r1pre.py 2>&1
echo "MARKER $MARK END"

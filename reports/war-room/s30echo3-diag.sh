#!/bin/bash
MARK=S30E3DIAG0312
echo "$MARK PAYLOAD-BEGIN"
date '+%F %T %Z'
cat /proc/loadavg
echo "--- BIGEVAL-PROCS ---"
ps -eo pid,lstart,etime,cmd | grep "bigeval.py" | grep -v grep | head -8
echo -n "bigeval-proc-count: "; pgrep -cf "bigeval.py" || echo 0
MPID=$(pgrep -f "bigeval.py" | head -1)
if [ -n "$MPID" ]; then
  echo "--- ENV-OF-PID $MPID ---"
  tr '\0' '\n' < /proc/$MPID/environ 2>/dev/null | grep -E "THREADS|EVO_WORKERS" || echo ENV-UNREADABLE
fi
echo "--- LIVE-FILES ---"
for T in echo3_m2 echo3_echo2; do
  F=/root/carbon/data/bigeval_live/$T.jsonl
  if [ -f "$F" ]; then wc -l "$F"; ls -l --time-style=full-iso "$F"; else echo "$T NO-FILE"; fi
done
echo "--- M2-SUMMARY-FROM-LIVE ---"
/root/carbon/.venv/bin/python - <<'PYEOF'
import json, statistics as st
rows = []
for ln in open('/root/carbon/data/bigeval_live/echo3_m2.jsonl'):
    try: rows.append(json.loads(ln))
    except Exception: pass
m = [r['margin'] for r in rows if r.get('margin') is not None]
err = sum(1 for r in rows if r.get('err'))
seeds = sorted(r['seed'] for r in rows if r.get('seed') is not None)
ts = [r['t'] for r in rows if r.get('t')]
ms = [r['ms'] for r in rows if r.get('ms') is not None]
wms = [r['wms'] for r in rows if r.get('wms') is not None]
def pct(a, q):
    a = sorted(a); return a[min(len(a)-1, int(q*len(a)))] if a else None
print('n_rows', len(rows), 'n_margins', len(m))
if m:
    se = st.stdev(m)/len(m)**.5 if len(m) > 1 else 0
    print('wins', sum(x > 0 for x in m), 'mean', round(st.mean(m), 1), 'se', round(se, 1), 'lo2se', round(st.mean(m)-2*se, 1), 'err_rows', err)
print('seeds', len(seeds), 'min', seeds[0] if seeds else None, 'max', seeds[-1] if seeds else None, 'distinct', len(set(seeds)))
print('t_first_in_file', ts[0] if ts else None, 't_last_in_file', ts[-1] if ts else None, 't_min', min(ts) if ts else None, 't_max', max(ts) if ts else None)
if ms:
    print('a_ms_per_step mean', round(st.mean(ms), 2), 'p50', pct(ms, .5), 'p95', pct(ms, .95), 'max', max(ms))
if wms:
    print('a_worst_step_ms max', max(wms))
print('serr_sum', sum(r.get('serr') or 0 for r in rows), 'srch_sum', sum(r.get('srch') or 0 for r in rows))
PYEOF
echo "--- SUMMARIES-IN-BIGEVAL-JSONL ---"
/root/carbon/.venv/bin/python - <<'PYEOF'
import json
try:
    for ln in open('/root/carbon/data/bigeval.jsonl'):
        try: s = json.loads(ln)
        except Exception: continue
        if s.get('tag') in ('echo3_m2', 'echo3_echo2'):
            s.pop('games', None)
            print(json.dumps(s, ensure_ascii=False))
except FileNotFoundError:
    print('NO-BIGEVAL-JSONL')
PYEOF
echo "--- LAUNCH-LOG-TAIL ---"
tail -4 /root/carbon/data/echo3_m2.launch.log 2>/dev/null || echo NO-LAUNCH-LOG
ls /root/carbon/data/*.launch.log 2>/dev/null
echo "$MARK PAYLOAD-END"

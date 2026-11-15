#!/bin/bash
MARK=S30E3RB0310
echo "$MARK PAYLOAD-BEGIN"
date '+%F %T %Z'
echo -n "bigeval-procs: "; pgrep -cf "bigeval.py" || echo 0
cat /proc/loadavg
echo "--- RECOMPUTE-BOTH ---"
/root/carbon/.venv/bin/python - <<'PYEOF'
import json, statistics as st
for tag, s0 in (('echo3_m2', 996000), ('echo3_echo2', 999000)):
    p = f'/root/carbon/data/bigeval_live/{tag}.jsonl'
    rows = []
    for ln in open(p):
        try: rows.append(json.loads(ln))
        except Exception: pass
    m = [r['margin'] for r in rows if r.get('margin') is not None]
    err = sum(1 for r in rows if r.get('err'))
    seeds = sorted(r['seed'] for r in rows if r.get('seed') is not None)
    missing = [s for s in range(s0, s0 + 200) if s not in set(seeds)]
    se = st.stdev(m) / len(m) ** .5 if len(m) > 1 else 0
    print(tag, 'rows', len(rows), 'wins', sum(x > 0 for x in m),
          'mean', round(st.mean(m), 1) if m else None, 'se', round(se, 1),
          'lo2se', round(st.mean(m) - 2 * se, 1) if m else None, 'err_rows', err,
          'seeds_distinct', len(set(seeds)), 'seed_min', seeds[0] if seeds else None,
          'seed_max', seeds[-1] if seeds else None, 'missing', missing[:10])
print('--- SUMMARIES ---')
for ln in open('/root/carbon/data/bigeval.jsonl'):
    try: s = json.loads(ln)
    except Exception: continue
    if s.get('tag') in ('echo3_m2', 'echo3_echo2'):
        s.pop('games', None)
        print(json.dumps(s, ensure_ascii=False))
PYEOF
echo "--- LAUNCH-LOGS ---"
cat /root/carbon/data/echo3_m2.launch.log 2>/dev/null || echo NO-M2-LAUNCHLOG
ls -lt /root/carbon/data/ | grep -iE "echo3|launch" | head -8
echo "--- FILES ---"
for T in echo3_m2 echo3_echo2; do
  F=/root/carbon/data/bigeval_live/$T.jsonl
  wc -l "$F"; md5sum "$F"; ls -l --time-style=full-iso "$F"
  echo -n "$T gz-b64-lines: "
  gzip -c "$F" | base64 -w 76 | wc -l
done
echo "$MARK PAYLOAD-END"

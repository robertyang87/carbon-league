#!/bin/bash
# Runs on h24 via cloud assistant. READ ONLY: smoke result check.
MARK="ECHO2PREPCHK3H24"
echo "MARKER $MARK START"
date '+%F %T %Z'; hostname
cd /root/carbon/code
echo "--- SMOKE PROC ---"
ps aux | grep echo2_smoke | grep -v grep | head -3 || true
echo "--- SMOKE LIVE ---"
timeout 30 wc -l /root/carbon/data/bigeval_live/echo2_smoke.jsonl 2>/dev/null || echo NO_LIVE_YET
echo "--- SMOKE AGG ---"
timeout 120 ../.venv/bin/python - <<'PYEOF'
import json
rows = []
try:
    for line in open('/root/carbon/data/bigeval_live/echo2_smoke.jsonl'):
        line = line.strip()
        if line:
            rows.append(json.loads(line))
except FileNotFoundError:
    print('no live file')
print('games', len(rows))
for r in rows:
    print('seed', r.get('seed'), 'side', r.get('side'), 'margin', r.get('margin'), 'err', r.get('err'), 'serr', r.get('serr'), 'wms', r.get('wms'), 'ms', r.get('ms'))
if rows:
    mg = [r['margin'] for r in rows if isinstance(r.get('margin'), (int, float))]
    print('margins', mg)
    print('mean_margin', round(sum(mg) / len(mg), 1) if mg else None)
    print('err_nonnull', sum(1 for r in rows if r.get('err')))
    print('max_wms', max((r.get('wms') or 0) for r in rows))
PYEOF
echo "--- SMOKE OUT TAIL ---"
timeout 30 tail -5 /root/carbon/data/echo2_smoke.out 2>/dev/null
echo "--- BIGEVAL SUMMARY ---"
timeout 30 grep -a 'echo2_smoke' /root/carbon/data/bigeval.jsonl 2>/dev/null || echo NO_SUMMARY_YET
echo "--- E0 FIELD STILL ALIVE ---"
timeout 30 wc -l /root/carbon/data/bigeval_live/m3s2r1_e0.jsonl
echo "MARKER $MARK END"

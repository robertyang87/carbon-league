#!/bin/bash
# spot25 step F10: final read - summary lines verbatim + live file state.
MARK="SPOT25F6K1"
echo "MARKER $MARK START"
date '+%F %T %Z'
echo "--- PROC ---"
ps -eo pid,etime,args | grep bigeval | grep -v grep || echo "NO-BIGEVAL-PROC"
echo "--- SUMMARY LINES (verbatim) ---"
grep 'spot25_smoke' /root/carbon/data/bigeval.jsonl 2>/dev/null || echo "NO-SUMMARY"
echo "--- LIVE FILE ---"
ls -la /root/carbon/data/bigeval_live/
python3 -c "
import json
for l in open('/root/carbon/data/bigeval_live/spot25_smoke.jsonl'):
    d = json.loads(l)
    print('seed=%s side=%s margin=%s err=%s t=%s serr=%s wms=%s' % (d.get('seed'), d.get('side'), d.get('margin'), d.get('err'), d.get('t'), d.get('serr'), d.get('wms')))"
echo "--- WEIGHTS DIR FINAL ---"
md5sum /root/carbon/data/weights/*
echo "MARKER $MARK END"

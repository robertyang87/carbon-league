#!/bin/bash
# spot25 step F: launch 2-game smoke detached, print launch state.
MARK="SPOT25F4K8"
echo "MARKER $MARK START"
date '+%F %T %Z'
cd /root/carbon/code || { echo "NO-CODE-DIR"; echo "MARKER $MARK END"; exit 1; }
ls exp/echo2.py exp/echo_e0.py exp/m3s2r1.py
EVO_WORKERS=2 setsid nohup ../.venv/bin/python bigeval.py exp/m3s2r1.py exp/echo2.py 999901 2 spot25_smoke > /root/carbon/data/spot25_smoke.out 2>&1 &
echo "SMOKE-LAUNCHED pid $!"
sleep 20
ps -eo pid,etime,args | grep -E 'bigeval' | grep -v grep || echo "NO-BIGEVAL-AFTER-20S"
tail -5 /root/carbon/data/spot25_smoke.out 2>/dev/null
echo "MARKER $MARK END"

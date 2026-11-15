#!/bin/bash
# h23 step C: launch E0 vs Master2-real baseline retest field (tag e0_m2_base).
MARK="H23SWPC5M"
echo "MARKER $MARK START"
date '+%F %T %Z'
echo "--- EXAMINER FILES ---"
ls -la /root/carbon/code/exp/echo_e0.py /root/carbon/code/exp/opp_master2r.py 2>&1
echo "--- PRE-LAUNCH BIGEVAL CHECK ---"
if pgrep -f 'bigeval.py exp/' > /dev/null; then
  echo "ABORT: a bigeval field is already running:"
  ps -eo pid,ppid,etime,args | grep 'bigeval.py exp/' | grep -v grep
  echo "MARKER $MARK END"
  exit 0
fi
echo "no bigeval running, launching"
cd /root/carbon/code || { echo "CD FAIL"; echo "MARKER $MARK END"; exit 0; }
EVO_WORKERS=7 setsid nohup ../.venv/bin/python bigeval.py exp/echo_e0.py exp/opp_master2r.py 999200 200 e0_m2_base >> /root/carbon/data/e0_m2_base.out 2>&1 &
sleep 10
echo "--- NEW PROCS (e0_m2_base) ---"
ps -eo pid,ppid,etime,args | grep 'e0_m2_base' | grep -v grep || echo "NO NEW PROC - LAUNCH FAILED"
echo "--- OUT FILE ---"
ls -la /root/carbon/data/e0_m2_base.out 2>/dev/null; head -5 /root/carbon/data/e0_m2_base.out 2>/dev/null
echo "MARKER $MARK END"

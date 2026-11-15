#!/bin/bash
# g1cal step A (read-only probe): confirm baseline field exited, record in-machine
# md5s of eval_v53.py/bigeval.py, verify m3s2g1 wrapper + weights pre-staged.
MARK="G1CALA9X2"
echo "MARKER $MARK START"
date '+%F %T %Z'
echo "--- PROC ---"
ps -eo pid,etime,args | grep -E 'bigeval' | grep -v grep || echo NO-BIGEVAL-PROC
ps -p 119669 -o pid,etime,args 2>/dev/null || echo "PID-119669-GONE"
echo "--- CODE MD5 ---"
md5sum /root/carbon/code/eval_v53.py /root/carbon/code/bigeval.py 2>&1
echo "--- GIT? ---"
git -C /root/carbon rev-parse HEAD 2>&1 | head -2
git -C /root/carbon/code rev-parse HEAD 2>&1 | head -2
echo "--- WRAPPERS ---"
ls -la /root/carbon/code/exp/m3s2g1.py /root/carbon/code/exp/opp_master2r.py 2>&1
md5sum /root/carbon/code/exp/m3s2g1.py /root/carbon/code/exp/opp_master2r.py 2>&1
grep -n "npz\|weights" /root/carbon/code/exp/m3s2g1.py | head -10
echo "--- WEIGHTS ---"
ls -la /root/carbon/data/weights/ 2>&1 | grep -i "m3s2g1" || echo NO-M3S2G1-WEIGHTS
md5sum /root/carbon/data/weights/cnet2_m3s2g1* 2>&1
echo "--- LIVE DIR (pre-existing m3s2g1_m2?) ---"
ls -la /root/carbon/data/bigeval_live/ 2>&1 | grep -i "m3s2g1" || echo NO-M3S2G1-LIVE-FILES
echo "--- VENV ---"
ls -la /root/carbon/.venv/bin/python 2>&1
echo "MARKER $MARK END"

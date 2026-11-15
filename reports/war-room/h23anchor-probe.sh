#!/bin/bash
# h23 probe (read-only): prereqs for Echo2 anchor chain (echo2_m2 + m3s2g1_echo2).
MARK="H23ANCHORPROBE2001A"
echo "${MARK}_BEGIN"
date '+%F %T %Z'
hostname
echo "--- NO BIGEVAL RUNNING? ---"
ps -eo pid,ppid,etime,args | grep 'bigeval.py exp/' | grep -v grep || echo "NO_BIGEVAL_RUNNING"
echo "--- G1 CHAIN STATUS (should be finished) ---"
cat /root/carbon/data/g1_chain.status 2>&1
echo "--- CODE MD5 ---"
md5sum /root/carbon/code/eval_v53.py /root/carbon/code/bigeval.py 2>&1
echo "--- WRAPPERS ---"
ls -la /root/carbon/code/exp/echo2.py /root/carbon/code/exp/m3s2g1.py /root/carbon/code/exp/opp_master2r.py 2>&1
md5sum /root/carbon/code/exp/echo2.py 2>&1
echo "--- opp_master2r.py weight refs ---"
grep -n "npz" /root/carbon/code/exp/opp_master2r.py 2>&1 | head -10
echo "--- echo2.py weight refs ---"
grep -n "npz" /root/carbon/code/exp/echo2.py 2>&1 | head -10
echo "--- WEIGHTS ---"
md5sum /root/carbon/data/weights/cnet2_m3s2g1L.npz /root/carbon/data/weights/cnet2_m3w5L.npz 2>&1
echo "--- NEW TAGS MUST NOT EXIST ---"
ls -la /root/carbon/data/bigeval_live/echo2_m2.jsonl /root/carbon/data/bigeval_live/m3s2g1_echo2.jsonl 2>&1
ls -la /root/carbon/data/echo2_m2.launcher.log /root/carbon/data/m3s2g1_echo2.launcher.log 2>&1
ls -la /root/carbon/data/echo2_anchor.status /root/carbon/data/echo2_anchor_runner.sh 2>&1
echo "--- MACHINE ---"
echo "nproc: $(nproc)"; free -m | head -2; cat /proc/loadavg
echo "${MARK}_END"

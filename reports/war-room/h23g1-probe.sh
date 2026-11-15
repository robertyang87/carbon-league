#!/bin/bash
# h23 probe (read-only): prereqs for g1 two-field chained run.
MARK="H23G1PROBE1953A"
echo "${MARK}_BEGIN"
date '+%F %T %Z'
hostname
echo "--- NO BIGEVAL RUNNING? ---"
ps -eo pid,ppid,etime,args | grep 'bigeval.py exp/' | grep -v grep || echo "NO_BIGEVAL_RUNNING"
echo "--- CODE MD5 ---"
md5sum /root/carbon/code/eval_v53.py /root/carbon/code/bigeval.py 2>&1
echo "--- WRAPPERS ---"
ls -la /root/carbon/code/exp/m3s2g1.py /root/carbon/code/exp/opp_master2r.py /root/carbon/code/exp/echo_e0.py 2>&1
md5sum /root/carbon/code/exp/echo_e0.py 2>&1
echo "--- echo_e0.py weight refs ---"
grep -n "npz" /root/carbon/code/exp/echo_e0.py 2>&1 | head -10
echo "--- WEIGHTS ---"
md5sum /root/carbon/data/weights/cnet2_m3s2g1L.npz /root/carbon/data/weights/cnet2_m3w3L.npz 2>&1
ls /root/carbon/data/weights/ 2>&1 | grep -E 's64|m3w3' || true
echo "--- OLD FIELD STATE ---"
wc -l /root/carbon/data/bigeval_live/m3s2g1_m2.jsonl 2>&1
tail -1 /root/carbon/data/bigeval_live/m3s2g1_m2.jsonl 2>&1 | head -c 400; echo
echo "--- NEW TAGS MUST NOT EXIST ---"
ls -la /root/carbon/data/bigeval_live/m3s2g1_m2r.jsonl /root/carbon/data/bigeval_live/m3s2g1_e0.jsonl 2>&1
ls -la /root/carbon/data/m3s2g1_m2r.launcher.log /root/carbon/data/m3s2g1_e0.launcher.log 2>&1
echo "--- MACHINE ---"
echo "nproc: $(nproc)"; free -m | head -2; cat /proc/loadavg
echo "${MARK}_END"

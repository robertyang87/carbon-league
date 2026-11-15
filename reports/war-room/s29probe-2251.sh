#!/bin/bash
# H29: spot-29 pre-flight inventory probe before Fable r1 launch.
MARK="S29PROBE2251"
echo "${MARK}_BEGIN"
date '+%F %T %Z'
hostname
echo "--- RESIDUAL ---"
ps -eo pid,ppid,etime,args | grep -E 'bigeval|snaploop' | grep -v grep || echo NO-RESIDUAL
echo "--- HARNESS MD5 ---"
md5sum /root/carbon/code/eval_v53.py /root/carbon/code/bigeval.py 2>&1
echo "--- OPPONENT WRAPPERS ---"
md5sum /root/carbon/code/exp/opp_master2r.py /root/carbon/code/exp/echo2.py /root/carbon/code/exp/m3s2r1.py 2>&1
echo "--- WEIGHTS COUNT ---"
find /root/carbon/data/weights -type f | wc -l
echo "--- TARGET WEIGHTS ---"
ls -l /root/carbon/data/weights/ | grep -iE 'fable|m7bsv2|carago' || echo NO-TARGET-WEIGHTS-LISTED
md5sum /root/carbon/data/weights/cnet2_fable_r1_fromstudio.npz 2>&1
md5sum /root/carbon/data/weights/cnet2_m3w5L.npz 2>&1
echo "--- LIVE DIR ---"
ls -l /root/carbon/data/bigeval_live/ 2>&1 | head -20
cat /proc/loadavg
free -m | head -2
echo "${MARK}_END"

#!/bin/bash
# h23 read-only probe for Line A paired calibration prep (wr1/wg1).
MARK="H23WPREP2013A"
echo "${MARK}_BEGIN"
date '+%F %T %Z'; hostname
echo "--- procs ---"
pgrep -af 'bigeval.py' || echo NO_BIGEVAL
echo "--- md5 core ---"
md5sum /root/carbon/code/eval_v53.py /root/carbon/code/bigeval.py
echo "--- weights m3s2/m3w5 ---"
ls -la /root/carbon/data/weights/ | grep -iE 'm3s2|m3w5'
md5sum /root/carbon/data/weights/cnet2_m3s2wr1_fromstudio.npz /root/carbon/data/weights/cnet2_m3s2wg1_fromstudio.npz 2>&1
echo "--- exp wrappers ---"
ls /root/carbon/code/exp/
echo "--- m3s2g1.py ---"
cat /root/carbon/code/exp/m3s2g1.py
echo "--- m3s2r1.py ---"
cat /root/carbon/code/exp/m3s2r1.py 2>&1
echo "--- opp_master2r.py head ---"
head -30 /root/carbon/code/exp/opp_master2r.py
echo "--- git ---"
cd /root/carbon && git rev-parse --abbrev-ref HEAD 2>&1; git remote -v 2>&1 | head -4; ls -d /root/carbon/.git 2>&1
echo "--- live tags (wr/wg must be absent) ---"
ls /root/carbon/data/bigeval_live/ | grep -E 'wr1|wg1' || echo NO_WRWG_TAGS
echo "--- disk ---"
df -h /root | tail -1
echo "${MARK}_END"

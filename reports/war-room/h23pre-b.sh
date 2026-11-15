#!/bin/bash
# H23 prestage probe B: runs ON h23. Read template wrapper + check state. READ-ONLY.
MARK=PREH23B7K9
echo "$MARK H23-BEGIN $(date '+%F %T %Z')"
hostname
echo '--- EXAM-PROC (do not touch) ---'
ps -eo pid,etime,args | grep -E 'bigeval' | grep -v grep || echo NO-BIGEVAL-PROC
echo '--- TARGET-WEIGHT-EXIST? ---'
ls -l /root/carbon/data/weights/cnet2_m3s2g1L.npz 2>&1 || echo TARGET-NOT-YET
ls -l /root/carbon/data/weights/ | head -20
echo '--- TEMPLATE m3s2r1pre.py ---'
cat /root/carbon/code/exp/m3s2r1pre.py 2>&1
echo '--- TEMPLATE-END ---'
ls -l /root/carbon/code/exp/ | head -30
echo "$MARK H23-END"

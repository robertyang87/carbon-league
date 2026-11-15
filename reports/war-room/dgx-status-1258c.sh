#!/bin/bash
# DGX status probe trip 12:58 - follow-up C. Marker DGXSTATUS1258C. Read-only.
echo "DGXSTATUS1258C $(date '+%H:%M:%S %Z')"
echo "== GEN1 WEIGHTS =="
ls -l /home/arm/junyang/carbon/data/weights/cnet2_gen1r1.npz /home/arm/junyang/carbon/data/weights/cnet2_gen1g1.npz 2>&1
md5sum /home/arm/junyang/carbon/data/weights/cnet2_gen1r1.npz /home/arm/junyang/carbon/data/weights/cnet2_gen1g1.npz 2>&1
echo "== GEN1 TRAIN LOGS =="
cat /home/arm/junyang/carbon/data/dq_logs/gen1r1.log 2>&1 | cut -c1-300
cat /home/arm/junyang/carbon/data/dq_logs/gen1g1.log 2>&1 | cut -c1-300
echo "== TRAIN PROCS NOW =="
ps -eo pid,etime,cmd | grep -E 'cnet2_train|train_heads|selfplay_gen|katago_chain' | grep -v grep || echo NO_TRAIN_PROC
uptime
echo "DGXSTATUS1258C-END $(date '+%H:%M:%S %Z')"

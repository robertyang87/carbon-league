#!/bin/bash
# DGX status probe 3: resolve ep10ship anomaly + g1 ship + replays count. 2026-11-12 ~13:57 CST
# Marker: DGXSTATUS1357KT9R (claim only output containing this marker)
MARK=DGXSTATUS1357KT9R
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
echo "DGXSTATUS1357KT9R DGX-BEGIN $(date '+%F %T %Z')"
echo '--- EP10SHIP-SCRIPT ---'
cat /home/arm/junyang/carbon/code/ep10ship.sh 2>/dev/null || echo NO-SCRIPT
echo '--- R1-EP19-MTIME ---'
ls -l --time-style=+%F_%H:%M:%S /home/arm/junyang/carbon/data/weights/cnet2_m3s2r1_ep19.npz 2>/dev/null || echo NO-R1-EP19
ls -l --time-style=+%F_%H:%M:%S /home/arm/junyang/carbon/data/weights/cnet2_m3s2r1_ep1[6-9].npz 2>/dev/null
echo '--- DONE-FLAGS ---'
ls -l --time-style=+%F_%H:%M:%S /home/arm/junyang/carbon/data/dq_logs/*.done 2>/dev/null || echo NO-DONE-FILES
echo '--- R1-NOW ---'
ps -eo pid,etime,args | grep -E 'cnet2_train' | grep -v grep || echo NO-TRAIN-PROC
tail -3 /home/arm/junyang/carbon/data/dq_logs/m3s2r1.log 2>/dev/null
echo '--- REPLAYS ---'
ls /home/arm/junyang/carbon/data/m3s2_corpus/
echo -n 'replays_dir_count='
ls /home/arm/junyang/carbon/data/m3s2_corpus/replays/ 2>/dev/null | wc -l
echo '--- M7BS-OUT-MTIME ---'
ls -l --time-style=+%F_%H:%M:%S /home/arm/junyang/carbon/data/m7bs.out 2>/dev/null
head -5 /home/arm/junyang/carbon/data/m7bs.out 2>/dev/null
echo '--- EP10SHIP-LOG-FULL ---'
cat /home/arm/junyang/carbon/data/dq_logs/ep10ship.log 2>/dev/null
echo "DGXSTATUS1357KT9R DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"

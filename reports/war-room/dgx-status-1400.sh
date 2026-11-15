#!/bin/bash
# DGX status probe 4: verify r1 ep19 landed+shipped, M7Bs auto-handoff started, g1 ship on L20. 2026-11-12 ~14:00 CST
# Marker: DGXSTATUS1400QZ8W (claim only output containing this marker)
MARK=DGXSTATUS1400QZ8W
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
echo '--- L20-WEIGHTS-m3s2 ---'
ls -l --time-style=+%F_%H:%M:%S /root/carbon/data/weights/cnet2_m3s2* 2>/dev/null | tail -12 || echo NO-L20-WEIGHTS
ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
echo "DGXSTATUS1400QZ8W DGX-BEGIN $(date '+%F %T %Z')"
hostname; uptime
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader
echo '--- PROC ---'
ps -eo pid,etime,args | grep -E 'cnet2_train|m7bs' | grep -v grep || echo NO-TRAIN-PROC
echo '--- R1-EP19 ---'
ls -l --time-style=+%F_%H:%M:%S /home/arm/junyang/carbon/data/weights/cnet2_m3s2r1_ep19.npz /home/arm/junyang/carbon/data/weights/cnet2_m3s2r1_ep20.npz 2>/dev/null || echo NO-R1-EP19
echo '--- DONE-FLAGS ---'
ls -l --time-style=+%F_%H:%M:%S /home/arm/junyang/carbon/data/dq_logs/*.done 2>/dev/null | tail -6
echo '--- R1-LOG-TAIL ---'
tail -5 /home/arm/junyang/carbon/data/dq_logs/m3s2r1.log 2>/dev/null
echo '--- G1-LOG-TAIL ---'
tail -3 /home/arm/junyang/carbon/data/dq_logs/m3s2g1.log 2>/dev/null
echo '--- M7BS ---'
ls -l --time-style=+%F_%H:%M:%S /home/arm/junyang/carbon/data/weights/cnet2_m7bs* 2>/dev/null | tail -6 || echo NO-M7BS-WEIGHTS
ls -l --time-style=+%F_%H:%M:%S /home/arm/junyang/carbon/data/m7bs.out 2>/dev/null
tail -8 /home/arm/junyang/carbon/data/m7bs.out 2>/dev/null
ls -lt /home/arm/junyang/carbon/data/dq_logs/ 2>/dev/null | head -12
echo '--- EP10SHIP-LOG ---'
tail -5 /home/arm/junyang/carbon/data/dq_logs/ep10ship.log 2>/dev/null
echo '--- REPLAYS ---'
python3 -c "import json;d=json.load(open('/home/arm/junyang/carbon/data/m3s2_corpus/index.json'));print('index_entries=',len(d if isinstance(d,list) else d.get('entries') or d))" 2>&1
echo -n 'replays_dir_count='
ls /home/arm/junyang/carbon/data/m3s2_corpus/replays/ 2>/dev/null | wc -l
echo "DGXSTATUS1400QZ8W DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"

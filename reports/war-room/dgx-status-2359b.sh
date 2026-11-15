#!/bin/bash
# DGX follow-up probe 2359b: is the smoke actually running? Marker DGXSTATUS0016B
MARK=DGXSTATUS0016B
echo "$MARK BEGIN $(date -u +%H:%M:%S)"
timeout -k 30 240 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
M=DGXSTATUS0016B
echo "$M DGX $(hostname) $(uptime)"
echo '--PSALL--'
ps -eo pid,etime,%cpu,args | grep -iE 'python|smoke|play_batch|fastsim|sim' | grep -v grep | head -15
echo '--SMOKEDIR--'
ls -lt --time-style=+%m%d_%H:%M:%S /home/arm/junyang/carbon/data/smoke_pb_final/ 2>/dev/null | head -15
echo '--SMOKEOUT--'
find /home/arm/junyang/carbon/data/smoke_pb_final /home/arm/junyang/carbon/output -mmin -60 -type f 2>/dev/null | head -10
echo '--GPU2--'
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader
echo "$M END"
DGXEOF
echo "$MARK END rc=$?"

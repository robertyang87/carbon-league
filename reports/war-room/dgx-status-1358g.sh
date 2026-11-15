#!/bin/bash
# DGX status probe trip 13:58 follow-up G - read-only. Marker DGXSTATUS1358G.
KEY="$HOME/.ssh/id_ed25519"
SSH_OPTS=(-i "$KEY" -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new -o "ProxyCommand=nc -X connect -x hatch-egress-proxy:3130 %h %p")
DGX="arm@100.84.148.73"
echo "DGXSTATUS1358G $(date '+%H:%M:%S %Z')"
timeout 90 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS1358G; nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv,noheader; echo ---TRAIN-PS---; ps -eo pid,etime,args | grep -E "train|cnet2|policy_verify|bigeval|gate|build|index|selfplay|lite_v7" | grep -v grep | head -20; echo ---BUILD-LOGS---; for f in /home/arm/junyang/carbon/data/dq_logs/build_lite_v7.log /home/arm/junyang/carbon/data/dq_logs/lite_v7r1.log /home/arm/junyang/carbon/data/dq_logs/lite_v7g1.log /home/arm/junyang/carbon/data/dq_logs/replica_chain_lite_v7.log; do echo "== $f"; stat -c "%y %s bytes" "$f"; tail -5 "$f"; done; echo ---DS-INDEX-DIRS---; ls -d /home/arm/junyang/carbon/data/ds_lite_v7* /home/arm/junyang/carbon/data/lite_v7_index* 2>/dev/null; ls /home/arm/junyang/carbon/data/lite_v7_corpus/replays/ 2>/dev/null | wc -l; echo ---END-G---' 2>&1
echo "rcG=$?"
echo "DGXSTATUS1358G-END $(date '+%H:%M:%S %Z')"

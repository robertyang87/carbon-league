#!/bin/bash
# DGX status probe trip 14:28 follow-up H - read-only. Marker DGXSTATUS1428H.
KEY="$HOME/.ssh/id_ed25519"
SSH_OPTS=(-i "$KEY" -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new -o "ProxyCommand=nc -X connect -x hatch-egress-proxy:3130 %h %p")
DGX="arm@100.84.148.73"
echo "DGXSTATUS1428H $(date '+%H:%M:%S %Z')"
timeout 90 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS1428H; nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv,noheader; echo ---R1-PS---; ps -eo pid,etime,args | grep -E "cnet2_train.*lite_v7r1" | grep -v grep; echo ---R1-LOG-TAIL---; tail -3 /home/arm/junyang/carbon/data/weights/cnet2_lite_v7r1_log.jsonl; echo ---R1-WEIGHTS---; ls -lt /home/arm/junyang/carbon/data/weights/ | grep -i lite_v7r1 | head -5; echo ---G1-FINAL---; ls -l /home/arm/junyang/carbon/data/weights/cnet2_lite_v7g1_ep19.npz /home/arm/junyang/carbon/data/weights/cnet2_lite_v7g1_last.npz; echo ---K4V2-NAMES---; find /home/arm/junyang/carbon/data /home/arm/junyang/carbon-repo/data -maxdepth 3 -iname "*k4v2*" 2>/dev/null | head -15; find /home/arm/junyang/carbon/data /home/arm/junyang/carbon-repo/data -maxdepth 2 -iname "*993*" 2>/dev/null | head -10; echo ---END-H---' 2>&1
echo "rcH=$?"
echo "DGXSTATUS1428H-END $(date '+%H:%M:%S %Z')"

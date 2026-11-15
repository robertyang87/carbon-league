#!/bin/bash
# DGX status probe trip 2026-11-13 18:43 - READ-ONLY. Marker DGXSTATUS1843A-G.
# Channel: Tailscale SSH arm@100.84.148.73 (cloud-assistant retired 2026-11-13 ~10:50, owner order).
# Since 18:28: it570 dual-arm training in progress (r1 ep4/20, g1 ep8/20 @18:29, GPU 95%);
# estimates: g1 ~18:37-18:39, r1 ~18:49-18:50 done. g1 may have finished by now - check.
# Also check it70 EVAL landing status via war-room only (no DGX file for it); HEAD lag 90 corpus commits.
KEY="$HOME/.ssh/id_ed25519"
SSH_OPTS=(-i "$KEY" -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new -o "ProxyCommand=nc -X connect -x hatch-egress-proxy:3130 %h %p")
DGX="arm@100.84.148.73"

echo "DGXSTATUS1843A $(date '+%H:%M:%S %Z')"
timeout 90 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS1843A; hostname; uptime; nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv,noheader; echo ---TRAIN-PS---; ps -eo pid,etime,args | grep -E "train|cnet2|policy_verify|bigeval|gate|build|index|selfplay|m9xj570|ladder" | grep -v grep | head -20; echo ---GIT-PS---; ps -eo pid,etime,args | grep -E "git (fetch|pull|checkout|clone)" | grep -v grep; echo ---END-A---' 2>&1
echo "rcA=$?"

echo "DGXSTATUS1843B $(date '+%H:%M:%S %Z')"
timeout 120 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS1843B; cd /home/arm/junyang/carbon-repo && git branch --show-current; echo LS-REMOTE-MAIN:; timeout 60 git ls-remote origin main 2>&1 | tail -1; echo LOCAL-HEAD:; git rev-parse HEAD; echo ---BEHIND-COUNT---; git rev-list --count HEAD..$(timeout 60 git ls-remote origin main 2>/dev/null | awk "{print \$1}") 2>/dev/null; echo ---BIGEVAL-STATUS---; git status --short data/bigeval.jsonl 2>/dev/null; echo ---END-B---' 2>&1
echo "rcB=$?"

echo "DGXSTATUS1843C $(date '+%H:%M:%S %Z')"
timeout 90 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS1843C; echo ---M9XJ570-WEIGHTS---; stat -c "%n %s %y" /home/arm/junyang/carbon/data/weights/cnet2_m9xj570r1.npz /home/arm/junyang/carbon/data/weights/cnet2_m9xj570g1.npz 2>&1; md5sum /home/arm/junyang/carbon/data/weights/cnet2_m9xj570r1.npz /home/arm/junyang/carbon/data/weights/cnet2_m9xj570g1.npz 2>/dev/null; echo ---TRAIN-LOGS-TAIL---; tail -5 /home/arm/junyang/carbon/data/dq_logs/m9xj570r1.log 2>&1; echo ---; tail -5 /home/arm/junyang/carbon/data/dq_logs/m9xj570g1.log 2>&1; echo ---LOG-MTIME---; stat -c "%n %s %y" /home/arm/junyang/carbon/data/dq_logs/m9xj570r1.log /home/arm/junyang/carbon/data/dq_logs/m9xj570g1.log /home/arm/junyang/carbon/data/dq_logs/replica_chain_m9xj570.log 2>&1; echo ---END-C---' 2>&1
echo "rcC=$?"

echo "DGXSTATUS1843D $(date '+%H:%M:%S %Z')"
timeout 90 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS1843D; echo ---DISK---; df -h / | tail -1; echo ---M9XJ570-CORPUS---; ls /home/arm/junyang/carbon/data/m9xj570_corpus/ 2>/dev/null | wc -l; ls /home/arm/junyang/carbon-repo/data/m9xj570_corpus/ 2>/dev/null | wc -l; echo ---END-D---' 2>&1
echo "rcD=$?"

echo "DGXSTATUS1843-END $(date '+%H:%M:%S %Z')"

#!/bin/bash
# Probe (runs on L20): find working ssh path L20 -> DGX, report DGX state.
MARK=M320PRB2130A
echo "${MARK}_BEGIN $(date '+%F %T %Z') host=$(hostname)"
ls ~/.ssh/ 2>/dev/null
echo '--- ssh config ---'
cat ~/.ssh/config 2>/dev/null || echo NO-CONFIG
echo '--- known hosts ---'
cut -d' ' -f1 ~/.ssh/known_hosts 2>/dev/null | head -20 || echo NO-KNOWN-HOSTS
for target in "arm@10.211.43.155" "arm@10.188.0.23"; do
  echo "TRY $target"
  timeout -k 5 40 ssh -n -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new "$target" 'echo DGX-OK $(hostname); nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader; pgrep -af cnet2_train || echo NO-TRAIN-PROC; pgrep -af "git (fetch|pull)" || echo NO-GIT-PROC; cd /home/arm/junyang/carbon-repo && git rev-parse --short HEAD && git status --porcelain | head -5' 2>&1 | head -25
  echo "ssh-rc=$?"
done
echo "${MARK}_END"

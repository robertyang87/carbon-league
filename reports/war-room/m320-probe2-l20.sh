#!/bin/bash
# Probe 2 (runs on L20): ssh to DGX with dgx_access key, report DGX state.
MARK=M320PRB2131B
echo "${MARK}_BEGIN $(date '+%F %T %Z') host=$(hostname)"
SSH="ssh -n -i ~/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new"
timeout -k 5 60 $SSH arm@10.211.43.155 'echo DGX-OK $(hostname); date "+%F %T %Z"; nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader; echo --- train procs ---; pgrep -af cnet2_train || echo NO-TRAIN-PROC; echo --- git procs ---; ps -eo pid,etimes,args | grep -E "git (fetch|pull)" | grep -v grep || echo NO-GIT-PROC; echo --- repo ---; cd /home/arm/junyang/carbon-repo && git rev-parse --short HEAD && git branch --show-current; echo --- workdir ---; ls /home/arm/junyang/carbon/data/ | head -30; ls /home/arm/junyang/carbon/code/build_m9xj_w.py /home/arm/junyang/carbon/code/replica_chain.sh 2>&1' 2>&1 | head -50
echo "ssh-rc=$?"
echo "${MARK}_END"

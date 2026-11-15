#!/bin/bash
LOG=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/dgx-sync-0844-ts.log
{
echo "DGXSYNC0844 PROBE-BEGIN $(date '+%F %T %Z')"
timeout -k 30 180 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30 -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o GlobalKnownHostsFile=/dev/null -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" arm@100.84.148.73 'bash -s' <<'STEOF'
echo "DGXSYNC0844 DGX-BEGIN $(date '+%F %T %Z')"
echo '--- GIT HEAD ---'
cd /home/arm/junyang/carbon && git rev-parse --short HEAD 2>&1; git log --oneline -3 2>&1 | cut -c1-120
echo '--- K0/A0A1 TRAIN PROCS ---'
pgrep -f 'cnet2_train|k0_train' | wc -l
ps -eo pid,ppid,etime,args | grep -E '[c]net2_train|[k]0_train' | cut -c1-220 || echo NO-TRAIN-PROC
echo '--- GPU ---'
nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu --format=csv 2>&1
echo '--- BRUSH + K0_TRAIN MD5 (code/) ---'
for f in code/bigeval.py code/eval_v53.py code/certify_pack.py code/k0_train.py; do md5sum /home/arm/junyang/carbon/$f 2>/dev/null || echo "MISSING $f"; done
ls -lt /home/arm/junyang/carbon/code/k0_train.py /home/arm/junyang/carbon/code/bigeval.py /home/arm/junyang/carbon/code/eval_v53.py /home/arm/junyang/carbon/code/certify_pack.py 2>/dev/null
echo '--- CORPUS PULL TARGETS (mtime after 08:30) ---'
ls -lt /home/arm/junyang/carbon/data/ 2>/dev/null | head -15
find /home/arm/junyang/carbon/data -maxdepth 2 \( -iname '*hakimi3pro*' -o -iname '*fable6*' -o -iname '*k4v2*' \) -newermt '2026-11-14 08:30' 2>/dev/null | head -20
echo '--- K0 WEIGHTS/LOGS (new since 08:30) ---'
ls -lt /home/arm/junyang/carbon/data/weights/ 2>/dev/null | head -6
find /home/arm/junyang/carbon/data -maxdepth 2 -name '*k0*' -newermt '2026-11-14 08:30' 2>/dev/null | head -10
echo '--- LOAD ---'
uptime
echo "DGXSYNC0844 DGX-END $(date '+%F %T %Z')"
STEOF
echo "DGXSYNC0844 SSH-RC=$?"
echo "DGXSYNC0844 PROBE-END $(date '+%F %T %Z')"
} | tee "$LOG"

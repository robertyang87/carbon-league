#!/bin/bash
LOG=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/dgx-sync-0844b-ts.log
{
echo "DGXSYNC0844B PROBE-BEGIN $(date '+%F %T %Z')"
timeout -k 30 180 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30 -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o GlobalKnownHostsFile=/dev/null -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" arm@100.84.148.73 'bash -s' <<'STEOF'
echo "DGXSYNC0844B DGX-BEGIN $(date '+%F %T %Z')"
echo '--- K0 TRAIN FULL CMDLINE + ETIME ---'
ps -eo pid,etime,args | grep '[k]0_train' | cut -c1-400
echo '--- K0 TRAIN LOG/OUT (files newer than 08:40 under carbon) ---'
find /home/arm/junyang/carbon -maxdepth 3 -newermt '2026-11-14 08:40' -type f 2>/dev/null | grep -v __pycache__ | head -30
echo '--- PULL PROCS / SCRIPTS ---'
ps -eo pid,etime,args | grep -iE '[p]ull|[l]adder' | cut -c1-200 || echo NO-PULL-PROC
ls -lt /home/arm/junyang/carbon/*.log /home/arm/junyang/carbon/data/*.log 2>/dev/null | head -8
echo '--- CORPUS TARGET FILES (any depth<=4, names) ---'
find /home/arm/junyang -maxdepth 4 \( -iname '*hakimi3pro*' -o -iname '*fable6*' -o -iname '*k4v2*' \) -newermt '2026-11-14 08:30' 2>/dev/null | head -20
find /home/arm/junyang/carbon/data -maxdepth 1 -iname '*matches*' 2>/dev/null | head; ls /home/arm/junyang/carbon/data/*.json 2>/dev/null | head
echo '--- REPO CLONES (git HEAD check, correct paths) ---'
ls -d /home/arm/junyang/carbon* 2>/dev/null; ls -d /home/arm/junyang/*/ 2>/dev/null | head -20
for d in /home/arm/junyang/carbon-repo /home/arm/junyang/carbon_repo; do [ -d "$d/.git" ] && (cd "$d" && git rev-parse --short HEAD); done 2>/dev/null
echo '--- GPU NOW ---'
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv 2>&1
echo "DGXSYNC0844B DGX-END $(date '+%F %T %Z')"
STEOF
echo "DGXSYNC0844B SSH-RC=$?"
echo "DGXSYNC0844B PROBE-END $(date '+%F %T %Z')"
} | tee "$LOG"

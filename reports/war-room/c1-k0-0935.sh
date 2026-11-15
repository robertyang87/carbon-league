#!/bin/bash
# C1 64-pair monitoring order receipt check (09:33 war-room order). READ-ONLY. Marker C1K00935.
LOG=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/c1-k0-0935.log
{
echo "C1K00935 PROBE-BEGIN $(date '+%F %T %Z')"
timeout -k 30 120 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30 -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o GlobalKnownHostsFile=/dev/null -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" root@100.65.8.27 'bash -s' <<'STEOF'
echo "C1K00935 C1-BEGIN $(date '+%F %T %Z')"
echo '--- K0 PACKS ON C1 ---'
ls -l /root/carbon/code/exp/k0_*.py 2>&1
md5sum /root/carbon/code/exp/k0_*.py 2>/dev/null
echo '--- BIGEVAL LIVE K0 FILES ---'
ls -lt /root/carbon/data/bigeval_live 2>/dev/null | head -12
echo '--- BIGEVAL PROCS ---'
ps -eo pid,etime,args | grep '[b]igeval' | cut -c1-220 || echo NO-BIGEVAL-PROC
echo '--- SEED 9901xx FILES ---'
ls /root/carbon/data/bigeval_live 2>/dev/null | grep -i k0 | head; find /root/carbon/data -maxdepth 3 -name '*k0*' -newermt '2026-11-14 09:30' 2>/dev/null | head
echo "C1K00935 C1-END $(date '+%F %T %Z')"
STEOF
echo "C1K00935 SSH-RC=$?"
echo "C1K00935 PROBE-END $(date '+%F %T %Z')"
} | tee "$LOG"

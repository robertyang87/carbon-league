#!/bin/bash
LOG=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/dgx-k0a1-0829-ts.log
{
echo "DGXK0A10829 PROBE-BEGIN $(date '+%F %T %Z')"
timeout -k 30 180 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30 -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o GlobalKnownHostsFile=/dev/null -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" arm@100.84.148.73 'bash -s' <<'STEOF'
echo "DGXK0A10829 DGX-BEGIN $(date '+%F %T %Z')"
echo '--- SELFPLAY PROC COUNT + OLDEST ---'
pgrep -f selfplay_gen | wc -l
ps -eo pid,ppid,pgid,etime,args | grep '[s]elfplay_gen' | head -3 | cut -c1-250
ps -eo pid,etime,args | grep '[s]elfplay_gen' | sort -k2 | head -2 | cut -c1-200
echo '--- CWD OF ONE WORKER ---'
P=$(pgrep -f selfplay_gen | head -1); echo "pid=$P"; readlink /proc/$P/cwd 2>/dev/null; ls -l /proc/$P/cwd/code/producer/selfplay_gen.py 2>/dev/null; md5sum /proc/$P/cwd/code/producer/selfplay_gen.py 2>/dev/null
echo '--- K0_A1 OUT DIR ---'
ls /home/arm/junyang/carbon/data/k0_a1 2>/dev/null | head -10; ls /home/arm/junyang/carbon/data/k0_a1 2>/dev/null | wc -l
find /home/arm/junyang/carbon/data/k0_a1 -type f -newermt '2026-11-14 05:00' 2>/dev/null | wc -l
ls -lt /home/arm/junyang/carbon/data/k0_a1 2>/dev/null | head -6
echo '--- PRODUCER LOGS (recent) ---'
ls -lt /home/arm/junyang/carbon/data/*.log /home/arm/junyang/carbon/data/k0_a1/*.log 2>/dev/null | head -8
find /home/arm/junyang/carbon -maxdepth 3 -name '*k0*' -newermt '2026-11-14 04:00' 2>/dev/null | head -10
echo "DGXK0A10829 DGX-END $(date '+%F %T %Z')"
STEOF
echo "DGXK0A10829 SSH-RC=$?"
echo "DGXK0A10829 PROBE-END $(date '+%F %T %Z')"
} | tee "$LOG"

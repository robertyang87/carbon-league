#!/bin/bash
# it74 sync-trip progress probe 20:18 (marker IT74SYNC2018)
timeout -k 30 120 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o ConnectTimeout=30 arm@100.84.148.73 <<'REMOTE'
echo MARKER=IT74SYNC2018
date '+%F %T %Z'
echo '--- IT74 PROCS ---'
ps -eo pid,etime,pcpu,args | grep -E 'bigeval|it74' | grep -v grep | head -10
echo '--- IT74 LIVE FILES ---'
for f in /home/arm/junyang/carbon/data/bigeval_live/it74_master2.jsonl /home/arm/junyang/carbon/data/bigeval_live/it74_echo2.jsonl; do echo "$f:"; stat -c '%y %s' "$f" 2>&1; wc -l < "$f" 2>/dev/null || echo "no file"; done
echo '--- K0 RETURNS ---'
ls /home/arm/junyang/k0_dist/returns/c1/ /home/arm/junyang/k0_dist/returns/c2/ 2>&1 | head -20
stat -c '%y' /home/arm/junyang/k0_dist/returns/c1/progress_c1.json /home/arm/junyang/k0_dist/returns/c2/progress.json 2>&1
echo '--- END MARKER=IT74SYNC2018 ---'
REMOTE

#!/bin/bash
# C1 spot-29 / C2 spot-31 liveness probe 14:48 (read-only). Marker C1C2ALIVE1448.
LOG=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/c1c2-alive-1448.log
{
echo "C1C2ALIVE1448 PROBE-BEGIN $(date '+%F %T %Z')"
for spec in "spot-29 100.65.8.27" "spot-31 100.98.67.20"; do
  set -- $spec; name=$1; ip=$2
  echo "--- $name $ip ---"
  timeout -k 30 240 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30 -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o GlobalKnownHostsFile=/dev/null -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" root@$ip 'hostname; uptime; cat /proc/loadavg; ls -l /root/carbon/data/bigeval_live 2>/dev/null | tail -3; ps -eo pid,etime,args | grep -E "[r]l2_ppo|[b]igeval|[s]naploop" | head -5; echo ---; ls /root/carbon/data/bigeval_live/*.jsonl 2>/dev/null | while read f; do echo "$f $(wc -l < "$f") $(stat -c %y "$f" | cut -d. -f1)"; done | tail -10' 2>&1
  echo "$name SSH-RC=$?"
done
echo "C1C2ALIVE1448 PROBE-END $(date '+%F %T %Z')"
} | tee "$LOG"

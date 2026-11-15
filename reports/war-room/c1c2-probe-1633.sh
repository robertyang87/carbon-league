#!/bin/bash
# C1 spot-29 / C2 spot-31 probe 16:18 (read-only). Marker C1C2PROBE1633.
# Checks: liveness, exam processes, bigeval_live files (probe: it95 pack vs m3s2g1 seeds 990000-990199), pack presence.
LOG=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/c1c2-probe-1633.log
{
echo "C1C2PROBE1633 PROBE-BEGIN $(date '+%F %T %Z')"
for spec in "spot-29 100.65.8.27" "spot-31 100.98.67.20"; do
  set -- $spec; name=$1; ip=$2
  echo "--- $name $ip ---"
  timeout -k 30 240 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30 -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o GlobalKnownHostsFile=/dev/null -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" root@$ip 'hostname; uptime; cat /proc/loadavg; echo "== exam procs =="; ps -eo pid,etime,args | grep -E "[b]igeval|[e]val_v53|[s]naploop|[q]_gate" | head -8; echo "== bigeval_live recent =="; ls -lt /root/carbon/data/bigeval_live/ 2>/dev/null | head -12; echo "== jsonl line counts =="; for f in /root/carbon/data/bigeval_live/*.jsonl; do [ -e "$f" ] && echo "$f $(wc -l < "$f") $(stat -c %y "$f" | cut -d. -f1)"; done | tail -12; echo "== it95 pack presence =="; ls -l /root/carbon/data/weights/*it95* /root/carbon/code/exp/*it95* 2>/dev/null; find /root/carbon -maxdepth 3 -name "*9e16085a*" 2>/dev/null | head -5; echo "== probe tag files =="; ls -lt /root/carbon/data/ 2>/dev/null | head -8' 2>&1
  echo "$name SSH-RC=$?"
done
echo "C1C2PROBE1633 PROBE-END $(date '+%F %T %Z')"
} | tee "$LOG"

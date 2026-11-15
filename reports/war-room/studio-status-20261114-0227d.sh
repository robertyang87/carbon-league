#!/bin/bash
MARK=STUDIOSTAT0227W2D9
echo "$MARK BEGIN $(date '+%F %T %Z')"
timeout -k 30 90 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes \
  -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o GlobalKnownHostsFile=/dev/null \
  -o BatchMode=yes -o ConnectTimeout=25 \
  -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" \
  arm@100.78.149.20 'bash -s' <<'STEOF'
echo "STUDIOSTAT0227W2D9 STUDIO-BEGIN $(date '+%F %T %Z')"
uptime
ps -p 41443 -o pid,%cpu,etime,time,command 2>/dev/null || echo PID41443-GONE
echo '--- TW-CANON ---'
tail -8 /tmp/tw_canon.log 2>/dev/null || echo NO-TW-LOG
echo '--- SELFPLAY-ERR-FULL ---'
wc -c /Users/arm/s1v4accept/automation/acceptance_r5/selfplay_err.log
cat /Users/arm/s1v4accept/automation/acceptance_r5/selfplay_err.log 2>/dev/null | head -40
echo "STUDIOSTAT0227W2D9 STUDIO-END"
STEOF
echo "$MARK END rc=$?"

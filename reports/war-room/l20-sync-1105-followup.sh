#!/bin/bash
# War-room-sync 11:06 trip (v132) follow-up: read E2 live row count at correct path. READ-ONLY. Marker L20SYNC1105B.
MARK=L20SYNC1105B
LOG=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/war-room-sync-1105-l20followup.log
{
echo "$MARK PROBE-BEGIN $(date '+%F %T %Z')"
timeout -k 30 120 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30 -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o GlobalKnownHostsFile=/dev/null -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" root@100.79.251.87 'bash -s' <<'STEOF'
echo "L20SYNC1105B L20-BEGIN $(date '+%F %T %Z')"
ls -l --time-style=+%F_%T /root/carbon/probe_m2/data/bigeval_live/ppou2_m_it20_m2.jsonl /root/carbon/probe_m2/data/bigeval_live/ppou2_m_it20_e2.jsonl
echo '--- ROWCOUNTS ---'
wc -l /root/carbon/probe_m2/data/bigeval_live/ppou2_m_it20_m2.jsonl /root/carbon/probe_m2/data/bigeval_live/ppou2_m_it20_e2.jsonl
echo '--- E2-TAIL ---'
tail -1 /root/carbon/probe_m2/data/bigeval_live/ppou2_m_it20_e2.jsonl | cut -c1-400
echo '--- E2-LOG-TAIL ---'
tail -5 /root/carbon/probe_m2/it20_e2.log 2>/dev/null | cut -c1-250
echo '--- M2-SUMMARY-CHECK ---'
grep -h 'mean' /root/carbon/probe_m2/it20_m2.log 2>/dev/null | tail -2 | cut -c1-250
echo "L20SYNC1105B L20-END $(date '+%F %T %Z')"
STEOF
echo "$MARK PROBE-END $(date '+%F %T %Z')"
} > "$LOG" 2>&1
echo "exit=$?"
cat "$LOG"

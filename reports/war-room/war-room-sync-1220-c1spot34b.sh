#!/bin/bash
MARK=C1SYNC1220B
LOG=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/war-room-sync-1220-c1spot34b.log
{
echo "$MARK PROBE-BEGIN $(date '+%F %T %Z')"
timeout -k 30 120 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30 -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o GlobalKnownHostsFile=/dev/null -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" root@100.119.67.98 'bash -s' <<'STEOF'
echo "C1SYNC1220B SPOT34-BEGIN $(date '+%F %T %Z')"
wc -l /root/carbon/data/bigeval_live/e4bare_m2.jsonl /root/carbon/data/bigeval_live/e4pack_m2.jsonl /root/carbon/data/bigeval_live/e4pack_e2.jsonl 2>&1
echo '--- TAILS ---'
for f in e4bare_m2 e4pack_m2 e4pack_e2; do echo "== $f"; tail -1 /root/carbon/data/bigeval_live/$f.jsonl | cut -c1-260; done
echo '--- ERR-COUNT ---'
grep -c '"err": [^n]' /root/carbon/data/bigeval_live/e4*.jsonl 2>/dev/null; grep -c '"err": null' /root/carbon/data/bigeval_live/e4*.jsonl 2>/dev/null
echo '--- EXAM-LOGS ---'
ls -lt --time-style=long-iso /root/carbon/data/bigeval_live/ /root/carbon/*.log /root/carbon/*.out 2>&1 | head -12
ls -lt --time-style=long-iso /root/carbon/ 2>&1 | head -8
echo "C1SYNC1220B SPOT34-END $(date '+%F %T %Z')"
STEOF
echo "$MARK SSH-RC=$?"
echo "$MARK PROBE-END $(date '+%F %T %Z')"
} | tee "$LOG"

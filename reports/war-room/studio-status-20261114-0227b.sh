#!/bin/bash
# Studio follow-up probe 2026-11-14 02:27 trip, part 2 (marker STUDIOSTAT0227Y7T1).
# 02:29 read found: (a) NEW process PID 41014 s1v4accept rl2_ppo_train (etime 00:01,
# CPU 80%); (b) pilot_du_v2.log + progress.json + PARTIAL glob NOT FOUND though
# 02:13 read saw them (2956 B / 492 B / 1268 B); (c) brush md5 only 6 lines —
# producer.py + guided/measure globs need explicit ls. Verify all three.
MARK=STUDIOSTAT0227Y7T1
echo "$MARK BEGIN $(date '+%F %T %Z')"
timeout -k 30 120 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes \
  -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o GlobalKnownHostsFile=/dev/null \
  -o BatchMode=yes -o ConnectTimeout=25 \
  -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" \
  arm@100.78.149.20 'bash -s' <<'STEOF'
echo "STUDIOSTAT0227Y7T1 STUDIO-BEGIN $(date '+%F %T %Z')"
echo '--- V4-PROC ---'
ps -p 41014 -o pid,%cpu,%mem,etime,time,command 2>/dev/null || echo PID41014-GONE
echo '--- V4-DIR ---'
ls -lt /Users/arm/s1v4accept/ 2>/dev/null | head -10
ls -lt /Users/arm/s1v4accept/code/ 2>/dev/null | head -10
echo '--- V4-LOGS ---'
find /Users/arm/s1v4accept -name '*.log' -newermt '2026-11-14 02:00' 2>/dev/null | head
ls -lt /var/folders/1r/fx5sv4hj63b4wh3chvjsqvjc0000gn/T/tmp7r2s9xc6/ 2>/dev/null | head -10
echo '--- PILOT-FIND ---'
find /Users/arm/junyang/carbon /Users/arm/s1v4accept /Users/arm/s1v3accept -name 'pilot_du_v2*' -o -name '*PARTIAL*' 2>/dev/null | head -20
echo '--- DATA-DIR-LS ---'
ls -lt /Users/arm/junyang/carbon/data/ 2>/dev/null | head -15
echo '--- CODE-FILES ---'
ls -l /Users/arm/junyang/carbon/code/producer.py /Users/arm/junyang/carbon/code/guided_*.py /Users/arm/junyang/carbon/code/measure*.py 2>&1
md5 -q /Users/arm/junyang/carbon/code/producer.py 2>/dev/null
for f in /Users/arm/junyang/carbon/code/guided_*.py /Users/arm/junyang/carbon/code/measure*.py; do echo "$f"; md5 -q "$f" 2>/dev/null; done
echo '--- QUARANTINE ---'
ls -lt /Users/arm/junyang/carbon/data/weights/quarantine_noncanonical/ 2>/dev/null | head -8
echo "STUDIOSTAT0227Y7T1 STUDIO-END"
STEOF
echo "$MARK END rc=$?"

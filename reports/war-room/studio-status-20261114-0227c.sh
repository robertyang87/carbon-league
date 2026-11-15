#!/bin/bash
# Studio follow-up probe 2026-11-14 02:27 trip, part 3 (marker STUDIOSTAT0227Z3Q8).
# Resolve: (a) pilot files in data/stage1_formal/ sizes+mtimes vs 02:13
# (2956 B/00:47:07, 492 B/00:19:42, 1268 B/00:25:15); (b) producer/guided
# paths under carbon/ (code/producer.py + guided_*.py do NOT exist — find real
# paths, md5 them); (c) s1v4accept acceptance_r5 results status + any process
# now running + current load; (d) disk delta 417->415Gi: du s1v4accept.
MARK=STUDIOSTAT0227Z3Q8
echo "$MARK BEGIN $(date '+%F %T %Z')"
timeout -k 30 120 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes \
  -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o GlobalKnownHostsFile=/dev/null \
  -o BatchMode=yes -o ConnectTimeout=25 \
  -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" \
  arm@100.78.149.20 'bash -s' <<'STEOF'
echo "STUDIOSTAT0227Z3Q8 STUDIO-BEGIN $(date '+%F %T %Z')"
uptime
echo '--- PILOT-FORMAL ---'
ls -l /Users/arm/junyang/carbon/data/stage1_formal/pilot_du_v2.log /Users/arm/junyang/carbon/data/stage1_formal/pilot_du_v2.progress.json /Users/arm/junyang/carbon/data/stage1_formal/pilot_du_v2.PARTIAL.md
echo '--- SIDECAR-CONTENT ---'
cat /Users/arm/junyang/carbon/data/stage1_formal/pilot_du_v2.progress.json
echo '--- PROC-NOW ---'
ps -ax -o pid,%cpu,%mem,etime,command | grep -E 'cnet2|train|fable|m3s2|wg1|wr1|m9xj|s1v4|accept|pilot' | grep -v grep || echo NO-TRAIN-PROC
pgrep -f spawn_main | wc -l
echo '--- PRODUCER-GUIDED-FIND ---'
find /Users/arm/junyang/carbon -maxdepth 3 -name 'producer*' -o -maxdepth 3 -name 'guided*' 2>/dev/null | head
ls /Users/arm/junyang/carbon/code/ | head -40
echo '--- R5-RESULTS ---'
ls -lt /Users/arm/s1v4accept/automation/acceptance_r5/results/ 2>/dev/null
for f in /Users/arm/s1v4accept/automation/acceptance_r5/results/stage0_ok.log /Users/arm/s1v4accept/automation/acceptance_r5/results/test_labels_r.log /Users/arm/s1v4accept/automation/acceptance_r5/results/test_ppo_wire_r.log /Users/arm/s1v4accept/automation/acceptance_r5/selfplay_err.log; do echo "== $f"; tail -3 "$f" 2>/dev/null || echo MISSING; done
echo '--- DU-V4 ---'
du -sh /Users/arm/s1v4accept 2>/dev/null
df -h / | tail -1
echo "STUDIOSTAT0227Z3Q8 STUDIO-END"
STEOF
echo "$MARK END rc=$?"

#!/bin/bash
# Studio status probe 0212: idle watch after trip-2 failure.
# Last read 02:00:20 (STUDIOSTAT0157): Studio normally idle, zero hakimi2
# traces 19 min after v2 dispatch; war room 02:03 confirmed trip-2 run
# 43272417220 ended in 50s (fake success, ssh layer unproven), third
# attempt awaits owner decision; Studio line continues 15-min direct-read
# standby, no self-start. This trip: any hakimi2 traces landed? any train
# proc? wr1 320 log still frozen 22:34? machine health.
# Marker: STUDIOSTAT0212HAKIMI
MARK=STUDIOSTAT0212HAKIMI
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout -k 30 420 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o UserKnownHostsFile=/root/.jishubuddy/network-checks/studio-known-hosts-20261112 arm@10.211.43.68 'bash -s' <<'STEOF'
echo "STUDIOSTAT0212HAKIMI STUDIO-BEGIN $(date '+%F %T %Z')"
hostname; uptime
echo '--- TRAIN-PROC ---'
ps -ax -o pid,%cpu,%mem,etime,command | grep -E 'cnet2|train|fable|m3s2|hakimi|wr1|k4v2|m9xj|320|distill' | grep -v grep || echo NO-TRAIN-PROC
echo '--- HAKIMI-TRACE ---'
find /Users/arm/junyang -maxdepth 4 -iname '*hakimi*' 2>/dev/null | head -20 || echo NO-HAKIMI-FILES
ls -l /Users/arm/junyang/carbon/data/hakimi2_matches.json 2>/dev/null || echo NO-HAKIMI-MATCHES
ls -d /Users/arm/junyang/carbon/data/hakimi2_corpus /Users/arm/junyang/carbon/data/ds_hakimi2 2>/dev/null || echo NO-HAKIMI-DATA
ls -lt /Users/arm/junyang/carbon/data/*hakimi* 2>/dev/null | head -5 || echo NO-HAKIMI-ARTIFACTS
echo '--- WR1-320-LOG ---'
ls -l /Users/arm/junyang/carbon/data/m9xj320wr1.log 2>/dev/null || echo NO-320WR1-LOG
tail -3 /Users/arm/junyang/carbon/data/m9xj320wr1.log 2>/dev/null || echo NO-320WR1-LOG
echo '--- NEW-LOGS ---'
ls -lt /Users/arm/junyang/carbon/data/*.log 2>/dev/null | head -6
echo '--- MEM-DISK ---'
df -h / | tail -1; vm_stat | head -3
echo "STUDIOSTAT0212HAKIMI STUDIO-END"
STEOF
echo "$MARK L20-END rc=$?"

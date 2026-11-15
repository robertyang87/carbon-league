#!/bin/bash
# Studio status probe 0242: trip-3 Stage 2 failed 02:25:37 (macOS lacks `timeout`, rc=127);
# war room now has the execution agent's own failure report (handoff ~02:36, exact root cause
# + fix direction: drop `timeout` on Studio side). No fixed re-dispatch visible in war room
# newest turns yet. This trip: verify by direct read — still idle? any hakimi traces?
# any new dispatch landed? it320 wr1 frozen 22:34? machine health.
# Marker: STUDIOSTAT0242HAKIMI
MARK=STUDIOSTAT0242HAKIMI
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout -k 30 420 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o UserKnownHostsFile=/root/.jishubuddy/network-checks/studio-known-hosts-20261112 arm@10.211.43.68 'bash -s' <<'STEOF'
echo "STUDIOSTAT0242HAKIMI STUDIO-BEGIN $(date '+%F %T %Z')"
hostname; uptime
echo '--- TRAIN-PROC ---'
ps -ax -o pid,%cpu,%mem,etime,command | grep -E 'cnet2|train|fable|hakimi|m3s2|wr1|320|distill' | grep -v grep || echo NO-TRAIN-PROC
echo '--- HAKIMI-TRACES ---'
ls -ld /Users/arm/junyang/matches /Users/arm/junyang/corpus /Users/arm/junyang/carbon/data/ds_hakimi2 /Users/arm/junyang/carbon/data/hakimi2_* 2>/dev/null || echo NO-HAKIMI-DIRS
find /Users/arm/junyang -maxdepth 3 -iname '*hakimi*' 2>/dev/null | head -20
echo '--- IT320-WR1 ---'
ls -l /Users/arm/junyang/carbon/data/m9xj320wr1.log 2>/dev/null
tail -4 /Users/arm/junyang/carbon/data/m9xj320wr1.log 2>/dev/null || echo NO-320WR1-LOG
echo '--- NEW-LOGS ---'
ls -lt /Users/arm/junyang/carbon/data/*.log 2>/dev/null | head -6
ls -lt /Users/arm/junyang/carbon/data/weights/ 2>/dev/null | head -5
echo '--- MEM-DISK ---'
df -h / | tail -1; vm_stat | head -3
echo "STUDIOSTAT0242HAKIMI STUDIO-END"
STEOF
echo "$MARK L20-END rc=$?"

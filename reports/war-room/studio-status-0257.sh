#!/bin/bash
# Studio status probe 0257: trip-4 approved 02:52 (owner) + execution agent dispatched,
# only change vs trip-3: drop all GNU `timeout` wrappers on Studio side (trip-3 died rc=127).
# This trip: direct read — has trip-4 Stage 1 probe / Stage 2 launch landed on Studio?
# Check training three-piece set (proc + MPS load + log/weight mtime) if anything started.
# Marker: STUDIOSTAT0257HAKIMI
MARK=STUDIOSTAT0257HAKIMI
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout -k 30 420 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o UserKnownHostsFile=/root/.jishubuddy/network-checks/studio-known-hosts-20261112 arm@10.211.43.68 'bash -s' <<'STEOF'
echo "STUDIOSTAT0257HAKIMI STUDIO-BEGIN $(date '+%F %T %Z')"
hostname; uptime
echo '--- TRAIN-PROC ---'
ps -ax -o pid,%cpu,%mem,etime,command | grep -E 'cnet2|train|fable|hakimi|m3s2|wr1|320|distill|build_top|set_uw' | grep -v grep || echo NO-TRAIN-PROC
echo '--- HAKIMI-TRACES ---'
ls -ld /Users/arm/junyang/carbon/data/hakimi2_* /Users/arm/junyang/carbon/data/ds_hakimi2 /Users/arm/junyang/matches /Users/arm/junyang/corpus 2>/dev/null || echo NO-HAKIMI-DIRS
find /Users/arm/junyang -maxdepth 4 -iname '*hakimi*' 2>/dev/null | head -30
echo '--- HAKIMI-LOGS ---'
ls -lt /Users/arm/junyang/carbon/data/hakimi2_*.log /Users/arm/junyang/carbon/data/hakimi2_*.json 2>/dev/null | head -10
tail -3 /Users/arm/junyang/carbon/data/hakimi2_chain.log 2>/dev/null || echo NO-CHAIN-LOG
tail -3 /Users/arm/junyang/carbon/data/hakimi2_verify.log 2>/dev/null || echo NO-VERIFY-LOG
echo '--- IT320-WR1 ---'
ls -l /Users/arm/junyang/carbon/data/m9xj320wr1.log 2>/dev/null
tail -2 /Users/arm/junyang/carbon/data/m9xj320wr1.log 2>/dev/null || echo NO-320WR1-LOG
echo '--- NEW-LOGS ---'
ls -lt /Users/arm/junyang/carbon/data/*.log 2>/dev/null | head -6
ls -lt /Users/arm/junyang/carbon/data/weights/ 2>/dev/null | head -5
echo '--- MEM-DISK ---'
df -h / | tail -1; vm_stat | head -3
echo "STUDIOSTAT0257HAKIMI STUDIO-END"
STEOF
echo "$MARK L20-END rc=$?"

#!/bin/bash
# Studio status probe 0357 (2026-11-13 ~03:57 CST trip).
# Context: trip-5 landed 03:39-03:42; Hakimi dual-arm training running at 03:45
# read (r1 PID 123 ep4/20 ~90s/ep -> done ~04:10 EST; g1 PID 125 ep11/20
# ~33s/ep -> done ~03:50 EST). This trip: epoch progress check, flag if
# 30 min without a new checkpoint. Read-only only.
# Marker: STUDIOSTAT0357HAKIMI
MARK=STUDIOSTAT0357HAKIMI
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout -k 30 420 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o UserKnownHostsFile=/root/.jishubuddy/network-checks/studio-known-hosts-20261112 arm@10.211.43.68 'bash -s' <<'STEOF'
echo "STUDIOSTAT0357HAKIMI STUDIO-BEGIN $(date '+%F %T %Z')"
hostname; uptime
echo '--- TRAIN-PROC ---'
ps -ax -o pid,%cpu,%mem,etime,command | grep -E 'cnet2|train|hakimi|fable|m3s2|wr1|320|distill|build_top|set_uw|ship_' | grep -v grep || echo NO-TRAIN-PROC
echo '--- HAKIMI-LOGS ---'
ls -lt /Users/arm/junyang/carbon/data/*.log 2>/dev/null | head -8
echo '--- HAKIMI-LOG-TAILS ---'
for f in /Users/arm/junyang/carbon/data/hakimi2_r1.log /Users/arm/junyang/carbon/data/hakimi2_g1.log; do echo "== $f"; ls -l "$f" 2>/dev/null; tail -4 "$f" 2>/dev/null || echo NO-LOG; done
echo '--- HAKIMI-CKPTS ---'
find /Users/arm/junyang -maxdepth 5 -iname '*hakimi*ep*.npz' 2>/dev/null | head -20
ls -lt /Users/arm/junyang/carbon/data/weights/ 2>/dev/null | head -8
echo '--- HAKIMI-DS ---'
ls -ld /Users/arm/junyang/carbon/data/ds_hakimi2 2>/dev/null; ls -l /Users/arm/junyang/carbon/data/ds_hakimi2/meta.json 2>/dev/null
echo '--- MEM-DISK ---'
df -h / | tail -1; vm_stat | head -3
echo "STUDIOSTAT0357HAKIMI STUDIO-END"
STEOF
echo "$MARK L20-END rc=$?"

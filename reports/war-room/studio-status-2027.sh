#!/bin/bash
# Studio status probe 2027: M9XJ-it260 triggered early at 97 games (owner order
# 20:22). War room dispatched DGX single-chain A1(r1)+A3(g1) and Studio A2
# weighted-attention arm cnet2_m9xj260wr1 (weighted set pushed from DGX chain,
# Studio local rebuild as fallback). This trip verifies by direct read whether
# Studio has taken over (A2 start three-piece set) or is still normally idle.
# Marker: STUDIOSTAT2027M9XJ
MARK=STUDIOSTAT2027M9XJ
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout -k 30 420 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o UserKnownHostsFile=/root/.jishubuddy/network-checks/studio-known-hosts-20261112 arm@10.211.43.68 'bash -s' <<'STEOF'
echo "STUDIOSTAT2027M9XJ STUDIO-BEGIN $(date '+%F %T %Z')"
hostname; uptime
echo '--- MEM-DISK ---'
df -h / | tail -1; vm_stat | head -6
echo '--- TRAIN-PROC ---'
ps -ax -o pid,%cpu,%mem,etime,command | grep -E 'cnet2|train|fable|m3s2|wg1|wr1|k4v2|m9xj|distill|relabel|lb_' | grep -v grep || echo NO-TRAIN-PROC
echo '--- SHIP-WATCHER ---'
ps -ax -o pid,etime,command | grep -E 'ship_' | grep -v grep || echo NO-SHIP-WATCHER
echo '--- NEW-LOGS ---'
ls -lt /Users/arm/junyang/carbon/data/*.log 2>/dev/null | head -10
echo '--- M9XJ-LOGS ---'
ls -lt /Users/arm/junyang/carbon/data/*m9xj* /Users/arm/junyang/carbon/data/*M9XJ* 2>/dev/null || echo NO-M9XJ-LOGS
echo '--- WG1-LOG-MTIME ---'
ls -l /Users/arm/junyang/carbon/data/m3s2wg1.log 2>/dev/null; tail -3 /Users/arm/junyang/carbon/data/m3s2wg1.log 2>/dev/null
echo '--- WEIGHTS-NEWEST ---'
ls -lt /Users/arm/junyang/carbon/data/weights/*.npz 2>/dev/null | head -8
echo '--- M9XJ-DATA ---'
ls -d /Users/arm/junyang/carbon/data/ds_m9xj* /Users/arm/junyang/carbon/data/*m9xj* 2>/dev/null || echo NO-M9XJ-DATA-YET
find /Users/arm/junyang/carbon/data -maxdepth 2 -iname '*m9xj*' 2>/dev/null | head -20 || true
echo '--- DS-M3S2-W ---'
du -sh /Users/arm/junyang/carbon/data/ds_m3s2_w 2>/dev/null
find /Users/arm/junyang/carbon/data/ds_m3s2_w -type f 2>/dev/null | wc -l
echo '--- DS-M3S2 (line B) ---'
du -sh /Users/arm/junyang/carbon/data/ds_m3s2 2>/dev/null || echo NO-DS-M3S2
find /Users/arm/junyang/carbon/data/ds_m3s2 -type f 2>/dev/null | wc -l
ls -lt /Users/arm/junyang/carbon/data/*relabel* /Users/arm/junyang/carbon/data/lb_* 2>/dev/null || echo NO-LB-FILES-YET
echo '--- DATA-TOTAL ---'
du -sh /Users/arm/junyang/carbon/data 2>/dev/null
echo "STUDIOSTAT2027M9XJ STUDIO-END"
STEOF
echo "$MARK L20-END rc=$?"

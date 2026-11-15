#!/bin/bash
# Studio status probe 1927: wg1 reported finished 19:16 (ep19 val 0.8643, best
# 0.8667, shipped to DGX md5 a0b2171c..., WG1_SHIP_DONE per war room). This trip
# verifies by direct read whether Studio is now normally idle (no train proc,
# no ship watcher) and that wg1 weights/logs are landed, awaiting M9XJ trigger
# (~20:25 per war room). Marker: STUDIOSTAT1927Q8Z2
MARK=STUDIOSTAT1927Q8Z2
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout -k 30 420 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o UserKnownHostsFile=/root/.jishubuddy/network-checks/studio-known-hosts-20261112 arm@10.211.43.68 'bash -s' <<'STEOF'
echo "STUDIOSTAT1927Q8Z2 STUDIO-BEGIN $(date '+%F %T %Z')"
hostname; uptime
echo '--- MEM-DISK ---'
df -h / | tail -1; vm_stat | head -6
echo '--- TRAIN-PROC ---'
ps -ax -o pid,%cpu,%mem,etime,command | grep -E 'cnet2|train|fable|m3s2|wg1|wr1|k4v2|m9xj|distill' | grep -v grep || echo NO-TRAIN-PROC
echo '--- SHIP-WATCHER ---'
ps -ax -o pid,etime,command | grep -E 'ship_' | grep -v grep || echo NO-SHIP-WATCHER
echo '--- WG1-LOGS ---'
for f in /Users/arm/junyang/carbon/data/m3s2wg1.log /Users/arm/junyang/carbon/data/m3s2wg1_ship.log; do [ -e "$f" ] || { echo "MISSING $f"; continue; }; echo "== $f =="; ls -l "$f"; tail -8 "$f"; done
ls -l /Users/arm/junyang/carbon/data/m3s2wg1.pid 2>/dev/null || echo NO-WG1-PIDFILE
echo '--- NEW-LOGS ---'
ls -lt /Users/arm/junyang/carbon/data/*.log 2>/dev/null | head -8
echo '--- WEIGHTS ---'
ls -lt /Users/arm/junyang/carbon/data/weights/cnet2_m3s2wg1* 2>/dev/null || echo NO-WG1-WEIGHTS
ls -lt /Users/arm/junyang/carbon/data/weights/*.npz 2>/dev/null | head -6
echo '--- DS-M3S2-W ---'
du -sh /Users/arm/junyang/carbon/data/ds_m3s2_w 2>/dev/null
find /Users/arm/junyang/carbon/data/ds_m3s2_w -type f 2>/dev/null | wc -l
echo '--- M9XJ-PRESTAGE ---'
ls -d /Users/arm/junyang/carbon/data/ds_m9xj* /Users/arm/junyang/carbon/data/*m9xj* 2>/dev/null || echo NO-M9XJ-DATA-YET
echo '--- DATA-TOTAL ---'
du -sh /Users/arm/junyang/carbon/data 2>/dev/null
echo "STUDIOSTAT1927Q8Z2 STUDIO-END"
STEOF
echo "$MARK L20-END rc=$?"

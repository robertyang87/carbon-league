#!/bin/bash
# Studio status probe 1912: war room dispatched wg1 (cnet2_m3s2wg1, ds_m3s2_w,
# BC_INIT=cnet2_b128m2t3.npz, 20 epochs) ~19:04. This trip verifies whether the
# execution agent has started it (three-piece: proc + fresh log/weight + load)
# or Studio is still idle awaiting start. Marker: STUDIOSTAT1912K7M3
MARK=STUDIOSTAT1912K7M3
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout -k 30 420 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o UserKnownHostsFile=/root/.jishubuddy/network-checks/studio-known-hosts-20261112 arm@10.211.43.68 'bash -s' <<'STEOF'
echo "STUDIOSTAT1912K7M3 STUDIO-BEGIN $(date '+%F %T %Z')"
hostname; uptime
echo '--- MEM-DISK ---'
df -h / | tail -1; vm_stat | head -6
echo '--- TRAIN-PROC ---'
ps -ax -o pid,%cpu,%mem,etime,command | grep -E 'cnet2|train|fable|m3s2|wg1|wr1|k4v2|m9xj|distill' | grep -v grep || echo NO-TRAIN-PROC
echo '--- SHIP-WATCHER ---'
ps -ax -o pid,etime,command | grep -E 'ship_' | grep -v grep || echo NO-SHIP-WATCHER
echo '--- WG1-LOGS ---'
for f in /Users/arm/junyang/carbon/data/m3s2wg1.log /Users/arm/junyang/carbon/data/m3s2wg1_ship.log; do [ -e "$f" ] || { echo "MISSING $f"; continue; }; echo "== $f =="; ls -l "$f"; tail -5 "$f"; done
ls -l /Users/arm/junyang/carbon/data/m3s2wg1.pid 2>/dev/null || echo NO-WG1-PIDFILE
echo '--- NEW-LOGS ---'
ls -lt /Users/arm/junyang/carbon/data/*.log 2>/dev/null | head -8
echo '--- WEIGHTS ---'
ls -lt /Users/arm/junyang/carbon/data/weights/cnet2_m3s2wg1* 2>/dev/null || echo NO-WG1-WEIGHTS
ls -lt /Users/arm/junyang/carbon/data/weights/*.npz 2>/dev/null | head -6
echo '--- DS-M3S2-W ---'
du -sh /Users/arm/junyang/carbon/data/ds_m3s2_w 2>/dev/null
cat /Users/arm/junyang/carbon/data/ds_m3s2_w/meta.json 2>/dev/null | head -20
find /Users/arm/junyang/carbon/data/ds_m3s2_w -type f 2>/dev/null | wc -l
echo '--- DATA-TOTAL ---'
du -sh /Users/arm/junyang/carbon/data 2>/dev/null
echo "STUDIOSTAT1912K7M3 STUDIO-END"
STEOF
echo "$MARK L20-END rc=$?"

#!/bin/bash
# Studio status probe 1827 (fresh script for this run, per task body):
# read-only check of Mac Studio. Weighted pilot line A (cnet2_m3s2wr1, PID 57882)
# was ep13/20 at 18:12-18:14 with est finish ~18:19-18:20; this trip checks
# whether it finished (ep19 checkpoint + ship watcher scp to DGX) and whether
# K4v2 staging (ds_k4v2 / k4v2 procs) has landed first-free after the pilot.
# Marker: STUDIOSTAT1827Q9W4 (claim only output containing this marker)
MARK=STUDIOSTAT1827Q9W4
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o UserKnownHostsFile=/root/.jishubuddy/network-checks/studio-known-hosts-20261112 arm@10.211.43.68 'bash -s' <<'STEOF'
echo "STUDIOSTAT1827Q9W4 STUDIO-BEGIN $(date '+%F %T %Z')"
hostname; uptime
echo '--- MEM-DISK ---'
df -h / | tail -1; vm_stat | head -6
echo '--- TRAIN-PROC ---'
ps -ax -o pid,%cpu,%mem,etime,command | grep -E 'cnet2|train|fable|m3s2|wr1|k4v2' | grep -v grep || echo NO-TRAIN-PROC
echo '--- SHIP-WATCHER ---'
ps -ax -o pid,etime,command | grep -E 'ship_' | grep -v grep || echo NO-SHIP-WATCHER
echo '--- LOGS ---'
ls -lt /Users/arm/junyang/carbon/data/*.log 2>/dev/null | head -12
for f in /Users/arm/junyang/carbon/data/m3s2wr1.log /Users/arm/junyang/carbon/data/m3s2wr1_ship.log /Users/arm/junyang/carbon/data/k4v2r1.log /Users/arm/junyang/carbon/data/k4v2g1.log /Users/arm/junyang/carbon/data/fable_r1.log /Users/arm/junyang/carbon/data/fable_g1.log; do [ -e "$f" ] || continue; echo "== $f =="; ls -l "$f"; tail -6 "$f"; done; echo PILOT-LOGS-DONE
echo '--- WEIGHTS ---'
ls -lt /Users/arm/junyang/carbon/data/weights 2>/dev/null | head -16
ls -lt /Users/arm/junyang/carbon/data 2>/dev/null | grep -iE 'm3s2w|wr1|fable|k4v2' | head -16
echo '--- DS ---'
for d in ds_m3s2_w ds_m3s2 ds_fable ds_k4v2; do echo "== $d =="; ls -ld /Users/arm/junyang/carbon/data/$d 2>/dev/null; du -sh /Users/arm/junyang/carbon/data/$d 2>/dev/null; find /Users/arm/junyang/carbon/data/$d -type f 2>/dev/null | wc -l; cat /Users/arm/junyang/carbon/data/$d/meta.json 2>/dev/null | head -5; done
echo '--- DATA-TOTAL ---'
du -sh /Users/arm/junyang/carbon/data 2>/dev/null
echo "STUDIOSTAT1827Q9W4 STUDIO-END"
STEOF
echo "$MARK L20-END rc=$?"

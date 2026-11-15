#!/bin/bash
# Studio status probe 1857: K4v2 done 18:42, Studio idle-normal, war room first
# notified 18:44. This trip verifies still idle (no new train proc, no new
# logs/weights since 18:43) and whether any new task/dataset has landed.
# Marker: STUDIOSTAT1857Q4F9 (claim only output containing this marker)
MARK=STUDIOSTAT1857Q4F9
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout -k 30 420 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o UserKnownHostsFile=/root/.jishubuddy/network-checks/studio-known-hosts-20261112 arm@10.211.43.68 'bash -s' <<'STEOF'
echo "STUDIOSTAT1857Q4F9 STUDIO-BEGIN $(date '+%F %T %Z')"
hostname; uptime
echo '--- MEM-DISK ---'
df -h / | tail -1; vm_stat | head -6
echo '--- TRAIN-PROC ---'
ps -ax -o pid,%cpu,%mem,etime,command | grep -E 'cnet2|train|fable|m3s2|wr1|k4v2|m9xj|distill' | grep -v grep || echo NO-TRAIN-PROC
echo '--- SHIP-WATCHER ---'
ps -ax -o pid,etime,command | grep -E 'ship_' | grep -v grep || echo NO-SHIP-WATCHER
echo '--- LOGS ---'
for f in /Users/arm/junyang/carbon/data/k4v2r1.log /Users/arm/junyang/carbon/data/k4v2g1.log /Users/arm/junyang/carbon/data/k4v2_ship.log; do [ -e "$f" ] || continue; echo "== $f =="; ls -l "$f"; tail -3 "$f"; done
echo '--- NEW-LOGS ---'
ls -lt /Users/arm/junyang/carbon/data/*.log 2>/dev/null | head -8
echo '--- WEIGHTS ---'
ls -lt /Users/arm/junyang/carbon/data/*.npz 2>/dev/null | head -10
ls -lt /Users/arm/junyang/carbon/data/weights/*.npz 2>/dev/null | head -10
echo '--- DS ---'
for d in ds_k4v2 ds_m3s2_w ds_m3s2 ds_fable; do echo "== $d =="; du -sh /Users/arm/junyang/carbon/data/$d 2>/dev/null; find /Users/arm/junyang/carbon/data/$d -type f 2>/dev/null | wc -l; done
echo '--- DATA-TOTAL ---'
du -sh /Users/arm/junyang/carbon/data 2>/dev/null
ls -lt /Users/arm/junyang/carbon/data 2>/dev/null | head -12
echo "STUDIOSTAT1857Q4F9 STUDIO-END"
STEOF
echo "$MARK L20-END rc=$?"

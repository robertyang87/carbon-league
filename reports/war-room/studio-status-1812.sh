#!/bin/bash
# Studio status probe 1812: 18:12 routine check — weighted pilot line A (cnet2_m3s2wr1) started 18:01 per war-room exec agent 9fe8523c receipt; 18:04 accept: PID 57882, ep0-2 done, val 0.8656, ~52s/epoch, est finish ~18:19, ship watcher PID 57914. This trip verifies the three-piece set mid-run (process in place + fresh log/weight mtime) and progress vs the ~18:19 estimate; also observes whether K4v2 staging (ds_k4v2 / k4v2 procs) has appeared, since war room dispatched the K4v2 exec agent first-free (Studio expected first after pilot ~18:19).
# Marker: STUDIOSTAT1812K7M3 (claim only output containing this marker)
MARK=STUDIOSTAT1812K7M3
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o UserKnownHostsFile=/root/.jishubuddy/network-checks/studio-known-hosts-20261112 arm@10.211.43.68 'bash -s' <<'STEOF'
echo "STUDIOSTAT1812K7M3 STUDIO-BEGIN $(date '+%F %T %Z')"
hostname; uptime
echo '--- MEM-DISK ---'
df -h / | tail -1; vm_stat | head -6
echo '--- TRAIN-PROC ---'
ps -ax -o pid,%cpu,%mem,etime,command | grep -E 'cnet2|distill|train|fable|m3s2|wr1|k4v2' | grep -v grep || echo NO-TRAIN-PROC
echo '--- SHIP-WATCHER ---'
ps -ax -o pid,etime,command | grep -E 'ship_m3s2wr1|ship_' | grep -v grep || echo NO-SHIP-WATCHER
echo '--- ALL-LOGS ---'
ls -lt /Users/arm/junyang/carbon/data/*.log 2>/dev/null | head -12
for f in /Users/arm/junyang/carbon/data/m3s2wr1.log /Users/arm/junyang/carbon/data/m3s2wr1_ship.log /Users/arm/junyang/carbon/data/fable_r1.log /Users/arm/junyang/carbon/data/fable_g1.log; do [ -e "$f" ] || continue; echo "== $f =="; ls -l "$f"; tail -6 "$f"; done; echo PILOT-LOGS-DONE
echo '--- WEIGHTS ---'
ls -lt /Users/arm/junyang/carbon/data/weights 2>/dev/null | head -16
ls -lt /Users/arm/junyang/carbon/data 2>/dev/null | grep -iE 'm3s2w|wr1|fable|k4v2' | head -16
echo '--- DS ---'
for d in ds_m3s2_w ds_m3s2 ds_fable ds_k4v2; do echo "== $d =="; ls -ld /Users/arm/junyang/carbon/data/$d 2>/dev/null; du -sh /Users/arm/junyang/carbon/data/$d 2>/dev/null; find /Users/arm/junyang/carbon/data/$d -type f 2>/dev/null | wc -l; cat /Users/arm/junyang/carbon/data/$d/meta.json 2>/dev/null | head -5; done
echo '--- DATA-TOTAL ---'
du -sh /Users/arm/junyang/carbon/data 2>/dev/null
echo "STUDIOSTAT1812K7M3 STUDIO-END"
STEOF
echo "$MARK L20-END rc=$?"

#!/bin/bash
# Studio status probe 1842 (fresh script for this run, per task body):
# read-only check of Mac Studio. K4v2 dual arms started ~18:25 (r1 PID 63855,
# g1 PID 63857, ship watcher 63860); est finish g1 ~18:37, r1 ~18:48.
# This trip checks whether g1 finished (ep19 + ship), r1 progress, and
# re-reads WEIGHTS mtime + ds_k4v2 file count (truncated last trip by the
# meta.json dump - now capped with head -c 300, weights section first).
# Marker: STUDIOSTAT1842K7M2 (claim only output containing this marker)
MARK=STUDIOSTAT1842K7M2
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout -k 30 420 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o UserKnownHostsFile=/root/.jishubuddy/network-checks/studio-known-hosts-20261112 arm@10.211.43.68 'bash -s' <<'STEOF'
echo "STUDIOSTAT1842K7M2 STUDIO-BEGIN $(date '+%F %T %Z')"
hostname; uptime
echo '--- MEM-DISK ---'
df -h / | tail -1; vm_stat | head -6
echo '--- TRAIN-PROC ---'
ps -ax -o pid,%cpu,%mem,etime,command | grep -E 'cnet2|train|fable|m3s2|wr1|k4v2' | grep -v grep || echo NO-TRAIN-PROC
echo '--- SHIP-WATCHER ---'
ps -ax -o pid,etime,command | grep -E 'ship_' | grep -v grep || echo NO-SHIP-WATCHER
echo '--- LOGS ---'
for f in /Users/arm/junyang/carbon/data/k4v2r1.log /Users/arm/junyang/carbon/data/k4v2g1.log /Users/arm/junyang/carbon/data/k4v2_ship.log; do [ -e "$f" ] || continue; echo "== $f =="; ls -l "$f"; tail -4 "$f"; done
echo '--- WEIGHTS ---'
ls -lt /Users/arm/junyang/carbon/data/*.npz 2>/dev/null | head -10
ls -lt /Users/arm/junyang/carbon/data/weights/*.npz 2>/dev/null | head -10
echo '--- DS ---'
for d in ds_k4v2 ds_m3s2_w ds_m3s2 ds_fable; do echo "== $d =="; du -sh /Users/arm/junyang/carbon/data/$d 2>/dev/null; find /Users/arm/junyang/carbon/data/$d -type f 2>/dev/null | wc -l; head -c 300 /Users/arm/junyang/carbon/data/$d/meta.json 2>/dev/null; echo; done
echo '--- DATA-TOTAL ---'
du -sh /Users/arm/junyang/carbon/data 2>/dev/null
echo "STUDIOSTAT1842K7M2 STUDIO-END"
STEOF
echo "$MARK L20-END rc=$?"

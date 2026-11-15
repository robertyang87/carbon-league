#!/bin/bash
# Studio status probe 2112: M9XJ-it260 A2 finish + ship verification.
# Last read 20:59: A2 cnet2_m9xj260wr1 RUNNING (PID 98547, ~20:47 start),
# ep15 in progress, weights ep0-ep14 landed, rate ~51s/ep, ep19 est ~21:03.
# This trip: verify finish (ep19 checkpoint + final/best val), ship guard
# delivery to DGX, and whether Studio is now normally idle (done) or still
# training. Line B (ds_m3s2 / relabel) re-check — last trip's window cut it.
# Marker: STUDIOSTAT2112M9XJ
MARK=STUDIOSTAT2112M9XJ
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout -k 30 420 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o UserKnownHostsFile=/root/.jishubuddy/network-checks/studio-known-hosts-20261112 arm@10.211.43.68 'bash -s' <<'STEOF'
echo "STUDIOSTAT2112M9XJ STUDIO-BEGIN $(date '+%F %T %Z')"
hostname; uptime
echo '--- TRAIN-PROC ---'
ps -ax -o pid,%cpu,%mem,etime,command | grep -E 'cnet2|train|fable|m3s2|wg1|wr1|k4v2|m9xj|distill|relabel|lb_' | grep -v grep || echo NO-TRAIN-PROC
echo '--- SHIP-WATCHER ---'
ps -ax -o pid,etime,command | grep -E 'ship_' | grep -v grep || echo NO-SHIP-WATCHER
echo '--- WR1-LOG ---'
ls -l /Users/arm/junyang/carbon/data/m9xj260wr1.log 2>/dev/null; tail -8 /Users/arm/junyang/carbon/data/m9xj260wr1.log 2>/dev/null
echo '--- WR1-EP ---'
grep -E 'ep[0-9]+|val|done|DONE|finish' /Users/arm/junyang/carbon/data/m9xj260wr1.log 2>/dev/null | tail -10
echo '--- M9XJ-WEIGHTS ---'
ls -lt /Users/arm/junyang/carbon/data/weights/*m9xj* 2>/dev/null | head -8 || echo NO-M9XJ-WEIGHTS
echo '--- SHIP-LOG ---'
ls -l /Users/arm/junyang/carbon/data/m9xj260wr1_ship.log 2>/dev/null; tail -5 /Users/arm/junyang/carbon/data/m9xj260wr1_ship.log 2>/dev/null
echo '--- MEM-DISK ---'
df -h / | tail -1; vm_stat | head -4
echo '--- DS-M3S2-W ---'
du -sh /Users/arm/junyang/carbon/data/ds_m3s2_w 2>/dev/null; find /Users/arm/junyang/carbon/data/ds_m3s2_w -type f 2>/dev/null | wc -l
echo '--- DS-M3S2-LINEB ---'
du -sh /Users/arm/junyang/carbon/data/ds_m3s2 2>/dev/null; find /Users/arm/junyang/carbon/data/ds_m3s2 -type f 2>/dev/null | wc -l; ls /Users/arm/junyang/carbon/data/lb_* 2>/dev/null | head -3 || echo NO-LB-FILES
echo '--- NEW-LOGS ---'
ls -lt /Users/arm/junyang/carbon/data/*.log 2>/dev/null | head -6
echo "STUDIOSTAT2112M9XJ STUDIO-END"
STEOF
echo "$MARK L20-END rc=$?"

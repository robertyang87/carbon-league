#!/bin/bash
# Studio status probe 2157: idle re-check + it320 arrival watch (freeze line 90).
# Last read 21:45: A2 cnet2_m9xj260wr1 DONE (ep19, best 0.8611, ship DONE 21:03:37,
# md5 f0f81835... match=YES, DGX confirmed weight landed 21:03:36). Studio idle
# (normal, task-done). Idle already reported to war room 21:15; do NOT re-report
# unless state changed. War room updates since: freeze conditions ALL MET 21:47
# (91 games, diff 0, Echo2 6 wins incl +1945); owner 21:44 ordered pipeline start;
# new DGX chain agent taking over (check old poller PID 264410 first, anti-double),
# then freeze/build ds_m9xj320_w. Studio A2 agent (dispatched 21:24) polling DGX
# ds_m9xj320_w READY. it320 was 88 games 21:38, 91 at 21:47.
# This trip: confirm still idle, watch for any it320 landing (expected NONE yet:
# DGX build only just starting ~21:47-21:50, build-to-process ~17 min; Studio pull
# comes after DGX build completes), line B re-check.
# Marker: STUDIOSTAT2157IT320
MARK=STUDIOSTAT2157IT320
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout -k 30 420 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o UserKnownHostsFile=/root/.jishubuddy/network-checks/studio-known-hosts-20261112 arm@10.211.43.68 'bash -s' <<'STEOF'
echo "STUDIOSTAT2157IT320 STUDIO-BEGIN $(date '+%F %T %Z')"
hostname; uptime
echo '--- TRAIN-PROC ---'
ps -ax -o pid,%cpu,%mem,etime,command | grep -E 'cnet2|train|fable|m3s2|wg1|wr1|k4v2|m9xj|320|distill|relabel|lb_' | grep -v grep || echo NO-TRAIN-PROC
echo '--- SHIP-WATCHER ---'
ps -ax -o pid,etime,command | grep -E 'ship_' | grep -v grep || echo NO-SHIP-WATCHER
echo '--- WR1-LOG-MTIME ---'
ls -l /Users/arm/junyang/carbon/data/m9xj260wr1.log /Users/arm/junyang/carbon/data/m9xj260wr1_ship.log 2>/dev/null
echo '--- IT320-ARRIVAL ---'
ls -d /Users/arm/junyang/carbon/data/*320* 2>/dev/null || echo NO-320-DIR
ls -lt /Users/arm/junyang/carbon/data/ 2>/dev/null | head -12
echo '--- NEW-LOGS ---'
ls -lt /Users/arm/junyang/carbon/data/*.log 2>/dev/null | head -6
echo '--- MEM-DISK ---'
df -h / | tail -1; vm_stat | head -3
echo '--- DS-M3S2-W ---'
du -sh /Users/arm/junyang/carbon/data/ds_m3s2_w 2>/dev/null; find /Users/arm/junyang/carbon/data/ds_m3s2_w -type f 2>/dev/null | wc -l
echo '--- DS-M3S2-LINEB ---'
du -sh /Users/arm/junyang/carbon/data/ds_m3s2 2>/dev/null; find /Users/arm/junyang/carbon/data/ds_m3s2 -type f 2>/dev/null | wc -l; ls /Users/arm/junyang/carbon/data/lb_* 2>/dev/null | head -3 || echo NO-LB-FILES
echo "STUDIOSTAT2157IT320 STUDIO-END"
STEOF
echo "$MARK L20-END rc=$?"

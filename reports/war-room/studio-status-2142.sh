#!/bin/bash
# Studio status probe 2142: idle re-check + it320 arrival watch (freeze line now 90).
# Last read 21:30: A2 cnet2_m9xj260wr1 DONE (ep19, best 0.8611, ship DONE 21:03:37,
# md5 f0f81835... match=YES, DGX confirmed weight landed 21:03:36). Studio idle
# (normal, task-done). Idle already reported to war room 21:15; do NOT re-report
# unless state changed. War room updates since: freeze line RAISED to 90 by owner
# 21:36 (reason 21:39: it320 playing Echo2, that game must be in freeze set;
# freeze needs diff 0 + >=90 games + Echo2 games = 6). DGX chain agent was
# mid-build of 85-version at 21:38:42, emergency-stopped, 85 scrapped, new chain
# agent waiting on 90-line. it320 at 88 games 21:38, 2 short. Studio A2 agent
# (dispatched 21:24) polling DGX ds_m9xj320_w READY.
# This trip: confirm still idle, watch for any it320 landing (expected NONE yet:
# freeze not reached), line B re-check.
# Marker: STUDIOSTAT2142IT320
MARK=STUDIOSTAT2142IT320
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout -k 30 420 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o UserKnownHostsFile=/root/.jishubuddy/network-checks/studio-known-hosts-20261112 arm@10.211.43.68 'bash -s' <<'STEOF'
echo "STUDIOSTAT2142IT320 STUDIO-BEGIN $(date '+%F %T %Z')"
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
echo "STUDIOSTAT2142IT320 STUDIO-END"
STEOF
echo "$MARK L20-END rc=$?"

#!/bin/bash
# Studio status probe 2127: A2 best-val reconfirm + idle re-check + it320 arrival watch.
# Last read 21:15: A2 cnet2_m9xj260wr1 DONE (ep19, best 0.8611, ship DONE 21:03:37,
# md5 f0f81835... match=YES, DGX confirmed weight landed 21:03:36). Studio idle
# (normal, task-done). Idle already reported to war room 21:15; do NOT re-report
# unless state changed. War room plan: it320 arms next when corpus ready
# (scout refresh ~21:26, corpus ~81 games) or line B r2 if it320 stalls.
# This trip: re-verify best val from Studio-side log, confirm still idle,
# watch for any it320 dataset/log/process landing, line B re-check.
# Marker: STUDIOSTAT2127IT320
MARK=STUDIOSTAT2127IT320
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout -k 30 420 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o UserKnownHostsFile=/root/.jishubuddy/network-checks/studio-known-hosts-20261112 arm@10.211.43.68 'bash -s' <<'STEOF'
echo "STUDIOSTAT2127IT320 STUDIO-BEGIN $(date '+%F %T %Z')"
hostname; uptime
echo '--- TRAIN-PROC ---'
ps -ax -o pid,%cpu,%mem,etime,command | grep -E 'cnet2|train|fable|m3s2|wg1|wr1|k4v2|m9xj|320|distill|relabel|lb_' | grep -v grep || echo NO-TRAIN-PROC
echo '--- SHIP-WATCHER ---'
ps -ax -o pid,etime,command | grep -E 'ship_' | grep -v grep || echo NO-SHIP-WATCHER
echo '--- WR1-BEST-VAL ---'
grep -E '^best|DONE|done' /Users/arm/junyang/carbon/data/m9xj260wr1.log 2>/dev/null | tail -3
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
echo "STUDIOSTAT2127IT320 STUDIO-END"
STEOF
echo "$MARK L20-END rc=$?"

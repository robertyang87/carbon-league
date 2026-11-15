#!/bin/bash
# Studio status probe 2327: idle watch, no change expected (last read 23:12 idle-normal).
# Last read 22:29: wr1 PID 22333 ep14 in progress, ep12 val 0.8646 best,
# est done ~22:33-22:34. Studio agent final report 22:39 claims: 20 epochs
# done 22:34, best val 0.8673 @ep8, ep19 0.8625, weight shipped to DGX
# (md5 71e8692f...). This trip: verify by direct read — ep19/DONE,
# log tail, checkpoints, ship log/out, train proc gone?, machine health,
# and whether Studio is now idle (normal idle = task done + shipped).
# Marker: STUDIOSTAT2327IT320
MARK=STUDIOSTAT2327IT320
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout -k 30 420 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o UserKnownHostsFile=/root/.jishubuddy/network-checks/studio-known-hosts-20261112 arm@10.211.43.68 'bash -s' <<'STEOF'
echo "STUDIOSTAT2327IT320 STUDIO-BEGIN $(date '+%F %T %Z')"
hostname; uptime
echo '--- TRAIN-PROC ---'
ps -ax -o pid,%cpu,%mem,etime,command | grep -E 'cnet2|train|fable|m3s2|wg1|wr1|k4v2|m9xj|320|distill|relabel|lb_' | grep -v grep || echo NO-TRAIN-PROC
echo '--- SHIP-WATCHER ---'
ps -ax -o pid,etime,command | grep -E 'ship_' | grep -v grep || echo NO-SHIP-WATCHER
echo '--- WR1-320-LOG ---'
ls -l /Users/arm/junyang/carbon/data/m9xj320wr1.log /Users/arm/junyang/carbon/data/m9xj320wr1_ship.log /Users/arm/junyang/carbon/data/m9xj320wr1_ship.out /Users/arm/junyang/carbon/data/m9xj320wr1.pid 2>/dev/null || echo NO-320WR1-LOG
echo '--- WR1-320-LOG-TAIL ---'
tail -8 /Users/arm/junyang/carbon/data/m9xj320wr1.log 2>/dev/null || echo NO-320WR1-LOG
echo '--- WR1-320-SHIP-LOG ---'
cat /Users/arm/junyang/carbon/data/m9xj320wr1_ship.log 2>/dev/null || echo NO-SHIP-LOG
echo '--- WR1-320-CKPT ---'
ls -lt /Users/arm/junyang/carbon/data/weights/ 2>/dev/null | grep -i 320 | head -10 || echo NO-320-WEIGHTS
ls -lt /Users/arm/junyang/carbon/data/cnet2_m9xj320* 2>/dev/null | head -8 || echo NO-320-CKPT
grep -o '"ep": 19[^}]*"val": {[^}]*}' /Users/arm/junyang/carbon/data/m9xj320wr1.log 2>/dev/null | tail -2
grep -o 'ep19[^,]*, "val": {[^}]*}' /Users/arm/junyang/carbon/data/m9xj320wr1_log.jsonl 2>/dev/null | tail -2
echo '--- NEW-LOGS ---'
ls -lt /Users/arm/junyang/carbon/data/*.log 2>/dev/null | head -6
echo '--- MEM-DISK ---'
df -h / | tail -1; vm_stat | head -3
echo '--- DS-320 ---'
du -sh /Users/arm/junyang/carbon/data/ds_m9xj320 /Users/arm/junyang/carbon/data/ds_m9xj320_w 2>/dev/null
echo '--- DS-M3S2-LINEB ---'
du -sh /Users/arm/junyang/carbon/data/ds_m3s2 2>/dev/null; ls /Users/arm/junyang/carbon/data/lb_* 2>/dev/null | head -3 || echo NO-LB-FILES
echo "STUDIOSTAT2327IT320 STUDIO-END"
STEOF
echo "$MARK L20-END rc=$?"

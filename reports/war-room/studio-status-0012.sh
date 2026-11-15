#!/bin/bash
# Studio status probe 0012 (2026-11-13 00:12 CST trip). Last direct read 00:00:12:
# Studio normally idle (it320 wr1 done 22:34, best val 0.8673 @ep8, weight shipped
# to DGX md5 71e8692f...); war room 22:45 confirmed standby; next job = Line B
# relabel (ds_m3s2 lb_*), hard-gated on K4 probe pass + override >=2%.
# War-room context 00:07: K4 probe both arms finished (r1 -623, g1 -544), both
# missed pre-registered PASS, so Line B gate NOT met; Studio stays standby.
# Two fix agents dispatched 23:59 for F01/F04/F03 + F02 (main only, no deploy).
# This trip: verify by direct read — any train proc? wr1 log unchanged?
# new logs? lb_* landed? machine health. Read-only only.
# Marker: STUDIOSTAT0012
MARK=STUDIOSTAT0012
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout -k 30 420 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o UserKnownHostsFile=/root/.jishubuddy/network-checks/studio-known-hosts-20261112 arm@10.211.43.68 'bash -s' <<'STEOF'
echo "STUDIOSTAT0012 STUDIO-BEGIN $(date '+%F %T %Z')"
hostname; uptime
echo '--- TRAIN-PROC ---'
ps -ax -o pid,%cpu,%mem,etime,command | grep -E 'cnet2|train|fable|m3s2|wg1|wr1|k4v2|m9xj|320|distill|relabel|lb_' | grep -v grep || echo NO-TRAIN-PROC
echo '--- SHIP-WATCHER ---'
ps -ax -o pid,etime,command | grep -E 'ship_' | grep -v grep || echo NO-SHIP-WATCHER
echo '--- WR1-320-LOG ---'
ls -l /Users/arm/junyang/carbon/data/m9xj320wr1.log /Users/arm/junyang/carbon/data/m9xj320wr1_ship.log 2>/dev/null || echo NO-320WR1-LOG
echo '--- WR1-320-LOG-TAIL ---'
tail -5 /Users/arm/junyang/carbon/data/m9xj320wr1.log 2>/dev/null || echo NO-320WR1-LOG
echo '--- WR1-320-CKPT ---'
ls -lt /Users/arm/junyang/carbon/data/weights/ 2>/dev/null | grep -i 320 | head -6 || echo NO-320-WEIGHTS
echo '--- NEW-LOGS ---'
ls -lt /Users/arm/junyang/carbon/data/*.log 2>/dev/null | head -6
echo '--- MEM-DISK ---'
df -h / | tail -1; vm_stat | head -3
echo '--- DS-M3S2-LINEB ---'
du -sh /Users/arm/junyang/carbon/data/ds_m3s2 2>/dev/null; ls /Users/arm/junyang/carbon/data/lb_* 2>/dev/null | head -3 || echo NO-LB-FILES
echo '--- FABLE-CHECK ---'
ls -l /Users/arm/junyang/carbon/data/fable_*.log 2>/dev/null || echo NO-FABLE-LOGS
echo "STUDIOSTAT0012 STUDIO-END"
STEOF
echo "$MARK L20-END rc=$?"

#!/bin/bash
# Studio status probe 2257: idle continuation watch after it320 wr1 done+shipped 22:34.
# Last read 22:44: wr1 done, best 0.8673 @ep8, shipped md5 match, Studio normal idle,
# war room 22:45 replied "standby, no new job" (next job = Line B relabel gated on
# K4 probe pass + override >=2%, probe running on spot-28/C2).
# This trip: verify idle persists — no train proc, no ship watcher, wr1 logs/ckpts
# mtime frozen at 22:34, no new landing (lb_*, new datasets/logs), machine health.
# Marker: STUDIOSTAT2257IT320
MARK=STUDIOSTAT2257IT320
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout -k 30 420 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o UserKnownHostsFile=/root/.jishubuddy/network-checks/studio-known-hosts-20261112 arm@10.211.43.68 'bash -s' <<'STEOF'
echo "STUDIOSTAT2257IT320 STUDIO-BEGIN $(date '+%F %T %Z')"
hostname; uptime
echo '--- TRAIN-PROC ---'
ps -ax -o pid,%cpu,%mem,etime,command | grep -E 'cnet2|train|fable|m3s2|wg1|wr1|k4v2|m9xj|320|distill|relabel|lb_' | grep -v grep || echo NO-TRAIN-PROC
echo '--- SHIP-WATCHER ---'
ps -ax -o pid,etime,command | grep -E 'ship_' | grep -v grep || echo NO-SHIP-WATCHER
echo '--- WR1-320-LOG-MTIME ---'
ls -l /Users/arm/junyang/carbon/data/m9xj320wr1.log /Users/arm/junyang/carbon/data/m9xj320wr1_ship.log 2>/dev/null
echo '--- WR1-320-LOG-TAIL ---'
tail -3 /Users/arm/junyang/carbon/data/m9xj320wr1.log 2>/dev/null
echo '--- WR1-320-CKPT ---'
ls -lt /Users/arm/junyang/carbon/data/weights/ 2>/dev/null | grep -i 320 | head -5
echo '--- NEW-LOGS ---'
ls -lt /Users/arm/junyang/carbon/data/*.log 2>/dev/null | head -8
echo '--- NEW-DIRS ---'
ls -lt /Users/arm/junyang/carbon/data/ 2>/dev/null | head -12
echo '--- MEM-DISK ---'
df -h / | tail -1; vm_stat | head -3
echo '--- DS-M3S2-LINEB ---'
du -sh /Users/arm/junyang/carbon/data/ds_m3s2 2>/dev/null; ls /Users/arm/junyang/carbon/data/lb_* 2>/dev/null | head -3 || echo NO-LB-FILES
echo "STUDIOSTAT2257IT320 STUDIO-END"
STEOF
echo "$MARK L20-END rc=$?"

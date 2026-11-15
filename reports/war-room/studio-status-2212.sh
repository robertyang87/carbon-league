#!/bin/bash
# Studio status probe 2212: it320 A2 wr1 landing watch (DGX READY 21:50:14).
# Last read 22:02: Studio idle, NO it320 landing (no *320* dir/log/process).
# DGX updates since: ds_m9xj320_w READY 21:50:14 (freeze 91 games, chain DONE);
# DGX g1 DONE 22:05:44 best 0.8673; DGX r1 ep15/20 at 22:09:16, best 0.8671,
# est ~22:12 done. Studio A2 agent (dispatched 21:24) polls DGX READY, then
# pulls ds_m9xj320_w via LAN and starts cnet2_m9xj320wr1. This trip: verify
# landing (ds dir / wr1 log / train proc / checkpoints); if still nothing,
# note elapsed-since-READY (~22 min) and whether any pull/poll process is visible.
# Marker: STUDIOSTAT2212IT320
MARK=STUDIOSTAT2212IT320
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout -k 30 420 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o UserKnownHostsFile=/root/.jishubuddy/network-checks/studio-known-hosts-20261112 arm@10.211.43.68 'bash -s' <<'STEOF'
echo "STUDIOSTAT2212IT320 STUDIO-BEGIN $(date '+%F %T %Z')"
hostname; uptime
echo '--- TRAIN-PROC ---'
ps -ax -o pid,%cpu,%mem,etime,command | grep -E 'cnet2|train|fable|m3s2|wg1|wr1|k4v2|m9xj|320|distill|relabel|lb_' | grep -v grep || echo NO-TRAIN-PROC
echo '--- SHIP-WATCHER ---'
ps -ax -o pid,etime,command | grep -E 'ship_' | grep -v grep || echo NO-SHIP-WATCHER
echo '--- POLL-PULL-PROC ---'
ps -ax -o pid,etime,command | grep -E 'poll|READY|scp|rsync|320wr1' | grep -v grep || echo NO-POLL-PROC
echo '--- WR1-LOG-MTIME ---'
ls -l /Users/arm/junyang/carbon/data/m9xj260wr1.log /Users/arm/junyang/carbon/data/m9xj260wr1_ship.log 2>/dev/null
echo '--- IT320-ARRIVAL ---'
ls -d /Users/arm/junyang/carbon/data/*320* 2>/dev/null || echo NO-320-DIR
ls -lt /Users/arm/junyang/carbon/data/ 2>/dev/null | head -12
echo '--- WR1-320-LOG ---'
ls -l /Users/arm/junyang/carbon/data/m9xj320wr1*.log /Users/arm/junyang/carbon/data/*320wr1* 2>/dev/null || echo NO-320WR1-LOG
ls -lt /Users/arm/junyang/carbon/data/cnet2_m9xj320* 2>/dev/null | head -8 || echo NO-320-CKPT
echo '--- NEW-LOGS ---'
ls -lt /Users/arm/junyang/carbon/data/*.log 2>/dev/null | head -6
echo '--- MEM-DISK ---'
df -h / | tail -1; vm_stat | head -3
echo '--- DS-M3S2-W ---'
du -sh /Users/arm/junyang/carbon/data/ds_m3s2_w 2>/dev/null; find /Users/arm/junyang/carbon/data/ds_m3s2_w -type f 2>/dev/null | wc -l
echo '--- DS-M3S2-LINEB ---'
du -sh /Users/arm/junyang/carbon/data/ds_m3s2 2>/dev/null; find /Users/arm/junyang/carbon/data/ds_m3s2 -type f 2>/dev/null | wc -l; ls /Users/arm/junyang/carbon/data/lb_* 2>/dev/null | head -3 || echo NO-LB-FILES
echo "STUDIOSTAT2212IT320 STUDIO-END"
STEOF
echo "$MARK L20-END rc=$?"

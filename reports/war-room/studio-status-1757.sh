#!/bin/bash
# Studio status probe 1757: 17:57 routine check — Fable both arms DONE 20/20 (g1 17:08 best 0.9336, r1 17:20 best 0.9348), Studio idle-normal reported to war room 17:31. War room then dispatched execution agent 9fe8523c for weighted pilot line A: build ds_m3s2_w from ds_m3s2 (train-set endgame rows step>=270 x3, val untouched), attention single arm cnet2_m3s2wr1, 20 epochs, ep19 scp back to DGX with md5. This trip verifies the three-piece set (process in place + fresh log/weight mtime) for the pilot; if no three-piece set within one cycle, report "dispatch not landed" to war room.
# Marker: STUDIOSTAT1757Q4K9 (claim only output containing this marker)
MARK=STUDIOSTAT1757Q4K9
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o UserKnownHostsFile=/root/.jishubuddy/network-checks/studio-known-hosts-20261112 arm@10.211.43.68 'bash -s' <<'STEOF'
echo "STUDIOSTAT1757Q4K9 STUDIO-BEGIN $(date '+%F %T %Z')"
hostname; uptime
echo '--- MEM-DISK ---'
df -h / | tail -1; vm_stat | head -6
echo '--- TRAIN-PROC ---'
ps -ax -o pid,%cpu,%mem,etime,command | grep -E 'cnet2|distill|train|fable|m3s2|wr1' | grep -v grep || echo NO-TRAIN-PROC
echo '--- ALL-LOGS ---'
ls -lt /Users/arm/junyang/carbon/data/*.log 2>/dev/null | head -12
for f in /Users/arm/junyang/carbon/data/*m3s2w*.log /Users/arm/junyang/carbon/data/*wr1*.log /Users/arm/junyang/carbon/data/fable_r1.log /Users/arm/junyang/carbon/data/fable_g1.log; do [ -e "$f" ] || continue; echo "== $f =="; ls -l "$f"; tail -5 "$f"; done; echo PILOT-LOGS-DONE
echo '--- WEIGHTS ---'
ls -lt /Users/arm/junyang/carbon/data/weights 2>/dev/null | head -16
ls -lt /Users/arm/junyang/carbon/data 2>/dev/null | grep -iE 'm3s2w|wr1|fable' | head -16
echo '--- DS ---'
for d in ds_m3s2_w ds_m3s2 ds_fable; do echo "== $d =="; ls -ld /Users/arm/junyang/carbon/data/$d 2>/dev/null; du -sh /Users/arm/junyang/carbon/data/$d 2>/dev/null; find /Users/arm/junyang/carbon/data/$d -type f 2>/dev/null | wc -l; cat /Users/arm/junyang/carbon/data/$d/meta.json 2>/dev/null | head -5; done
echo '--- DATA-TOTAL ---'
du -sh /Users/arm/junyang/carbon/data 2>/dev/null
echo "STUDIOSTAT1757Q4K9 STUDIO-END"
STEOF
echo "$MARK L20-END rc=$?"

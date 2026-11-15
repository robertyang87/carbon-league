#!/bin/bash
# Studio status probe 1727: 17:27 routine check — 17:14 verified r1 ep13 done, ep14 in progress, g1 DONE 20/20; r1 ETA ~17:20-17:24 so this trip checks whether r1 finished and the ship watcher shipped weights. This trip independently verifies the three-piece (process + MPS load + fresh logs/checkpoints) and measures epoch progress.
# Marker: STUDIOSTAT1727K5M9 (claim only output containing this marker)
MARK=STUDIOSTAT1727K5M9
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o UserKnownHostsFile=/root/.jishubuddy/network-checks/studio-known-hosts-20261112 arm@10.211.43.68 'bash -s' <<'STEOF'
echo "STUDIOSTAT1727K5M9 STUDIO-BEGIN $(date '+%F %T %Z')"
hostname; uptime
echo '--- MEM-DISK ---'
df -h / | tail -1; vm_stat | head -6
echo '--- TRAIN-PROC ---'
ps -ax -o pid,%cpu,%mem,etime,command | grep -E 'cnet2|distill|train|fable' | grep -v grep || echo NO-TRAIN-PROC
echo '--- FABLE-LOGS ---'
ls -lt /Users/arm/junyang/carbon/data/*.log 2>/dev/null | head -8
for f in /Users/arm/junyang/carbon/data/fable_r1.log /Users/arm/junyang/carbon/data/fable_g1.log; do echo "== $f =="; ls -l "$f" 2>/dev/null; tail -4 "$f" 2>/dev/null; done; echo FABLE-LOGS-DONE
echo '--- FABLE-EP-PROGRESS ---'
for f in /Users/arm/junyang/carbon/data/fable_r1.log /Users/arm/junyang/carbon/data/fable_g1.log; do echo "== $f =="; grep -o '"ep": [0-9]*' "$f" 2>/dev/null | tail -3; grep -o '"val": {"all": [0-9.]*' "$f" 2>/dev/null | tail -2; done
echo '--- FABLE-WEIGHTS ---'
ls -lt /Users/arm/junyang/carbon/data/weights 2>/dev/null | head -12
ls -lt /Users/arm/junyang/carbon/data 2>/dev/null | grep -i fable | head -10
echo '--- DS_FABLE ---'
ls /Users/arm/junyang/carbon/data/ds_fable 2>/dev/null | head -5; du -sh /Users/arm/junyang/carbon/data/ds_fable 2>/dev/null; find /Users/arm/junyang/carbon/data/ds_fable -type f 2>/dev/null | wc -l
echo '--- DATA-TOTAL ---'
du -sh /Users/arm/junyang/carbon/data 2>/dev/null
echo "STUDIOSTAT1727K5M9 STUDIO-END"
STEOF
echo "$MARK L20-END rc=$?"

#!/bin/bash
# Studio status probe 1657: 16:57 routine check — war room 16:52 claims Fable dual-arm launched (r1 PID 40277 ep0 val 0.9171, g1 PID 40278 ep0 val 0.9148, ep2). This trip independently verifies the three-piece (process + MPS load + fresh logs/checkpoints) and measures epoch progress.
# Marker: STUDIOSTAT1657K9M2 (claim only output containing this marker)
MARK=STUDIOSTAT1657K9M2
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o UserKnownHostsFile=/root/.jishubuddy/network-checks/studio-known-hosts-20261112 arm@10.211.43.68 'bash -s' <<'STEOF'
echo "STUDIOSTAT1657K9M2 STUDIO-BEGIN $(date '+%F %T %Z')"
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
echo "STUDIOSTAT1657K9M2 STUDIO-END"
STEOF
echo "$MARK L20-END rc=$?"

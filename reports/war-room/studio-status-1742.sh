#!/bin/bash
# Studio status probe 1742: 17:42 routine check — 17:29 verified Fable both arms DONE 20/20 (g1 best 0.9336, r1 best 0.9348), Studio idle-normal, already reported to war room 17:31. War room 17:19 says next job is K4v2 (~17:45 trigger) or M9XJ (~18:30). This trip checks whether anything new has started, and re-verifies idle state.
# Marker: STUDIOSTAT1742H7Q2 (claim only output containing this marker)
MARK=STUDIOSTAT1742H7Q2
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o UserKnownHostsFile=/root/.jishubuddy/network-checks/studio-known-hosts-20261112 arm@10.211.43.68 'bash -s' <<'STEOF'
echo "STUDIOSTAT1742H7Q2 STUDIO-BEGIN $(date '+%F %T %Z')"
hostname; uptime
echo '--- MEM-DISK ---'
df -h / | tail -1; vm_stat | head -6
echo '--- TRAIN-PROC ---'
ps -ax -o pid,%cpu,%mem,etime,command | grep -E 'cnet2|distill|train|fable|k4|m9' | grep -v grep || echo NO-TRAIN-PROC
echo '--- ALL-LOGS ---'
ls -lt /Users/arm/junyang/carbon/data/*.log 2>/dev/null | head -10
for f in /Users/arm/junyang/carbon/data/fable_r1.log /Users/arm/junyang/carbon/data/fable_g1.log; do echo "== $f =="; ls -l "$f" 2>/dev/null; tail -3 "$f" 2>/dev/null; done; echo FABLE-LOGS-DONE
echo '--- WEIGHTS ---'
ls -lt /Users/arm/junyang/carbon/data/weights 2>/dev/null | head -12
ls -lt /Users/arm/junyang/carbon/data 2>/dev/null | grep -iE 'fable|k4|m9' | head -12
echo '--- DS_FABLE ---'
du -sh /Users/arm/junyang/carbon/data/ds_fable 2>/dev/null; find /Users/arm/junyang/carbon/data/ds_fable -type f 2>/dev/null | wc -l
echo '--- DATA-TOTAL ---'
du -sh /Users/arm/junyang/carbon/data 2>/dev/null
echo "STUDIOSTAT1742H7Q2 STUDIO-END"
STEOF
echo "$MARK L20-END rc=$?"

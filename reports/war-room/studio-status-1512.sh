#!/bin/bash
# Studio status probe 1512: 15:12 routine check — r1 due ~15:07, check done + auto-ship to DGX, g1 unchanged
# Marker: STUDIOSTAT1512QW7R (claim only output containing this marker)
MARK=STUDIOSTAT1512QW7R
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o UserKnownHostsFile=/root/.jishubuddy/network-checks/studio-known-hosts-20261112 arm@10.211.43.68 'bash -s' <<'STEOF'
echo "STUDIOSTAT1512QW7R STUDIO-BEGIN $(date '+%F %T %Z')"
hostname; uptime
echo '--- MEM-DISK ---'
df -h / | tail -1
echo '--- TRAIN-PROC ---'
ps -ax -o pid,%cpu,%mem,etime,command | grep -E 'cnet2|distill' | grep -v grep || echo NO-DISTILL-PROC
echo '--- G1-LOG-TAIL ---'
tail -2 /Users/arm/junyang/carbon/data/distill_g1.log 2>/dev/null
echo '--- R1-LOG-TAIL ---'
tail -5 /Users/arm/junyang/carbon/data/distill_r1.log 2>/dev/null
echo '--- R1-SHIPPED ---'
grep -E 'SHIPPED|DONE|COMPLETE' /Users/arm/junyang/carbon/data/distill_r1.log 2>/dev/null | tail -3
echo '--- R1-JSONL-LAST3 ---'
tail -3 /Users/arm/junyang/carbon/data/weights/cnet2_distill_r1_log.jsonl 2>/dev/null
echo '--- WEIGHTS-NOW ---'
ls -lt /Users/arm/junyang/carbon/data/weights 2>/dev/null | head -8
echo '--- DATASETS ---'
du -sh /Users/arm/junyang/carbon/data 2>/dev/null
echo "STUDIOSTAT1512QW7R STUDIO-END"
STEOF
echo "$MARK L20-END rc=$?"

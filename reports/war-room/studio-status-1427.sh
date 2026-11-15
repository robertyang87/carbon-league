#!/bin/bash
# Studio status probe 1427: routine 15-min check, distill dual-arm progress (g1 ETA ~14:43, r1 ETA ~15:28 per 14:20 read)
# Marker: STUDIOSTAT1427K9M2 (claim only output containing this marker)
MARK=STUDIOSTAT1427K9M2
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o UserKnownHostsFile=/root/.jishubuddy/network-checks/studio-known-hosts-20261112 arm@10.211.43.68 'bash -s' <<'STEOF'
echo "STUDIOSTAT1427K9M2 STUDIO-BEGIN $(date '+%F %T %Z')"
hostname; uptime
echo '--- MEM-DISK ---'
df -h / | tail -1
echo '--- TRAIN-PROC ---'
ps -ax -o pid,%cpu,%mem,etime,command | grep -E 'cnet2|distill' | grep -v grep || echo NO-DISTILL-PROC
echo '--- R1-LOG-TAIL ---'
tail -6 /Users/arm/junyang/carbon/data/distill_r1.log 2>/dev/null
echo '--- G1-LOG-TAIL ---'
tail -6 /Users/arm/junyang/carbon/data/distill_g1.log 2>/dev/null
echo '--- R1-JSONL ---'
cat /Users/arm/junyang/carbon/data/weights/cnet2_distill_r1_log.jsonl 2>/dev/null
echo '--- G1-JSONL ---'
cat /Users/arm/junyang/carbon/data/weights/cnet2_distill_g1_log.jsonl 2>/dev/null
echo '--- WEIGHTS-NOW ---'
ls -lt /Users/arm/junyang/carbon/data/weights/ 2>/dev/null | head -8
echo '--- DATASETS ---'
du -sh /Users/arm/junyang/carbon/data 2>/dev/null
echo "STUDIOSTAT1427K9M2 STUDIO-END"
STEOF
echo "$MARK L20-END rc=$?"

#!/bin/bash
# Studio status probe 1454: war-room reconciliation 14:53 — g1 final md5 + best val, r1 progress vs ~15:28 ETA
# Marker: STUDIOSTAT1454ZQ8W (claim only output containing this marker)
MARK=STUDIOSTAT1454ZQ8W
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o UserKnownHostsFile=/root/.jishubuddy/network-checks/studio-known-hosts-20261112 arm@10.211.43.68 'bash -s' <<'STEOF'
echo "STUDIOSTAT1454ZQ8W STUDIO-BEGIN $(date '+%F %T %Z')"
hostname; uptime
echo '--- MEM-DISK ---'
df -h / | tail -1
echo '--- TRAIN-PROC ---'
ps -ax -o pid,%cpu,%mem,etime,command | grep -E 'cnet2|distill' | grep -v grep || echo NO-DISTILL-PROC
echo '--- G1-LOG-TAIL ---'
tail -4 /Users/arm/junyang/carbon/data/distill_g1.log 2>/dev/null
echo '--- R1-LOG-TAIL ---'
tail -4 /Users/arm/junyang/carbon/data/distill_r1.log 2>/dev/null
echo '--- G1-JSONL-LAST3 ---'
tail -3 /Users/arm/junyang/carbon/data/weights/cnet2_distill_g1_log.jsonl 2>/dev/null
echo '--- R1-JSONL-LAST3 ---'
tail -3 /Users/arm/junyang/carbon/data/weights/cnet2_distill_r1_log.jsonl 2>/dev/null
echo '--- G1-FINAL-MD5 ---'
W=/Users/arm/junyang/carbon/data/weights
ls -l "$W"/cnet2_distill_g1_ep19.npz "$W"/cnet2_distill_g1_last.npz 2>/dev/null
md5 -q "$W"/cnet2_distill_g1_ep19.npz 2>/dev/null || md5sum "$W"/cnet2_distill_g1_ep19.npz 2>/dev/null
echo '--- WEIGHTS-NOW ---'
ls -lt "$W" 2>/dev/null | head -8
echo '--- DATASETS ---'
du -sh /Users/arm/junyang/carbon/data 2>/dev/null
echo "STUDIOSTAT1454ZQ8W STUDIO-END"
STEOF
echo "$MARK L20-END rc=$?"

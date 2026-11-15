#!/bin/bash
# Studio log-tail follow-up: distill dual-arm epoch val/loss (probe 1 had proc + weights mtime; tail was missing)
# Marker: STUDIOSTAT1415L3P8 (claim only output containing this marker)
MARK=STUDIOSTAT1415L3P8
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o UserKnownHostsFile=/root/.jishubuddy/network-checks/studio-known-hosts-20261112 arm@10.211.43.68 'bash -s' <<'STEOF'
echo "STUDIOSTAT1415L3P8 STUDIO-BEGIN $(date '+%F %T %Z')"
echo '--- R1-LOG-TAIL ---'
tail -6 /Users/arm/junyang/carbon/data/distill_r1.log 2>/dev/null
echo '--- G1-LOG-TAIL ---'
tail -6 /Users/arm/junyang/carbon/data/distill_g1.log 2>/dev/null
echo '--- R1-JSONL ---'
cat /Users/arm/junyang/carbon/data/weights/cnet2_distill_r1_log.jsonl 2>/dev/null
echo '--- G1-JSONL ---'
cat /Users/arm/junyang/carbon/data/weights/cnet2_distill_g1_log.jsonl 2>/dev/null
echo '--- WEIGHTS-NOW ---'
ls -lt /Users/arm/junyang/carbon/data/weights/ 2>/dev/null | head -6
echo "STUDIOSTAT1415L3P8 STUDIO-END"
STEOF
echo "$MARK L20-END rc=$?"

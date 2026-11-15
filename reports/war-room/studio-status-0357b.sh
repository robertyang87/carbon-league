#!/bin/bash
# Studio status probe 0357b (follow-up, 2026-11-13 ~04:03 CST).
# 04:01 read: NO train procs; r1 ep19 (final) 03:56; g1 stuck at ep15,
# g1 log mtime 03:49. Need: g1 ckpt mtimes, both log tails (correct names
# hakimi2r1.log / hakimi2g1.log — no underscore), ship log tail.
# Marker: STUDIOSTAT0357B
MARK=STUDIOSTAT0357B
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout -k 30 240 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o UserKnownHostsFile=/root/.jishubuddy/network-checks/studio-known-hosts-20261112 arm@10.211.43.68 'bash -s' <<'STEOF'
echo "STUDIOSTAT0357B STUDIO-BEGIN $(date '+%F %T %Z')"
ps -ax -o pid,%cpu,%mem,etime,command | grep -E 'cnet2|hakimi|python3' | grep -v grep || echo NO-TRAIN-PROC
echo '--- G1-CKPTS ---'
ls -lt /Users/arm/junyang/carbon/data/weights/cnet2_hakimi2g1* 2>/dev/null
echo '--- R1-TAIL ---'
tail -6 /Users/arm/junyang/carbon/data/hakimi2r1.log
echo '--- G1-TAIL ---'
tail -8 /Users/arm/junyang/carbon/data/hakimi2g1.log
echo '--- SHIP-TAIL ---'
tail -6 /Users/arm/junyang/carbon/data/hakimi2_ship.log
echo '--- R1-JSONL-TAIL ---'
tail -3 /Users/arm/junyang/carbon/data/weights/cnet2_hakimi2r1_log.jsonl 2>/dev/null; ls -l /Users/arm/junyang/carbon/data/weights/cnet2_hakimi2g1_log.jsonl 2>/dev/null
echo "STUDIOSTAT0357B STUDIO-END"
STEOF
echo "$MARK L20-END rc=$?"

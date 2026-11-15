#!/bin/bash
# Studio status probe 0442: idle watch after trip-5 done.
# Last confirmed: Hakimi2 trip-5 dual-arm 20/20 ep done, weights shipped to DGX
# md5-verified, HAKIMI2_SHIP_DONE 03:56:29; Studio normally idle since then.
# Channel outage: cloud-assistant.yml start-fail fleet-wide since ~04:13,
# latest run 43302451306 @04:39:21 CST also failure. This trip: one fresh
# dispatch to test whether the channel recovered; if it runs, verify idle.
# Marker: STUDIOSTAT0442
MARK=STUDIOSTAT0442
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout -k 30 420 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o UserKnownHostsFile=/root/.jishubuddy/network-checks/studio-known-hosts-20261112 arm@10.211.43.68 'bash -s' <<'STEOF'
echo "STUDIOSTAT0442 STUDIO-BEGIN $(date '+%F %T %Z')"
hostname; uptime
echo '--- TRAIN-PROC ---'
ps -ax -o pid,%cpu,%mem,etime,command | grep -E 'cnet2|train|fable|m3s2|wg1|wr1|k4v2|m9xj|320|distill|hakimi|calib' | grep -v grep || echo NO-TRAIN-PROC
echo '--- MPS-LOAD ---'
echo '--- FABLE-LOGS ---'
ls -l /Users/arm/junyang/carbon/data/fable_r1.log /Users/arm/junyang/carbon/data/fable_g1.log 2>/dev/null || echo NO-FABLE-LOGS
tail -3 /Users/arm/junyang/carbon/data/fable_r1.log 2>/dev/null; tail -3 /Users/arm/junyang/carbon/data/fable_g1.log 2>/dev/null
echo '--- FABLE-CKPT ---'
ls -lt /Users/arm/junyang/carbon/data/weights/ 2>/dev/null | grep -i fable | head -6 || echo NO-FABLE-WEIGHTS
ls -l /Users/arm/junyang/carbon/data/HAKIMI2_SHIP_DONE 2>/dev/null || echo NO-SHIP-DONE
echo '--- NEW-LOGS ---'
ls -lt /Users/arm/junyang/carbon/data/*.log 2>/dev/null | head -6
echo '--- MEM-DISK ---'
df -h / | tail -1; vm_stat | head -3
echo "STUDIOSTAT0442 STUDIO-END"
STEOF
echo "$MARK L20-END rc=$?"

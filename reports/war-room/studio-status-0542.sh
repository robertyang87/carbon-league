#!/bin/bash
# Studio status probe 0542: idle watch + queue readiness after war-room 05:36/05:38 assignment.
# Last confirmed: Hakimi2 trip-5 dual-arm 20/20 ep done, weights shipped to DGX
# md5-verified, HAKIMI2_SHIP_DONE 03:56:29; Studio normally idle since then.
# War room 05:36/05:38 assigned Studio a >2h queue (owner 05:38: every machine >=2h):
#  1. sp-probe (seed 991000)  2. head smoke 2nd arm (MPS, after DGX 1st arm)
#  3. teacher consistency offline eval (ds_m3s2, 35k samples)  4. ds_m3s2 head train full arm (BC_DEVICE=mps)
# Channel outage: cloud-assistant.yml start-fail fleet-wide since ~04:13,
# still broken at 05:31 patrol (76 min). This trip: one fresh dispatch to test
# whether the channel recovered; if it runs, verify idle + queue readiness.
# Marker: STUDIOSTAT0542
MARK=STUDIOSTAT0542
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout -k 30 420 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o UserKnownHostsFile=/root/.jishubuddy/network-checks/studio-known-hosts-20261112 arm@10.211.43.68 'bash -s' <<'STEOF'
echo "STUDIOSTAT0542 STUDIO-BEGIN $(date '+%F %T %Z')"
hostname; uptime
echo '--- TRAIN-PROC ---'
ps -ax -o pid,%cpu,%mem,etime,command | grep -E 'cnet2|train|fable|m3s2|wg1|wr1|k4v2|m9xj|320|distill|hakimi|calib' | grep -v grep || echo NO-TRAIN-PROC
echo '--- MPS-LOAD ---'
powermetrics -n 1 -s gpu_power 2>/dev/null | head -5 || echo NO-POWERMETRICS
echo '--- FABLE-LOGS ---'
ls -l /Users/arm/junyang/carbon/data/fable_r1.log /Users/arm/junyang/carbon/data/fable_g1.log 2>/dev/null || echo NO-FABLE-LOGS
tail -3 /Users/arm/junyang/carbon/data/fable_r1.log 2>/dev/null; tail -3 /Users/arm/junyang/carbon/data/fable_g1.log 2>/dev/null
echo '--- FABLE-CKPT ---'
ls -lt /Users/arm/junyang/carbon/data/weights/ 2>/dev/null | grep -i fable | head -6 || echo NO-FABLE-WEIGHTS
ls -l /Users/arm/junyang/carbon/data/HAKIMI2_SHIP_DONE 2>/dev/null || echo NO-SHIP-DONE
echo '--- QUEUE-READINESS ---'
ls -l /Users/arm/junyang/carbon/data/weights/cnet2_m3s2r1L.npz 2>/dev/null || echo NO-M3S2R1L-WEIGHT
ls -ld /Users/arm/junyang/carbon/data/ds_m3s2 2>/dev/null || echo NO-DS-M3S2
find /Users/arm/junyang/carbon -maxdepth 3 -name 'cnet2_m3s2r1L.npz' 2>/dev/null | head -3 || echo NO-M3S2R1L-ANYWHERE
echo '--- NEW-LOGS ---'
ls -lt /Users/arm/junyang/carbon/data/*.log 2>/dev/null | head -6
echo '--- MEM-DISK ---'
df -h / | tail -1; vm_stat | head -3
echo "STUDIOSTAT0542 STUDIO-END"
STEOF
echo "$MARK L20-END rc=$?"

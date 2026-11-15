#!/bin/bash
# Studio status probe 0142 (2026-11-13 01:42 CST trip).
# Context: last verified direct read 01:14:45 — Studio normally idle
# (it320 wr1 done 22:34, best val 0.8673 @ep8). War room approved ONE
# corrective re-dispatch ~01:40-01:42 (hakimi2-studio-launch2.sh v2:
# no git clone on Studio - pull main via LAN from DGX instead).
# This trip checks whether the v2 executor has landed (any hakimi2
# traces). New 15-min clock starts ~01:41; no escalation this trip
# (clock just started, build+train lead ~17 min). Also watch pack transit.
# Read-only only. Marker: STUDIOSTAT0142
MARK=STUDIOSTAT0142
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout -k 30 420 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o UserKnownHostsFile=/root/.jishubuddy/network-checks/studio-known-hosts-20261112 arm@10.211.43.68 'bash -s' <<'STEOF'
echo "STUDIOSTAT0142 STUDIO-BEGIN $(date '+%F %T %Z')"
hostname; uptime
echo '--- TRAIN-PROC ---'
ps -ax -o pid,%cpu,%mem,etime,command | grep -E 'cnet2|train|fable|hakimi|m3s2|wg1|wr1|k4v2|m9xj|320|distill|relabel|lb_|pack' | grep -v grep || echo NO-TRAIN-PROC
echo '--- HAKIMI2-VERIFY ---'
ls -l /Users/arm/junyang/carbon/data/hakimi2* 2>/dev/null || echo NO-HAKIMI2-IN-DATA
find /Users/arm/junyang -maxdepth 4 -name 'hakimi2_matches.json' 2>/dev/null | head -3
echo '--- HAKIMI2-COUNT ---'
for f in $(find /Users/arm/junyang -maxdepth 4 -name 'hakimi2_matches.json' 2>/dev/null | head -1); do python3 -c "import json;print('$f', len(json.load(open('$f'))))" 2>/dev/null || echo COUNT-FAILED; done
echo '--- HAKIMI2-CORPUS ---'
for d in $(find /Users/arm/junyang -maxdepth 4 -type d -name 'hakimi2_corpus' 2>/dev/null | head -1); do echo "$d $(ls "$d" | wc -l) files"; done
echo '--- HAKIMI2-DS ---'
ls -ld /Users/arm/junyang/carbon/data/ds_hakimi2* 2>/dev/null || echo NO-DS-HAKIMI2
ls -l /Users/arm/junyang/carbon/data/*hakimi* 2>/dev/null | head -10 || echo NO-HAKIMI-LOGS
echo '--- HAKIMI2-NPZ-META ---'
ls -lt /Users/arm/junyang/carbon/data/*hakimi*.npz 2>/dev/null | head -5 || echo NO-HAKIMI-NPZ
for m in $(find /Users/arm/junyang -maxdepth 5 -path '*ds_hakimi2*' -name 'meta.json' 2>/dev/null | head -2); do echo "$m"; head -c 400 "$m"; echo; done
echo '--- PACK-TRANSIT ---'
ls -lt /Users/arm/junyang/*pack* /Users/arm/junyang/carbon/*pack* 2>/dev/null | head -5 || echo NO-PACK-FILES
find /Users/arm/junyang -maxdepth 3 -name '*n1r*pack*' -o -maxdepth 3 -name '*pack*n1r*' 2>/dev/null | head -5
echo '--- WR1-320-LOG ---'
tail -3 /Users/arm/junyang/carbon/data/m9xj320wr1.log 2>/dev/null || echo NO-320WR1-LOG
echo '--- NEW-LOGS ---'
ls -lt /Users/arm/junyang/carbon/data/*.log 2>/dev/null | head -6
echo '--- MEM-DISK ---'
df -h / | tail -1; vm_stat | head -3
echo '--- FABLE-CHECK ---'
ls -l /Users/arm/junyang/carbon/data/fable_*.log 2>/dev/null || echo NO-FABLE-LOGS
echo "STUDIOSTAT0142 STUDIO-END"
STEOF
echo "$MARK L20-END rc=$?"

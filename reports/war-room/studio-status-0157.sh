#!/bin/bash
# Studio status probe 0157 (2026-11-13 01:57 CST trip).
# Context: last verified direct read 01:44:46 — Studio normally idle,
# zero hakimi2 traces (v2 executor re-dispatched ~01:41, only ~3 min in
# then). New 15-min clock from ~01:41 expires ~01:56; this trip is past
# expiry. Check whether v2 executor has landed (any hakimi2 traces:
# matches count / corpus count / ds / logs / npz / train proc).
# If still zero traces, this trip sends THIRD '派活未落地' escalation
# naming the v2 agent (per studio-status-last.json next_trip note).
# Read-only only. Marker: STUDIOSTAT0157
MARK=STUDIOSTAT0157
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout -k 30 420 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o UserKnownHostsFile=/root/.jishubuddy/network-checks/studio-known-hosts-20261112 arm@10.211.43.68 'bash -s' <<'STEOF'
echo "STUDIOSTAT0157 STUDIO-BEGIN $(date '+%F %T %Z')"
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
echo '--- HAKIMI2-ANYWHERE ---'
find /Users/arm/junyang -maxdepth 3 -name '*hakimi2*' 2>/dev/null | head -10 || echo NO-HAKIMI2-ANYWHERE
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
echo "STUDIOSTAT0157 STUDIO-END"
STEOF
echo "$MARK L20-END rc=$?"

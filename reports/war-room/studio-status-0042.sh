#!/bin/bash
# Studio status probe 0042 (2026-11-13 00:42 CST trip, run ~00:49).
# Context: last verified direct read 00:00:12 — Studio normally idle
# (it320 wr1 done 22:34, best val 0.8673 @ep8). 00:12 trip probe
# (STUDIOSTAT0012, run 43256791669) completed but log unclaimed:
# GitHub API core quota 0/5000 + Actions log-download channel fault.
# War room has since NAMED the hakimi2 executor (作战室刚派出的执行代理,
# synced to this chat ~00:35): hakimi2 v2 pro replica dual arms, Carago
# priority; hard gate = local verify first (matches list 100 / corpus 102)
# then build dataset + dual-arm train (20 ep, BC_PCT=0.1, set_uw 1 1 1).
# This trip: verify by direct read — has the executor started on Studio?
# (local verification counts? dataset build? any train proc?) plus
# standard health checks. Read-only only.
# Marker: STUDIOSTAT0042
MARK=STUDIOSTAT0042
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout -k 30 420 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o UserKnownHostsFile=/root/.jishubuddy/network-checks/studio-known-hosts-20261112 arm@10.211.43.68 'bash -s' <<'STEOF'
echo "STUDIOSTAT0042 STUDIO-BEGIN $(date '+%F %T %Z')"
hostname; uptime
echo '--- TRAIN-PROC ---'
ps -ax -o pid,%cpu,%mem,etime,command | grep -E 'cnet2|train|fable|hakimi|m3s2|wg1|wr1|k4v2|m9xj|320|distill|relabel|lb_' | grep -v grep || echo NO-TRAIN-PROC
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
echo '--- WR1-320-LOG ---'
tail -3 /Users/arm/junyang/carbon/data/m9xj320wr1.log 2>/dev/null || echo NO-320WR1-LOG
echo '--- NEW-LOGS ---'
ls -lt /Users/arm/junyang/carbon/data/*.log 2>/dev/null | head -6
echo '--- MEM-DISK ---'
df -h / | tail -1; vm_stat | head -3
echo '--- FABLE-CHECK ---'
ls -l /Users/arm/junyang/carbon/data/fable_*.log 2>/dev/null || echo NO-FABLE-LOGS
echo "STUDIOSTAT0042 STUDIO-END"
STEOF
echo "$MARK L20-END rc=$?"

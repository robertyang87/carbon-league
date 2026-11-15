#!/bin/bash
# Studio status probe 0227: verify trip-3 Stage 1 probe traces + Stage 2
# launch state. War-room context: owner approved trip-3 02:16 (two-stage
# hardened, 3 dispatches). Execution agent's own claim logs (transcribed,
# NOT my read): Stage 1 probe HAKIMI2P3 OK 02:21:44 (STUDIO-REACHABLE);
# Stage 2 launch run 43279534274 FAILED 02:25:37, "FAIL dgx-pull-ssh
# rc=127" + "bash: line 33: timeout: command not found" (timeout binary
# missing on the Studio/macOS side). This trip: independent direct read -
# any hakimi2 traces landed? any train proc? wr1 320 log still frozen
# 22:34? machine health. NOTE: no `timeout` inside the ssh heredoc
# (macOS lacks GNU timeout) - timeout only wraps ssh on the L20 side.
# Marker: STUDIOSTAT0227HAKIMI
MARK=STUDIOSTAT0227HAKIMI
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout -k 30 420 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o UserKnownHostsFile=/root/.jishubuddy/network-checks/studio-known-hosts-20261112 arm@10.211.43.68 'bash -s' <<'STEOF'
echo "STUDIOSTAT0227HAKIMI STUDIO-BEGIN $(date '+%F %T %Z')"
hostname; uptime
echo '--- TRAIN-PROC ---'
ps -ax -o pid,%cpu,%mem,etime,command | grep -E 'cnet2|train|fable|m3s2|hakimi|wr1|k4v2|m9xj|320|distill' | grep -v grep || echo NO-TRAIN-PROC
echo '--- HAKIMI-TRACE ---'
find /Users/arm/junyang -maxdepth 4 -iname '*hakimi*' 2>/dev/null | head -20 || echo NO-HAKIMI-FILES
ls -l /Users/arm/junyang/carbon/data/hakimi2_matches.json 2>/dev/null || echo NO-HAKIMI-MATCHES
ls -d /Users/arm/junyang/carbon/data/hakimi2_corpus /Users/arm/junyang/carbon/data/ds_hakimi2 2>/dev/null || echo NO-HAKIMI-DATA
ls -lt /Users/arm/junyang/carbon/data/*hakimi* 2>/dev/null | head -5 || echo NO-HAKIMI-ARTIFACTS
echo '--- WR1-320-LOG ---'
ls -l /Users/arm/junyang/carbon/data/m9xj320wr1.log 2>/dev/null || echo NO-320WR1-LOG
tail -3 /Users/arm/junyang/carbon/data/m9xj320wr1.log 2>/dev/null || echo NO-320WR1-LOG
echo '--- NEW-LOGS ---'
ls -lt /Users/arm/junyang/carbon/data/*.log 2>/dev/null | head -6
echo '--- MEM-DISK ---'
df -h / | tail -1; vm_stat | head -3
echo "STUDIOSTAT0227HAKIMI STUDIO-END"
STEOF
echo "$MARK L20-END rc=$?"

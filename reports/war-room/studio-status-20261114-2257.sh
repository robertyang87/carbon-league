#!/bin/bash
# Studio status probe 2257 (2026-11-14 22:57 CST).
# Last read 22:43: q gate formal RUNNING, launcher 72802 / gate 72804 etime 05:07:04,
# 20 spawn_main workers, ~281/1072 roots (26.2%), ETA center 2026-11-15 14:30 (estimate).
# War-room sync v82 22:50 confirms same (etime 05:10:03, ~282/1072).
# This trip: direct Tailscale SSH read, verify gate still running + progress.
# Marker: STUDIOSTAT2257Q9K4
MARK=STUDIOSTAT2257Q9K4
HF=~/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files
echo "$MARK PROBE-BEGIN $(date '+%F %T %Z')"
timeout -k 30 240 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes \
  -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" \
  -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts \
  -o StrictHostKeyChecking=accept-new -o BatchMode=yes -o ConnectTimeout=30 \
  arm@100.78.149.20 'bash -s' <<'STEOF'
echo "MARKER STUDIOSTAT2257Q9K4"
hostname; uptime
echo "--- MEM-DISK ---"
df -h / | tail -1
vm_stat | head -8
echo "--- GATE-PROCS ---"
ps -o pid,etime,pcpu,pmem,command -p 72802,72804 2>/dev/null || echo "PIDS 72802/72804 NOT FOUND"
ps -o pid,etime,command -p 72804 2>/dev/null | tail -1
echo "--- WORKERS ---"
pgrep -f 'spawn_mai[n]' | wc -l
ps -eo pid,etime,time,pcpu,command | grep 'spawn_main' | grep -v grep | awk '{print $1, $2, $3, $4}' | head -5
echo "--- STAGEB_Q DIR ---"
ls -l ~/junyang/carbon/data/stageB_q/q_du_formal.log ~/junyang/carbon/data/stageB_q/q_du_formal.json 2>&1
ls -lt ~/junyang/carbon/data/stageB_q 2>/dev/null | head -6
find ~/junyang/carbon/data/stageB_q -type f -mmin -20 2>/dev/null | head -5; echo "FRESH20-DONE"
date '+%Y-%m-%d %H:%M:%S %Z'
echo "MARKER STUDIOSTAT2257Q9K4 END"
STEOF
echo "$MARK SSH-RC=$?"
echo "$MARK PROBE-END $(date '+%F %T %Z')"

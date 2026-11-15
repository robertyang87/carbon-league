#!/bin/bash
# Studio status probe 1542 — 15:42 routine check
# Marker: STUDIOSTAT1542TS (claim only output containing this marker)
MARK=STUDIOSTAT1542TS
LOG=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/studio-status-20261113-1542.log
{
echo "$MARK PROBE-BEGIN $(date '+%F %T %Z')"
timeout -k 30 240 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30 -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o GlobalKnownHostsFile=/dev/null -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" arm@100.78.149.20 'bash -s' <<'STEOF'
echo "STUDIOSTAT1542TS STUDIO-BEGIN $(date '+%F %T %Z')"
hostname; uptime
echo '--- MEM-DISK ---'
df -h / | tail -1; vm_stat | head -6
echo '--- TRAIN-TEST-PROC ---'
ps -ax -o pid,%cpu,%mem,etime,command | grep -E 'cnet2|distill|train|s12test|test_s12|teacher' | grep -v grep || echo NO-TRAIN-TEST-PROC
echo '--- S12TEST ---'
ls -lt /Users/arm/s12test 2>/dev/null | head -12
echo '--- CODE-MD5 ---'
cd /Users/arm/junyang/carbon/code 2>/dev/null && md5 -q cnet2_build.py cnet2_train.py set_uw.py cnet2_model.py producer/selfplay_gen.py 2>/dev/null; ls -l cnet2_build.py cnet2_train.py set_uw.py cnet2_model.py producer/selfplay_gen.py 2>/dev/null
ls -ld /Users/arm/junyang/carbon/code/code_backup_pre_b7e1_20261113 2>/dev/null
echo '--- WEIGHTS-NOW ---'
ls -lt /Users/arm/junyang/carbon/data/weights 2>/dev/null | head -8
echo '--- DATA-TOP ---'
ls -lt /Users/arm/junyang/carbon/data 2>/dev/null | head -10
echo "STUDIOSTAT1542TS STUDIO-END"
STEOF
echo "$MARK SSH-RC=$?"
echo "$MARK PROBE-END $(date '+%F %T %Z')"
} | tee "$LOG"

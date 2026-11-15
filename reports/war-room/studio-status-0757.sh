#!/bin/bash
# Studio status probe 2026-11-15 07:57 trip. Read-only.
MARK=STUDIOSTAT0757Q8Z4
LOG=~/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/studio-status-20261115-0757-ts.log
{
echo "$MARK PROBE-BEGIN $(date '+%Y-%m-%d %H:%M:%S %Z')"
timeout -k 30 150 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes \
  -o StrictHostKeyChecking=accept-new \
  -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o GlobalKnownHostsFile=/dev/null \
  -o ConnectTimeout=30 \
  -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" \
  arm@100.78.149.20 "echo MARKER=$MARK; hostname; date; uptime; df -h /; vm_stat | head -8; echo ---GATE-PROCS---; ps -eo pid,etime,time,pcpu,args | grep -E 'stage1_teacher_gate|q_launch' | grep -v grep; echo ---WORKERS---; ps -eo pid,etime,time,pcpu,args | grep 'spawn_main' | grep -v grep; echo ---WORKER-COUNT---; ps -eo args | grep -c '[s]pawn_main'; echo ---STAGEB---; ls -la ~/junyang/carbon/data/stageB_q/; ls -l ~/junyang/carbon/data/stageB_q/q_du_formal.json 2>&1; echo ---Q-LOG-TAIL---; tail -c 1200 ~/junyang/carbon/data/stageB_q/q_du_formal.log; echo; echo END-MARKER=$MARK"
echo "SSH-EXIT $?"
} 2>&1 | tee "$LOG"

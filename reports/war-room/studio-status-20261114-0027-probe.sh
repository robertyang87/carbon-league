#!/bin/bash
# Studio status probe 2026-11-14 00:27 (Tailscale direct; same pattern as 0012 trip).
# Context since last read 00:14:07: war room ordered STOP v2 pilot (run 49b518f6) at
# 00:19 (owner 00:18 authorization, §18 withdrawal); Studio chat reports stop done
# ~00:22 (zero residue, PARTIAL archived, sidecar est 36/113), non-canonical heads
# db623416... disposed 00:19, brush 4acbbaea done 00:21-00:22 (8 md5 all match,
# py_compile pass). Studio idle, awaiting v3 pilot / 12-game smoke / terminal pilot
# order. This trip: verify idle (no training proc, no MPS load), brush md5 still
# 4acbbaea, pilot log/progress final state, no new order landed on disk.
# READ-ONLY.

MARK=STUDIOSTAT0027Q9M4
LOG=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/studio-status-20261114-0027-ts.log
{
echo "$MARK PROBE-BEGIN $(date '+%F %T %Z')"
timeout -k 30 240 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30 -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o GlobalKnownHostsFile=/dev/null -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" arm@100.78.149.20 'bash -s' <<'STEOF'
echo "STUDIOSTAT0027Q9M4 STUDIO-BEGIN $(date '+%F %T %Z')"
hostname; uptime
echo '--- MEM-DISK ---'
df -h / | tail -1; vm_stat | head -4
echo '--- PILOT-MAIN-91954 ---'
ps -p 91954 -o pid,%cpu,%mem,etime,command 2>/dev/null || echo "PID-91954-NOT-FOUND"
echo '--- ANY-TRAINING-PROC ---'
ps aux | grep -iE 'python.*(stage1|stage0|selfplay|rl2_ppo|teacher|train)' | grep -v grep || echo "NO-TRAINING-PROC"
echo '--- TOP-CPU ---'
ps aux -r | head -8
echo '--- S1-FILES-MD5 ---'
md5 /Users/arm/junyang/carbon/code/stage0_teacher.py /Users/arm/junyang/carbon/code/stage1_teacher_gate.py /Users/arm/junyang/carbon/code/cnet2_build.py /Users/arm/junyang/carbon/code/producer/selfplay_gen.py /Users/arm/junyang/carbon/code/rl2_ppo_train.py /Users/arm/junyang/carbon/code/rl2_ppo_cnet2.py /Users/arm/junyang/carbon/code/stage0_guided_match.py /Users/arm/junyang/carbon/code/stage0_measure_teacher.py 2>/dev/null
echo '--- V2-PILOT-FINAL-STATE ---'
ls -lT /Users/arm/junyang/carbon/data/stage1_formal/pilot_du_v2.progress.json /Users/arm/junyang/carbon/data/stage1_formal/pilot_du_v2.log 2>/dev/null || echo "NO-v2-pilot-files"
tail -5 /Users/arm/junyang/carbon/data/stage1_formal/pilot_du_v2.log 2>/dev/null || echo "NO-v2-pilot-log"
echo '--- HEADS-DISPOSAL ---'
ls -lT /Users/arm/junyang/carbon/data/weights/ 2>/dev/null | head -20
echo '--- V3-ACCEPT-LOG-MTIME ---'
ls -lT ~/s1v3accept_run.log 2>/dev/null || echo "NO-accept-log"
echo "STUDIOSTAT0027Q9M4 STUDIO-END"
STEOF
echo "$MARK SSH-RC=$?"
echo "$MARK PROBE-END $(date '+%F %T %Z')"
} | tee "$LOG"

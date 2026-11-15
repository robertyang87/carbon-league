#!/bin/bash
# Studio status probe 2026-11-14 02:12 (Tailscale direct; same pattern as 0057/0157 trips).
# Context since last direct read 01:58: idle normal holds (load 1.49, disk 417Gi,
# brush md5 all 4acbbaea, pilot log 2956 B, sidecar 36/113 est); war room 01:59
# sync already noted Studio 01:58 zero change; hakimi v3pro trigger 4a closed
# (list+corpus 100/100 in repo) — no new Studio order visible. This trip: verify
# idle holds (no training proc, no spawn_main workers, load normal), brush md5
# still 4acbbaea, pilot log/sidecar unchanged, no new order/code-round files landed.
# READ-ONLY.

MARK=STUDIOSTAT0212Q9Z4
LOG=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/studio-status-20261114-0212-ts.log
{
echo "$MARK PROBE-BEGIN $(date '+%F %T %Z')"
timeout -k 30 240 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30 -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o GlobalKnownHostsFile=/dev/null -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" arm@100.78.149.20 'bash -s' <<'STEOF'
echo "STUDIOSTAT0212Q9Z4 STUDIO-BEGIN $(date '+%F %T %Z')"
hostname; uptime
echo '--- MEM-DISK ---'
df -h / | tail -1; vm_stat | head -4
echo '--- SPAWN-MAIN-COUNT ---'
ps aux | grep -c '[s]pawn_main' || echo "SPAWN-MAIN-COUNT-0"
echo '--- ANY-TRAINING-PROC ---'
ps aux | grep -iE 'python.*(stage1|stage0|selfplay|rl2_ppo|teacher|train)' | grep -v grep || echo "NO-TRAINING-PROC"
echo '--- TOP-CPU ---'
ps aux -r | head -8
echo '--- S1-FILES-MD5 ---'
md5 /Users/arm/junyang/carbon/code/stage0_teacher.py /Users/arm/junyang/carbon/code/stage1_teacher_gate.py /Users/arm/junyang/carbon/code/cnet2_build.py /Users/arm/junyang/carbon/code/producer/selfplay_gen.py /Users/arm/junyang/carbon/code/rl2_ppo_train.py /Users/arm/junyang/carbon/code/rl2_ppo_cnet2.py /Users/arm/junyang/carbon/code/stage0_guided_match.py /Users/arm/junyang/carbon/code/stage0_measure_teacher.py 2>/dev/null
echo '--- V2-PILOT-FINAL-STATE ---'
ls -lT /Users/arm/junyang/carbon/data/stage1_formal/pilot_du_v2.progress.json /Users/arm/junyang/carbon/data/stage1_formal/pilot_du_v2.log /Users/arm/junyang/carbon/data/stage1_formal/pilot_du_v2.PARTIAL.md 2>/dev/null || echo "NO-v2-pilot-files"
echo '--- NEW-PILOT-OR-ORDER-FILES ---'
ls -lT /Users/arm/junyang/carbon/data/stage1_formal/ 2>/dev/null | tail -8
echo '--- V3-ACCEPT-LOG-MTIME ---'
ls -lT ~/s1v3accept_run.log 2>/dev/null || echo "NO-accept-log"
echo "STUDIOSTAT0212Q9Z4 STUDIO-END"
STEOF
echo "$MARK SSH-RC=$?"
echo "$MARK PROBE-END $(date '+%F %T %Z')"
} | tee "$LOG"

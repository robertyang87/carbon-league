#!/bin/bash
# Studio status probe 2026-11-13 23:57 (Tailscale direct; same pattern as 2342 trip).
# Context since last read 23:44:58: v2 pilot running (run 49b518f6, main PID 91954 + 12
# workers, 113 roots / 34 games, --belief sample, teacher v2, gate dd921e57); sidecar
# first write 23:19:42 only, next write expected ~23:49 (30-min cadence); ETA ~02:35-02:40
# (est, from formal pilot wall 3h20m50s for same roots). War-room 23:48 sync: L20 it78
# switch done (PID 30478, utility head), DGX gen-1 shard1 180/200 ~23:57, shard0 PASS.
# v2 pilot stays baseline record; Stage B N + terminal pilot move to teacher v3 after
# (i) acceptance conclusion + brush order. This trip: verify pilot still running with
# the three-piece check (process + load + fresh log/file mtime) + read progress sidecar
# (expect second write ~23:49 with roots_completed).
# READ-ONLY.

MARK=STUDIOSTAT2357M9Q8
LOG=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/studio-status-20261113-2357-ts.log
{
echo "$MARK PROBE-BEGIN $(date '+%F %T %Z')"
timeout -k 30 240 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30 -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o GlobalKnownHostsFile=/dev/null -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" arm@100.78.149.20 'bash -s' <<'STEOF'
echo "STUDIOSTAT2357M9Q8 STUDIO-BEGIN $(date '+%F %T %Z')"
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
echo '--- V2-PILOT-FILES ---'
ls -lT /Users/arm/junyang/carbon/data/stage1_formal/pilot_du_v2* 2>/dev/null || echo "NO-v2-pilot-files"
echo '--- V2-PILOT-PROGRESS-CONTENT ---'
cat /Users/arm/junyang/carbon/data/stage1_formal/pilot_du_v2.progress.json 2>/dev/null | tail -c 1500 || echo "NO-progress-json"
echo '--- V2-PILOT-LOG-TAIL ---'
tail -5 /Users/arm/junyang/carbon/data/stage1_formal/pilot_du_v2.log 2>/dev/null || ls -lT /Users/arm/junyang/carbon/data/stage1_formal/ 2>/dev/null | head -25
echo "STUDIOSTAT2357M9Q8 STUDIO-END"
STEOF
echo "$MARK SSH-RC=$?"
echo "$MARK PROBE-END $(date '+%F %T %Z')"
} | tee "$LOG"

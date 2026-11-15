#!/bin/bash
# Studio status probe 2026-11-14 00:12 (Tailscale direct; same pattern as 2357 trip).
# Context since last read 23:59:41: v2 pilot running (run 49b518f6, main PID 91954 + 12
# workers, sidecar 2nd write 23:49:42 roots_completed_est 19/113, ETA ~02:35-02:40 est).
# v3 acceptance suite ~/s1v3accept started 23:55 (PID 2148), was on last item
# test_teacher_information_set.py at 23:59 — this trip: read its result log.
# War-room 00:05: §18 teacher v3 (4acbbaea) PASS-WITH-NOTES + whole-team brush order
# issued; DGX shard1 build FAIL (994577 alignment, strategy group ruled (i) local
# patch at 00:12). This trip: verify pilot still running (three-piece) + v3 accept
# result + whether Studio code set still a4dd8329 or already re-brushed.
# READ-ONLY.

MARK=STUDIOSTAT0012X7K3
LOG=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/studio-status-20261114-0012-ts.log
{
echo "$MARK PROBE-BEGIN $(date '+%F %T %Z')"
timeout -k 30 240 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30 -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o GlobalKnownHostsFile=/dev/null -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" arm@100.78.149.20 'bash -s' <<'STEOF'
echo "STUDIOSTAT0012X7K3 STUDIO-BEGIN $(date '+%F %T %Z')"
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
echo '--- V2-PILOT-PROGRESS-CONTENT ---'
cat /Users/arm/junyang/carbon/data/stage1_formal/pilot_du_v2.progress.json 2>/dev/null | tail -c 1500 || echo "NO-progress-json"
echo '--- V2-PILOT-LOG-TAIL ---'
tail -5 /Users/arm/junyang/carbon/data/stage1_formal/pilot_du_v2.log 2>/dev/null || echo "NO-v2-pilot-log"
echo '--- V3-ACCEPT-SUITE ---'
ps aux | grep -iE 's1v3accept|test_teacher|pytest' | grep -v grep || echo "NO-ACCEPT-PROC"
ls -lT ~/s1v3accept_run.log 2>/dev/null || echo "NO-accept-log"
tail -40 ~/s1v3accept_run.log 2>/dev/null || echo "NO-accept-log-content"
echo "STUDIOSTAT0012X7K3 STUDIO-END"
STEOF
echo "$MARK SSH-RC=$?"
echo "$MARK PROBE-END $(date '+%F %T %Z')"
} | tee "$LOG"

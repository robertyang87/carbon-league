#!/bin/bash
# Studio status probe 2026-11-13 23:42 (Tailscale direct; same pattern as 2327 trip).
# Context since last read 23:29:20: v2 pilot running (run 49b518f6, main PID 91954 + 12
# workers, 113 roots / 34 games, --belief sample, teacher v2, gate dd921e57); sidecar
# first write 23:19:42, next expected ~23:49; ETA ~02:35-02:40 (est).
# New since last trip (war-room / MEMORY): DGX gen-1 shard0 ACCEPTANCE PASS 23:43
# (200/200 games, 119,600 pos, endgame 39,600, mirror 0.0, U distinct 384, mean 0.54);
# shard1 continuing. Terminal pilot feed now ready, but order still waits for (i)
# acceptance conclusion + brush order, teacher v3. v2 pilot continues as baseline
# record; hold a4dd8329 until then. This trip: verify pilot still running with the
# three-piece check (process + load + fresh log/file mtime) + read progress sidecar.
# READ-ONLY.

MARK=STUDIOSTAT2342M7Q2
LOG=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/studio-status-20261113-2342-ts.log
{
echo "$MARK PROBE-BEGIN $(date '+%F %T %Z')"
timeout -k 30 240 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30 -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o GlobalKnownHostsFile=/dev/null -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" arm@100.78.149.20 'bash -s' <<'STEOF'
echo "STUDIOSTAT2342M7Q2 STUDIO-BEGIN $(date '+%F %T %Z')"
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
echo '--- V2-PILOT-SIGNS ---'
ls -lT /Users/arm/junyang/carbon/data/stage1_formal/pilot_du_v2* 2>/dev/null || echo "NO-v2-pilot-files"
echo '--- V2-PILOT-PROGRESS-CONTENT ---'
cat /Users/arm/junyang/carbon/data/stage1_formal/pilot_du_v2.progress.json 2>/dev/null | tail -c 1500 || echo "NO-progress-json"
echo '--- V2-PILOT-LOG-TAIL ---'
tail -5 /Users/arm/junyang/carbon/data/stage1_formal/pilot_du_v2.log 2>/dev/null || ls -lT /Users/arm/junyang/carbon/data/stage1_formal/ 2>/dev/null | head -25
echo "STUDIOSTAT2342M7Q2 STUDIO-END"
STEOF
echo "$MARK SSH-RC=$?"
echo "$MARK PROBE-END $(date '+%F %T %Z')"
} | tee "$LOG"

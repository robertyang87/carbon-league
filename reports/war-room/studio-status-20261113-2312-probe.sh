#!/bin/bash
# Studio status probe 2026-11-13 23:12 (Tailscale direct; same pattern as 2257 trip).
# Context: SS16 acceptance suite running since 22:59 (wrapper PID 87466, ~/s1accept):
#   twoworld small 37/37 PASS EXIT=0 (23:00:33); --canon in progress (PID 87657, ~1 core);
#   ppo re-run pending after canon (old /tmp/acc_ppo.log 22:57:45 EXIT=1 missing carbon.json;
#   strategy group says env_src fixed 22:59, wrapper will re-run ppo).
# War-room 23:08-23:09 dispatch (new since last trip): Studio brush to main a4dd8329 code set
#   (teacher 2c2541c6, gate dd921e57, build a3f16063, producer 15d24544, ppo_train 1f1430b4,
#   ppo_cnet2 62a9adc7, guided ed43fda8, measure d0de8a12) + v2 pilot re-run (same 34-game pool,
#   --belief sample, gate dd921e57); sigma-hat + du_mean back to strategy group to confirm Stage B N.
#   Execution agent is war-room's; this trip only checks whether anything has started/moved. READ-ONLY.

MARK=STUDIOSTAT2312M7Q2
LOG=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/studio-status-20261113-2312-ts.log
{
echo "$MARK PROBE-BEGIN $(date '+%F %T %Z')"
timeout -k 30 240 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30 -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o GlobalKnownHostsFile=/dev/null -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" arm@100.78.149.20 'bash -s' <<'STEOF'
echo "STUDIOSTAT2312M7Q2 STUDIO-BEGIN $(date '+%F %T %Z')"
hostname; uptime
echo '--- MEM-DISK ---'
df -h / | tail -1; vm_stat | head -4
echo '--- WRAPPER-87466 ---'
ps -p 87466 -o pid,%cpu,%mem,etime,command 2>/dev/null || echo "PID-87466-NOT-FOUND"
echo '--- CANON-87657 ---'
ps -p 87657 -o pid,%cpu,%mem,etime,command 2>/dev/null || echo "PID-87657-NOT-FOUND"
echo '--- ANY-TRAINING-PROC ---'
ps aux | grep -iE 'python.*(stage1|stage0|selfplay|rl2_ppo|teacher|train)' | grep -v grep || echo "NO-TRAINING-PROC"
echo '--- ACC-LOGS ---'
for f in /tmp/acc_twoworld_small.log /tmp/acc_twoworld_canon.log /tmp/acc_ppo.log; do echo "== $f =="; ls -lT "$f" 2>/dev/null; tail -5 "$f" 2>/dev/null || echo "MISSING"; done
echo '--- S1-FILES-MD5 ---'
md5 /Users/arm/junyang/carbon/code/stage0_teacher.py /Users/arm/junyang/carbon/code/stage1_teacher_gate.py /Users/arm/junyang/carbon/code/cnet2_build.py /Users/arm/junyang/carbon/code/producer/selfplay_gen.py /Users/arm/junyang/carbon/code/rl2_ppo_train.py /Users/arm/junyang/carbon/code/rl2_ppo_cnet2.py /Users/arm/junyang/carbon/code/stage0_guided_match.py /Users/arm/junyang/carbon/code/stage0_measure_teacher.py 2>/dev/null
echo '--- FORMAL-DIR ---'
ls -lT /Users/arm/junyang/carbon/data/stage1_formal/ 2>/dev/null | head -20
echo '--- V2-PILOT-SIGNS ---'
ls -lT /Users/arm/junyang/carbon/data/stage1_formal/pilot_du_v2* 2>/dev/null || echo "NO-v2-pilot-files"
ls -ld /Users/arm/junyang/carbon/code_backup_pre_s1v16_20261113 /Users/arm/junyang/carbon/code_backup_pre_* 2>/dev/null
echo "STUDIOSTAT2312M7Q2 STUDIO-END"
STEOF
echo "$MARK SSH-RC=$?"
echo "$MARK PROBE-END $(date '+%F %T %Z')"
} | tee "$LOG"

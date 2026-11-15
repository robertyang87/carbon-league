#!/bin/bash
# Studio status probe 2026-11-14 01:42 (Tailscale direct; same pattern as 0127 trip).
# Context since last read 01:29: Studio verified idle (normal) since 00:47 —
# v2 pilot stopped by war-room order 00:22, PARTIAL archived, cleanup of 12
# orphan spawn workers verified, brush md5 all 4acbbaea, disk 417Gi. Idle
# already acknowledged by war room (00:47, 01:09 sync, 01:19 sync, 01:33 sync:
# "Studio 自 00:47 空出（等 C 包，v3 pilot）"). War room 01:33: L20 snipe_n1
# relaunched from it79 (new PID 34214, it80 est ~02:10-02:14); DGX shard1
# still stuck (raw label alignment, ~70 min no output, waiting §19 B pack).
# No new Studio order in war-room messages read this trip. This trip:
# verify idle still holds (no training proc, no spawn_main workers, load
# normal), brush md5 still 4acbbaea, pilot log/sidecar final state unchanged,
# no new order landed on disk. READ-ONLY.

MARK=STUDIOSTAT0142M8D4
LOG=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/studio-status-20261114-0142-ts.log
{
echo "$MARK PROBE-BEGIN $(date '+%F %T %Z')"
timeout -k 30 240 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30 -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o GlobalKnownHostsFile=/dev/null -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" arm@100.78.149.20 'bash -s' <<'STEOF'
echo "STUDIOSTAT0142M8D4 STUDIO-BEGIN $(date '+%F %T %Z')"
hostname; uptime
echo '--- MEM-DISK ---'
df -h / | tail -1; vm_stat | head -4
echo '--- CLEANUP-PIDS-EXPLICIT ---'
ps -p 91954,91958,91959,91960,91961,91962,91963,91964,91965,91966,91967,91968,91969,91970,92588 -o pid,etime,time,comm 2>/dev/null || echo "CLEANUP-PIDS-NOT-FOUND"
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
echo "STUDIOSTAT0142M8D4 STUDIO-END"
STEOF
echo "$MARK SSH-RC=$?"
echo "$MARK PROBE-END $(date '+%F %T %Z')"
} | tee "$LOG"

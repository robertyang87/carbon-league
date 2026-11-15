#!/bin/bash
# Studio status probe 2026-11-14 17:57: routine check — formal q gate RESTARTED 17:36:48 (new run 51972702c4f81c21, launcher 72802 / gate 72804, 20 workers per 17:38 receipt; old run c132bf81 VOID, zero residue). This trip re-verifies the three-piece set for the NEW run (process + load + fresh files) and watches for first game output / q_du_formal.json.
# Channel: Tailscale direct via ~/workspace/studio_ssh.sh (arm@100.78.149.20).
# Marker: STUDIOSTAT1757Q2Z9 (claim only output containing this marker)
MARK=STUDIOSTAT1757Q2Z9
echo "$MARK PROBE-BEGIN $(date '+%F %T %Z')"
/home/hatch/workspace/studio_ssh.sh "bash -s" <<'STEOF'
echo "STUDIOSTAT1757Q2Z9 STUDIO-BEGIN $(date '+%F %T %Z')"
hostname; uptime
echo '--- GATE-LAUNCHER-72802 ---'
ps -p 72802 -o pid,etime,time,%cpu,args 2>/dev/null || echo NO-LAUNCHER-72802
echo '--- GATE-PROC-72804 ---'
ps -p 72804 -o pid,etime,time,%cpu,args 2>/dev/null || echo NO-GATE-72804
echo '--- GATE-ANY (stage1_teacher_gate / q_launch) ---'
ps -ax -o pid,%cpu,%mem,etime,command | grep -E 'stage1_teacher_gate|q_launch' | grep -v grep || echo NO-GATE-PROC
echo '--- WORKERS ---'
pgrep -f 'spawn_mai[n]' | wc -l
ps -ax -o pid,etime,time,%cpu,command | grep spawn_main | grep -v grep | head -22
echo '--- OLD-RUN-RESIDUE (51489/51491) ---'
ps -p 51489,51491 -o pid,etime,args 2>/dev/null || echo OLD-PIDS-GONE
echo '--- LOAD ---'
uptime
echo '--- STAGEB_Q DIR ---'
ls -lt /Users/arm/junyang/carbon/data/stageB_q 2>/dev/null | head -12
echo '--- FRESH FILES (stageB_q, mmin -20) ---'
find /Users/arm/junyang/carbon/data/stageB_q -type f -mmin -20 2>/dev/null | head; echo FRESH-DONE
echo '--- Q_DU_FORMAL OUTPUT ---'
ls -l /Users/arm/junyang/carbon/data/stageB_q/q_du_formal.json 2>/dev/null || echo NO-Q_DU_FORMAL-JSON
ls -l /Users/arm/junyang/carbon/data/stageB_q/q_du_formal.log 2>/dev/null; head -c 400 /Users/arm/junyang/carbon/data/stageB_q/q_du_formal.log 2>/dev/null; echo
echo '--- Q_LAUNCH JSON ---'
cat /Users/arm/junyang/carbon/data/stageB_q/q_launch.json 2>/dev/null; echo
echo '--- DISK ---'
df -h / | tail -1
echo "STUDIOSTAT1757Q2Z9 STUDIO-END $(date '+%F %T %Z')"
STEOF
echo "$MARK SSH-RC=$?"
echo "$MARK PROBE-END $(date '+%F %T %Z')"

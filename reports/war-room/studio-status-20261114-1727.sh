#!/bin/bash
# Studio status probe 2026-11-14 17:27: routine check — formal q gate (launcher 51489 / gate 51491, 12 workers) verified 17:13, etime 01:02:18, load ~13.3, q_du_formal.log PREREG-only 13,843 B mtime 16:11, ETA ~2026-11-15 22:00 (estimate). This trip re-verifies the three-piece set (process + load + fresh files) and watches for first game output / q_du_formal.json.
# Channel: Tailscale direct via ~/workspace/studio_ssh.sh (arm@100.78.149.20). The L20 ssh channel in older same-name scripts is stale; do not use.
# Marker: STUDIOSTAT1727B7K2 (claim only output containing this marker)
MARK=STUDIOSTAT1727B7K2
echo "$MARK PROBE-BEGIN $(date '+%F %T %Z')"
/home/hatch/workspace/studio_ssh.sh "bash -s" <<'STEOF'
echo "STUDIOSTAT1727B7K2 STUDIO-BEGIN $(date '+%F %T %Z')"
hostname; uptime
echo '--- GATE-LAUNCHER-51489 ---'
ps -p 51489 -o pid,etime,time,%cpu,args 2>/dev/null || echo NO-LAUNCHER-51489
echo '--- GATE-PROC-51491 ---'
ps -p 51491 -o pid,etime,time,%cpu,args 2>/dev/null || echo NO-GATE-51491
echo '--- GATE-ANY (stage1_teacher_gate / q_launch) ---'
ps -ax -o pid,%cpu,%mem,etime,command | grep -E 'stage1_teacher_gate|q_launch' | grep -v grep || echo NO-GATE-PROC
echo '--- WORKERS ---'
pgrep -f spawn_main | wc -l
ps -ax -o pid,etime,time,%cpu,command | grep spawn_main | grep -v grep | head -14
echo '--- LOAD ---'
uptime
echo '--- STAGEB_Q DIR ---'
ls -lt /Users/arm/junyang/carbon/data/stageB_q 2>/dev/null | head -12
echo '--- FRESH FILES (stageB_q, mmin -20) ---'
find /Users/arm/junyang/carbon/data/stageB_q -type f -mmin -20 2>/dev/null | head; echo FRESH-DONE
echo '--- Q_DU_FORMAL OUTPUT ---'
ls -l /Users/arm/junyang/carbon/data/stageB_q/q_du_formal.json 2>/dev/null || echo NO-Q_DU_FORMAL-JSON
ls -l /Users/arm/junyang/carbon/data/stageB_q/q_du_formal.log 2>/dev/null; head -c 400 /Users/arm/junyang/carbon/data/stageB_q/q_du_formal.log 2>/dev/null; echo
echo '--- DISK ---'
df -h / | tail -1
echo "STUDIOSTAT1727B7K2 STUDIO-END $(date '+%F %T %Z')"
STEOF
echo "$MARK SSH-RC=$?"
echo "$MARK PROBE-END $(date '+%F %T %Z')"

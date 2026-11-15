#!/bin/bash
# Studio status probe 2026-11-13 17:57 (Tailscale direct; cloud-assistant dead since 04:13, GitHub Actions usage-blocked).
# Context: Stage A pilot started 16:26, run_id 7a189805a8fbc6ec, PID 89571 + 12 workers, 47 scenes du.
# 17:43 trip: PID 89571 etime 01:16:52, 5 of 12 workers STOPPED accumulating CPU (89577/89579/89583/89584/89586, 0.0%, CPU time 3-8 min behind elapsed), 7 still at 100%; load 1-min 7.80.
# pilot_du.log mtime still 16:26:57, pilot_du.json NOT landed (76 min 52 s no output at 17:43:49).
# S1 v1.3 disk files confirmed (teacher 77ba3677..., gate bdd3caed...). Disk 421Gi free.
# War-room 17:48 sync: pilot verdict still open item #1 (wait vs stop-switch to v1.3 formal pilot); DGX idle ~149 min; it320 pool switch waits it70 EVAL (est 18:20-18:40).
# This trip: pilot still running? more workers stopped / any resumed? pilot_du.json landed? Any new process (owner decision executed)?
MARK=STUDIOSTAT1757Q2F8
LOG=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/studio-status-20261113-1757-ts.log
{
echo "$MARK PROBE-BEGIN $(date '+%F %T %Z')"
timeout -k 30 240 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30 -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o GlobalKnownHostsFile=/dev/null -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" arm@100.78.149.20 'bash -s' <<'STEOF'
echo "STUDIOSTAT1757Q2F8 STUDIO-BEGIN $(date '+%F %T %Z')"
hostname; uptime
echo '--- MEM-DISK ---'
df -h / | tail -1; vm_stat | head -4
echo '--- PILOT-PROC ---'
ps -p 89571 -o pid,%cpu,%mem,etime,command 2>/dev/null || echo "PID-89571-NOT-FOUND"
echo "children-of-89571: $(pgrep -P 89571 2>/dev/null | wc -l)"
ps -ax -o pid,ppid,%cpu,time,etime,command | grep -E 'stage1_teacher_gate|probe_pilot|p3a_harness|test_s12' | grep -v grep || echo NO-PILOT-RELATED-PROC
echo '--- PILOT-DIR ---'
ls -lT /Users/arm/junyang/carbon/data/stage1_pilot/ 2>/dev/null | head -20
echo '--- PILOT-LOG ---'
ls -lT /Users/arm/junyang/carbon/data/stage1_pilot/pilot_du.log /Users/arm/junyang/carbon/data/stage1_pilot/pilot_du.json 2>/dev/null
tail -5 /Users/arm/junyang/carbon/data/stage1_pilot/pilot_du.log 2>/dev/null
echo '--- PILOT-JSON-HEAD ---'
head -c 600 /Users/arm/junyang/carbon/data/stage1_pilot/pilot_du.json 2>/dev/null || echo "NO-pilot_du.json"
echo
echo '--- S1-FILES-MD5 ---'
md5 /Users/arm/junyang/carbon/code/stage0_teacher.py /Users/arm/junyang/carbon/code/stage1_teacher_gate.py 2>/dev/null
echo "STUDIOSTAT1757Q2F8 STUDIO-END"
STEOF
echo "$MARK SSH-RC=$?"
echo "$MARK PROBE-END $(date '+%F %T %Z')"
} | tee "$LOG"

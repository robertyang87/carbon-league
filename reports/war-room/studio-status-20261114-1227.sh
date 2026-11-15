#!/bin/bash
# Studio status 12:12 trip: verify v3 full pilot running (launched 12:10:49, gate PID 89592) + residual check. READ-ONLY. Marker STUDIOSTAT1227.
MARK=STUDIOSTAT1227
LOG=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/studio-status-20261114-1227-ts.log
{
echo "$MARK PROBE-BEGIN $(date '+%F %T %Z')"
timeout -k 30 150 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30 -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o GlobalKnownHostsFile=/dev/null -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" arm@100.78.149.20 'bash -s' <<'STEOF'
echo "STUDIOSTAT1227 STUDIO-BEGIN $(date '+%F %T %Z')"
hostname; uptime
echo '--- V3 GATE PROC 89592 ---'
ps -p 89592 -o pid,etime,%cpu,args 2>/dev/null | cut -c1-220 || echo PID89592-GONE
echo '--- V3 LAUNCHER 89589 ---'
ps -p 89589 -o pid,etime,%cpu,args 2>/dev/null | cut -c1-220 || echo PID89589-GONE
echo '--- SPAWN_MAIN COUNT ---'
pgrep -f 'spawn_mai[n]' | wc -l
ps -eo pid,etime,%cpu,args | grep -E 'spawn_mai[n]' | cut -c1-200 | head -15
echo '--- V3 PILOT PROCS (any) ---'
ps -eo pid,etime,%cpu,args | grep -E 'pilot_du_v3|v3_launch|stage1_teacher_gate|stage1_formal' | grep -v grep | cut -c1-220 | head -10 || echo NO-V3-PROCS
echo '--- A4 RESIDUAL ---'
ps -eo pid,etime,args | grep -E 'a4_select|cnet2_train' | grep -v grep | cut -c1-200 | head -5 || echo NO-A4-RESIDUAL
echo '--- STAGE1_FORMAL DIR ---'
ls -lt /Users/arm/junyang/carbon/data/stage1_formal/ | cut -c1-160 | head -20
echo '--- V3 LAUNCH RECORD ---'
cat /Users/arm/junyang/carbon/data/stage1_formal/v3_launch.json 2>/dev/null | head -30 || echo NO-V3-LAUNCH-JSON
echo '--- PILOT DU V3 LOG ---'
for f in /Users/arm/junyang/carbon/data/stage1_formal/pilot_du_v3.log /Users/arm/junyang/carbon/data/pilot_du_v3.log; do
  if [ -f "$f" ]; then echo "LOG $f:"; ls -l "$f"; echo '--- tail ---'; tail -20 "$f"; fi
done
echo '--- PILOT DU V3 JSON ---'
for f in /Users/arm/junyang/carbon/data/stage1_formal/pilot_du_v3.json /Users/arm/junyang/carbon/data/pilot_du_v3.json; do
  if [ -f "$f" ]; then echo "JSON $f:"; ls -l "$f"; python3 -c "import json;d=json.load(open('$f'));print('keys:',sorted(d.keys()));print('scenes_done:',d.get('scenes_done'),d.get('n_scenes'));print('du_mean:',d.get('du_mean'))" 2>&1 | head -10; fi
done
echo '--- DISK ---'
df -h / | tail -1
echo "STUDIOSTAT1227 STUDIO-END $(date '+%F %T %Z')"
STEOF
echo "$MARK SSH-RC=$?"
echo "$MARK PROBE-END $(date '+%F %T %Z')"
} | tee "$LOG"

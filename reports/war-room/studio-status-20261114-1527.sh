#!/bin/bash
# Studio status 15:27 trip: verify v3 full pilot running (launched 12:10:49, gate PID 89592, launcher 89589, 12 workers). READ-ONLY. Marker STUDIOSTAT1527.
MARK=STUDIOSTAT1527
LOG=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/studio-status-20261114-1527-ts.log
{
echo "$MARK PROBE-BEGIN $(date '+%F %T %Z')"
timeout -k 30 150 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30 -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o GlobalKnownHostsFile=/dev/null -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" arm@100.78.149.20 'bash -s' <<'STEOF'
echo "STUDIOSTAT1527 STUDIO-BEGIN $(date '+%F %T %Z')"
hostname; uptime
echo '--- V3 GATE PROC 89592 ---'
ps -p 89592 -o pid,etime,cputime,%cpu,args 2>/dev/null | cut -c1-220 || echo PID89592-GONE
echo '--- V3 LAUNCHER 89589 ---'
ps -p 89589 -o pid,etime,cputime,%cpu,args 2>/dev/null | cut -c1-220 || echo PID89589-GONE
echo '--- SPAWN_MAIN COUNT ---'
pgrep -f 'spawn_mai[n]' | wc -l
echo '--- WORKERS (pid, etime, cputime, %cpu) ---'
pgrep -f 'spawn_mai[n]' | tr '\n' ' ' | xargs ps -o pid,etime,cputime,%cpu,args 2>/dev/null | cut -c1-200
echo '--- TOTAL WORKER CPU SECONDS ---'
pgrep -f 'spawn_mai[n]' | xargs ps -o cputime= -p 2>/dev/null | python3 -c "
import sys
tot=0; n=0
for line in sys.stdin:
    s=line.strip()
    if not s: continue
    d=0
    if '-' in s:
        dd, s = s.split('-',1); d=int(dd)
    parts=[float(x) for x in s.split(':')]
    while len(parts)<3: parts=[0]+parts
    h,m,sec=parts
    tot+=((d*24+h)*3600+m*60+sec); n+=1
print('WORKERS_COUNT', n)
print('TOTAL_WORKER_CPU_S', int(tot))
"
echo '--- V3 PILOT PROCS (any) ---'
ps -eo pid,etime,%cpu,args | grep -E 'pilot_du_v3|v3_launch|stage1_teacher_gate|stage1_formal' | grep -v grep | cut -c1-220 | head -10 || echo NO-V3-PROCS
echo '--- A4 / M7BSR1 RESIDUAL ---'
ps -eo pid,etime,args | grep -E 'a4_select|cnet2_train|m7bsr1' | grep -v grep | cut -c1-200 | head -5 || echo NO-A4-RESIDUAL
echo '--- LOAD ---'
uptime
echo '--- MEMORY ---'
vm_stat | head -12
sysctl -n hw.memsize
echo '--- STAGE1_FORMAL DIR ---'
ls -lt /Users/arm/junyang/carbon/data/stage1_formal/ | cut -c1-160 | head -25
echo '--- PILOT DU V3 LOG ---'
for f in /Users/arm/junyang/carbon/data/stage1_formal/pilot_du_v3.log /Users/arm/junyang/carbon/data/pilot_du_v3.log; do
  if [ -f "$f" ]; then echo "LOG $f:"; ls -l "$f"; wc -l "$f"; echo '--- tail ---'; tail -5 "$f" | cut -c1-300; fi
done
echo '--- PILOT DU V3 JSON ---'
for f in /Users/arm/junyang/carbon/data/stage1_formal/pilot_du_v3.json /Users/arm/junyang/carbon/data/pilot_du_v3.json; do
  if [ -f "$f" ]; then echo "JSON $f:"; ls -l "$f"; fi
done
echo '--- SIDECAR / SCENE / PROGRESS FILES ---'
ls -lt /Users/arm/junyang/carbon/data/stage1_formal/ | grep -iE 'sidecar|scene|root|progress|pilot_du' | cut -c1-160 | head -15
echo '--- M7BSR1 SELECTION STAGING ---'
ls -lt /Users/arm/junyang/carbon/ 2>/dev/null | grep -iE 'm7bsr1|a4|select' | cut -c1-160 | head -10
find /Users/arm/junyang/carbon -maxdepth 3 -iname '*m7bsr1*' -o -maxdepth 3 -iname '*a4_select*' 2>/dev/null | head -10
echo '--- DISK ---'
df -h / | tail -1
echo "STUDIOSTAT1527 STUDIO-END $(date '+%F %T %Z')"
STEOF
echo "$MARK SSH-RC=$?"
echo "$MARK PROBE-END $(date '+%F %T %Z')"
} > "$LOG" 2>&1
echo "probe done rc=$? log=$LOG"

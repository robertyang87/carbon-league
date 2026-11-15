#!/bin/bash
# Studio status 15:42 trip: verify m7bsr1 A4 selection running (launched 15:39:43, driver PID/PGID 42374, 16 workers). READ-ONLY. Marker STUDIOSTAT1542.
MARK=STUDIOSTAT1542
LOG=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/studio-status-20261114-1542-ts.log
{
echo "$MARK PROBE-BEGIN $(date '+%F %T %Z')"
timeout -k 30 150 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30 -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o GlobalKnownHostsFile=/dev/null -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" arm@100.78.149.20 'bash -s' <<'STEOF'
echo "STUDIOSTAT1542 STUDIO-BEGIN $(date '+%F %T %Z')"
hostname; uptime
echo '--- A4 DRIVER 42374 ---'
ps -p 42374 -o pid,etime,cputime,%cpu,args 2>/dev/null | cut -c1-220 || echo PID42374-GONE
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
echo '--- A4 / M7BSR1 PROCS (any) ---'
ps -eo pid,etime,%cpu,args | grep -E 'a4_select|cnet2_train|m7bsr1' | grep -v grep | cut -c1-220 | head -10 || echo NO-A4-PROCS
echo '--- LOAD ---'
uptime
echo '--- MEMORY ---'
vm_stat | head -12
sysctl -n hw.memsize
echo '--- A4 SELECT LOG ---'
f=/Users/arm/junyang/a4_select_m7bsr1_20261114.log
if [ -f "$f" ]; then ls -l "$f"; wc -l "$f"; echo '--- tail ---'; tail -8 "$f" | cut -c1-300; echo '--- epoch lines ---'; grep -iE 'ep[0-9]+|epoch' "$f" | tail -6 | cut -c1-300; fi
echo '--- SELECTION DIR ---'
ls -lt /Users/arm/junyang/a4_selection_20261114/ 2>/dev/null | cut -c1-160 | head -15
echo '--- V3 PILOT RESIDUAL ---'
ps -eo pid,etime,args | grep -E 'pilot_du_v3|stage1_teacher_gate' | grep -v grep | cut -c1-200 | head -5 || echo NO-V3-RESIDUAL
echo '--- DISK ---'
df -h / | tail -1
echo "STUDIOSTAT1542 STUDIO-END $(date '+%F %T %Z')"
STEOF
echo "$MARK SSH-RC=$?"
echo "$MARK PROBE-END $(date '+%F %T %Z')"
} > "$LOG" 2>&1
echo "probe done rc=$? log=$LOG"

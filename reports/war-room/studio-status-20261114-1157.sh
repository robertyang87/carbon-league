#!/bin/bash
# Studio status report 11:57 trip: A4 r1 selection progress/winner check + v3 pilot readiness. READ-ONLY. Marker STUDIOSTAT1157.
MARK=STUDIOSTAT1157
LOG=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/studio-status-20261114-1157-ts.log
{
echo "$MARK PROBE-BEGIN $(date '+%F %T %Z')"
timeout -k 30 150 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30 -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o GlobalKnownHostsFile=/dev/null -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" arm@100.78.149.20 'bash -s' <<'STEOF'
echo "STUDIOSTAT1157 STUDIO-BEGIN $(date '+%F %T %Z')"
hostname; uptime
echo '--- A4 DRIVER PROCS ---'
ps -eo pid,etime,%cpu,args | grep -E 'a4_select|spawn_mai[n]|cnet2_train' | grep -v grep | cut -c1-220 | head -20 || echo NO-A4-PROCS
echo '--- R1 DRIVER 82551 ---'
ps -p 82551 -o pid,etime,%cpu,args 2>/dev/null | cut -c1-220 || echo PID82551-GONE
echo '--- SPAWN COUNT ---'
pgrep -f 'spawn_mai[n]' | wc -l
echo '--- TOP CPU ---'
ps -eo pid,%cpu,etime,args | sort -k2 -rn | head -5 | cut -c1-200
echo '--- A4 SELECTION DIR ---'
ls -lt /Users/arm/junyang/a4_selection_20261114/ | cut -c1-160 | head -25
echo '--- R1 META SUMMARY ---'
python3 -c "
import json
try:
    d=json.load(open('/Users/arm/junyang/a4_selection_20261114/r1_selection.json'))
    print('keys:', sorted(d.keys()))
    for k in ('winner','pick_code','pick_operative','winner_epoch','epochs_done','epochs','n_epochs','mean_u','pick','duel','tie_band'):
        if k in d: print(k, '=', d[k])
    eps=d.get('epochs') or d.get('results') or d.get('per_epoch') or []
    if isinstance(eps,list):
        for e in eps:
            if isinstance(e,dict):
                print('EP', e.get('epoch', e.get('ep')), 'mean_u', e.get('mean_u'), 'se', e.get('se'), 'errors', e.get('errors'), 'ok', e.get('ok'), 'wall', e.get('wall_s'))
    else:
        print('epochs type', type(eps))
except Exception as ex:
    print('R1-META-ERR', ex)
" 2>&1 | head -60
echo '--- R1 META RAW HEAD ---'
head -c 1500 /Users/arm/junyang/a4_selection_20261114/r1_selection.json 2>/dev/null || echo NO-R1-META-YET
echo '--- V3 PILOT STATE ---'
ls -lt /Users/arm/junyang/carbon/data/ 2>/dev/null | grep -i -E 'pilot_du_v3|v3' | cut -c1-160 | head
ls -lt /Users/arm/junyang/carbon/data/pilot_du_v3.json /Users/arm/junyang/carbon/data/*pilot* 2>/dev/null | cut -c1-160 | head
ps -eo pid,etime,args | grep -E 'pilot_du_v3|v3_pilot|gen1r.*pilot' | grep -v grep | cut -c1-200 | head || echo NO-V3-PILOT-PROC
echo '--- DISK ---'
df -h / | tail -1
echo "STUDIOSTAT1157 STUDIO-END $(date '+%F %T %Z')"
STEOF
echo "$MARK SSH-RC=$?"
echo "$MARK PROBE-END $(date '+%F %T %Z')"
} | tee "$LOG"

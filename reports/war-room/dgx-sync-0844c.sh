#!/bin/bash
LOG=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/dgx-sync-0844c-ts.log
{
echo "DGXSYNC0844C PROBE-BEGIN $(date '+%F %T %Z')"
timeout -k 30 180 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30 -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o GlobalKnownHostsFile=/dev/null -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" arm@100.84.148.73 'bash -s' <<'STEOF'
echo "DGXSYNC0844C DGX-BEGIN $(date '+%F %T %Z')"
echo '--- PULL COUNTS (repo data dir) ---'
python3 -c "
import json
for n in ['hakimi3pro','fable6','k4v2']:
    p=f'/home/arm/junyang/carbon-repo/data/{n}_matches.json'
    try:
        d=json.load(open(p)); print(n, 'entries=', len(d) if isinstance(d,list) else 'dict:'+str(len(d)))
    except Exception as e: print(n, 'ERR', repr(e)[:120])
"
ls -lt /home/arm/junyang/carbon-repo/data/ | head -10
echo '--- PULL MANIFESTS ---'
head -c 600 /home/arm/junyang/pull0030_hakimi3pro_gap2.manifest; echo
head -c 600 /home/arm/junyang/pull0030_fable6_gap2.manifest; echo
head -c 600 /home/arm/junyang/pull0030_k4v2_gap2.manifest; echo
echo '--- K0 OUT DIR ---'
ls -lt /home/arm/junyang/k0_a1_out/ 2>/dev/null | head -12 || echo NO-K0-OUT-DIR
echo '--- K0 TRAIN PROC NOW ---'
ps -eo pid,etime,args | grep '[k]0_train' | cut -c1-200
ps -p 1079386 -o pid,etime,args 2>/dev/null | cut -c1-160 || echo OLD-PID-1079386-GONE
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv 2>&1
echo "DGXSYNC0844C DGX-END $(date '+%F %T %Z')"
STEOF
echo "DGXSYNC0844C SSH-RC=$?"
echo "DGXSYNC0844C PROBE-END $(date '+%F %T %Z')"
} | tee "$LOG"

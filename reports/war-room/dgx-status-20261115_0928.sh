#!/bin/bash
# DGX status probe — TEMPLATE (2026-11-15, war-room order: hard-code relay key)
# Each trip: copy this file, replace DGXSTATUS20261115_0928 with the dated marker
# (e.g. DGXSTATUS20261115_0843), run on DGX via: ssh arm@dgx 'bash -s' < file.
# DO NOT rewrite the spot-32 line below; append extra checks only.
# Read-only. Spot-32 probe uses the relay key -i id_ed25519_carbon_relay ALWAYS.
echo "MARKER=DGXSTATUS20261115_0928 START $(date '+%F %T %Z')"
hostname; uptime
echo "--- nvidia-smi ---"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader 2>&1
echo "--- load ---"
cat /proc/loadavg
echo "--- duel/training processes ---"
ps -eo pid,etime,%cpu,args | grep -E 'duel|k0_|ppo_train|bigeval' | grep -v grep | head -25 || echo NO_DUEL_PROC
echo "--- git fetch/pull/checkout processes ---"
ps -eo pid,etime,args | grep -E 'git (fetch|pull|checkout|clone)' | grep -v grep || echo NO_GIT_PROC
echo "--- k0_calib duels dir ---"
ls -la --time-style=long-iso /home/arm/junyang/k0_calib/duels/ 2>&1
echo "--- duel log mtimes ---"
ls -la --time-style=long-iso /home/arm/junyang/k0_calib/*.log /home/arm/junyang/k0_calib/duels/*.log 2>&1 | head -10
echo "--- corpus counts ---"
python3 -c "
import json,os
for d in ['m3s2_corpus','carago_corpus','fable_corpus']:
    p=f'/home/arm/junyang/carbon/data/{d}/index.json'
    try: n=len(json.load(open(p)))
    except Exception as e: n='ERR '+str(e)
    rp=f'/home/arm/junyang/carbon/data/{d}/replays'
    c=len(os.listdir(rp)) if os.path.isdir(rp) else 'NO-DIR'
    print(d,'index=',n,'replays=',c)
for d in ['ds_carago','ds_fable']:
    p=f'/home/arm/junyang/carbon/data/{d}/meta.json'
    try:
        m=json.load(open(p)); print(d,'meta=',{k:m.get(k) for k in ('samples','n','count','total') if k in m})
    except Exception as e: print(d,'meta ERR',e)
"
echo "--- repo HEAD vs remote ---"
cd /home/arm/junyang/carbon-repo && git rev-parse HEAD && timeout 60 git ls-remote origin HEAD 2>&1 | head -2
echo "--- recent file changes (last 60 min, excl logs) ---"
find /home/arm/junyang/carbon /home/arm/junyang/carbon-repo -type f -mmin -60 2>/dev/null | grep -v -E '\.log$' | head -15
echo "--- tailscale status ---"
tailscale status 2>&1 | head -12
echo "--- spot-32 relay-key ssh (read-only probe; relay key HARD-CODED, never default key) ---"
timeout 25 ssh -o BatchMode=yes -o ConnectTimeout=10 -o IdentitiesOnly=yes -i /home/arm/.ssh/id_ed25519_carbon_relay -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null root@100.95.45.69 'echo SPOT32-RELAY-OK; hostname; uptime; ps -eo pid,etime,%cpu,args | grep -E "bigeval|eval_v53" | grep -v grep | head -5 || echo NO_EXAM_PROC' 2>&1 | head -8
echo "MARKER-END DGXSTATUS20261115_0928"
echo "--- spot-33 tailnet visibility (appended 20261115_0928) ---"
tailscale status 2>&1 | grep -E 'spot-33|100.90.180.19' || echo SPOT33-NOT-IN-DGX-TAILSCALE-LIST

#!/bin/bash
# DGX status probe — marker DGXSTATUS0758 (2026-11-15 07:58 CST run)
# Read-only. Run on DGX via: ssh arm@dgx 'bash -s' < this file
echo "MARKER=DGXSTATUS0758 START $(date '+%F %T %Z')"
hostname; uptime
echo "--- nvidia-smi ---"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader 2>&1
echo "--- load ---"
cat /proc/loadavg
echo "--- duel/training processes ---"
ps -eo pid,etime,%cpu,args | grep -E 'duel|k0_|ppo_train|bigeval' | grep -v grep | head -25 || echo NO_DUEL_PROC
echo "--- git fetch/pull/checkout processes ---"
ps -eo pid,etime,args | grep -E 'git (fetch|pull|checkout|clone)' | grep -v grep || echo NO_GIT_PROC
echo "--- duels dir ---"
ls -la --time-style=long-iso /home/arm/junyang/carbon/data/duels/ 2>&1 | tail -6
echo "--- student npz ---"
ls -la --time-style=long-iso /home/arm/junyang/carbon/data/weights/ 2>&1 | tail -6
md5sum /home/arm/junyang/carbon/data/weights/*.npz 2>/dev/null | head -4
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
git rev-list --count HEAD..origin/main 2>/dev/null || echo count-err
echo "--- recent file changes (last 60 min, excl logs) ---"
find /home/arm/junyang/carbon /home/arm/junyang/carbon-repo -type f -mmin -60 2>/dev/null | grep -v -E '\.log$' | head -15
echo "--- tailscale status ---"
tailscale status 2>&1 | head -20
echo "--- spot-32 ssh retest (tailnet, batch, 20s) ---"
timeout 25 ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null root@100.95.45.69 'echo SPOT32-OK' 2>&1 | head -3
echo "MARKER-END DGXSTATUS0758"

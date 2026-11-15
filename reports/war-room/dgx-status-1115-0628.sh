#!/bin/bash
# DGX status probe 2026-11-15 06:28 trip — read-only. Tailscale SSH from this VM to DGX.
# Markers DGXSTATUS0628A (main), DGXSTATUS0628B (spot32/tailnet), DGXSTATUS0628C (k0 detail).
LOG=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/dgx-status-1115-0628-ts.log
{
echo "DGXSTATUS0628A PROBE-BEGIN $(date '+%F %T %Z')"
timeout -k 30 420 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30 -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o GlobalKnownHostsFile=/dev/null -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" arm@100.84.148.73 'bash -s' <<'DGXEOF'
echo "DGXSTATUS0628A DGX-BEGIN $(date '+%F %T %Z')"
hostname
uptime
date
echo "--- gpu ---"
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader
echo "--- load ---"
cat /proc/loadavg
echo "--- training procs ---"
ps -eo pid,etime,%cpu,cmd | grep -E "k0_calib|k0_duel|rl2|ppo|train|bigeval|fastsim" | grep -v grep || echo NOPROC
echo "--- git residue ---"
ps -eo pid,etime,cmd | grep -E "git (fetch|pull|checkout)" | grep -v grep || echo NOGIT
echo "--- python procs ---"
ps -eo pid,etime,%cpu,cmd | grep python | grep -v grep | head -30 || echo NOPY
echo "--- k0 calib cells ---"
ls -lt /home/arm/junyang/k0_calib/cells/ 2>/dev/null | head -6
echo "--- k0 calib duels dir ---"
ls -l /home/arm/junyang/k0_calib/duels/ 2>/dev/null | tail -5
echo "--- duel log ---"
stat -c '%y %s %n' /home/arm/junyang/k0_calib/duel_lr0.0002_s200_b1_seed7.log 2>/dev/null; tail -3 /home/arm/junyang/k0_calib/duel_lr0.0002_s200_b1_seed7.log 2>/dev/null
echo "--- corpus counts ---"
cd /home/arm/junyang/carbon
for c in m3s2_corpus carago_corpus fable_corpus; do
  if [ -d "data/$c" ]; then
    idx=$(python3 -c "import json;print(len(json.load(open('data/$c/index.json'))))" 2>/dev/null || echo ERR)
    rep=$(ls data/$c/replays 2>/dev/null | wc -l)
    echo "$c idx=$idx replays=$rep"
  else
    echo "$c MISSING"
  fi
done
echo "--- ds meta ---"
for d in ds_carago ds_fable; do
  if [ -f "data/$d/meta.json" ]; then
    echo "$d $(stat -c '%y' data/$d/meta.json | cut -c1-16) $(python3 -c "import json;m=json.load(open('data/$d/meta.json'));print('games=',m.get('games'),'total=',m.get('total'))" 2>/dev/null || echo ERR)"
  else
    echo "$d MISSING"
  fi
done
echo "--- newest npz ---"
find /home/arm/junyang -name "*.npz" -type f -printf "%T@ %p\n" 2>/dev/null | sort -rn | head -5
echo "--- files changed last 60m ---"
find /home/arm/junyang/carbon /home/arm/junyang/k0_calib -type f -mmin -60 2>/dev/null | head -20
echo "--- git HEAD ---"
cd /home/arm/junyang/carbon-repo 2>/dev/null && git rev-parse HEAD && git branch --show-current && timeout 60 git ls-remote origin HEAD || echo "NOGITREPO"
echo "DGXSTATUS0628B TAILSCALE-FULL"
tailscale status 2>&1 | head -25
echo "--- SPOT32-MARKERS ---"
find /home/arm/junyang -maxdepth 3 \( -iname '*spot*32*' -o -iname '*spot32*' -o -iname '*exam*ready*' -o -iname '*EXAM-READY*' \) 2>/dev/null | head -20
echo "--- SPOT32-PROC ---"
ps -eo pid,etime,args | grep -E '[1]00\.95\.45\.69|[s]pot-32' | head -5; echo NO_SPOT32_PROC
echo "--- SPOT32-SSH-PROBE ---"
timeout 25 ssh -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o GlobalKnownHostsFile=/dev/null arm@100.95.45.69 'hostname; uptime' 2>&1 | head -5; echo "SPOT32-SSH-RC=$?"
echo "DGXSTATUS0628C K0-DETAIL"
echo "--- cells npz md5 ---"
md5sum /home/arm/junyang/k0_calib/cells/lr0.0002_s200_b1_seed7.npz 2>/dev/null; stat -c '%y %s' /home/arm/junyang/k0_calib/cells/lr0.0002_s200_b1_seed7.npz 2>/dev/null
echo "--- duels json ---"
ls -l /home/arm/junyang/k0_calib/duels/ 2>/dev/null | tail -5
echo "DGXSTATUS0628A DGX-END $(date '+%F %T %Z')"
DGXEOF
echo "DGXSTATUS0628A SSH-RC=$?"
echo "DGXSTATUS0628A PROBE-END $(date '+%F %T %Z')"
} | tee "$LOG"

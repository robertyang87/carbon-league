#!/bin/bash
# DGX status report 04:58 trip (this run): read-only Tailscale SSH direct. Marker DGXSTATUS0458B.
MARK=DGXSTATUS0458B
LOG=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/dgx-status-0458b-ts.log
{
echo "$MARK PROBE-BEGIN $(date '+%F %T %Z')"
timeout -k 30 240 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30 -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o GlobalKnownHostsFile=/dev/null -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" arm@100.84.148.73 'bash -s' <<'STEOF'
echo "DGXSTATUS0458B DGX-BEGIN $(date '+%F %T %Z')"
echo '--- HOST ---'
hostname; uptime
echo '--- GPU ---'
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv 2>&1 | head -5
echo '--- GRID2-TRAIN-PROC ---'
ps -eo pid,etime,%cpu,args | grep -E '[k]0_distill|[l]r0.0002_s200' | cut -c1-220 | head -10
echo '--- TRAIN-PID-1671928 ---'
ps -p 1671928 -o pid,etime,%cpu,args 2>&1 | cut -c1-220
echo '--- CALIB-DRIVER ---'
ps -eo pid,etime,%cpu,args | grep '[k]0_calib_driver' | cut -c1-220 | head -5
echo '--- K0-CALIB-DIR-RECENT ---'
ls -lt --time-style=+%F_%T /home/arm/junyang/k0_calib/ 2>&1 | head -20
echo '--- CELLS-DIR ---'
ls -lt --time-style=+%F_%T /home/arm/junyang/k0_calib/cells/ 2>&1 | head -10
echo '--- DUELS-DIR ---'
ls -l --time-style=+%F_%T /home/arm/junyang/k0_calib/duels/ 2>&1 | head -10
echo '--- DRIVER-LOG-TAIL ---'
tail -6 /home/arm/junyang/k0_calib/driver.log 2>&1 | cut -c1-250
echo '--- TRAIN-LOG-DISCOVER ---'
ls -lt --time-style=+%F_%T /home/arm/junyang/k0_calib/*.log* 2>&1 | head -10
echo '--- S200-LOG-TAIL ---'
for f in /home/arm/junyang/k0_calib/*s200*.log* /home/arm/junyang/k0_calib/*s200*.jsonl; do [ -f "$f" ] && echo "== $f" && tail -4 "$f" | cut -c1-250; done
echo '--- CORPUS ---'
cd /home/arm/junyang/carbon && for c in m3s2_corpus carago_corpus fable_corpus; do echo -n "$c idx="; python3 -c "import json;print(len(json.load(open('data/$c/index.json'))))" 2>/dev/null || echo -n '?'; echo -n ' replays='; ls data/$c/replays 2>/dev/null | wc -l; done
echo '--- DS-META ---'
for d in ds_carago ds_fable; do echo -n "$d "; ls -l --time-style=+%m%d_%H:%M data/$d/meta.json 2>/dev/null; python3 -c "import json;m=json.load(open('data/$d/meta.json'));print('games='+str(len(m.get('games',[])))+' total='+str(m.get('total')))" 2>/dev/null || echo '?'; done
echo '--- GIT-RESIDUAL ---'
ps -eo pid,etime,args | grep -E '[g]it (fetch|pull|checkout)' | head -5
echo '--- GIT-HEAD ---'
cd /home/arm/junyang/carbon-repo && git rev-parse --short HEAD 2>&1 && timeout 60 git ls-remote origin HEAD 2>&1 | cut -c1-12
echo '--- TAILNET-SPOT32 ---'
tailscale status 2>&1 | grep -E 'spot-32|100.95.45.69|spot-29|spot-31' | head -10
echo '--- EXAM-DEPLOY-PROCS ---'
ps -eo pid,etime,args | grep -iE '[b]igeval|[e]xam|[s]pot-32|[d]eploy' | cut -c1-220 | head -10
echo '--- RECENT-FILES-60M ---'
find /home/arm/junyang -maxdepth 2 -type f -mmin -60 2>/dev/null | grep -v -E 'carbon-repo/|\.cache' | head -20
echo "DGXSTATUS0458B DGX-END $(date '+%F %T %Z')"
STEOF
echo "$MARK SSH-RC=$?"
echo "$MARK PROBE-END $(date '+%F %T %Z')"
} | tee "$LOG"

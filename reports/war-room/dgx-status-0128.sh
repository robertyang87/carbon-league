#!/bin/bash
# DGX status probe 0128 (2026-11-15 01:28 CST). Marker DGXSTATUS0128A. Read-only.
MARK=DGXSTATUS0128A
echo "$MARK BEGIN $(date -u +%H:%M:%S)"
timeout -k 30 240 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=20 -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o GlobalKnownHostsFile=/dev/null arm@100.84.148.73 'bash -s' <<'DGXEOF'
M=DGXSTATUS0128A
echo "$M DGX $(hostname) $(uptime)"
echo '--GPU--'
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader
echo '--PS-PYTHON--'
ps -eo pid,etime,%cpu,args | grep -iE 'k0_duel|k0_annot|train|duel|rl2|ppo|build|pull|fastsim' | grep -v grep | head -40
echo '--DRIVERS--'
ps -p 1605013 -o pid,etime,args 2>/dev/null || echo "1605013 NOT RUNNING"
ps -p 1613140 -o pid,etime,args 2>/dev/null || echo "1613140 NOT RUNNING"
echo '--DUEL-CWD--'
for p in $(pgrep -f k0_duel.py | head -3); do echo "pid=$p cwd=$(readlink /proc/$p/cwd)"; done
echo '--DUEL-DIR--'
D=$(readlink /proc/$(pgrep -f k0_duel.py | head -1)/cwd 2>/dev/null)
echo "duel_cwd=$D"
ls -la "$D" 2>/dev/null | head -30
echo '--DUEL-OUT--'
ls -la "$D/duels" 2>/dev/null | head -20; echo "duels_count=$(ls "$D/duels" 2>/dev/null | wc -l)"
find "$D" -maxdepth 2 -name '*.log' -newermt '2026-11-15 01:00' -exec ls -la {} \; 2>/dev/null | head -20
echo '--GITRESID--'
ps -eo pid,etime,args | grep -E 'git (fetch|pull|checkout)' | grep -v grep; echo "(end gitresid)"
echo '--HEAD--'
cd /home/arm/junyang/carbon-repo && timeout 60 git rev-parse HEAD && timeout 60 git rev-parse --abbrev-ref HEAD && timeout 60 git rev-parse origin/HEAD 2>/dev/null; timeout 60 git ls-remote origin HEAD | awk '{print "remote:", $1}'
echo '--CORPUS--'
for d in m3s2_corpus carago_corpus fable_corpus krypton-csplant_corpus; do echo -n "$d total_files="; ls /home/arm/junyang/carbon-repo/data/$d/ 2>/dev/null | wc -l; echo -n "$d gz="; ls /home/arm/junyang/carbon-repo/data/$d/*.json.gz 2>/dev/null | wc -l; echo -n "$d index="; python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(len(d))' /home/arm/junyang/carbon-repo/data/$d/index.json 2>/dev/null || echo NA; done
for d in ds_carago ds_fable; do echo -n "$d meta="; python3 -c 'import json,sys; m=json.load(open(sys.argv[1])); print(m.get("total", m))' /home/arm/junyang/carbon-repo/data/$d/meta.json 2>/dev/null || echo NA; done
echo '--K0-OUT--'
find /home/arm/junyang/carbon-repo /home/arm/junyang/carbon -maxdepth 4 -name 'complete.json' -newermt '2026-11-15 00:30' 2>/dev/null | head; echo "(end k0-out)"
find /home/arm/junyang/carbon-repo /home/arm/junyang/carbon -maxdepth 4 -name '*.npz' -newermt '2026-11-15 00:30' 2>/dev/null | head; echo "(end npz)"
echo "$M END"
DGXEOF
echo "$MARK END rc=$?"

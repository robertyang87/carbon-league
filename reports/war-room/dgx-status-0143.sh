#!/bin/bash
# DGX status probe 0143 (2026-11-14 01:43 CST). Marker DGXSTATUS0143A. Read-only.
MARK=DGXSTATUS0143A
echo "$MARK BEGIN $(date -u +%H:%M:%S)"
timeout -k 30 240 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=20 -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o GlobalKnownHostsFile=/dev/null arm@100.84.148.73 'bash -s' <<'DGXEOF'
M=DGXSTATUS0143A
echo "$M DGX $(hostname) $(uptime)"
echo '--GPU--'
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader
echo '--PS--'
ps -eo pid,etime,%cpu,args | grep -iE 'python|train|build|pull|fastsim|sim|agent_mcp' | grep -v grep | head -20
echo '--SPAWN--'
pgrep -a spawn_main 2>/dev/null; echo "(end spawn)"
echo '--GITRESID--'
ps -eo pid,etime,args | grep -E 'git (fetch|pull|checkout)' | grep -v grep; echo "(end gitresid)"
echo '--HEAD--'
cd /home/arm/junyang/carbon-repo && timeout 60 git rev-parse HEAD && timeout 60 git ls-remote origin HEAD | awk '{print "remote:", $1}'
echo '--CORPUS--'
for d in m3s2_corpus carago_corpus fable_corpus; do echo -n "$d replays="; ls /home/arm/junyang/carbon-repo/data/$d/ 2>/dev/null | wc -l; echo -n "$d index="; python3 -c "import json;print(len(json.load(open('/home/arm/junyang/carbon-repo/data/$d/index.json'))))" 2>/dev/null || echo NA; done
for d in ds_carago ds_fable; do for base in /home/arm/junyang/carbon-repo/data /home/arm/junyang/carbon/data; do f="$base/$d/meta.json"; if [ -f "$f" ]; then echo -n "$d meta($base)="; python3 -c "import json;m=json.load(open('$f'));print(m.get('n_samples',m.get('samples',m)))" 2>/dev/null || echo NA; fi; done; done
echo '--SHARD1--'
ls -lt /home/arm/junyang/carbon-repo/data/ds_gen1r_shard1/ 2>/dev/null | head -8
find /home/arm/junyang/carbon-repo -name '*shard1*' -newermt '2026-11-14 00:20' -type f 2>/dev/null | head -5; echo "(end shard1-new)"
grep -rl 'raw label alignment failed' /home/arm/junyang/carbon-repo/data/ 2>/dev/null | head -3
ls -lt /home/arm/junyang/carbon-repo/data/ 2>/dev/null | head -8
echo '--FASTSIM-SPOT--'
md5sum /home/arm/junyang/carbon-repo/code/fastsim.py /home/arm/junyang/carbon/code/fastsim.py 2>/dev/null
echo "$M END"
DGXEOF
echo "$MARK END rc=$?"

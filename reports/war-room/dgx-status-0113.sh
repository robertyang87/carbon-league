#!/bin/bash
# DGX status probe 0113 (2026-11-14 01:13 CST). Marker DGXSTATUS0113A. Read-only.
MARK=DGXSTATUS0113A
echo "$MARK BEGIN $(date -u +%H:%M:%S)"
timeout -k 30 240 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=20 -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o GlobalKnownHostsFile=/dev/null arm@100.84.148.73 'bash -s' <<'DGXEOF'
M=DGXSTATUS0113A
echo "$M DGX $(hostname) $(uptime)"
echo '--GPU--'
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader
echo '--PS--'
ps -eo pid,etime,%cpu,args | grep -iE 'python|train|build|pull|fastsim|sim|agent_mcp' | grep -v grep | head -20
echo '--GITRESID--'
ps -eo pid,etime,args | grep -E 'git (fetch|pull|checkout)' | grep -v grep; echo "(end gitresid)"
echo '--HEAD--'
cd /home/arm/junyang/carbon-repo && timeout 60 git rev-parse HEAD && timeout 60 git ls-remote origin HEAD | awk '{print "remote:", $1}'
echo '--PULL0030--'
tail -5 /home/arm/junyang/pull0030.log 2>/dev/null
ls -l /home/arm/junyang/pull0030_*.manifest 2>/dev/null
ls /home/arm/junyang/carbon-repo/data/fable_corpus/ 2>/dev/null | grep -c '0d4872c2' ; ls -l /home/arm/junyang/carbon-repo/data/fable_corpus/0d4872c256d7d36a0650f3c7ca73f92d-1.json.gz 2>/dev/null
echo '--CORPUS--'
for d in m3s2_corpus carago_corpus fable_corpus; do echo -n "$d replays="; ls /home/arm/junyang/carbon-repo/data/$d/ 2>/dev/null | wc -l; echo -n "$d index="; python3 -c "import json;print(len(json.load(open('/home/arm/junyang/carbon-repo/data/$d/index.json'))))" 2>/dev/null || echo NA; done
for d in ds_carago ds_fable; do echo -n "$d meta="; python3 -c "import json;m=json.load(open('/home/arm/junyang/carbon-repo/data/$d/meta.json'));print(m)" 2>/dev/null || echo NA; done
echo '--SHARD1--'
tail -3 /home/arm/junyang/carbon-repo/data/ds_gen1r_shard1/*.log 2>/dev/null; ls /home/arm/junyang/carbon-repo/data/ds_gen1r_shard1/ 2>/dev/null | head; ls /home/arm/junyang/carbon-repo/data/ds_gen1r_shard1/meta.json 2>/dev/null || echo "shard1 meta MISSING"
find /home/arm/junyang/carbon-repo -name '*shard1*' -newermt '2026-11-13 23:56' -type f 2>/dev/null | head -5; echo "(end shard1-new)"
echo '--FASTSIM-SPOT--'
md5sum /home/arm/junyang/carbon-repo/code/fastsim.py /home/arm/junyang/carbon/code/fastsim.py /home/arm/junyang/s26stage/code/fastsim.py 2>/dev/null
echo '--BUILDMD5--'
md5sum /home/arm/junyang/carbon-repo/code/cnet2_build.py 2>/dev/null
echo "$M END"
DGXEOF
echo "$MARK END rc=$?"

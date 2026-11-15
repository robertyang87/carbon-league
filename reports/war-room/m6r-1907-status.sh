#!/bin/bash
echo "M6CHAIN1907R2 STATUS START $(date -u +%H:%M:%S)"
DGX="arm@10.211.43.155"
KEY="/root/.ssh/dgx_access"
echo "--- zt reachability ---"
ping -c 1 -W 5 10.211.43.155 || echo "PING FAILED"
timeout -k 30 240 ssh -i "$KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=30 "$DGX" 'bash -s' <<'EOS'
echo "--- repo pull ---"
cd /home/arm/junyang/carbon-repo && timeout 60 git pull --ff-only 2>&1 | tail -5
git -C /home/arm/junyang/carbon-repo rev-parse --short HEAD
git -C /home/arm/junyang/carbon-repo branch --show-current
echo "--- procs ---"
ps aux | grep -E "cnet2_train|ladder_corpus|build_top|set_uw|replica_chain|bigeval" | grep -v grep || echo "no training/build procs"
echo "--- gpu ---"
nvidia-smi --query-gpu=utilization.gpu,temperature.gpu,memory.used --format=csv,noheader
echo "--- m6 counts repo ---"
python3 -c "import json; d=json.load(open('/home/arm/junyang/carbon-repo/data/m6_matches.json')); print('repo matches:', len(d))"
ls /home/arm/junyang/carbon-repo/data/m6_corpus/ | wc -l
echo "--- m6 counts workdir ---"
python3 -c "import json; d=json.load(open('/home/arm/junyang/carbon/data/m6_matches.json')); print('workdir matches:', len(d))"
ls /home/arm/junyang/carbon/data/m6_corpus/ | wc -l
ls /home/arm/junyang/carbon/data/ | grep -i m6 || echo "no m6 entries in workdir data"
ls /home/arm/junyang/carbon/data/ds_m6 2>/dev/null | head -5 || echo "no ds_m6 dir in workdir"
EOS
echo "SSH_RC=$?"
echo "M6CHAIN1907R2 STATUS END"

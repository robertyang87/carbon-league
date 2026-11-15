#!/bin/bash
# DGX status probe 2026-11-14 13:13, marker DGXSTATUS11141313 (read-only)
SSH_OPTS=(-i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o ConnectTimeout=30)
timeout -k 30 240 ssh "${SSH_OPTS[@]}" arm@100.84.148.73 '
echo MARKER=DGXSTATUS11141313
hostname; uptime
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader
echo ---PS-TRAIN---
ps aux | grep -E "rl2_ppo_train|cnet2_train|train|eval|select" | grep -v grep | head -20
echo ---SPAWN---
pgrep -f "spawn_mai[n]" | wc -l
echo ---GITPROC---
ps -eo pid,etime,args | grep -E "git (fetch|pull|checkout|clone)" | grep -v grep
echo ---HEAD---
cd /home/arm/junyang/carbon-repo && git rev-parse HEAD && timeout 60 git ls-remote origin HEAD
echo ---PACKS---
ls -l --time-style=+%F_%T code/packs/m3s2g1_a4.py code/packs/m3s2g1_base.py code/packs/m3s2r1_a4.py code/packs/m3s2r1_base.py
md5sum code/packs/m3s2g1_a4.py code/packs/m3s2g1_base.py code/packs/m3s2r1_a4.py code/packs/m3s2r1_base.py
echo ---CORPUS---
echo -n "m3s2 files: "; ls data/m3s2_corpus/replays/*.json.gz 2>/dev/null | wc -l
echo -n "carago files: "; ls data/carago_corpus/replays/*.json.gz 2>/dev/null | wc -l
echo -n "fable files: "; ls data/fable_corpus/replays/*.json.gz 2>/dev/null | wc -l
python3 -c "import json;print(\"m3s2 idx\",len(json.load(open(\"data/m3s2_corpus/index.json\"))));print(\"carago idx\",len(json.load(open(\"data/carago_corpus/index.json\"))));print(\"fable idx\",len(json.load(open(\"data/fable_corpus/index.json\"))))"
echo ---FABLE6---
ls /home/arm/junyang/gap4/ 2>/dev/null | head; python3 -c "import json;m=json.load(open(\"/home/arm/junyang/gap4/pull0032_fable6_gap4.manifest\"));print(\"fable6 manifest\",len(m))" 2>&1
echo ---DS---
ls -l --time-style=+%F_%T data/ds_carago/meta.json data/ds_fable/meta.json
python3 -c "import json;d=json.load(open(\"data/ds_carago/meta.json\"));print(\"ds_carago\",{k:d.get(k) for k in list(d)[:8]})" 2>&1
python3 -c "import json;d=json.load(open(\"data/ds_fable/meta.json\"));print(\"ds_fable\",{k:d.get(k) for k in list(d)[:8]})" 2>&1
'

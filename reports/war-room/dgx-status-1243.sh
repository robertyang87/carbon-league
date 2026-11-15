#!/bin/bash
# DGX status probe trip 12:43 - read-only. Marker DGXSTATUS1243A.
echo "DGXSTATUS1243A $(date '+%H:%M:%S %Z')"
hostname; uptime; df -h / | tail -1
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv,noheader
echo "== TRAIN =="
ps -eo pid,etime,cmd | grep -E 'cnet2_train|stage0_train|selfplay_gen|eval_v53|train_heads|deploy_stack|stage0_guided|gate_pilot' | grep -v grep || echo NO_TRAIN_PROC
echo "== GIT =="
ps -eo pid,etime,cmd | grep -E 'git fetch|git pull|git checkout' | grep -v grep || echo NO_GIT_PROC
echo "== GATE PILOT =="
ls -lt /home/arm/junyang/carbon/gate_pilot* /home/arm/junyang/carbon/data/gate_pilot* 2>&1 | head -12
tail -n 8 /home/arm/junyang/carbon/gate_pilot_guided.log 2>&1
echo ---
ls -lt /home/arm/junyang/carbon/*.log 2>&1 | head -12
echo "== HEADS V2 =="
ls -l /home/arm/junyang/carbon/data/weights/cnet2_m3s2r1_heads_v2.npz 2>&1
md5sum /home/arm/junyang/carbon/data/weights/cnet2_m3s2r1_heads_v2.npz 2>&1
tail -n 3 /home/arm/junyang/train_heads_v2.log 2>&1
echo "== HEAD =="
cd /home/arm/junyang/carbon-repo && git rev-parse HEAD && git log --oneline -3 && timeout 30 git ls-remote origin HEAD
echo "== LAST_STATS =="
git log --oneline -5 -- code/cnet2_search.py
grep -rn 'search_games' /home/arm/junyang/carbon-repo/code/cnet2_search.py | head -5
echo "== CORPORA WORKDIR =="
for c in m3s2_corpus carago_corpus fable_corpus; do echo -n "$c replays: "; ls /home/arm/junyang/carbon/data/$c/replays 2>/dev/null | wc -l; echo -n "$c index: "; python3 -c "import json;print(len(json.load(open('/home/arm/junyang/carbon/data/$c/index.json'))))" 2>&1 | tail -1; done
echo "== DS META =="
for d in ds_carago ds_fable ds_m3s2 ds_gen1; do echo -n "$d: "; head -c 200 /home/arm/junyang/carbon/data/$d/meta.json 2>&1; echo; done
echo "== REPO DATA LITE-V7 / IT460 =="
ls /home/arm/junyang/carbon-repo/data/lite_v7_corpus 2>/dev/null | wc -l
ls /home/arm/junyang/carbon-repo/data/m9xj460_corpus 2>/dev/null | wc -l
python3 -c "import json;print('m9xj460_matches:',len(json.load(open('/home/arm/junyang/carbon-repo/data/m9xj460_matches.json'))))" 2>&1 | tail -1
echo "== GOLDEN / DEPLOY =="
ls -lt /home/arm/junyang/golden/ | head -8
ps -eo pid,etime,cmd | grep -E 'deploy_stack|bootstrap' | grep -v grep || echo NO_DEPLOY_PROC
echo "DGXSTATUS1243A-END $(date '+%H:%M:%S %Z')"

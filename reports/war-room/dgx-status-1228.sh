#!/bin/bash
# DGX status probe trip 12:28 - read-only. Marker DGXSTATUS1228A.
echo "DGXSTATUS1228A $(date '+%H:%M:%S %Z')"
hostname; uptime; df -h / | tail -1
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv,noheader
echo "== TRAIN =="
ps -eo pid,etime,cmd | grep -E 'cnet2_train|stage0_train|selfplay_gen|eval_v53|train_heads|deploy_stack' | grep -v grep || echo NO_TRAIN_PROC
echo "== GIT =="
ps -eo pid,etime,cmd | grep -E 'git fetch|git pull|git checkout' | grep -v grep || echo NO_GIT_PROC
echo "== GEN1 LOGS =="
ls -lt /home/arm/junyang/carbon/dq_logs/ 2>&1 | head -10
tail -n 5 /home/arm/junyang/carbon/dq_logs/gen1g1.log 2>&1
echo ---
tail -n 5 /home/arm/junyang/carbon/dq_logs/gen1r1.log 2>&1
echo "== GEN1 WEIGHTS =="
ls -lt /home/arm/junyang/carbon/data/weights/ 2>&1 | head -8
echo "== HEAD =="
cd /home/arm/junyang/carbon-repo && git rev-parse HEAD && git log --oneline -3 && timeout 30 git ls-remote origin HEAD
echo "== LAST_STATS =="
grep -n 'last_stats' /home/arm/junyang/carbon-repo/code/cnet2_search.py | head -8
grep -rn 'last_stats' /home/arm/junyang/carbon-repo/code/eval_v53.py | head -5
git log --oneline -5 -- code/cnet2_search.py
echo "== CORPORA WORKDIR =="
for c in m3s2_corpus carago_corpus fable_corpus; do echo -n "$c replays: "; ls /home/arm/junyang/carbon/data/$c/replays 2>/dev/null | wc -l; echo -n "$c index: "; python3 -c "import json;print(len(json.load(open('/home/arm/junyang/carbon/data/$c/index.json'))))" 2>&1 | tail -1; done
echo "== DS META =="
for d in ds_carago ds_fable ds_m3s2 ds_gen1; do echo -n "$d: "; head -c 200 /home/arm/junyang/carbon/data/$d/meta.json 2>&1; echo; done
echo "== REPO DATA =="
ls /home/arm/junyang/carbon-repo/data/ | head -40
echo "== LITE-V7 / IT460 =="
ls /home/arm/junyang/carbon-repo/data/lite_v7_corpus 2>/dev/null | wc -l
ls -d /home/arm/junyang/carbon-repo/data/*it460* /home/arm/junyang/carbon/data/*it460* 2>/dev/null || echo NO_IT460_DIR
echo "== GOLDEN / DEPLOY =="
ls -lt /home/arm/junyang/golden/ | head -8
echo "DGXSTATUS1228A-END $(date '+%H:%M:%S %Z')"

#!/bin/bash
# DGX status probe trip 13:28 - read-only. Marker DGXSTATUS1328A.
echo "DGXSTATUS1328A $(date '+%H:%M:%S %Z')"
hostname; uptime; df -h / | tail -1
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv,noheader
echo "== TRAIN/GATE/DEPLOY PROCS =="
ps -eo pid,etime,cmd | grep -E 'katago_chain|stage0_guided|gate|policy_verify|bigeval|selfplay_gen|deploy_stack|train_heads|eval_v53|hakimi' | grep -v grep || echo NO_TRAIN_PROC
echo "== GIT PROCS =="
ps -eo pid,etime,cmd | grep -E 'git (fetch|pull|checkout|merge|rebase)' | grep -v grep || echo NO_GIT_PROC
echo "== DEPLOY STACK DETAIL =="
ps -eo pid,etime,cmd | grep -i deploy | grep -v grep || echo NO_DEPLOY_PROC
ls -lt /home/arm/junyang/golden/ 2>&1 | head -5
echo "== LAST_STATS VERIFY FIELD (repo tree is the live one) =="
ls -lt /home/arm/junyang/carbon-repo/data/bigeval_live/ 2>&1 | head -8
for f in /home/arm/junyang/carbon-repo/data/bigeval_live/*.jsonl; do [ -f "$f" ] && echo "$f lines=$(wc -l < "$f") mtime=$(stat -c %y "$f" | cut -c1-19)"; done 2>/dev/null
echo "-- margin rows --"
grep -h '"margin"' /home/arm/junyang/carbon-repo/data/bigeval_live/*.jsonl 2>/dev/null | wc -l
tail -n 3 /home/arm/junyang/carbon-repo/data/bigeval_live/verify_search.jsonl 2>/dev/null | cut -c1-400
echo "-- verify_search log tail --"
tail -n 3 /home/arm/junyang/verify_search.log 2>&1 | cut -c1-300
echo "== HEAD =="
cd /home/arm/junyang/carbon-repo && git branch --show-current && git rev-parse HEAD && git log --oneline -3 && git status --porcelain=v1 --branch 2>&1 | head -5
timeout 30 git ls-remote origin HEAD 2>&1
echo "== CODE VERSION (a7ed8f95 check) =="
md5sum /home/arm/junyang/carbon-repo/code/cnet2_model.py /home/arm/junyang/carbon-repo/code/producer/selfplay_gen.py 2>&1 | cut -c1-80
grep -m1 -i 'version\|a7ed' /home/arm/junyang/carbon-repo/code/CODE_VERSION 2>/dev/null || echo NO_CODE_VERSION_FILE
echo "== CORPORA =="
for c in m3s2_corpus carago_corpus fable_corpus; do echo -n "$c replays: "; ls /home/arm/junyang/carbon/data/$c/replays 2>/dev/null | wc -l; echo -n "$c index: "; python3 -c "import json;print(len(json.load(open('/home/arm/junyang/carbon/data/$c/index.json'))))" 2>&1 | tail -1; done
python3 -c "
import json
for d in ['ds_carago','ds_fable','ds_m3s2','ds_gen1']:
    try:
        m=json.load(open(f'/home/arm/junyang/carbon/data/{d}/meta.json'))
        print(d, 'games=', len(m.get('games',[])))
    except Exception as e:
        print(d, 'ERR', e)
" 2>&1
echo "== REPO CORPORA =="
ls /home/arm/junyang/carbon-repo/data/lite_v7_corpus 2>/dev/null | wc -l
ls /home/arm/junyang/carbon-repo/data/m9xj460_corpus 2>/dev/null | wc -l
ls /home/arm/junyang/carbon-repo/data/hakimi3_corpus 2>/dev/null | wc -l
echo "DGXSTATUS1328A-END $(date '+%H:%M:%S %Z')"

#!/bin/bash
# DGX status probe trip 12:58 - read-only. Marker DGXSTATUS1258A.
echo "DGXSTATUS1258A $(date '+%H:%M:%S %Z')"
hostname; uptime; df -h / | tail -1
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv,noheader
echo "== TRAIN/GATE/DEPLOY PROCS =="
ps -eo pid,etime,cmd | grep -E 'katago_chain|stage0_guided|gate|policy_verify|bigeval|selfplay_gen|deploy_stack|train_heads|eval_v53' | grep -v grep || echo NO_TRAIN_PROC
echo "== GIT PROCS =="
ps -eo pid,etime,cmd | grep -E 'git (fetch|pull|checkout|merge|rebase)' | grep -v grep || echo NO_GIT_PROC
echo "== GATE 400-GAME FIELD =="
ls -lt /home/arm/junyang/carbon/data/gate_pilot_guided /home/arm/junyang/carbon/data/gate_pilot_greedy 2>&1 | head -20
cat /home/arm/junyang/carbon/data/gate_pilot_guided/summary.json 2>&1; echo
ls -lt /home/arm/junyang/carbon/ /home/arm/junyang/ 2>&1 | grep -Ei 'gate|teacher|guided' | head -10
find /home/arm/junyang/carbon /home/arm/junyang -maxdepth 2 -name '*gate*.log' -o -maxdepth 2 -name '*teacher*.log' 2>/dev/null | head
echo "== SPOT-31 DEPLOY =="
ls -lt /home/arm/junyang/golden/ 2>&1 | head -8
ls -lt /home/arm/junyang/*.log /home/arm/junyang/carbon/*.log 2>/dev/null | head -8
echo "== LAST_STATS VERIFY FIELD =="
ls -lt /home/arm/junyang/carbon/data/ 2>&1 | head -15
find /home/arm/junyang/carbon -maxdepth 3 -name '*policy_verify*' -o -maxdepth 3 -name 'bigeval_live*' 2>/dev/null | head
tail -n 5 /home/arm/junyang/carbon/bigeval_live.jsonl 2>&1 | cut -c1-300
echo "== GEN-1 ROUND 2 =="
ls /home/arm/junyang/carbon/data/gen1_shard0/replays 2>/dev/null | wc -l
python3 -c "import json;print('gen1 index:',len(json.load(open('/home/arm/junyang/carbon/data/gen1_shard0/index.json'))))" 2>&1 | tail -1
ls -lt /home/arm/junyang/*gen1* /home/arm/junyang/carbon/*gen1* 2>/dev/null | head -6
tail -n 4 /home/arm/junyang/gen1b.log 2>&1 | cut -c1-300
echo "== HEAD =="
cd /home/arm/junyang/carbon-repo && git branch --show-current && git rev-parse HEAD && git log --oneline -5 && git status --porcelain=v1 --branch 2>&1 | head -8
timeout 30 git ls-remote origin HEAD 2>&1
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
echo "== REPO DATA =="
ls /home/arm/junyang/carbon-repo/data/lite_v7_corpus 2>/dev/null | wc -l
ls /home/arm/junyang/carbon-repo/data/m9xj460_corpus 2>/dev/null | wc -l
echo "DGXSTATUS1258A-END $(date '+%H:%M:%S %Z')"

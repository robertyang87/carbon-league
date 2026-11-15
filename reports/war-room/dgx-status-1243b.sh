#!/bin/bash
# DGX status probe trip 12:43 - follow-up B. Marker DGXSTATUS1243B. Read-only.
echo "DGXSTATUS1243B $(date '+%H:%M:%S %Z')"
echo "== GATE PILOT SUMMARIES =="
cat /home/arm/junyang/carbon/data/gate_pilot_greedy/summary.json 2>&1; echo
cat /home/arm/junyang/carbon/data/gate_pilot_greedy/games.jsonl 2>&1; echo
cat /home/arm/junyang/carbon/data/gate_pilot_guided/summary.json 2>&1; echo
cat /home/arm/junyang/carbon/data/gate_pilot_guided/games.jsonl 2>&1; echo
echo "== REMOTE HEAD OBJECT =="
cd /home/arm/junyang/carbon-repo && git cat-file -t a9287a4ca3adf871651c74c6e05f350c07bcf1c7 2>&1
git rev-list --count 4f12f901f74ce3b0e7d3fea1465a655e0904de29..a9287a4ca3adf871651c74c6e05f350c07bcf1c7 2>&1
git log --oneline a9287a4ca3adf871651c74c6e05f350c07bcf1c7 -5 2>&1
echo "== LAST_STATS COMMITS =="
git log --oneline -10 --grep=last_stats 2>&1
git log --oneline -5 2>&1
echo "== DS COUNTS =="
python3 -c "
import json
for d in ['ds_carago','ds_fable','ds_m3s2','ds_gen1']:
    m=json.load(open(f'/home/arm/junyang/carbon/data/{d}/meta.json'))
    g=m.get('games',[])
    print(d, 'games=', len(g))
" 2>&1
echo "== LITE-V7 REPO =="
ls -lt /home/arm/junyang/carbon-repo/data/lite_v7_corpus 2>&1 | head -5
echo "== GPU NOW =="
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv,noheader
ps -eo pid,etime,cmd | grep -E 'cnet2_train|stage0_train|selfplay_gen|eval_v53|train_heads|deploy_stack|stage0_guided' | grep -v grep || echo NO_TRAIN_PROC
echo "DGXSTATUS1243B-END $(date '+%H:%M:%S %Z')"

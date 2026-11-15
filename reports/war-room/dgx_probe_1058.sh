#!/bin/bash
# DGX status probe - read-only. Marker: DGXSTATUS1058A (trip scheduled 2026-11-13 10:58 CST)
echo "MARK DGXSTATUS1058A"
hostname
uptime
date '+%F %T %Z'
echo "--- GPU ---"
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader || echo GPU_QUERY_FAIL
echo "--- TRAIN/PULL PROCS ---"
ps -eo pid,etime,args | grep -E 'rl2_ppo|bigeval|eval_v53|pull_all|carago|fable|godslayer|train' | grep -v grep || echo NONE_TRAIN_PULL
echo "--- PULL PID 517459 ---"
ps -p 517459 -o pid,etime,args 2>/dev/null || echo PULL517459_GONE
echo "--- GIT RESIDUE ---"
ps -eo pid,etime,args | grep -E 'git (fetch|pull|checkout|clone)' | grep -v grep || echo NO_GIT_RESIDUE
echo "--- REPO ---"
cd /home/arm/junyang/carbon-repo && git rev-parse HEAD && git branch --show-current && git log --oneline -2 && timeout 60 git ls-remote origin HEAD
echo "--- WORKDIR CORPUS ---"
cd /home/arm/junyang/carbon
for d in data/m3s2_corpus data/carago_corpus data/fable_corpus; do
  echo "$d files=$(ls $d 2>/dev/null | wc -l) index=$(python3 -c "import json;print(len(json.load(open('$d/index.json'))))" 2>/dev/null || echo NO_INDEX)"
done
for ds in ds_carago ds_fable; do
  echo "$ds meta=$(python3 -c "import json;m=json.load(open('$ds/meta.json'));print(m.get('total',m))" 2>/dev/null || echo NO_META)"
done
echo "--- PULL LOG TAIL ---"
tail -5 /home/arm/junyang/carbon/pull_all.log 2>/dev/null || tail -5 /home/arm/pull_all.log 2>/dev/null || echo NO_PULL_LOG
echo "--- CARAGO1 CORPUS ---"
ls /home/arm/junyang/carbon/data/carago1_corpus 2>/dev/null | wc -l
echo "--- END DGXSTATUS1058A ---"

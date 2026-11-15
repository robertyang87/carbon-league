#!/bin/bash
# DGX status probe - read-only. Marker: DGXSTATUS1128A (trip scheduled 2026-11-13 11:28 CST)
echo "MARK DGXSTATUS1128A"
hostname
uptime
date '+%F %T %Z'
echo "--- GPU ---"
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader || echo GPU_QUERY_FAIL
echo "--- GPU PROCS ---"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader || echo NO_GPU_PROCS
echo "--- TRAIN PROCS ---"
ps -eo pid,etime,args | grep -E 'train_heads|rl2_ppo|bigeval|eval_v53|pull_all|heads_patch|smoke' | grep -v grep || echo NONE_TRAIN
echo "--- TRAIN PID 527818 ---"
ps -p 527818 -o pid,etime,args 2>/dev/null || echo TRAIN527818_GONE
echo "--- GIT RESIDUE ---"
ps -eo pid,etime,args | grep -E 'git (fetch|pull|checkout|clone)' | grep -v grep || echo NO_GIT_RESIDUE
echo "--- REPO ---"
cd /home/arm/junyang/carbon-repo && git rev-parse HEAD && git branch --show-current && git log --oneline -2 && timeout 60 git ls-remote origin HEAD
echo "--- WORKDIR CORPUS ---"
cd /home/arm/junyang/carbon
for d in data/m3s2_corpus data/carago_corpus data/fable_corpus data/carago1_corpus data/godslayer_cvi_corpus data/m9xj320_corpus data/lite_v7_corpus; do
  echo "$d files=$(ls $d 2>/dev/null | wc -l) index=$(python3 -c "import json;print(len(json.load(open('$d/index.json'))))" 2>/dev/null || echo NO_INDEX)"
done
for ds in data/ds_carago data/ds_fable data/ds_m3s2; do
  echo "$ds meta=$(python3 -c "import json;m=json.load(open('$ds/meta.json'));print(m.get('total',m))" 2>/dev/null || echo NO_META)"
done
echo "--- REPO-SIDE NEW BATCHES ---"
for d in data/carago1_corpus data/godslayer_cvi_corpus data/m9xj320_corpus data/lite_v7_corpus; do
  echo "repo/$d files=$(ls /home/arm/junyang/carbon-repo/$d 2>/dev/null | wc -l)"
done
echo "--- TRAIN HEADS LOG TAIL ---"
tail -25 /home/arm/junyang/train_heads.log 2>/dev/null || echo NO_TRAIN_HEADS_LOG
echo "--- HEADS WEIGHT OUTPUT ---"
ls -l --time-style=full-iso /home/arm/junyang/carbon/data/weights/cnet2_m3s2r1_heads.npz 2>/dev/null || echo NO_HEADS_WEIGHT_YET
echo "--- TAR/SCP TO SPOT-30 (100.99.41.71) ---"
ps -eo pid,etime,args | grep -E 'scp|rsync|tar|zstd' | grep -v grep || echo NO_TAR_SCP_RUNNING
ls -l --time-style=full-iso /home/arm/junyang/golden/ 2>/dev/null | head -10 || echo NO_GOLDEN_DIR
echo "--- DISK ---"
df -h / | tail -1
echo "--- END DGXSTATUS1128A ---"

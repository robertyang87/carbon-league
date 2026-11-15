#!/bin/bash
# DGX status probe 2243: post-main-switch verify, GPU idle, it320 wr1 weight, corpus counts, git residue, HEAD.
# Read-only. Marker: DGXSTATUS2243A
MARK=DGXSTATUS2243A
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout -k 30 240 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
echo "DGXSTATUS2243A DGX-BEGIN $(date '+%F %T %Z')"
hostname; uptime
echo '--- TRAIN-PROCS ---'
ps -eo pid,etime,args | grep -E 'cnet2_train|rl2_ppo|train' | grep -v grep || echo NO-TRAIN-PROC
echo '--- GPU ---'
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader
echo '--- GIT-RESIDUE ---'
ps -eo pid,etime,args | grep -E 'git (fetch|pull|checkout)' | grep -v grep || echo NO-GIT-PROC
echo '--- HEAD-LOCAL ---'
cd /home/arm/junyang/carbon-repo && timeout 60 git rev-parse HEAD && timeout 60 git rev-parse --abbrev-ref HEAD && timeout 60 git status --porcelain=v1 | head -5; echo HEAD-RC=$?
echo '--- HEAD-REMOTE ---'
cd /home/arm/junyang/carbon-repo && timeout 60 git ls-remote origin main | cut -c1-7; timeout 60 git ls-remote origin cnet2-search-v12 | cut -c1-7; timeout 60 git fetch --dry-run 2>&1 | head -3; echo FETCH-RC=$?
echo '--- ATTIC ---'
ls -l --time-style=+%F_%T /home/arm/junyang/carbon/data/attic/ 2>/dev/null | head -5 || echo NO-ATTIC
echo '--- WR1-WEIGHT ---'
ls -l --time-style=+%F_%T /home/arm/junyang/carbon/data/weights/cnet2_m9xj320wr1_fromstudio.npz 2>/dev/null && md5sum /home/arm/junyang/carbon/data/weights/cnet2_m9xj320wr1_fromstudio.npz 2>/dev/null || echo NO-WR1-WEIGHT
echo '--- WEIGHTS-NEWEST ---'
ls -lt --time-style=+%F_%T /home/arm/junyang/carbon/data/weights/cnet2_m9xj320*.npz 2>/dev/null | head -8
echo '--- CHAIN-STATE ---'
cat /home/arm/junyang/carbon/data/m9xj320_chain.state 2>/dev/null || echo NO-CHAIN-STATE
echo '--- CORPUS-COUNTS ---'
cd /home/arm/junyang/carbon
for c in m3s2_corpus carago_corpus fable_corpus m9xj_corpus m9xj320_corpus; do
  echo -n "$c idx="; python3 -c "import json;print(len(json.load(open('data/$c/index.json'))))" 2>/dev/null || echo -n MISSING; echo -n " replays="; ls data/$c/replays 2>/dev/null | wc -l
done
echo '--- DS-META ---'
for d in ds_m9xj320 ds_m9xj320_w ds_carago ds_fable; do echo -n "$d: "; ls -d data/$d 2>/dev/null && python3 -c "import json;m=json.load(open('data/$d/meta.json'));print(m.get('num_samples') or m.get('n') or list(m)[:6])" 2>/dev/null || echo MISSING; done
echo '--- RECENT-FILES-30M ---'
find /home/arm/junyang/carbon/data -mmin -30 -type f 2>/dev/null | head -10 || echo NONE-RECENT
echo '--- S26STAGE ---'
ls -lt --time-style=+%F_%T /home/arm/junyang/s26stage 2>/dev/null | head -5 || echo NO-S26STAGE
echo '--- LOADAVG ---'
cat /proc/loadavg
echo "DGXSTATUS2243A DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"

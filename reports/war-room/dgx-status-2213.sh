#!/bin/bash
# DGX status probe 2213: r1 completion, GPU, chain state, corpus counts, git residue, HEAD.
# Read-only. Marker: DGXSTATUS2213A
MARK=DGXSTATUS2213A
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout -k 30 240 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
echo "DGXSTATUS2213A DGX-BEGIN $(date '+%F %T %Z')"
hostname; uptime
echo '--- R1-TRAIN-LOG-TAIL ---'
tail -30 /home/arm/junyang/carbon/data/dq_logs/m9xj320r1.log 2>/dev/null || echo NO-R1-LOG
echo '--- G1-TRAIN-LOG-TAIL ---'
tail -5 /home/arm/junyang/carbon/data/dq_logs/m9xj320g1.log 2>/dev/null || echo NO-G1-LOG
echo '--- CHAIN-OUTER-TAIL ---'
tail -12 /home/arm/junyang/carbon/data/dq_logs/m320n_chain_outer.log 2>/dev/null || echo NO-OUTER-LOG
echo '--- CHAIN-STATE ---'
cat /home/arm/junyang/carbon/data/m9xj320_chain.state 2>/dev/null || echo NO-CHAIN-STATE
echo '--- WEIGHTS-NEWEST ---'
ls -lt --time-style=+%F_%T /home/arm/junyang/carbon/data/weights/cnet2_m9xj320*.npz 2>/dev/null | head -8
echo '--- TRAIN-PROCS ---'
ps -eo pid,etime,args | grep -E 'cnet2_train|rl2_ppo|train' | grep -v grep || echo NO-TRAIN-PROC
echo '--- GPU ---'
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader
echo '--- GIT-RESIDUE ---'
ps -eo pid,etime,args | grep -E 'git (fetch|pull|checkout)' | grep -v grep || echo NO-GIT-PROC
echo '--- HEAD-LOCAL ---'
cd /home/arm/junyang/carbon-repo && timeout 60 git rev-parse HEAD && timeout 60 git rev-parse --abbrev-ref HEAD && timeout 60 git status --porcelain=v1 | head -5; echo HEAD-RC=$?
echo '--- CORPUS-COUNTS ---'
cd /home/arm/junyang/carbon
for c in m3s2_corpus carago_corpus fable_corpus m9xj_corpus m9xj320_corpus; do
  echo -n "$c idx="; python3 -c "import json;print(len(json.load(open('data/$c/index.json'))))" 2>/dev/null || echo -n MISSING; echo -n " replays="; ls data/$c/replays 2>/dev/null | wc -l
done
echo '--- DS-META ---'
for d in ds_m9xj320 ds_m9xj320_w; do echo -n "$d: "; ls -d data/$d 2>/dev/null && python3 -c "import json;m=json.load(open('data/$d/meta.json'));print(m.get('num_samples') or m.get('n') or list(m)[:6])" 2>/dev/null || echo MISSING; done
echo "DGXSTATUS2213A DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"

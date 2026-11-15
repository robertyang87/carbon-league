#!/bin/bash
# DGX status probe 1913: verify REPLACEMENT M6 agent start triple + GPU idle duration + HEAD + git residue + corpus counts.
# Read-only. Marker: DGXSTATUS1913Q8R2
MARK=DGXSTATUS1913Q8R2
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout -k 30 240 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
echo "DGXSTATUS1913Q8R2 DGX-BEGIN $(date '+%F %T %Z')"
hostname; uptime
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader
echo '--- TRAIN-PROCS ---'
ps -eo pid,etime,args | grep -E 'cnet2_train|ladder_corpus|set_uw|replica_chain' | grep -v grep || echo NO-TRAIN-PROC
echo '--- ALL-PYTHON-PROCS ---'
ps -eo pid,etime,args | grep -E 'python3|python ' | grep -v grep | head -20 || echo NO-PY-PROC
echo '--- ALL-BASH-PROCS (chain scripts) ---'
ps -eo pid,etime,args | grep -E 'replica|code/' | grep -v grep | head -10 || echo NO-CHAIN-PROC
echo '--- GIT-RESIDUE ---'
ps -eo pid,etime,args | grep -E 'git (fetch|pull|checkout)' | grep -v grep || echo NO-GIT-PROC
echo '--- M6-DS-AND-LOGS ---'
date '+%F %T %Z'
ls -ld /home/arm/junyang/carbon/data/ds_m6 2>/dev/null || echo NO-DS-M6
python3 -c "import json;d=json.load(open('/home/arm/junyang/carbon/data/ds_m6/meta.json'));print('m6 games=',d.get('n_games',len(d.get('games',[]))),'samples=',d.get('n_samples',d.get('samples')))" 2>&1 | head -2
ls -lt --time-style=+%F_%T /home/arm/junyang/carbon/data/dq_logs/ 2>/dev/null | head -12
for f in /home/arm/junyang/carbon/data/dq_logs/m6r1.log /home/arm/junyang/carbon/data/dq_logs/m6g1.log /home/arm/junyang/carbon/data/dq_logs/m6*.log; do [ -e "$f" ] || continue; echo "== $f"; ls -l --time-style=+%F_%T "$f"; tail -3 "$f"; done
echo '--- M6-WEIGHTS ---'
ls -lt --time-style=+%F_%T /home/arm/junyang/carbon/data/weights/cnet2_m6*.npz 2>/dev/null | head -6 || echo NO-M6-CKPT
echo '--- K4V2-ON-DGX ---'
ls -lt --time-style=+%F_%T /home/arm/junyang/carbon/data/weights/cnet2_k4v2*.npz 2>/dev/null | head -4 || echo NO-K4V2-WEIGHT
ls -l --time-style=+%F_%T /home/arm/junyang/carbon/data/weights/K4V2_SHIP_DONE 2>/dev/null || echo NO-K4V2-SHIP-FILE
echo '--- DS-META ---'
for d in ds_m7bs_v2 ds_k4v2 ds_m3s2; do python3 -c "import json;dd=json.load(open('/home/arm/junyang/carbon/data/$d/meta.json'));print('$d games=',dd.get('n_games',len(dd.get('games',[]))),'samples=',dd.get('n_samples',dd.get('samples')))" 2>&1 | head -1; done
echo '--- CORPUS-COUNTS ---'
cd /home/arm/junyang/carbon/data 2>/dev/null
for d in m3s2_corpus carago_corpus fable_corpus k4v2_corpus m6_corpus; do echo -n "$d replays-dir="; ls "$d/replays" 2>/dev/null | wc -l; echo -n "$d top-gz="; ls "$d"/*.json.gz 2>/dev/null | wc -l; done
for f in m3s2_corpus/index.json carago_corpus/index.json fable_corpus/index.json k4v2_corpus/index.json m6_corpus/index.json; do echo -n "$f: "; python3 -c "import json;d=json.load(open('$f'));print(len(d) if isinstance(d,list) else len(d.get('entries',d)))" 2>/dev/null || echo MISSING; done
echo '--- HEAD-VS-REMOTE (branch tip) ---'
cd /home/arm/junyang/carbon-repo 2>/dev/null && git rev-parse --abbrev-ref HEAD && git rev-parse HEAD && git log -1 --format='%ci %s' && timeout 60 git ls-remote origin cnet2-search-v12 2>&1 | head -2
echo "DGXSTATUS1913Q8R2 DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"

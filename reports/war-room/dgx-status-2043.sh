#!/bin/bash
# DGX status probe 2043: M9XJ it260 single-chain landing check (dispatched ~20:26, now ~17 min, M6 precedent ~17 min to visible process);
# it260 behavior-profile 97-round precise version re-check; GPU idle watch (4th consecutive idle run if still idle).
# Read-only. Marker: DGXSTATUS2043K2N8
MARK=DGXSTATUS2043K2N8
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout -k 30 240 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
echo "DGXSTATUS2043K2N8 DGX-BEGIN $(date '+%F %T %Z')"
hostname; uptime
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader
echo '--- TRAIN-PROCS ---'
ps -eo pid,etime,args | grep -E 'cnet2_train|ladder_corpus|set_uw|replica_chain' | grep -v grep || echo NO-TRAIN-PROC
echo '--- ALL-PYTHON-PROCS ---'
ps -eo pid,etime,args | grep -E 'python3|python ' | grep -v grep | head -20 || echo NO-PY-PROC
echo '--- GIT-RESIDUE ---'
ps -eo pid,etime,args | grep -E 'git (fetch|pull|checkout)' | grep -v grep || echo NO-GIT-PROC
echo '--- M9XJ-LOGS-AND-WEIGHTS ---'
date '+%F %T %Z'
for f in /home/arm/junyang/carbon/data/dq_logs/m9xj260r1.log /home/arm/junyang/carbon/data/dq_logs/m9xj260g1.log /home/arm/junyang/carbon/data/dq_logs/m9xj260wr1.log; do [ -e "$f" ] || continue; echo "== $f"; ls -l --time-style=+%F_%T "$f"; tail -3 "$f"; done
ls -lt --time-style=+%F_%T /home/arm/junyang/carbon/data/weights/cnet2_m9xj*.npz 2>/dev/null | head -6 || echo NO-M9XJ-CKPT
echo '--- NEWEST-WEIGHTS ---'
ls -lt --time-style=+%F_%T /home/arm/junyang/carbon/data/weights/*.npz 2>/dev/null | head -6
echo '--- RECENT-FILES-LAST-30MIN ---'
find /home/arm/junyang/carbon/data -mmin -30 -type f 2>/dev/null | head -40
echo '--- M9XJ-DIRS ---'
ls -d /home/arm/junyang/carbon/data/*m9xj* /home/arm/junyang/carbon/data/*it260* 2>/dev/null || echo NO-M9XJ-DIRS
for d in /home/arm/junyang/carbon/data/m9xj_corpus /home/arm/junyang/carbon/data/ds_m9xj /home/arm/junyang/carbon/data/ds_m9xj_w /home/arm/junyang/carbon/data/m9xj-trigger97; do [ -e "$d" ] || continue; echo "== $d"; ls "$d" 2>/dev/null | head -5; done
echo '--- DS-META ---'
for d in ds_m6 ds_m7bs_v2 ds_k4v2 ds_m3s2 ds_m9xj ds_m9xj_w; do python3 -c "import json;dd=json.load(open('/home/arm/junyang/carbon/data/$d/meta.json'));print('$d games=',dd.get('n_games',len(dd.get('games',[]))),'samples=',dd.get('n_samples',dd.get('samples')))" 2>&1 | head -1; done
echo '--- BEHAVIOR-PROFILE ---'
ls -lt --time-style=+%F_%T /home/arm/junyang/carbon/reports/live/ 2>/dev/null | head -6
echo '--- CORPUS-COUNTS ---'
cd /home/arm/junyang/carbon/data 2>/dev/null
for d in m3s2_corpus carago_corpus fable_corpus k4v2_corpus m6_corpus m9xj_corpus; do echo -n "$d replays-dir="; ls "$d/replays" 2>/dev/null | wc -l; echo -n "$d top-gz="; ls "$d"/*.json.gz 2>/dev/null | wc -l; done
for f in m3s2_corpus/index.json carago_corpus/index.json fable_corpus/index.json k4v2_corpus/index.json m6_corpus/index.json m9xj_corpus/index.json; do echo -n "$f: "; python3 -c "import json;d=json.load(open('$f'));print(len(d) if isinstance(d,list) else len(d.get('entries',d)))" 2>/dev/null || echo MISSING; done
echo '--- HEAD-VS-REMOTE ---'
cd /home/arm/junyang/carbon-repo 2>/dev/null && git rev-parse --abbrev-ref HEAD && git rev-parse HEAD && git log -3 --format='%h %ci %s' && timeout 60 git ls-remote origin cnet2-search-v12 2>&1 | head -2
echo "DGXSTATUS2043K2N8 DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"

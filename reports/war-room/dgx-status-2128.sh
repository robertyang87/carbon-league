#!/bin/bash
# DGX status probe 2128: it320 dispatch check (user 21:16: dispatched directly, no 100-game
# wait; freeze corpus at diff 0 (~81 games, scout topping up), DGX dual arms + Studio
# weighted arm start in sync). Check corpus/build/train progress; M9XJ DONE unchanged.
# Read-only. Marker: DGXSTATUS2128R7T3
MARK=DGXSTATUS2128R7T3
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout -k 30 240 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
echo "DGXSTATUS2128R7T3 DGX-BEGIN $(date '+%F %T %Z')"
hostname; uptime
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader
echo '--- TRAIN-PROCS ---'
ps -eo pid,etime,args | grep -E 'cnet2_train|ladder_corpus|set_uw|replica_chain' | grep -v grep || echo NO-TRAIN-PROC
echo '--- ALL-PYTHON-PROCS ---'
ps -eo pid,etime,args | grep -E 'python3|python ' | grep -v grep | head -20 || echo NO-PY-PROC
echo '--- GIT-RESIDUE ---'
ps -eo pid,etime,args | grep -E 'git (fetch|pull|checkout)' | grep -v grep || echo NO-GIT-PROC
echo '--- DATA-DIR-LISTING ---'
ls -lt --time-style=+%F_%T /home/arm/junyang/carbon/data/ | head -30
echo '--- IT320-CANDIDATES ---'
ls -ld /home/arm/junyang/carbon/data/*320* /home/arm/junyang/carbon/data/*it320* /home/arm/junyang/carbon/data/ds_*320* 2>/dev/null || echo NO-IT320-DIRS
ls -lt --time-style=+%F_%T /home/arm/junyang/carbon/data/dq_logs/ 2>/dev/null | head -15
echo '--- NEWEST-WEIGHTS ---'
ls -lt --time-style=+%F_%T /home/arm/junyang/carbon/data/weights/*.npz 2>/dev/null | head -10
echo '--- CHAIN-STATES ---'
ls -lt --time-style=+%F_%T /home/arm/junyang/carbon/data/*.state 2>/dev/null | head -8
for s in /home/arm/junyang/carbon/data/*.state; do echo "== $s"; tail -3 "$s" 2>/dev/null; done
echo '--- RECENT-FILES-LAST-30MIN ---'
find /home/arm/junyang/carbon/data -mmin -30 -type f 2>/dev/null | head -40
echo '--- DS-META ---'
for d in ds_m6 ds_m7bs_v2 ds_k4v2 ds_m3s2 ds_m9xj ds_m9xj_w ds_carago ds_fable ds_m320 ds_it320; do python3 -c "import json;dd=json.load(open('/home/arm/junyang/carbon/data/$d/meta.json'));print('$d games=',dd.get('n_games',len(dd.get('games',[]))),'samples=',dd.get('n_samples',dd.get('samples')))" 2>&1 | head -1; done
echo '--- CORPUS-COUNTS ---'
cd /home/arm/junyang/carbon/data 2>/dev/null
for d in m3s2_corpus carago_corpus fable_corpus k4v2_corpus m6_corpus m9xj_corpus m320_corpus it320_corpus; do [ -d "$d" ] || continue; echo -n "$d replays-dir="; ls "$d/replays" 2>/dev/null | wc -l; echo -n "$d top-gz="; ls "$d"/*.json.gz 2>/dev/null | wc -l; done
for f in m3s2_corpus/index.json carago_corpus/index.json fable_corpus/index.json k4v2_corpus/index.json m6_corpus/index.json m9xj_corpus/index.json m320_corpus/index.json it320_corpus/index.json; do [ -e "$f" ] || continue; echo -n "$f: "; python3 -c "import json;d=json.load(open('$f'));print(len(d) if isinstance(d,list) else len(d.get('entries',d)))" 2>/dev/null || echo MISSING; done
echo '--- M9XJ-WEIGHTS-STILL-THERE ---'
ls -l --time-style=+%F_%T /home/arm/junyang/carbon/data/weights/cnet2_m9xj*.npz 2>/dev/null
echo '--- HEAD-VS-REMOTE ---'
cd /home/arm/junyang/carbon-repo 2>/dev/null && git rev-parse --abbrev-ref HEAD && git rev-parse HEAD && git log -3 --format='%h %ci %s' && timeout 60 git ls-remote origin cnet2-search-v12 2>&1 | head -2
echo "DGXSTATUS2128R7T3 DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"

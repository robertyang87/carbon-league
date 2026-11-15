#!/bin/bash
# DGX status probe 2158: it320 freeze/build/train trio verification.
# War room 21:49: freeze 3 conditions met 21:47 (91 games, list b84832a2,
# corpus 91 diff 0, Echo2 6 games all won); owner 21:44 ordered pipeline start;
# new DGX chain agent took over (check old poll PID 264410 first, then freeze/build/train).
# Branch rule changed 21:51: main only. Read-only. Marker: DGXSTATUS2158Q4M9
MARK=DGXSTATUS2158Q4M9
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout -k 30 240 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
echo "DGXSTATUS2158Q4M9 DGX-BEGIN $(date '+%F %T %Z')"
hostname; uptime
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader
echo '--- TRAIN-PROCS ---'
ps -eo pid,etime,args | grep -E 'cnet2_train|ladder_corpus|set_uw|replica_chain|build_top|build_m9xj' | grep -v grep || echo NO-TRAIN-PROC
echo '--- OLD-POLL-PID-264410 ---'
ps -p 264410 -o pid,etime,args 2>/dev/null || echo PID-264410-NOT-RUNNING
echo '--- ALL-PYTHON-PROCS ---'
ps -eo pid,etime,args | grep -E 'python3|python ' | grep -v grep | head -20 || echo NO-PY-PROC
echo '--- GIT-RESIDUE ---'
ps -eo pid,etime,args | grep -E 'git (fetch|pull|checkout)' | grep -v grep || echo NO-GIT-PROC
echo '--- DATA-DIR-LISTING ---'
ls -lt --time-style=+%F_%T /home/arm/junyang/carbon/data/ | head -30
echo '--- IT320-DIRS ---'
ls -ld /home/arm/junyang/carbon/data/*m9xj320* /home/arm/junyang/carbon/data/*320* /home/arm/junyang/carbon/data/ds_m9xj320* 2>/dev/null || echo NO-M9XJ320-DIRS
echo '--- NEWEST-WEIGHTS ---'
ls -lt --time-style=+%F_%T /home/arm/junyang/carbon/data/weights/*.npz 2>/dev/null | head -10
echo '--- CHAIN-STATES ---'
ls -lt --time-style=+%F_%T /home/arm/junyang/carbon/data/*.state 2>/dev/null | head -8
for s in /home/arm/junyang/carbon/data/*.state; do echo "== $s"; tail -3 "$s" 2>/dev/null; done
echo '--- RECENT-FILES-LAST-30MIN ---'
find /home/arm/junyang/carbon/data -mmin -30 -type f 2>/dev/null | head -40
echo '--- CHAIN-LOGS ---'
ls -lt --time-style=+%F_%T /home/arm/junyang/carbon/data/m320n_chain_outer.log /home/arm/junyang/carbon/data/build_m9xj320.log /home/arm/junyang/carbon/data/replica_chain_m9xj320.log 2>/dev/null
echo '--- M320N-OUTER-TAIL ---'
tail -15 /home/arm/junyang/carbon/data/m320n_chain_outer.log 2>/dev/null || echo NO-OUTER-LOG
echo '--- BUILD-LOG-TAIL ---'
tail -10 /home/arm/junyang/carbon/data/build_m9xj320.log 2>/dev/null || echo NO-BUILD-LOG
echo '--- REPLICA-CHAIN-LOG-TAIL ---'
tail -10 /home/arm/junyang/carbon/data/replica_chain_m9xj320.log 2>/dev/null || echo NO-REPLICA-LOG
echo '--- DS-META ---'
for d in ds_m6 ds_m7bs_v2 ds_k4v2 ds_m3s2 ds_m9xj ds_m9xj_w ds_carago ds_fable ds_m9xj320 ds_m9xj320_w; do python3 -c "import json;dd=json.load(open('/home/arm/junyang/carbon/data/$d/meta.json'));print('$d games=',dd.get('n_games',len(dd.get('games',[]))),'samples=',dd.get('n_samples',dd.get('samples')))" 2>&1 | head -1; done
echo '--- READY-MARKERS ---'
ls -l --time-style=+%F_%T /home/arm/junyang/carbon/data/ds_m9xj320_w/READY /home/arm/junyang/carbon/data/ds_m9xj320/READY 2>/dev/null || echo NO-M9XJ320-READY
echo '--- CORPUS-COUNTS ---'
cd /home/arm/junyang/carbon/data 2>/dev/null
for d in m3s2_corpus carago_corpus fable_corpus k4v2_corpus m6_corpus m9xj_corpus m9xj320_corpus; do [ -d "$d" ] || continue; echo -n "$d replays-dir="; ls "$d/replays" 2>/dev/null | wc -l; echo -n "$d top-gz="; ls "$d"/*.json.gz 2>/dev/null | wc -l; done
for f in m3s2_corpus/index.json carago_corpus/index.json fable_corpus/index.json k4v2_corpus/index.json m6_corpus/index.json m9xj_corpus/index.json m9xj320_corpus/index.json; do [ -e "$f" ] || continue; echo -n "$f: "; python3 -c "import json;d=json.load(open('$f'));print(len(d) if isinstance(d,list) else len(d.get('entries',d)))" 2>/dev/null || echo MISSING; done
echo '--- M9XJ320-CORPUS-ECHO2 ---'
python3 -c "
import json
d=json.load(open('/home/arm/junyang/carbon/data/m9xj320_corpus/index.json'))
e=d if isinstance(d,list) else d.get('entries',d)
print('m9xj320_corpus index entries=',len(e))
def is_echo2(x):
    for k in ('opponent','opponent_id','name_a','name_b','player_a','player_b','a','b'):
        if 'echo2' in str(x.get(k,'')).lower(): return True
    return False
n=[x for x in e if is_echo2(x)]
print('vs Echo2 games=',len(n))
for x in n:
    print(' ', x.get('name_a'), 'vs', x.get('name_b'), x.get('score_a'), '-', x.get('score_b'), x.get('result',''))
" 2>&1 || echo NO-M9XJ320-CORPUS
echo '--- HEAD-VS-REMOTE-MAIN ---'
cd /home/arm/junyang/carbon-repo 2>/dev/null && git rev-parse --abbrev-ref HEAD && git rev-parse HEAD && git log -3 --format='%h %ci %s' && TIP=$(timeout 60 git ls-remote origin main | cut -f1) && echo "remote main tip=$TIP" && git rev-list --count HEAD..$TIP 2>&1 | head -1
echo '--- TRAIN-LOGS-TAIL ---'
ls -lt --time-style=+%F_%T /home/arm/junyang/carbon/data/logs/ 2>/dev/null | head -10
echo "DGXSTATUS2158Q4M9 DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"

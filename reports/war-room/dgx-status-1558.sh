#!/bin/bash
# DGX status probe 14 (scheduled 15:58 run): stage1-fail watch + stage2 watch + silent-corruption blind-spot checks.
# War room latest (read 15:58): stage1 ran 15:51 and FAILED fast (both build_top rc=1, no training launched).
# Suspect: index strategies flat vs nested shape; true traceback in build_carago.log head, war room reading it.
# Stage2 fix script (nested-shape index rebuild + rebuild ds + launch Carago dual-arm) is ready, dispatch pending
# on log confirmation. Last direct read 15:46:13: GPU 0%/38C, no training; Fable landed (182), Carago replays 0.
# NEW per 2026-11-12 owner mechanism rule: git fetch/pull/checkout leftover check (>5 min alive = anomaly)
# + DGX repo HEAD vs remote cnet2-search-v12 latest commit (git ls-remote, timeout-bounded).
# This probe is READ-ONLY: process + GPU + weight/log mtime trio, md5 reads only.
# Marker: DGXSTATUS1558Q9M4 (claim only output containing this marker)
MARK=DGXSTATUS1558Q9M4
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
echo '--- L20-WEIGHTS-MD5 ---'
md5sum /root/carbon/data/weights/cnet2_m3s2r1L.npz /root/carbon/data/weights/cnet2_m7bsr1L.npz /root/carbon/data/weights/cnet2_m7bsg1L.npz 2>&1
echo '--- DGX (via ssh) ---'
ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
echo "DGXSTATUS1558Q9M4 DGX-BEGIN $(date '+%F %T %Z')"
hostname; uptime
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader
echo '--- PROC-TRAIN ---'
ps -eo pid,etime,args | grep -E 'cnet2_train|m7bs|train|carago|fable' | grep -v grep || echo NO-TRAIN-PROC
echo '--- ALL-PYTHON ---'
ps -eo pid,etime,args | grep -E 'python' | grep -v grep | head -10 || echo NO-PYTHON-PROC
echo '--- PROBE-LEFTOVER ---'
ps -eo pid,etime,args | grep -E 'probe_n1r|bigeval|r1fable|stage1|stage2' | grep -v grep || echo NO-PROBE-PROC
echo '--- GIT-LEFTOVER (any fetch/pull/checkout alive >5min = ANOMALY) ---'
ps -eo pid,etime,args | grep -E 'git (fetch|pull|checkout)' | grep -v grep || echo NO-GIT-PROC
echo '--- REPO-HEAD-VS-REMOTE ---'
cd /home/arm/junyang/carbon 2>/dev/null && git rev-parse HEAD 2>&1 && git log -1 --format='%h %ci %s' 2>&1
echo -n 'remote cnet2-search-v12: '
timeout 60 git ls-remote origin cnet2-search-v12 2>&1 || echo LSREMOTE-FAILED
echo '--- STAGE-LOGS ---'
ls -l --time-style=+%F_%H:%M:%S /home/arm/junyang/carbon/data/dq_logs/stage*.log /home/arm/junyang/carbon/data/stage*.log 2>/dev/null || echo NO-STAGE-LOGS
echo '== build_carago.log HEAD (traceback hunt) =='
head -30 /home/arm/junyang/carbon/data/dq_logs/build_carago.log 2>/dev/null || echo NO-BUILD-CARAGO-LOG
echo '== build_carago.log TAIL =='
tail -6 /home/arm/junyang/carbon/data/dq_logs/build_carago.log 2>/dev/null
echo '== build_fable.log TAIL =='
tail -6 /home/arm/junyang/carbon/data/dq_logs/build_fable.log 2>/dev/null || echo NO-BUILD-FABLE-LOG
echo '--- DS-DIRS ---'
for d in ds_carago ds_fable; do echo "== $d"; ls -l --time-style=+%F_%H:%M:%S /home/arm/junyang/carbon/data/$d/meta.json 2>/dev/null || echo NO-META; du -sh /home/arm/junyang/carbon/data/$d 2>/dev/null; done
echo '--- CARAGO-TRAIN-LOGS ---'
for f in /home/arm/junyang/carbon/data/dq_logs/caragor1.log /home/arm/junyang/carbon/data/dq_logs/caragog1.log; do echo "== $f"; ls -l --time-style=+%F_%H:%M:%S "$f" 2>/dev/null || echo MISSING; tail -3 "$f" 2>/dev/null; done
echo '--- M7BS-LOGS-TAIL ---'
for f in /home/arm/junyang/carbon/data/dq_logs/m7bsr1.log /home/arm/junyang/carbon/data/dq_logs/m7bsg1.log; do echo "== $f"; ls -l --time-style=+%F_%H:%M:%S "$f" 2>/dev/null; tail -2 "$f" 2>/dev/null; done
echo '--- WEIGHTS-NEWEST-ALL ---'
ls -lt /home/arm/junyang/carbon/data/weights/ 2>/dev/null | head -8
md5sum /home/arm/junyang/carbon/data/weights/cnet2_distill_r1_fromstudio.npz /home/arm/junyang/carbon/data/weights/cnet2_distill_g1_fromstudio.npz 2>&1
echo '--- DQ-LOGS-NEWEST ---'
ls -lt /home/arm/junyang/carbon/data/dq_logs/ 2>/dev/null | head -8
echo '--- CARAGO-FABLE-CORPUS ---'
for d in carago_corpus fable_corpus; do
  echo "== $d"; ls -l --time-style=+%F_%H:%M:%S /home/arm/junyang/carbon/data/${d}_matches.json 2>/dev/null
  echo -n 'replays='; ls /home/arm/junyang/carbon/data/$d/replays/ 2>/dev/null | wc -l
  python3 -c "import json;d=json.load(open('/home/arm/junyang/carbon/data/$d/index.json'));print('index_entries=',len(d));print('first_strategies=',json.dumps(d[0].get('strategies'),ensure_ascii=False)[:200])" 2>&1
done
echo '--- REPLAYS-M3S2 ---'
python3 -c "import json;d=json.load(open('/home/arm/junyang/carbon/data/m3s2_corpus/index.json'));print('index_entries=',len(d if isinstance(d,list) else d.get('entries') or d))" 2>&1
echo -n 'replays_dir_count='
ls /home/arm/junyang/carbon/data/m3s2_corpus/replays/ 2>/dev/null | wc -l
echo '--- RECENTLY-MODIFIED-UNDER-DATA (30min) ---'
find /home/arm/junyang/carbon/data -mmin -30 -type f 2>/dev/null | head -20; echo "recent-find-done"
echo "DGXSTATUS1558Q9M4 DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"

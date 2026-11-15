#!/bin/bash
# DGX status probe 17 (scheduled 16:43 run): Carago dual-arm progress watch.
# Last direct read 16:29:09-16:29:27: Carago dual-arm running (PIDs 125877/125878),
# GPU 95%/47C, r1 ep8 (ep7 val all 0.9043), g1 ep15 (ep14 val all 0.9092).
# Epoch pace measured: r1 ~3-4 min/ep, g1 ~1.5-2 min/ep; est finish g1 ~16:37-16:40, r1 ~17:05-17:17.
# This run: check whether g1 finished (est window just passed), r1 progress,
# HEAD vs remote at carbon-repo (local 476b7ef vs remote 62f8db7, count unquantified last run),
# Fable pull side-check (sshd sessions ~16:23/16:26 last run).
# READ-ONLY: process + GPU + weight/log mtime trio, md5 reads only.
# Marker: DGXSTATUS1643Q8W2 (claim only output containing this marker)
MARK=DGXSTATUS1643Q8W2
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
echo '--- L20-WEIGHTS-MD5 ---'
md5sum /root/carbon/data/weights/cnet2_m3s2r1L.npz /root/carbon/data/weights/cnet2_m7bsr1L.npz /root/carbon/data/weights/cnet2_m7bsg1L.npz 2>&1
echo '--- DGX (via ssh) ---'
ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
echo "DGXSTATUS1643Q8W2 DGX-BEGIN $(date '+%F %T %Z')"
hostname; uptime
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader
echo '--- PROC-TRAIN ---'
ps -eo pid,etime,args | grep -E 'cnet2_train|m7bs|train|carago|fable' | grep -v grep || echo NO-TRAIN-PROC
echo '--- ALL-PYTHON ---'
ps -eo pid,etime,args | grep -E 'python' | grep -v grep | head -10 || echo NO-PYTHON-PROC
echo '--- TRAIN-PROC-CWD ---'
for p in $(pgrep -f cnet2_train); do echo "pid $p cwd=$(readlink /proc/$p/cwd 2>/dev/null)"; done
echo '--- PROBE-LEFTOVER ---'
ps -eo pid,etime,args | grep -E 'probe_n1r|bigeval|r1fable|stage1|stage2' | grep -v grep || echo NO-PROBE-PROC
echo '--- GIT-LEFTOVER (any fetch/pull/checkout alive >5min = ANOMALY) ---'
ps -eo pid,etime,args | grep -E 'git (fetch|pull|checkout)' | grep -v grep || echo NO-GIT-PROC
echo '--- FABLE-PULL-SIDE (scp/rsync/sshd serving L20 pull?) ---'
ps -eo pid,etime,args | grep -E 'scp|rsync|sshd: arm' | grep -v grep | head -5 || echo NO-SCP-RSYNC-PROC
echo '--- REPO-HEAD-VS-REMOTE (carbon-repo, branch cnet2-search-v12) ---'
if [ -d /home/arm/junyang/carbon-repo/.git ]; then
  (cd /home/arm/junyang/carbon-repo && git rev-parse HEAD 2>&1 && git log -1 --format='%h %ci %s' 2>&1 && git branch --show-current 2>&1 && echo -n 'remote cnet2-search-v12: ' && timeout 60 git ls-remote origin cnet2-search-v12 2>&1 || echo LSREMOTE-FAILED)
else echo NO-CARBON-REPO-GIT; fi
echo '--- STAGE-LOGS ---'
ls -l --time-style=+%F_%H:%M:%S /home/arm/junyang/carbon/data/dq_logs/stage*.log /home/arm/junyang/carbon/data/stage*.log 2>/dev/null || echo NO-STAGE-LOGS
echo '== build_carago.log TAIL =='
tail -4 /home/arm/junyang/carbon/data/dq_logs/build_carago.log 2>/dev/null
echo '== build_fable.log TAIL =='
tail -4 /home/arm/junyang/carbon/data/dq_logs/build_fable.log 2>/dev/null || echo NO-BUILD-FABLE-LOG
echo '--- DS-DIRS ---'
for d in ds_carago ds_fable; do echo "== $d"; ls -l --time-style=+%F_%H:%M:%S /home/arm/junyang/carbon/data/$d/meta.json 2>/dev/null || echo NO-META; du -sh /home/arm/junyang/carbon/data/$d 2>/dev/null; done
echo '--- CARAGO-TRAIN-LOGS ---'
for f in /home/arm/junyang/carbon/data/dq_logs/caragor1.log /home/arm/junyang/carbon/data/dq_logs/caragog1.log; do echo "== $f"; ls -l --time-style=+%F_%H:%M:%S "$f" 2>/dev/null || echo MISSING; tail -8 "$f" 2>/dev/null; grep -i 'best\|done\|finish\|complete' "$f" 2>/dev/null | tail -3; done
echo '--- M7BS-LOGS-TAIL ---'
for f in /home/arm/junyang/carbon/data/dq_logs/m7bsr1.log /home/arm/junyang/carbon/data/dq_logs/m7bsg1.log; do echo "== $f"; ls -l --time-style=+%F_%H:%M:%S "$f" 2>/dev/null; tail -2 "$f" 2>/dev/null; done
echo '--- WEIGHTS-NEWEST-ALL ---'
ls -lt /home/arm/junyang/carbon/data/weights/ 2>/dev/null | head -14
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
echo "DGXSTATUS1643Q8W2 DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"

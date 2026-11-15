#!/bin/bash
# DGX status probe 18 (scheduled 16:58 run): Carago r1 finish check + M7Bs v2接档 watch.
# Last direct read 16:45:20-16:45:33: Carago g1 DONE 20/20 best val 0.9101 (ep18);
# r1 solo ep16 (ep15 val all 0.9083), pace ~1.5 min/ep, est finish ~16:51-16:53.
# War room (16:47 sync): next job M7Bs v2 重训 dispatched (142局刷新语料重建 ds_m7bs_v2, 双臂原配方重训),
# starts when DGX frees up (after r1). Fable dual-arm started on Studio 16:52 (not DGX).
# This run: check r1 DONE?, M7Bs v2 stage1/stage2 or training started?, HEAD vs remote,
# Fable回传权重是否落地 (cnet2_fable_*_fromstudio.npz), git leftover, GPU.
# READ-ONLY: process + GPU + weight/log mtime trio, md5 reads only.
# Marker: DGXSTATUS1658K4M9 (claim only output containing this marker)
MARK=DGXSTATUS1658K4M9
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
echo '--- L20-WEIGHTS-MD5 ---'
md5sum /root/carbon/data/weights/cnet2_m3s2r1L.npz /root/carbon/data/weights/cnet2_m7bsr1L.npz /root/carbon/data/weights/cnet2_m7bsg1L.npz 2>&1
echo '--- DGX (via ssh) ---'
ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
echo "DGXSTATUS1658K4M9 DGX-BEGIN $(date '+%F %T %Z')"
hostname; uptime
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader
echo '--- PROC-TRAIN ---'
ps -eo pid,etime,args | grep -E 'cnet2_train|m7bs|train|carago|fable' | grep -v grep || echo NO-TRAIN-PROC
echo '--- ALL-PYTHON ---'
ps -eo pid,etime,args | grep -E 'python' | grep -v grep | head -10 || echo NO-PYTHON-PROC
echo '--- TRAIN-PROC-CWD ---'
for p in $(pgrep -f cnet2_train); do echo "pid $p cwd=$(readlink /proc/$p/cwd 2>/dev/null)"; done
echo '--- PROBE-LEFTOVER ---'
ps -eo pid,etime,args | grep -E 'probe_n1r|bigeval|r1fable|stage1|stage2|build_' | grep -v grep || echo NO-PROBE-PROC
echo '--- GIT-LEFTOVER (any fetch/pull/checkout alive >5min = ANOMALY) ---'
ps -eo pid,etime,args | grep -E 'git (fetch|pull|checkout)' | grep -v grep || echo NO-GIT-PROC
echo '--- SCP-RSYNC-SSHD ---'
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
echo '== build_m7bs_v2.log TAIL =='
tail -6 /home/arm/junyang/carbon/data/dq_logs/build_m7bs_v2.log 2>/dev/null || echo NO-BUILD-M7BSV2-LOG
echo '--- DS-DIRS ---'
for d in ds_carago ds_fable ds_m7bs ds_m7bs_v2; do echo "== $d"; ls -l --time-style=+%F_%H:%M:%S /home/arm/junyang/carbon/data/$d/meta.json 2>/dev/null || echo NO-META; du -sh /home/arm/junyang/carbon/data/$d 2>/dev/null; python3 -c "import json;m=json.load(open('/home/arm/junyang/carbon/data/$d/meta.json'));print('meta:',{k:m.get(k) for k in ('games','samples','total','rows') if k in m})" 2>/dev/null; done
echo '--- CARAGO-TRAIN-LOGS ---'
for f in /home/arm/junyang/carbon/data/dq_logs/caragor1.log /home/arm/junyang/carbon/data/dq_logs/caragog1.log; do echo "== $f"; ls -l --time-style=+%F_%H:%M:%S "$f" 2>/dev/null || echo MISSING; tail -8 "$f" 2>/dev/null; grep -i 'best\|done\|finish\|complete' "$f" 2>/dev/null | tail -3; done
echo '--- M7BS-LOGS-TAIL ---'
for f in /home/arm/junyang/carbon/data/dq_logs/m7bsr1.log /home/arm/junyang/carbon/data/dq_logs/m7bsg1.log /home/arm/junyang/carbon/data/dq_logs/m7bsv2r1.log /home/arm/junyang/carbon/data/dq_logs/m7bsv2g1.log; do echo "== $f"; ls -l --time-style=+%F_%H:%M:%S "$f" 2>/dev/null || echo MISSING; tail -3 "$f" 2>/dev/null; done
echo '--- WEIGHTS-NEWEST-ALL ---'
ls -lt /home/arm/junyang/carbon/data/weights/ 2>/dev/null | head -16
md5sum /home/arm/junyang/carbon/data/weights/cnet2_distill_r1_fromstudio.npz /home/arm/junyang/carbon/data/weights/cnet2_distill_g1_fromstudio.npz /home/arm/junyang/carbon/data/weights/cnet2_fable_r1_fromstudio.npz /home/arm/junyang/carbon/data/weights/cnet2_fable_g1_fromstudio.npz 2>&1
echo '--- DQ-LOGS-NEWEST ---'
ls -lt /home/arm/junyang/carbon/data/dq_logs/ 2>/dev/null | head -10
echo '--- CARAGO-FABLE-CORPUS ---'
for d in carago_corpus fable_corpus; do
  echo "== $d"
  echo -n 'replays='; ls /home/arm/junyang/carbon/data/$d/replays/ 2>/dev/null | wc -l
  python3 -c "import json;d=json.load(open('/home/arm/junyang/carbon/data/$d/index.json'));print('index_entries=',len(d))" 2>&1
done
echo '--- REPLAYS-M3S2 ---'
python3 -c "import json;d=json.load(open('/home/arm/junyang/carbon/data/m3s2_corpus/index.json'));print('index_entries=',len(d if isinstance(d,list) else d.get('entries') or d))" 2>&1
echo -n 'replays_dir_count='
ls /home/arm/junyang/carbon/data/m3s2_corpus/replays/ 2>/dev/null | wc -l
echo '--- RECENTLY-MODIFIED-UNDER-DATA (30min) ---'
find /home/arm/junyang/carbon/data -mmin -30 -type f 2>/dev/null | head -20; echo "recent-find-done"
echo "DGXSTATUS1658K4M9 DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"

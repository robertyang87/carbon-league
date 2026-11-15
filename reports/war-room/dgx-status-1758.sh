#!/bin/bash
# DGX status probe 22 (scheduled 17:58 run): M7Bs v2 train-start watch.
# Last direct read 17:47:16 (run 43194502220): ds_m7bs_v2 built 17:31:43
# (142 games / 42,458 samples / val 17); HEAD a44acf4, remote ca410e28 (lag 6,
# data/docs only); NO train proc, NO m7bsv2 logs, GPU 0%/36C idle since 16:51
# (~56 min at 17:47); queue non-empty (M7Bs v2).
# War room since: confirmed ds_m7bs_v2 built; asked exec agent for train-start
# three-piece confirmation (start immediately if not started). Queue after v2:
# M9XJ-it260 / K4v2 first-free. M6 insurance pull closed 17:46 (191/191).
# This run: did m7bsv2 train start (proc+GPU+log mtime trio)? GPU idle time?
# HEAD vs remote now? m6_corpus landed on DGX?
# READ-ONLY: process + GPU + weight/log mtime trio, md5 reads only.
# Marker: DGXSTATUS1758M7V2 (claim only output containing this marker)
MARK=DGXSTATUS1758M7V2
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
echo '--- L20-WEIGHTS-MD5 ---'
md5sum /root/carbon/data/weights/cnet2_m3s2r1L.npz /root/carbon/data/weights/cnet2_m7bsr1L.npz /root/carbon/data/weights/cnet2_m7bsg1L.npz 2>&1
echo '--- DGX (via ssh) ---'
ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
echo "DGXSTATUS1758M7V2 DGX-BEGIN $(date '+%F %T %Z')"
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
echo '--- M7BS-CORPUS ---'
echo -n 'm7bs replays='; ls /home/arm/junyang/carbon/data/m7bs_corpus/replays/ 2>/dev/null | wc -l
ls -l --time-style=+%F_%H:%M:%S /home/arm/junyang/carbon/data/m7bs_corpus/m7bs_matches.json /home/arm/junyang/carbon/data/m7bs_corpus/index.json 2>/dev/null || echo NO-M7BS-MATCHES-OR-INDEX
python3 -c "import json;d=json.load(open('/home/arm/junyang/carbon/data/m7bs_corpus/index.json'));print('m7bs index_entries=',len(d if isinstance(d,list) else d.get('entries') or d))" 2>&1
echo '--- M6-CORPUS (scout closed 17:46, 191/191) ---'
echo -n 'm6 replays='; ls /home/arm/junyang/carbon/data/m6_corpus/replays/ 2>/dev/null | wc -l
ls -l --time-style=+%F_%H:%M:%S /home/arm/junyang/carbon/data/m6_corpus/ 2>/dev/null | head -5
echo '--- DS-DIRS ---'
for d in ds_carago ds_fable ds_m7bs ds_m7bs_v2; do echo "== $d"; ls -l --time-style=+%F_%H:%M:%S /home/arm/junyang/carbon/data/$d/meta.json 2>/dev/null || echo NO-META; du -sh /home/arm/junyang/carbon/data/$d 2>/dev/null; done
echo '--- M7BSV2-TRAIN-LOGS ---'
for f in /home/arm/junyang/carbon/data/dq_logs/m7bsv2r1.log /home/arm/junyang/carbon/data/dq_logs/m7bsv2g1.log /home/arm/junyang/carbon/data/dq_logs/m7bsv2.log; do echo "== $f"; ls -l --time-style=+%F_%H:%M:%S "$f" 2>/dev/null || echo MISSING; tail -5 "$f" 2>/dev/null; done
echo '--- CARAGO/M7BS-OLD-LOGS ---'
for f in /home/arm/junyang/carbon/data/dq_logs/caragor1.log /home/arm/junyang/carbon/data/dq_logs/m7bsr1.log; do echo "== $f"; ls -l --time-style=+%F_%H:%M:%S "$f" 2>/dev/null || echo MISSING; tail -2 "$f" 2>/dev/null; done
echo '--- WEIGHTS-NEWEST-ALL ---'
ls -lt /home/arm/junyang/carbon/data/weights/ 2>/dev/null | head -16
md5sum /home/arm/junyang/carbon/data/weights/cnet2_fable_r1_fromstudio.npz /home/arm/junyang/carbon/data/weights/cnet2_fable_g1_fromstudio.npz 2>&1
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
echo "DGXSTATUS1758M7V2 DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"

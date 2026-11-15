#!/bin/bash
# DGX status probe 0258: 02:58 periodic report. Read-only. Marker: DGXSTATUS0258A
# Focus: Echo3 pack on s26stage (verify md5/size/mtime independently),
# deployment A (ff to 639253c0 + new harness), GPU/training state, git residue,
# HEAD vs remote, corpus counts.
MARK=DGXSTATUS0258A
echo "$MARK BEGIN $(date -u +%H:%M:%S)"
timeout -k 30 240 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
M=DGXSTATUS0258A
echo "$M DGX $(hostname) $(uptime)"
echo '--PROC--'
ps -eo pid,etime,args | grep -E 'cnet2_train|rl2_ppo|play_batch|bigeval|train' | grep -v grep | head -5 || echo NOTRAIN
echo '--PROCBROAD--'
ps -eo pid,etime,args | grep -E 'python|smoke' | grep -v grep | head -8 || echo NOPROC
echo '--GPU--'
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader
echo '--GITRES--'
ps -eo pid,etime,args | grep -E 'git (fetch|pull|checkout)' | grep -v grep || echo NOGIT
echo '--HEAD--'
cd /home/arm/junyang/carbon-repo && echo "HEAD $(timeout 60 git rev-parse --short HEAD) $(timeout 60 git rev-parse --abbrev-ref HEAD) $(timeout 60 git rev-parse HEAD)" && R=$(timeout 60 git ls-remote origin main | cut -f1) && echo "REMOTE ${R:0:8}"
echo '--DEPLOY-A--'
cd /home/arm/junyang/carbon-repo && timeout 60 git log --oneline -3 2>/dev/null; timeout 60 git merge-base --is-ancestor 639253c0 HEAD && echo "639253c0 IS ANCESTOR" || echo "639253c0 NOT ANCESTOR"
echo '--S26STAGE--'
ls -lt --time-style=+%m%d_%H:%M /home/arm/junyang/s26stage/ 2>&1 | head -15
echo '== echo3 pack:'
ls -l --time-style=+%m%d_%H:%M /home/arm/junyang/s26stage/echo3_int8.py 2>&1
md5sum /home/arm/junyang/s26stage/echo3_int8.py 2>&1
echo '== echo3 files anywhere:'
timeout 60 find /home/arm/junyang -iname '*echo3*' -type f 2>/dev/null | head -10
echo '--SMOKE--'
ls -lt --time-style=+%m%d_%H:%M /home/arm/junyang/carbon/data/smoke_pb_final/ 2>/dev/null | head -6
echo '--WNEW--'
ls -lt --time-style=+%m%d_%H:%M /home/arm/junyang/carbon/data/weights/*.npz 2>/dev/null | head -4
echo '--CORPUS--'
cd /home/arm/junyang/carbon && for c in m3s2_corpus carago_corpus fable_corpus; do echo -n "$c idx="; python3 -c "import json;print(len(json.load(open('data/$c/index.json'))))" 2>/dev/null || echo -n '?'; echo -n ' replays='; ls data/$c/replays 2>/dev/null | wc -l; done
echo '--DSMETA--'
for d in ds_carago ds_fable; do echo -n "$d "; python3 -c "import json;m=json.load(open('data/$d/meta.json'));print('games='+str(len(m.get('games',[])))+' total='+str(m.get('total')))" 2>/dev/null || echo '?'; done
echo '--RECENT30M--'
find /home/arm/junyang/carbon/data /home/arm/junyang/s26stage -mmin -30 -type f 2>/dev/null | head -10
echo '--LOAD--'
cat /proc/loadavg
echo "$M END"
DGXEOF
echo "$MARK END rc=$?"

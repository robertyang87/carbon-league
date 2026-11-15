#!/bin/bash
# DGX status probe 0019: compact output + tolerant claim. Read-only. Marker: DGXSTATUS0019A
MARK=DGXSTATUS0019A
echo "$MARK BEGIN $(date -u +%H:%M:%S)"
timeout -k 30 240 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
M=DGXSTATUS0019A
echo "$M DGX $(hostname) $(uptime)"
echo '--PROC--'
ps -eo pid,etime,args | grep -E 'cnet2_train|rl2_ppo|play_batch|bigeval|train' | grep -v grep | head -5 || echo NOTRAIN
echo '--PROCBROAD--'
ps -eo pid,etime,args | grep -E 'python|smoke' | grep -v grep | head -8 || echo NOPROC
echo '--SMOKE--'
ls -lt --time-style=+%m%d_%H:%M /home/arm/junyang/carbon/data/smoke_pb_final/ 2>/dev/null | head -8
for f in /home/arm/junyang/carbon/data/smoke_pb_final/*.json; do echo "== $f"; cat "$f" 2>/dev/null; echo; done
echo '--GPU--'
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader
echo '--GITRES--'
ps -eo pid,etime,args | grep -E 'git (fetch|pull|checkout)' | grep -v grep || echo NOGIT
echo '--HEAD--'
cd /home/arm/junyang/carbon-repo && echo "HEAD $(timeout 60 git rev-parse --short HEAD) $(timeout 60 git rev-parse --abbrev-ref HEAD)" && R=$(timeout 60 git ls-remote origin main | cut -f1) && echo "REMOTE ${R:0:8} BEHIND $(timeout 60 git rev-list --count HEAD..$R 2>/dev/null)"
echo '--WNEW--'
ls -lt --time-style=+%m%d_%H:%M /home/arm/junyang/carbon/data/weights/*.npz 2>/dev/null | head -4
echo '--CORPUS--'
cd /home/arm/junyang/carbon && for c in m3s2_corpus carago_corpus fable_corpus; do echo -n "$c idx="; python3 -c "import json;print(len(json.load(open('data/$c/index.json'))))" 2>/dev/null || echo -n '?'; echo -n ' replays='; ls data/$c/replays 2>/dev/null | wc -l; done
echo '--DSMETA--'
for d in ds_carago ds_fable; do echo -n "$d "; python3 -c "import json;m=json.load(open('data/$d/meta.json'));print(m.get('num_samples') or m.get('n') or '?')" 2>/dev/null || echo '?'; done
echo '--STEP0--'
cd /home/arm/junyang/carbon-repo && echo "HEAD $(timeout 60 git rev-parse HEAD) $(timeout 60 git rev-parse --abbrev-ref HEAD)"
md5sum /home/arm/junyang/carbon/code/rl2_ppo_rollout.py /home/arm/junyang/carbon/code/eval_v53.py /home/arm/junyang/carbon/code/cnet2_search.py /home/arm/junyang/carbon/code/cnet2_np.py 2>&1
md5sum /home/arm/junyang/carbon/data/weights/cnet2_b128m2t3.npz 2>&1
echo '--RECENT30M--'
find /home/arm/junyang/carbon/data -mmin -30 -type f 2>/dev/null | head -5
echo '--LOAD--'
cat /proc/loadavg
echo "$M END"
DGXEOF
echo "$MARK END rc=$?"

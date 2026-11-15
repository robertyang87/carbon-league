#!/bin/bash
# DGX status probe 0010: compact output (log-truncation workaround) + tolerant claim.
# Read-only. Marker: DGXSTATUS0010A
MARK=DGXSTATUS0010A
echo "$MARK BEGIN $(date -u +%H:%M:%S)"
timeout -k 30 240 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
M=DGXSTATUS0010A
echo "$M DGX $(hostname) $(uptime)"
echo '--PROC--'
ps -eo pid,etime,args | grep -E 'cnet2_train|rl2_ppo|play_batch|bigeval|selfplay_gen|gen1r_chain|replay' | grep -v grep | head -10 || echo NOTRAIN
echo '--GPU--'
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader
echo '--GITRES--'
ps -eo pid,etime,args | grep -E 'git (fetch|pull|checkout)' | grep -v grep || echo NOGIT
echo '--HEAD--'
cd /home/arm/junyang/carbon-repo && echo "HEAD $(timeout 60 git rev-parse --short HEAD) $(timeout 60 git rev-parse --abbrev-ref HEAD)" && R=$(timeout 60 git ls-remote origin main | cut -f1) && echo "REMOTE ${R:0:8} BEHIND $(timeout 60 git rev-list --count HEAD..$R 2>/dev/null || echo NA-nolocalobj)"
echo '--CODEMD5--'
cd /home/arm/junyang/carbon-repo && for f in code/cnet2_build.py code/selfplay_gen.py code/stage0_teacher.py code/stage1_teacher_gate.py; do echo -n "$f "; timeout 30 md5sum $f 2>/dev/null | cut -c1-8; done
echo '--GEN1R--'
cd /home/arm/junyang/carbon
for s in shard0 shard1; do echo -n "gen1r_$s idx="; python3 -c "import json;print(len(json.load(open('data/gen1r_$s/index.json'))))" 2>/dev/null || echo -n '?'; echo -n ' replays='; ls data/gen1r_$s/replays/*.json.gz 2>/dev/null | wc -l; done
for s in shard0 shard1; do echo -n "ds_gen1r_$s meta="; python3 -c "import json;m=json.load(open('data/ds_gen1r_$s/meta.json'));print({k:m.get(k) for k in ('num_samples','mirror_gate','mirror_share') if k in m})" 2>/dev/null | head -c 300 || echo -n '?'; echo; done
echo "chain.log tail:"; tail -4 data/dq_logs/gen1r_chain.log 2>&1
echo "build_shard1.log tail:"; tail -6 data/dq_logs/gen1r_build_shard1.log 2>&1
echo '--REPO-CORPUS--'
cd /home/arm/junyang/carbon-repo && for c in k4v3_corpus fable6_corpus m3s3z_corpus hakimi31_corpus hakimi3_corpus krypton_corpus; do echo -n "$c="; ls data/$c 2>/dev/null | wc -l; done; echo
echo '--CORPUS--'
cd /home/arm/junyang/carbon && for c in m3s2_corpus carago_corpus fable_corpus; do echo -n "$c idx="; python3 -c "import json;print(len(json.load(open('data/$c/index.json'))))" 2>/dev/null || echo -n '?'; echo -n ' replays='; ls data/$c/replays 2>/dev/null | wc -l; done
echo '--DSMETA--'
for d in ds_carago ds_fable; do echo -n "$d "; python3 -c "import json;m=json.load(open('data/$d/meta.json'));print(m.get('num_samples') or m.get('n') or '?')" 2>/dev/null || echo '?'; done
echo '--RECENT30M--'
find /home/arm/junyang/carbon/data /home/arm/junyang/carbon-repo/data -mmin -30 -type f 2>/dev/null | head -10
echo '--DISK--'
df -h / | tail -1
echo '--LOAD--'
cat /proc/loadavg
echo "$M END"
DGXEOF
echo "$MARK END rc=$?"

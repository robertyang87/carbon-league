#!/bin/bash
# DGX status probe 0012 (read-only). Marker DGXSTATUS0012A.
MARK=DGXSTATUS0012A
W=/home/arm/junyang/carbon
R=/home/arm/junyang/carbon-repo
echo "$MARK BEGIN $(date '+%H:%M:%S')"
hostname; uptime
echo "--PROC--"
ps -eo pid,etime,args | grep -E 'cnet2|rl2_ppo|selfplay_gen|gen1r|build_top|train|python3' | grep -v grep || echo NOPROC
echo "--GPU--"
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader 2>&1 || echo NOGPU
echo "--GITRES--"
ps -eo pid,etime,args | grep -E 'git (fetch|pull|checkout|merge)' | grep -v grep || echo NOGIT
echo "--HEAD--"
cd "$R" 2>/dev/null && git rev-parse HEAD && git rev-parse --abbrev-ref HEAD
timeout 30 git ls-remote origin HEAD 2>&1 | head -2
echo "--CODEMD5--"
cd "$R/code" 2>/dev/null && md5sum cnet2_build.py producer/selfplay_gen.py selfplay_gen.py stage0_teacher.py stage1_teacher_gate.py rl2_ppo_train.py 2>&1 | awk '{print $2, substr($1,1,8)}'
echo "--FASTSIM--"
find /home/arm/junyang -name fastsim.py -not -path '*/node_modules/*' 2>/dev/null | while read f; do echo "$f $(md5sum "$f" | cut -c1-8) $(stat -c%s "$f")"; done
echo "--GEN1R--"
cd "$W" 2>/dev/null
for s in shard0 shard1; do
  echo "gen1r_$s idx=$(python3 -c "import json;print(len(json.load(open('data/gen1r_$s/index.json'))))" 2>&1)"
  echo " replays=$(ls data/gen1r_$s/replays 2>/dev/null | wc -l)"
done
python3 -c "import json;m=json.load(open('data/ds_gen1r_shard0/meta.json'));print('ds_gen1r_shard0 meta=',m.get('mirror_gate'))" 2>&1
echo "ds_gen1r_shard1 meta:"; python3 -c "import json;print(json.load(open('data/ds_gen1r_shard1/meta.json')))" 2>&1 | head -2
ls data/ds_gen1r_shard1 2>&1
echo "chain.log tail:"; tail -6 data/gen1r_chain.log 2>/dev/null || tail -6 data/dq_logs/gen1r_chain.log 2>/dev/null || ls data/*chain* data/dq_logs/*chain* 2>&1
echo "build_shard1.log tail:"; tail -8 data/dq_logs/gen1r_build_shard1.log 2>/dev/null || tail -8 data/gen1r_build_shard1.log 2>/dev/null || echo NOBUILDLOG
echo "compat shard1 log:"; ls -la data/dq_logs/gen1r_build_shard1_compat.log 2>&1; tail -6 data/dq_logs/gen1r_build_shard1_compat.log 2>/dev/null || echo NOCOMPATLOG
echo "compat shard0 logs:"; ls -la data/dq_logs/gen1r_compat_shard0*.log 2>&1 | head -5
echo "dq_logs recent:"; ls -lat data/dq_logs/ 2>/dev/null | head -8
echo "--REPO-CORPUS--"
cd "$R/data" 2>/dev/null && for d in k4v3_corpus fable6_corpus m3s3z_corpus hakimi31_corpus hakimi3_corpus krypton_corpus; do echo "$d=$(ls $d/replays 2>/dev/null | wc -l)"; done
echo "--CORPUS--"
cd "$W" 2>/dev/null
for c in m3s2 carago fable; do
  echo "${c}_corpus idx=$(python3 -c "import json;print(len(json.load(open('data/${c}_corpus/index.json'))))" 2>&1)"
  echo " replays=$(ls data/${c}_corpus/replays 2>/dev/null | wc -l)"
done
echo "--DSMETA--"
python3 -c "
import json
for n in ('ds_carago','ds_fable'):
    try:
        m=json.load(open(f'data/{n}/meta.json'))
        g=m.get('games',[])
        print(n,'games=',len(g),'samples=',sum(x.get('n',0) for x in g))
    except Exception as e:
        print(n,'ERR',e)
" 2>&1
echo "--RECENT30M--"
find data -mmin -30 -type f 2>/dev/null | head -10
echo "--DISK--"
df -h / | tail -1
echo "--LOAD--"
cat /proc/loadavg
echo "$MARK END rc=$?"

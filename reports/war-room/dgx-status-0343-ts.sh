#!/bin/bash
# DGX status probe 0343 (2026-11-14 03:43 CST periodic report). Read-only. Marker DGXSTATUS0343A.
MARK=DGXSTATUS0343A
W=/home/arm/junyang/carbon
R=/home/arm/junyang/carbon-repo
echo "$MARK BEGIN $(date '+%H:%M:%S %Z')"
hostname; uptime
echo "--PROC--"
ps -eo pid,etime,args | grep -E 'cnet2|rl2_ppo|selfplay_gen|gen1r|build_top|train|spawn_main|python3' | grep -v grep || echo NOPROC
echo "--GPU--"
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader 2>&1 || echo NOGPU
echo "--GPU-PROC--"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>&1 || echo NOGPUPROC
echo "--GITRES--"
ps -eo pid,etime,args | grep -E 'git (fetch|pull|checkout|merge)' | grep -v grep || echo NOGIT
echo "--HEAD--"
cd "$R" 2>/dev/null && git rev-parse HEAD && git rev-parse --abbrev-ref HEAD
timeout 60 git ls-remote origin HEAD 2>&1 | head -2
echo "--BIGEVAL--"
md5sum "$R/code/bigeval.py" "$W/code/bigeval.py" 2>&1 | awk '{print $2, substr($1,1,8), $1}' ; ls -l --time-style=+%m%d_%H:%M "$R/code/bigeval.py" "$W/code/bigeval.py" 2>&1
echo "--FASTSIM--"
find /home/arm/junyang -name fastsim.py -not -path '*/node_modules/*' 2>/dev/null | while read f; do echo "$f $(md5sum "$f" | cut -c1-8) $(stat -c%s "$f")"; done
echo "--CODEMD5--"
cd "$R/code" 2>/dev/null && md5sum rl2_ppo_train.py stage1_teacher_gate.py cnet2_build.py 2>&1 | awk '{print $2, substr($1,1,8)}'
echo "--SHARD1--"
ls -l --time-style=+%m%d_%H:%M "$W/data/ds_gen1r_shard1/meta.json" 2>&1
python3 -c "import json;m=json.load(open('$W/data/ds_gen1r_shard1/meta.json'));print('shard1 meta games=',len(m.get('games',[])),'samples=',m.get('total'))" 2>&1
ls -lt --time-style=+%m%d_%H:%M "$W/data/ds_gen1r_shard1/" 2>/dev/null | head -5
echo "--CORPUS-W--"
cd "$W" 2>/dev/null
for c in m3s2 carago fable hakimi3pro; do
  echo -n "${c}_corpus idx="; python3 -c "import json;print(len(json.load(open('data/${c}_corpus/index.json'))))" 2>&1 | tr -d '\n'; echo -n ' replays='; ls data/${c}_corpus/replays 2>/dev/null | wc -l
done
echo "--CORPUS-R--"
cd "$R/data" 2>/dev/null && for d in m3s2_corpus carago_corpus fable_corpus hakimi3pro_corpus; do echo "$d files=$(ls $d 2>/dev/null | wc -l)"; done
echo "--DSMETA--"
cd "$W" 2>/dev/null
for d in ds_carago ds_fable; do echo -n "$d "; ls -l --time-style=+%m%d_%H:%M data/$d/meta.json 2>/dev/null; python3 -c "import json;m=json.load(open('data/$d/meta.json'));print('games='+str(len(m.get('games',[])))+' total='+str(m.get('total')))" 2>/dev/null || echo '?'; done
echo "--WNEW--"
ls -lt --time-style=+%m%d_%H:%M "$W/data/weights/"*.npz 2>/dev/null | head -4
echo "--RECENT30M--"
find "$W/data" /home/arm/junyang/s26stage -mmin -30 -type f 2>/dev/null | head -10
echo "--LOAD--"
cat /proc/loadavg
echo "--DISK--"
df -h / | tail -1
echo "$MARK END rc=$?"

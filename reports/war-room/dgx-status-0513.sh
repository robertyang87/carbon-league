#!/bin/bash
# DGX status probe 0428 — read-only. Runs on L20, ssh to DGX.
echo "DGXSTATUS0513A L20-BEGIN $(date -u +%FT%TZ)"
timeout -k 30 420 ssh -o ConnectTimeout=30 -o BatchMode=yes -i /root/.ssh/dgx_access arm@10.211.43.155 'bash -s' <<'DGXEOF'
echo "DGXSTATUS0513A DGX-BEGIN"
hostname
uptime
date
echo "--- gpu ---"
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader
echo "--- load ---"
cat /proc/loadavg
echo "--- training procs ---"
ps -eo pid,etime,cmd | grep -E "rl2|ppo|train|rollout|build_top|cnet2|selfplay" | grep -v grep || echo NOPROC
echo "--- git residue ---"
ps -eo pid,etime,cmd | grep -E "git (fetch|pull|checkout)" | grep -v grep || echo NOGIT
echo "--- all python procs ---"
ps -eo pid,etime,cmd | grep python3 | grep -v grep | head -20 || echo NOPY
cd /home/arm/junyang/carbon 2>/dev/null || echo "NOWORKDIR"
pwd
echo "--- corpus counts ---"
for c in m3s2_corpus carago_corpus fable_corpus; do
  if [ -d "data/$c" ]; then
    idx=$(python3 -c "import json;print(len(json.load(open('data/$c/index.json'))))" 2>/dev/null || echo ERR)
    rep=$(ls data/$c/replays 2>/dev/null | wc -l)
    echo "$c idx=$idx replays=$rep"
  else
    echo "$c MISSING"
  fi
done
echo "--- ds meta ---"
for d in ds_carago ds_fable; do
  if [ -f "data/$d/meta.json" ]; then
    echo "$d $(stat -c '%y' data/$d/meta.json | cut -c1-16) $(python3 -c "import json;m=json.load(open('data/$d/meta.json'));print('games=',m.get('games'),'total=',m.get('total'))" 2>/dev/null || echo ERR)"
  else
    echo "$d MISSING"
  fi
done
echo "--- hakimi weights ---"
find /home/arm/junyang -name "cnet2_hakimi2*_fromstudio.npz" 2>/dev/null | while read f; do md5sum "$f"; stat -c "%y %s" "$f" | cut -c1-16,21-; done
echo "--- echo3 pack ---"
find /home/arm/junyang -name "echo3_int8.py" 2>/dev/null | while read f; do md5sum "$f"; stat -c "%y %s" "$f" | cut -c1-16,21-; done
echo "--- newest weights ---"
find /home/arm/junyang -name "*.npz" -type f -printf "%T@ %p\n" 2>/dev/null | sort -rn | head -5
echo "--- files changed last 60m ---"
find /home/arm/junyang/carbon /home/arm/junyang/s26stage -type f -mmin -60 2>/dev/null | head -30
echo "--- git HEAD ---"
cd /home/arm/junyang/carbon-repo 2>/dev/null && git rev-parse HEAD && git branch --show-current && timeout 60 git ls-remote origin HEAD || echo "NOGITREPO"
echo "DGXSTATUS0513A DGX-END"
DGXEOF
echo "DGXSTATUS0513A L20-END $(date -u +%FT%TZ)"

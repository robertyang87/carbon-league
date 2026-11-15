#!/bin/bash
# DGX status probe 01:58 CST, marker DGXSTATUS0158A
set -u
M=DGXSTATUS0158A
echo "=== $M start $(date '+%F %T %Z') ==="
hostname; uptime
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv 2>&1 | head -5
echo "--- duel/calibration processes ---"
ps -eo pid,ppid,etime,time,pcpu,comm,args | grep -E 'k0_duel|k0_calib|duel' | grep -v grep | head -30
echo "--- load ---"
cat /proc/loadavg
echo "--- duel outputs ---"
ls -la /home/arm/junyang/carbon/data/k0_calib/ 2>/dev/null | head -20
find /home/arm/junyang/carbon -maxdepth 4 -name '*duel*' -newermt '2026-11-15 01:11' 2>/dev/null | head -20
echo "--- cell1 artifacts ---"
ls -la /home/arm/junyang/carbon/data/k0_calib/cell1* 2>/dev/null | head; find /home/arm/junyang/carbon -maxdepth 5 -name '*.npz' -newermt '2026-11-15 01:00' 2>/dev/null | head
echo "--- git residue ---"
ps -eo pid,etime,args | grep -E 'git (fetch|pull|checkout)' | grep -v grep || echo "no git fetch/pull/checkout procs"
echo "--- repo HEAD ---"
cd /home/arm/junyang/carbon-repo && git rev-parse HEAD && timeout 60 git ls-remote origin HEAD && timeout 60 git ls-remote origin refs/heads/sync/h29
echo "--- corpora workdir ---"
for d in m3s2_corpus carago_corpus fable_corpus; do
  echo "$d: index=$(python3 -c "import json;print(len(json.load(open('/home/arm/junyang/carbon/data/$d/index.json'))))" 2>&1) replays=$(ls /home/arm/junyang/carbon/data/$d/replays 2>/dev/null | wc -l)"
done
for d in ds_carago ds_fable; do
  echo "$d: $(python3 -c "import json;print(json.load(open('/home/arm/junyang/carbon/data/$d/meta.json')))" 2>&1 | head -c 300)"
done
echo "--- clone counts ---"
for d in m3s2_corpus carago_corpus fable_corpus krypton-csplant; do
  echo "$d: $(ls /home/arm/junyang/carbon-repo/data/$d 2>/dev/null | wc -l) files"
done
echo "=== $M end ==="

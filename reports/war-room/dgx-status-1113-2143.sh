#!/bin/bash
# DGX status probe 2143 (read-only). Marker: DGXSTATUS2143
MARK=DGXSTATUS2143
echo "$MARK BEGIN $(date '+%F %T %Z')"
echo '--- A: HOST/GPU/PS ---'
hostname; uptime
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv,noheader
echo '--- TRAIN-PS ---'
ps -eo pid,etime,args | grep -E 'cnet2_train|selfplay_gen|cnet2_build|pack' | grep -v grep || echo NO-TRAIN-PROC
echo '--- ALL-PYTHON ---'
ps -eo pid,etime,args | grep -E 'python3|python ' | grep -v grep | head -20 || echo NO-PY-PROC
echo '--- GIT-RESIDUE ---'
ps -eo pid,etime,args | grep -E 'git (fetch|pull|checkout)' | grep -v grep || echo NO-GIT-PROC
echo '--- B: HEAD-VS-REMOTE (main) ---'
cd /home/arm/junyang/carbon-repo 2>/dev/null && git rev-parse --abbrev-ref HEAD && git rev-parse HEAD && git log -3 --format='%h %ci %s' && timeout 60 git ls-remote origin main 2>&1 | head -2
echo '--- WAUX: AUX-HEAD TRAIN (waux1, started ~21:41 per DGX line) ---'
ls -l --time-style=+%F_%T /home/arm/junyang/carbon/data/weights/*waux* 2>&1 | head -30
for f in /home/arm/junyang/carbon/data/weights/*waux*log*.jsonl /home/arm/junyang/carbon/data/weights/*waux*.jsonl; do [ -f "$f" ] && echo "== $f ==" && tail -5 "$f"; done
echo '--- C: PACKS MD5 ---'
ls -l --time-style=+%F_%T /home/arm/junyang/carbon-repo/code/exp/m9xj570wr1.py /home/arm/junyang/carbon-repo/code/exp/m9xj570r1.py /home/arm/junyang/carbon-repo/code/exp/m9xj570g1.py /home/arm/junyang/carbon-repo/code/exp/snap_70.py 2>&1
md5sum /home/arm/junyang/carbon-repo/code/exp/m9xj570wr1.py /home/arm/junyang/carbon-repo/code/exp/m9xj570r1.py /home/arm/junyang/carbon-repo/code/exp/m9xj570g1.py 2>&1
echo '--- D: DISK ---'
df -h / | tail -1
echo '--- E: CORPUS-COUNTS ---'
cd /home/arm/junyang/carbon/data 2>/dev/null
for d in m3s2_corpus carago_corpus fable_corpus; do echo -n "$d replays="; ls "$d/replays" 2>/dev/null | wc -l; done
for f in m3s2_corpus/index.json carago_corpus/index.json fable_corpus/index.json; do echo -n "$f: "; python3 -c "import json;d=json.load(open('$f'));print(len(d) if isinstance(d,list) else len(d.get('entries',d)))" 2>/dev/null || echo MISSING; done
echo '--- F: WR1 WEIGHTS/LOG ---'
ls -l --time-style=+%F_%T /home/arm/junyang/carbon/data/weights/cnet2_m9xj570wr1.npz /home/arm/junyang/carbon/data/weights/cnet2_m9xj570wr1_last.npz /home/arm/junyang/carbon/data/weights/cnet2_m9xj570wr1_log.jsonl 2>&1
md5sum /home/arm/junyang/carbon/data/weights/cnet2_m9xj570wr1.npz /home/arm/junyang/carbon/data/weights/cnet2_m9xj570wr1_last.npz 2>&1
tail -2 /home/arm/junyang/carbon/data/weights/cnet2_m9xj570wr1_log.jsonl 2>&1
echo '--- G: PRODUCER-GATE FILES (frozen, expect dcaa2d89 / 07769df8) ---'
md5sum /home/arm/junyang/carbon-repo/code/producer/selfplay_gen.py /home/arm/junyang/carbon-repo/code/cnet2_build.py 2>&1
ls -l --time-style=+%F_%T /home/arm/junyang/carbon-repo/code/producer/selfplay_gen.py /home/arm/junyang/carbon-repo/code/cnet2_build.py 2>&1
echo '--- G2: DS_GEN1 ISOLATION MARKS ---'
ls -l --time-style=+%F_%T /home/arm/junyang/carbon/data/ds_gen1/USAGE_MARK.json /home/arm/junyang/carbon/data/gen1_shard0/USAGE_MARK.json 2>&1
md5sum /home/arm/junyang/carbon/data/ds_gen1/USAGE_MARK.json /home/arm/junyang/carbon/data/gen1_shard0/USAGE_MARK.json 2>&1
echo '--- H: RECENT FILES (last 30 min) ---'
find /home/arm/junyang/carbon/data /home/arm/junyang/carbon-repo/code/exp -mmin -30 -type f 2>/dev/null | head -40
echo "$MARK END $(date '+%F %T %Z')"

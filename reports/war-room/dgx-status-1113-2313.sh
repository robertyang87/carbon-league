#!/bin/bash
# DGX status probe 2313 (read-only). Marker: DGXSTATUS2313
# Context: war-room §16 brush order issued ~23:06; DGX exec agent started
# archiving frozen gates, brushing to a4dd8329 code set, then backtest on ds_gen1.
# This probe watches for brush progress; it performs NO writes.
MARK=DGXSTATUS2313
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
echo '--- B2: WORKING TREE (read-only status, brush in progress?) ---'
cd /home/arm/junyang/carbon-repo 2>/dev/null && git status --short 2>&1 | head -20; echo "(end status)"
echo '--- WAUX: WEIGHTS/LOG (by name) ---'
ls -l --time-style=+%F_%T /home/arm/junyang/carbon/data/weights/cnet2_m9xj570waux1.npz /home/arm/junyang/carbon/data/weights/cnet2_m9xj570waux1_last.npz /home/arm/junyang/carbon/data/weights/cnet2_m9xj570waux1_auxheads.npz /home/arm/junyang/carbon/data/weights/cnet2_m9xj570waux1_log.jsonl 2>&1
md5sum /home/arm/junyang/carbon/data/weights/cnet2_m9xj570waux1.npz /home/arm/junyang/carbon/data/weights/cnet2_m9xj570waux1_last.npz /home/arm/junyang/carbon/data/weights/cnet2_m9xj570waux1_auxheads.npz 2>&1
tail -2 /home/arm/junyang/carbon/data/weights/cnet2_m9xj570waux1_log.jsonl 2>&1
echo '--- C: PACKS MD5 (by name, incl waux1) ---'
ls -l --time-style=+%F_%T /home/arm/junyang/carbon-repo/code/exp/m9xj570waux1.py /home/arm/junyang/carbon-repo/code/exp/m9xj570wr1.py /home/arm/junyang/carbon-repo/code/exp/m9xj570r1.py /home/arm/junyang/carbon-repo/code/exp/m9xj570g1.py /home/arm/junyang/carbon-repo/code/exp/snap_70.py 2>&1
md5sum /home/arm/junyang/carbon-repo/code/exp/m9xj570waux1.py /home/arm/junyang/carbon-repo/code/exp/m9xj570wr1.py /home/arm/junyang/carbon-repo/code/exp/m9xj570r1.py /home/arm/junyang/carbon-repo/code/exp/m9xj570g1.py 2>&1
echo '--- C2: STAGEB FORMAL SPLIT (must NOT exist) ---'
ls -l /home/arm/junyang/carbon/data/stageB_split_N492.json 2>&1
ls -l --time-style=+%F_%T /home/arm/junyang/carbon/data/split_stageB_pool.py 2>&1
md5sum /home/arm/junyang/carbon/data/split_stageB_pool.py 2>&1
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
echo '--- G: PRODUCER-GATE FILES (frozen expect dcaa2d89 / 07769df8; brushed expect 15d24544 / a3f16063) ---'
md5sum /home/arm/junyang/carbon-repo/code/producer/selfplay_gen.py /home/arm/junyang/carbon-repo/code/cnet2_build.py 2>&1
ls -l --time-style=+%F_%T /home/arm/junyang/carbon-repo/code/producer/selfplay_gen.py /home/arm/junyang/carbon-repo/code/cnet2_build.py 2>&1
echo '--- G2: DS_GEN1 ISOLATION MARKS ---'
ls -l --time-style=+%F_%T /home/arm/junyang/carbon/data/ds_gen1/USAGE_MARK.json /home/arm/junyang/carbon/data/gen1_shard0/USAGE_MARK.json 2>&1
md5sum /home/arm/junyang/carbon/data/ds_gen1/USAGE_MARK.json /home/arm/junyang/carbon/data/gen1_shard0/USAGE_MARK.json 2>&1
echo '--- H: RECENT FILES (last 30 min) ---'
find /home/arm/junyang/carbon/data /home/arm/junyang/carbon-repo/code/exp -mmin -30 -type f 2>/dev/null | head -40
echo '--- H2: RECENT FILES IN REPO (last 120 min, brush watch) ---'
find /home/arm/junyang/carbon-repo -mmin -120 -type f 2>/dev/null | grep -v '.git/' | head -40
echo '--- H3: /tmp PATCH SCRIPTS (frozen-gate archive source) ---'
ls -l --time-style=+%F_%T /tmp/*.py /tmp/*.sh 2>/dev/null | head -20; echo "(end tmp)"
echo '--- I: TEACHER V2 LANDING CHECK (frozen v1.3 md5 77ba3677; brushed v2 md5 2c2541c6) ---'
md5sum /home/arm/junyang/carbon-repo/code/stage0_teacher.py 2>&1
grep -c -- '--belief' /home/arm/junyang/carbon-repo/code/stage0_teacher.py 2>&1
cd /home/arm/junyang/carbon-repo && git log -1 --format='%h %ci %s' -- code/stage0_teacher.py 2>&1
echo '--- J: BRUSH 8-FILE MD5 (expect after brush: teacher 2c2541c6, gate dd921e57, guided ed43fda8, measure d0de8a12, build a3f16063, producer 15d24544, ppo_train 1f1430b4, ppo_cnet2 62a9adc7) ---'
md5sum /home/arm/junyang/carbon-repo/code/stage0_teacher.py /home/arm/junyang/carbon-repo/code/stage1_teacher_gate.py /home/arm/junyang/carbon-repo/code/stage0_guided_match.py /home/arm/junyang/carbon-repo/code/stage0_measure_teacher.py /home/arm/junyang/carbon-repo/code/cnet2_build.py /home/arm/junyang/carbon-repo/code/producer/selfplay_gen.py /home/arm/junyang/carbon-repo/code/rl2_ppo_train.py /home/arm/junyang/carbon-repo/code/rl2_ppo_cnet2.py 2>&1
echo "$MARK END $(date '+%F %T %Z')"

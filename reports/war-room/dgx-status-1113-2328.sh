#!/bin/bash
# DGX status probe 2328 (read-only). Marker: DGXSTATUS2328
# Context: gen-1 rebuild order landed 23:17:56 (spec fd2e791d); DGX exec agent
# dispatched ~23:18:52, shard0 (seeds 994000-994199, 200 games) reported started.
# This probe verifies rebuild running + shard0 progress; it performs NO writes.
MARK=DGXSTATUS2328
echo "$MARK BEGIN $(date '+%F %T %Z')"
echo '--- A: HOST/GPU/PS ---'
hostname; uptime
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv,noheader
echo '--- TRAIN-PS (gen1 rebuild: selfplay_gen / cnet2_build / gen1r) ---'
ps -eo pid,etime,args | grep -E 'selfplay_gen|cnet2_build|gen1r|build_top' | grep -v grep || echo NO-GEN1-PROC
echo '--- TRAIN-PS (any train/pack) ---'
ps -eo pid,etime,args | grep -E 'cnet2_train|pack' | grep -v grep || echo NO-TRAIN-PROC
echo '--- ALL-PYTHON ---'
ps -eo pid,etime,args | grep -E 'python3|python ' | grep -v grep | head -20 || echo NO-PY-PROC
echo '--- GIT-RESIDUE ---'
ps -eo pid,etime,args | grep -E 'git (fetch|pull|checkout)' | grep -v grep || echo NO-GIT-PROC
echo '--- B: HEAD-VS-REMOTE (main) ---'
cd /home/arm/junyang/carbon-repo 2>/dev/null && git rev-parse --abbrev-ref HEAD && git rev-parse HEAD && git log -3 --format='%h %ci %s' && timeout 60 git ls-remote origin main 2>&1 | head -2
echo '--- B2: WORKING TREE (read-only status) ---'
cd /home/arm/junyang/carbon-repo 2>/dev/null && git status --short 2>&1 | head -20; echo "(end status)"
echo '--- G: BRUSH 8-FILE MD5 (expect: teacher 2c2541c6, gate dd921e57, guided ed43fda8, measure d0de8a12, build a3f16063, producer 15d24544, ppo_train 1f1430b4, ppo_cnet2 62a9adc7) ---'
md5sum /home/arm/junyang/carbon-repo/code/stage0_teacher.py /home/arm/junyang/carbon-repo/code/stage1_teacher_gate.py /home/arm/junyang/carbon-repo/code/stage0_guided_match.py /home/arm/junyang/carbon-repo/code/stage0_measure_teacher.py /home/arm/junyang/carbon-repo/code/cnet2_build.py /home/arm/junyang/carbon-repo/code/producer/selfplay_gen.py /home/arm/junyang/carbon-repo/code/rl2_ppo_train.py /home/arm/junyang/carbon-repo/code/rl2_ppo_cnet2.py 2>&1
echo '--- GEN1R: SHARD0 CORPUS ---'
ls -ld /home/arm/junyang/carbon/data/gen1r_shard0 /home/arm/junyang/carbon/data/gen1r_shard1 /home/arm/junyang/carbon/data/ds_gen1r_shard0 /home/arm/junyang/carbon/data/ds_gen1r_shard1 2>&1
echo -n "gen1r_shard0 replays="; ls /home/arm/junyang/carbon/data/gen1r_shard0/replays 2>/dev/null | wc -l
echo -n "gen1r_shard0 files="; find /home/arm/junyang/carbon/data/gen1r_shard0 -type f 2>/dev/null | wc -l
ls -l --time-style=+%F_%T /home/arm/junyang/carbon/data/gen1r_shard0 2>&1 | head -20
echo '--- GEN1R: INDEX / META ---'
for f in /home/arm/junyang/carbon/data/gen1r_shard0/index.json /home/arm/junyang/carbon/data/ds_gen1r_shard0/meta.json /home/arm/junyang/carbon/data/gen1r_shard0/meta.json; do echo "== $f"; ls -l --time-style=+%F_%T "$f" 2>&1; python3 -c "import json;d=json.load(open('$f'));print('entries:',len(d) if isinstance(d,list) else d if isinstance(d,dict) and len(d)<8 else list(d.keys())[:8])" 2>&1 | head -5; done
echo '--- GEN1R: POSITIONS + ENDGAME (step>=200) COUNT ---'
python3 - <<'EOF' 2>&1 | head -20
import json, os, glob
idx="/home/arm/junyang/carbon/data/gen1r_shard0/index.json"
try:
    d=json.load(open(idx))
    items=d if isinstance(d,list) else d.get("entries",d.get("games",[]))
    print("index_items:",len(items))
    steps=[it.get("step") for it in items if isinstance(it,dict) and "step" in it]
    if steps:
        print("positions_with_step:",len(steps),"step>=200:",sum(1 for s in steps if s>=200))
    else:
        keys=sorted({k for it in items if isinstance(it,dict) for k in it})[:12]
        print("item_keys:",keys)
        print("first_item:",items[0] if items else None)
except Exception as e:
    print("index_read_error:",e)
EOF
echo '--- GEN1R: PRODUCER LOG (recent files under gen1r + /home/arm logs) ---'
find /home/arm/junyang/carbon/data/gen1r_shard0 /home/arm/junyang/carbon/data/ds_gen1r_shard0 -mmin -120 -type f 2>/dev/null | head -20
ls -lt --time-style=+%F_%T /home/arm/junyang/*.log 2>/dev/null | head -10
echo '--- GEN1R: BACKTEST LOG (pending direct check) ---'
ls -l --time-style=+%F_%T /home/arm/junyang/gen1_mirror_gate_backtest_20261113.log 2>&1
tail -5 /home/arm/junyang/gen1_mirror_gate_backtest_20261113.log 2>&1
echo '--- FROZEN-GATE ARCHIVE (pending direct check) ---'
ls -l --time-style=+%F_%T /home/arm/junyang/frozen-gates-archive-20261113/ 2>&1
cat /home/arm/junyang/frozen-gates-archive-20261113/MANIFEST* 2>/dev/null | head -20
md5sum /home/arm/junyang/frozen-gates-archive-20261113/* 2>/dev/null | head -10
echo '--- C2: STAGEB FORMAL SPLIT (must NOT exist) ---'
ls -l /home/arm/junyang/carbon/data/stageB_split_N492.json 2>&1
md5sum /home/arm/junyang/carbon/data/split_stageB_pool.py 2>&1
echo '--- D: DISK ---'
df -h / | tail -1
echo '--- E: CORPUS-COUNTS ---'
cd /home/arm/junyang/carbon/data 2>/dev/null
for d in m3s2_corpus carago_corpus fable_corpus; do echo -n "$d replays="; ls "$d/replays" 2>/dev/null | wc -l; done
for f in m3s2_corpus/index.json carago_corpus/index.json fable_corpus/index.json; do echo -n "$f: "; python3 -c "import json;d=json.load(open('$f'));print(len(d) if isinstance(d,list) else len(d.get('entries',d)))" 2>/dev/null || echo MISSING; done
echo '--- H: RECENT FILES (last 30 min, data + repo code) ---'
find /home/arm/junyang/carbon/data /home/arm/junyang/carbon-repo/code/exp -mmin -30 -type f 2>/dev/null | head -40
echo '--- H2: /tmp collision-check files ---'
ls -l --time-style=+%F_%T /tmp/collide*.py 2>/dev/null; echo "(end tmp)"
echo "$MARK END $(date '+%F %T %Z')"

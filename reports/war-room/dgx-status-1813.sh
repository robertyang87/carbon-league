#!/bin/bash
# DGX status probe 1813: M7Bs v2 dual-arm progress + corpus counts + HEAD vs remote + git residue + k4v2 intake.
# Marker: DGXSTATUS1813K7M2
MARK=DGXSTATUS1813K7M2
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
echo "DGXSTATUS1813K7M2 DGX-BEGIN $(date '+%F %T %Z')"
hostname; uptime
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader
echo '--- TRAIN-PROCS ---'
ps -eo pid,etime,args | grep -E 'cnet2_train|train' | grep -v grep || echo NO-TRAIN-PROC
echo '--- GIT-RESIDUE ---'
ps -eo pid,etime,args | grep -E 'git (fetch|pull|checkout)' | grep -v grep || echo NO-GIT-PROC
echo '--- M7BSV2-LOGS ---'
date '+%F %T %Z'
for f in /home/arm/junyang/carbon/data/dq_logs/m7bsv2r1.log /home/arm/junyang/carbon/data/dq_logs/m7bsv2g1.log; do echo "== $f"; ls -l --time-style=+%F_%T "$f" 2>/dev/null; tail -3 "$f" 2>/dev/null || echo NO-LOG; done
echo '--- M7BSV2-CKPTS ---'
ls -lt --time-style=+%F_%T /home/arm/junyang/carbon/data/weights/cnet2_m7bsv2*.npz 2>/dev/null | head -12 || echo NO-CKPT
echo '--- DS-META ---'
python3 -c "import json;d=json.load(open('/home/arm/junyang/carbon/data/ds_m7bs_v2/meta.json'));print('m7bsv2', {k:d.get(k) for k in ('n_train','n_val','n','samples','games') if k in d})" 2>&1
echo '--- CORPUS-COUNTS ---'
cd /home/arm/junyang/carbon/data 2>/dev/null
for d in m3s2_corpus carago_corpus fable_corpus m6_corpus k4v2_corpus; do echo -n "$d replays="; ls "$d" 2>/dev/null | wc -l; done
for f in m3s2_corpus/index.json carago_corpus/index.json fable_corpus/index.json k4v2_matches.json; do echo -n "$f: "; python3 -c "import json;print(len(json.load(open('$f'))))" 2>/dev/null || echo MISSING; done
echo '--- STUDIO-RETURN ---'
ls -l --time-style=+%F_%T /home/arm/junyang/carbon/data/weights/cnet2_m3s2wr1_fromstudio.npz 2>/dev/null || echo NO-STUDIO-RETURN
echo '--- HEAD-VS-REMOTE ---'
cd /home/arm/junyang/carbon-repo 2>/dev/null && git rev-parse --abbrev-ref HEAD && git rev-parse --short HEAD && timeout 60 git ls-remote origin HEAD 2>&1 | head -2
echo "DGXSTATUS1813K7M2 DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"

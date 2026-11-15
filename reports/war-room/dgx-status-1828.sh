#!/bin/bash
# DGX status probe 1828: M7Bs v2 (g1 completion check + r1 progress) + m6 post-sync verify + HEAD branch-tip + git residue + Studio return.
# Read-only. Marker: DGXSTATUS1828Q7K3
MARK=DGXSTATUS1828Q7K3
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout -k 30 240 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
echo "DGXSTATUS1828Q7K3 DGX-BEGIN $(date '+%F %T %Z')"
hostname; uptime
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader
echo '--- TRAIN-PROCS ---'
ps -eo pid,etime,args | grep -E 'cnet2_train' | grep -v grep || echo NO-TRAIN-PROC
echo '--- GIT-RESIDUE ---'
ps -eo pid,etime,args | grep -E 'git (fetch|pull|checkout)' | grep -v grep || echo NO-GIT-PROC
echo '--- M7BSV2-LOGS ---'
date '+%F %T %Z'
for f in /home/arm/junyang/carbon/data/dq_logs/m7bsv2r1.log /home/arm/junyang/carbon/data/dq_logs/m7bsv2g1.log; do echo "== $f"; ls -l --time-style=+%F_%T "$f" 2>/dev/null; tail -2 "$f" 2>/dev/null || echo NO-LOG; done
ls /home/arm/junyang/carbon/data/dq_logs/m7bsv2*.done 2>/dev/null || echo NO-M7BSV2-DONE
echo '--- M7BSV2-CKPTS ---'
ls -lt --time-style=+%F_%T /home/arm/junyang/carbon/data/weights/cnet2_m7bsv2*.npz 2>/dev/null | head -20 || echo NO-CKPT
echo '--- DS-META ---'
python3 -c "import json;d=json.load(open('/home/arm/junyang/carbon/data/ds_m7bs_v2/meta.json'));print('m7bsv2 n_games=', len(d.get('games',[])), 'samples=', d.get('samples', d.get('n')))" 2>&1
echo '--- CORPUS-COUNTS ---'
cd /home/arm/junyang/carbon/data 2>/dev/null
for d in m3s2_corpus carago_corpus fable_corpus k4v2_corpus; do echo -n "$d replays-dir="; ls "$d/replays" 2>/dev/null | wc -l; echo -n "$d top-gz="; ls "$d"/*.json.gz 2>/dev/null | wc -l; done
for f in m3s2_corpus/index.json carago_corpus/index.json fable_corpus/index.json; do echo -n "$f: "; python3 -c "import json;d=json.load(open('$f'));print(len(d) if isinstance(d,list) else len(d.get('entries',d)))" 2>/dev/null || echo MISSING; done
echo '--- M6-POSTSYNC ---'
echo -n "workdir top .json.gz="; ls /home/arm/junyang/carbon/data/m6_corpus/*.json.gz 2>/dev/null | wc -l
echo -n "workdir replays/ .json="; ls /home/arm/junyang/carbon/data/m6_corpus/replays/*.json 2>/dev/null | wc -l
echo -n "workdir index entries="; python3 -c "import json;d=json.load(open('/home/arm/junyang/carbon/data/m6_corpus/index.json'));print(len(d) if isinstance(d,list) else len(d.get('entries',d)))" 2>/dev/null || echo MISSING
ls -l --time-style=+%F_%T /home/arm/junyang/carbon/data/m6_corpus/index.json 2>/dev/null
echo -n "repo top .json.gz="; ls /home/arm/junyang/carbon-repo/data/m6_corpus/*.json.gz 2>/dev/null | wc -l
echo -n "repo m6_matches entries="; python3 -c "import json;print(len(json.load(open('/home/arm/junyang/carbon-repo/data/m6_matches.json'))))" 2>/dev/null || echo MISSING
echo '--- STUDIO-RETURN ---'
ls -l --time-style=+%F_%T /home/arm/junyang/carbon/data/weights/cnet2_m3s2wr1_fromstudio.npz 2>/dev/null || echo NO-STUDIO-RETURN
md5sum /home/arm/junyang/carbon/data/weights/cnet2_m3s2wr1_fromstudio.npz 2>/dev/null || true
echo '--- HEAD-VS-REMOTE (branch tip, not origin HEAD) ---'
cd /home/arm/junyang/carbon-repo 2>/dev/null && git rev-parse --abbrev-ref HEAD && git rev-parse HEAD && git log -1 --format='%ci %s' && timeout 60 git ls-remote origin cnet2-search-v12 2>&1 | head -2
echo "DGXSTATUS1828Q7K3 DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"

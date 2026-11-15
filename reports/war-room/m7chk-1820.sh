#!/bin/bash
MARK=M7CHK1820X4P9
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout 240 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
echo "M7CHK1820X4P9 DGX-BEGIN $(date '+%F %T %Z')"
hostname; uptime
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader
echo '--- PROC-193396-193397 ---'
ps -p 193396,193397 -o pid,etime,args 2>/dev/null || echo PIDS-NOT-FOUND
echo '--- ALL-TRAIN-PROCS ---'
ps -eo pid,etime,args | grep -E 'cnet2_train' | grep -v grep || echo NO-TRAIN-PROC
echo '--- GIT-RESIDUE ---'
ps -eo pid,etime,args | grep -E 'git (fetch|pull|checkout)' | grep -v grep || echo NO-GIT-PROC
D=/home/arm/junyang/carbon/data
echo '--- LOGS ---'
for f in $D/dq_logs/m7bsv2r1.log $D/dq_logs/m7bsv2g1.log; do echo "== $f"; ls -l --time-style=+%F_%T "$f" 2>/dev/null || echo NO-LOG; echo "head:"; head -2 "$f" 2>/dev/null | cut -c1-300; echo "tail:"; tail -3 "$f" 2>/dev/null | cut -c1-300; done
echo '--- WEIGHTS ---'
ls -lt --time-style=+%F_%T $D/weights/cnet2_m7bsv2*.npz 2>/dev/null | head -12 || echo NO-CKPT
echo '--- DS ---'
python3 -c "import json;d=json.load(open('$D/ds_m7bs_v2/meta.json'));g=d.get('games');print('games',len(g) if isinstance(g,list) else g,'samples',d.get('samples'))" 2>&1
python3 -c "import json;print('index entries',len(json.load(open('$D/m7bs_corpus/index.json'))))" 2>&1
echo -n 'replays='; ls $D/m7bs_corpus/replays/*.json.gz 2>/dev/null | wc -l
echo "M7CHK1820X4P9 DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"

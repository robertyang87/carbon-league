echo "DGXSTATUS0658A START $(date '+%F %T %Z')"
hostname
uptime
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader
echo "--- ps train ---"
ps -eo pid,etime,cmd | grep -E 'ppo_train|train\.py|cnet2' | grep -v grep || echo NO_TRAIN_PROC
echo "--- ps git ---"
ps -eo pid,etime,cmd | grep -E 'git (fetch|pull|checkout)' | grep -v grep || echo NO_GIT_PROC
echo "--- repo HEAD ---"
cd /home/arm/junyang/carbon-repo && git rev-parse HEAD && timeout 60 git ls-remote origin HEAD | awk '{print "remote:", $1}'
echo "--- corpora ---"
cd /home/arm/junyang/carbon-repo
python3 -c "import json; d=json.load(open('data/m3s2_corpus/index.json')); print('m3s2_index', len(d))" 2>&1 | head -2
ls data/m3s2_corpus/replays 2>/dev/null | wc -l
python3 -c "import json; d=json.load(open('data/carago_corpus/index.json')); print('carago_index', len(d))" 2>&1 | head -2
ls data/carago_corpus 2>/dev/null | wc -l
python3 -c "import json; d=json.load(open('data/fable_corpus/index.json')); print('fable_index', len(d))" 2>&1 | head -2
ls data/fable_corpus 2>/dev/null | wc -l
python3 -c "import json; print('ds_carago', json.load(open('data/ds_carago/meta.json')))" 2>&1 | head -2
python3 -c "import json; print('ds_fable', json.load(open('data/ds_fable/meta.json')))" 2>&1 | head -2
echo "DGXSTATUS0658A END"

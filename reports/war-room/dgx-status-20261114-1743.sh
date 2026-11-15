echo "MARK DGXSTATUS11141743"
hostname; date '+%F %T %Z'; uptime
df -h /home/arm 2>/dev/null | tail -n +2
echo "== gpu =="
nvidia-smi --query-gpu=utilization.gpu,temperature.gpu,memory.used,memory.total --format=csv,noheader
echo "== procs =="
ps -eo pid,etime,pcpu,args | grep -E 'python3|train|rl2|ppo|eval|pull|ladder|krypton|git' | grep -v grep | head -40
echo "== spawn_main =="
pgrep -f 'spawn_mai[n]' | wc -l
echo "== git procs =="
ps -eo pid,etime,args | grep -E 'git (fetch|pull|checkout|ls-remote)' | grep -v grep || echo "git-procs none"
echo "== corpora (workdir /home/arm/junyang/carbon) =="
cd /home/arm/junyang/carbon 2>/dev/null || cd ~/junyang/carbon
pwd
python3 -c "import json;print('m3s2 index',len(json.load(open('data/m3s2_corpus/index.json'))))" 2>&1
echo "m3s2 files: $(find data/m3s2_corpus -type f 2>/dev/null | wc -l)"
python3 -c "import json;print('carago index',len(json.load(open('data/carago_corpus/index.json'))))" 2>&1
echo "carago files: $(find data/carago_corpus -type f 2>/dev/null | wc -l)"
python3 -c "import json;print('fable index',len(json.load(open('data/fable_corpus/index.json'))))" 2>&1
echo "fable files: $(find data/fable_corpus -type f 2>/dev/null | wc -l)"
python3 -c "import json;m=json.load(open('data/ds_carago/meta.json'));print('ds_carago meta',m if not isinstance(m,dict) else {k:m[k] for k in list(m)[:6]})" 2>&1 | head -c 400; echo
python3 -c "import json;m=json.load(open('data/ds_fable/meta.json'));print('ds_fable meta',m if not isinstance(m,dict) else {k:m[k] for k in list(m)[:6]})" 2>&1 | head -c 400; echo
echo "== split =="
ls data/ | grep -i stageB
md5sum data/stageB_split_N323*.json data/stageB_pool.json 2>/dev/null
ls data/ | grep -i -E 'verify|split'
md5sum data/*verify* 2>/dev/null
echo "== X10 v2 =="
md5sum data/k0_consumed_gen1r_shard1_registry_v2.json 2>/dev/null; stat -c '%y %s' data/k0_consumed_gen1r_shard1_registry_v2.json 2>/dev/null
echo "== it95 =="
find /home/arm/junyang/carbon /home/arm/junyang/carbon-repo -maxdepth 4 -name 'ckpt_95.npz' 2>/dev/null
md5sum $(find /home/arm/junyang/carbon /home/arm/junyang/carbon-repo -maxdepth 4 -name 'ckpt_95.npz' 2>/dev/null | head -1) 2>/dev/null
echo "== HEAD =="
git -C /home/arm/junyang/carbon-repo rev-parse HEAD
timeout 60 git -C /home/arm/junyang/carbon-repo ls-remote origin main
echo "== krypton pull =="
ps -eo pid,etime,args | grep -iE 'krypton|pull0035|ladder' | grep -v grep || echo "krypton-pull-procs none"
ls -d /home/arm/junyang/carbon/data/*krypton* /home/arm/junyang/carbon-repo/data/*krypton* 2>/dev/null
for d in /home/arm/junyang/carbon/data/*krypton* /home/arm/junyang/carbon-repo/data/*krypton*; do [ -d "$d" ] && echo "$d files: $(find "$d" -type f 2>/dev/null | wc -l)"; done
ls -t /home/arm/junyang/carbon/data/ 2>/dev/null | head -15
echo "== recently modified under data (30min) =="
find /home/arm/junyang/carbon/data -mmin -30 -type f 2>/dev/null | head -20; echo "recent-find-done"
echo "== ds_gen1 mark =="
find /home/arm/junyang/carbon /home/arm/junyang/carbon-repo -maxdepth 4 -name 'USAGE_MARK.json' 2>/dev/null
md5sum $(find /home/arm/junyang/carbon /home/arm/junyang/carbon-repo -maxdepth 4 -name 'USAGE_MARK.json' 2>/dev/null | head -2) 2>/dev/null
echo "== resident =="
ps -eo args | grep -E 'agent_mcp|vox_bridge|gitlab-runner' | grep -v grep | head
echo "== MARK END DGXSTATUS11141743 =="

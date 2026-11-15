echo MARK_DGXSTATUS1958KQ7M
hostname; date '+%F %T %Z'; uptime
echo --- GPU ---
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,temperature.gpu --format=csv,noheader
echo --- TRAIN-ANNOTATE-PROCS ---
ps -eo pid,etime,pcpu,args | grep -E 'k0_annotate|rl2_ppo|train\.py|python3.*carbon' | grep -v grep | head -20
echo --- GIT-RESIDUE ---
ps -eo pid,etime,args | grep -E 'git (fetch|pull|checkout|clone)' | grep -v grep
echo --- RETURNS ---
ls -la /home/arm/junyang/k0_dist/returns/c1/ /home/arm/junyang/k0_dist/returns/c2/ 2>&1
echo '--- C1-PROGRESS ---'
python3 -c "import json;p=json.load(open('/home/arm/junyang/k0_dist/returns/c1/progress_c1.json'));print('n_shards',len(p.get('shards',[])) if isinstance(p.get('shards'),list) else p.get('n_shards'));print('games_done',sum(1 for v in p.get('games',{}).values() if v));print('keys',sorted(p.keys()))" 2>&1
echo '--- C2-PROGRESS ---'
python3 -c "import json;p=json.load(open('/home/arm/junyang/k0_dist/returns/c2/progress.json'));print('n_shards',len(p.get('shards',[])) if isinstance(p.get('shards'),list) else p.get('n_shards'));print('games_done',sum(1 for v in p.get('games',{}).values() if v));print('keys',sorted(p.keys()))" 2>&1
echo --- L20-ARCHIVE ---
ls -la /home/arm/junyang/l20_ppo_u_archive_20261114/ 2>&1
echo 'run_dir files:'; find /home/arm/junyang/l20_ppo_u_archive_20261114/run_dir -type f 2>/dev/null | wc -l
echo 'ppo_attrib files:'; find /home/arm/junyang/l20_ppo_u_archive_20261114/ppo_attrib -type f 2>/dev/null | wc -l
echo 'value-diag files:'; find /home/arm/junyang/l20_ppo_u_archive_20261114/ppo_attrib/value-diag -type f 2>/dev/null | wc -l; ls /home/arm/junyang/l20_ppo_u_archive_20261114/ppo_attrib/value-diag/ 2>&1 | head -5
echo 'MANIFEST:'; md5sum /home/arm/junyang/l20_ppo_u_archive_20261114/MANIFEST.md5 2>&1; wc -l /home/arm/junyang/l20_ppo_u_archive_20261114/MANIFEST.md5 2>&1
echo 'total archive files:'; find /home/arm/junyang/l20_ppo_u_archive_20261114 -type f 2>/dev/null | wc -l; du -sb /home/arm/junyang/l20_ppo_u_archive_20261114 2>/dev/null
echo --- REPO-HEAD ---
cd /home/arm/junyang/carbon-repo && git rev-parse HEAD && git log -1 --format='%ci %s' && timeout 30 git ls-remote origin HEAD 2>&1 | head -2
echo --- WORKTREE ---
cd /home/arm/junyang/carbon-repo && git status --porcelain 2>&1 | head -15
echo --- END MARK_DGXSTATUS1958KQ7M ---

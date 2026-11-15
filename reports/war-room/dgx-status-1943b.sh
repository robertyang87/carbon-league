echo MARK_DGXSTATUS1943ZQ8R
hostname; date '+%F %T %Z'; uptime
echo --- GPU ---
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,temperature.gpu --format=csv,noheader
echo --- TRAIN-ANNOTATE-PROCS ---
ps -eo pid,etime,pcpu,args | grep -E 'k0_annotate|rl2_ppo|train\.py|python3.*carbon' | grep -v grep | head -20
echo --- GIT-RESIDUE ---
ps -eo pid,etime,args | grep -E 'git (fetch|pull|checkout|clone)' | grep -v grep
echo --- RETURNS ---
ls -la /home/arm/junyang/k0_dist/returns/c1/ /home/arm/junyang/k0_dist/returns/c2/ 2>&1
echo --- L20-ARCHIVE ---
ls -la /home/arm/junyang/l20_ppo_u_archive_20261114/run_dir/ 2>&1 | head -50
ls -la /home/arm/junyang/l20_ppo_u_archive_20261114/run_dir/ppo_attrib/ 2>&1 | head -5
ls -la /home/arm/junyang/l20_ppo_u_archive_20261114/MANIFEST.md5 2>&1
echo --- REPO-HEAD ---
cd /home/arm/junyang/carbon-repo && git rev-parse HEAD && git log -1 --format='%ci %s' && timeout 30 git ls-remote origin HEAD 2>&1 | head -2
echo --- WORKTREE ---
cd /home/arm/junyang/carbon-repo && git status --porcelain 2>&1 | head -15
echo --- END MARK_DGXSTATUS1943ZQ8R ---

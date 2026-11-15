echo MARK_L20STATUS1943X7KQ
hostname; date '+%F %T %Z'; uptime
echo --- GPU ---
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,temperature.gpu --format=csv,noheader
echo --- VALUE-DIAG-PROCS ---
ps -eo pid,etime,args | grep -E 'run_all|value.diag|realized_u|rl2_ppo' | grep -v grep | head -20
echo --- VALUE-DIAG-DIR ---
ls -la /root/carbon/data/value-diag/ 2>&1 | head -30
ls -la /root/carbon/data/value-diag/out/ 2>&1 | head -10
echo --- END MARK_L20STATUS1943X7KQ ---

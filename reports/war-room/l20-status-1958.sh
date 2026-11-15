echo MARK_L20STATUS1958QW3T
hostname; date '+%F %T %Z'; uptime
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader 2>&1 | head -3
ps -eo pid,etime,args | grep -E 'rl2_ppo_train|snipe' | grep -v grep | head -5
ls -la /root/ppo_attrib/value-diag/out/ 2>&1 | tail -5
ls /root/ppo_attrib/value-diag/ 2>&1 | head -5
echo --- END MARK_L20STATUS1958QW3T ---

#!/bin/bash
# DGX direct read 21:51 trip: K0 shard watch + idle + L20 archive count
echo "MARK WARROOMSYNC2151_DGX"
date '+%F %T %Z'
echo "--- GPU ---"
nvidia-smi --query-gpu=utilization.gpu,temperature.gpu,memory.used --format=csv,noheader 2>&1 | head -3
echo "--- K0 returns c1 ---"
ls -la /home/arm/junyang/k0_dist/returns/c1/ 2>&1 | head -20
echo "--- K0 returns c2 ---"
ls -la /home/arm/junyang/k0_dist/returns/c2/ 2>&1 | head -20
echo "--- progress mtimes ---"
stat -c '%y %n' /home/arm/junyang/k0_dist/returns/c1/* /home/arm/junyang/k0_dist/returns/c2/* 2>&1 | head -10
echo "--- L20 archive count ---"
ls /home/arm/junyang/l20_ppo_u_archive_20261114/ 2>/dev/null | wc -l
find /home/arm/junyang/l20_ppo_u_archive_20261114/ -type f 2>/dev/null | wc -l
echo "--- DGX repo HEAD ---"
git -C /home/arm/junyang/carbon-repo rev-parse HEAD 2>&1 | head -2
echo "--- k0 procs on DGX (expect none) ---"
ps aux | grep -E 'k0_annotate|train|bigeval' | grep -v grep | head -5; echo "(end)"

#!/bin/bash
# DGX status probe - DGXSTATUS11141843
ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o ConnectTimeout=30 arm@100.84.148.73 <<'REMOTE'
echo MARKER=DGXSTATUS11141843
hostname
uptime
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader
echo ---PROC---
ps -o pid,ppid,pgid,etime,pcpu,time,comm -p 1394500
ps -eo pid,ppid,etime,pcpu,time,comm | grep -E 'annotate|k0_annot' | grep -v grep | head -20
echo ---LOAD---
cat /proc/loadavg
echo ---SHARDS---
ls /home/arm/junyang/carbon/data/gen1r_shard1/shards/ 2>/dev/null | wc -l
ls -la /home/arm/junyang/carbon/data/gen1r_shard1/ 2>/dev/null | head -30
echo ---PROG---
stat -c '%y %s %n' /home/arm/junyang/carbon/data/gen1r_shard1/progress.json 2>/dev/null
head -c 2000 /home/arm/junyang/carbon/data/gen1r_shard1/progress.json 2>/dev/null; echo
echo ---LOG---
find /home/arm/junyang/carbon -maxdepth 5 -name 'k0_annot_shard1.log' 2>/dev/null | head
echo ---GITRES---
ps -eo pid,etime,args | grep -E 'git (fetch|pull|checkout)' | grep -v grep
echo ---DISK---
df -h /home/arm/junyang/carbon | tail -1
REMOTE

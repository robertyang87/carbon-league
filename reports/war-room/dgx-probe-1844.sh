#!/bin/bash
# DGX probe 2 - DGXSTATUS11141844
ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o ConnectTimeout=30 arm@100.84.148.73 <<'REMOTE'
echo MARKER=DGXSTATUS11141844
echo ---PGROUP---
ps -o pid,ppid,etime,pcpu,time,args -g 1394500 | head -30
echo ---WORKERS-PPID---
ps --ppid 1394500 -o pid,etime,pcpu,time,args | head -25
echo ---FIND-PROGRESS---
find /home/arm/junyang/carbon /home/arm/junyang/carbon-repo /home/arm/junyang/k0_prod_stage -maxdepth 6 -name 'progress.json' 2>/dev/null
find /home/arm/junyang/carbon /home/arm/junyang/carbon-repo /home/arm/junyang/k0_prod_stage -maxdepth 6 -name 'k0_annot*' 2>/dev/null
echo ---STAGE---
ls -la /home/arm/junyang/k0_prod_stage/ 2>/dev/null | head -20
echo ---ANNOT-DIRS---
ls /home/arm/junyang/carbon/data/ | head -40
echo ---TOPCPU---
ps -eo pid,ppid,etime,pcpu,time,args --sort=-pcpu | head -22
REMOTE

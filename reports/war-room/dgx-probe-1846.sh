#!/bin/bash
# DGX probe 4 - DGXSTATUS11141846
ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o ConnectTimeout=30 arm@100.84.148.73 <<'REMOTE'
echo MARKER=DGXSTATUS11141846
echo ---CWD-FD---
readlink /proc/1394500/cwd
ls -l /proc/1394500/fd 2>/dev/null | grep -v -E 'socket|pipe|/dev/null' | head -20
echo ---HAKIMI---
ls /home/arm/junyang/carbon/data/ | grep -i hakimi
find /home/arm/junyang/carbon/data -maxdepth 2 -iname '*hakimi*' 2>/dev/null
echo ---RECENT-LOGS---
find /home/arm/junyang/carbon /home/arm/junyang/k0_prod_stage -maxdepth 2 -name '*.log' -newermt '2026-11-14 18:00' 2>/dev/null
ls -la /home/arm/junyang/carbon/*.log 2>/dev/null | tail -5
echo ---OUT-MTIME---
stat -c '%y %s' /home/arm/junyang/carbon/data/k0_annot_gen1r_shard1/progress.json
date
REMOTE

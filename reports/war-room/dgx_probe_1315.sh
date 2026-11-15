#!/bin/bash
# DGX status probe round 2, 2026-11-14 13:15, marker DGXSTATUS11141315 (read-only, path discovery)
SSH_OPTS=(-i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o ConnectTimeout=30)
timeout -k 30 240 ssh "${SSH_OPTS[@]}" arm@100.84.148.73 '
echo MARKER=DGXSTATUS11141315
echo ---REPO-TOP---
ls /home/arm/junyang/carbon-repo
echo ---FIND-PACKS---
find /home/arm/junyang/carbon-repo /home/arm/junyang/carbon -maxdepth 4 -name "m3s2*_a4.py" -o -maxdepth 4 -name "m3s2*_base.py" 2>/dev/null
echo ---DATA-LAYOUT---
ls /home/arm/junyang/carbon-repo/data 2>/dev/null; ls /home/arm/junyang/carbon/data 2>/dev/null | head -30
echo ---M3S2-LOC---
find /home/arm/junyang/carbon-repo /home/arm/junyang/carbon -maxdepth 3 -name "m3s2_corpus" -o -maxdepth 3 -name "carago_corpus" -o -maxdepth 3 -name "fable_corpus" -o -maxdepth 3 -name "ds_carago" -o -maxdepth 3 -name "ds_fable" 2>/dev/null
'

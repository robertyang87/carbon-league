#!/bin/bash
# DGX probe 5 - DGXSTATUS11141847
ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o ConnectTimeout=30 arm@100.84.148.73 <<'REMOTE'
echo MARKER=DGXSTATUS11141847
L=/home/arm/junyang/k0_annot_shard1.log
stat -c '%y %s %n' $L
tail -c 1500 $L; echo
echo ---HAKIMI3---
ls /home/arm/junyang/carbon/data/hakimi3_corpus/replays 2>/dev/null | wc -l
python3 -c "import json;print('hakimi3 index', len(json.load(open('/home/arm/junyang/carbon/data/hakimi3_corpus/index.json'))))" 2>/dev/null
echo ---TIME---
date
REMOTE

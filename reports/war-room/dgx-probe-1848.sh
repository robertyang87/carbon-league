#!/bin/bash
# DGX probe 6 - DGXSTATUS11141848 (final)
ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o ConnectTimeout=30 arm@100.84.148.73 <<'REMOTE'
echo MARKER=DGXSTATUS11141848
find /home/arm/junyang/carbon /home/arm/junyang/carbon-repo -maxdepth 3 -iname '*hakimi3pro*' 2>/dev/null | head
python3 -c "import json;d=json.load(open('/home/arm/junyang/carbon/data/hakimi3_matches.json'));print('hakimi3_matches entries:',len(d) if isinstance(d,list) else len(d.get('matches',d)))" 2>/dev/null
ls /home/arm/junyang/carbon/data/hakimi3_corpus/ ; ls /home/arm/junyang/carbon/data/hakimi3_corpus/replays 2>/dev/null | wc -l
stat -c '%y %s' /home/arm/junyang/carbon/data/k0_annot_gen1r_shard1/progress.json
ps -o pid,etime,pcpu,time,args -g 1394500 | wc -l
date
REMOTE

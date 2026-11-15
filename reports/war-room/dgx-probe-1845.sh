#!/bin/bash
# DGX probe 3 - DGXSTATUS11141845
ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o ConnectTimeout=30 arm@100.84.148.73 <<'REMOTE'
echo MARKER=DGXSTATUS11141845
O=/home/arm/junyang/carbon/data/k0_annot_gen1r_shard1
echo ---OUTDIR---
ls -la $O
echo ---SHARDS---
ls $O/shards/ 2>/dev/null | wc -l
ls -la $O/shards/ 2>/dev/null | head -10
echo ---PROG---
stat -c '%y %s %n' $O/progress.json 2>/dev/null
head -c 3000 $O/progress.json 2>/dev/null; echo
echo ---LOGFILES---
find $O -maxdepth 2 -name '*.log' -o -maxdepth 2 -name '*.txt' 2>/dev/null | head
ls -la /home/arm/junyang/carbon/data/k0_annot_gen1r_shard1/ 2>/dev/null
echo ---LOGTAIL---
for f in /home/arm/junyang/k0_prod_stage/k0_annot_shard1.log /home/arm/junyang/carbon/k0_annot_shard1.log $O/k0_annot_shard1.log; do [ -f "$f" ] && echo "== $f ==" && stat -c '%y %s' "$f" && tail -c 500 "$f"; done
echo ---CORPORA---
cd /home/arm/junyang/carbon/data
for d in m3s2_corpus carago_corpus fable_corpus hakimi3pro_corpus; do echo -n "$d replays="; ls $d/replays 2>/dev/null | wc -l; echo -n "  index="; python3 -c "import json;print(len(json.load(open('$d/index.json'))))" 2>/dev/null || echo "?"; done
echo ---DSMETA---
python3 -c "import json; m=json.load(open('/home/arm/junyang/carbon/data/ds_carago/meta.json')); print('ds_carago', m.get('total', m.get('n', m)))" 2>/dev/null
python3 -c "import json; m=json.load(open('/home/arm/junyang/carbon/data/ds_fable/meta.json')); print('ds_fable', m.get('total', m.get('n', m)))" 2>/dev/null
ls /home/arm/junyang/carbon/data/ds_carago/ 2>/dev/null | head -5; ls /home/arm/junyang/carbon/data/ds_fable/ 2>/dev/null | head -5
echo ---HEAD---
cd /home/arm/junyang/carbon-repo && timeout 30 git rev-parse HEAD && timeout 30 git ls-remote origin main | awk '{print $1}'
REMOTE

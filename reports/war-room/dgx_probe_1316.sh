#!/bin/bash
# DGX status probe round 3, 2026-11-14 13:16, marker DGXSTATUS11141316 (read-only)
SSH_OPTS=(-i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o ConnectTimeout=30)
timeout -k 30 240 ssh "${SSH_OPTS[@]}" arm@100.84.148.73 '
echo MARKER=DGXSTATUS11141316
R=/home/arm/junyang/carbon-repo
W=/home/arm/junyang/carbon
echo ---PACKS---
ls -l --time-style=+%F_%T $R/code/exp/m3s2g1_a4.py $R/code/exp/m3s2g1_base.py $R/code/exp/m3s2r1_a4.py $R/code/exp/m3s2r1_base.py
md5sum $R/code/exp/m3s2g1_a4.py $R/code/exp/m3s2g1_base.py $R/code/exp/m3s2r1_a4.py $R/code/exp/m3s2r1_base.py
echo ---CORPUS-REPO---
for c in m3s2_corpus carago_corpus fable_corpus fable6_corpus; do echo -n "$c files: "; find $R/data/$c -name "*.json.gz" 2>/dev/null | wc -l; echo -n "$c index: "; python3 -c "import json;d=json.load(open(\"$R/data/$c/index.json\"));print(len(d))" 2>&1; done
echo ---CORPUS-WORKDIR---
for c in m3s2_corpus carago_corpus fable_corpus; do echo -n "work $c files: "; find $W/data/$c -name "*.json.gz" 2>/dev/null | wc -l; done
echo ---DS-WORKDIR---
ls -l --time-style=+%F_%T $W/data/ds_carago/meta.json $W/data/ds_fable/meta.json
python3 -c "import json;d=json.load(open(\"$W/data/ds_carago/meta.json\"));print(\"ds_carago keys\",list(d)[:10]);print({k:d[k] for k in d if isinstance(d[k],(int,float))})" 2>&1
python3 -c "import json;d=json.load(open(\"$W/data/ds_fable/meta.json\"));print(\"ds_fable keys\",list(d)[:10]);print({k:d[k] for k in d if isinstance(d[k],(int,float))})" 2>&1
echo ---FABLE6-MANIFEST---
head -c 400 /home/arm/junyang/gap4/pull0032_fable6_gap4.manifest; echo; wc -l /home/arm/junyang/gap4/pull0032_fable6_gap4.manifest
echo -n "fable6 repo files: "; find $R/data/fable6_corpus -name "*.json.gz" 2>/dev/null | wc -l
echo ---NEWEST-REPLAYS---
ls -lt $R/data/fable_corpus/*.json.gz 2>/dev/null | head -3; ls -lt $R/data/m3s2_corpus/*.json.gz 2>/dev/null | head -3
'

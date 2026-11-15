#!/bin/bash
# M6 probe 4: locate the index-builder code (K4v2/M7V2 method) and verify the
# pre-existing ds_m6 (complete? current? shards on disk?). Counts only, no big cats.
# Marker: M6PRB41855K3
MARK=M6PRB41855K3
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout -k 30 240 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
echo "M6PRB41855K3 DGX-BEGIN $(date '+%F %T %Z')"
echo '--- FIND-INDEX-BUILDER ---'
grep -rln "INDEX-BUILT\|orphan replays" /home/arm/junyang/carbon/code/ 2>/dev/null | head
ls -lt /tmp/*.sh 2>/dev/null | head -10
grep -rln "REBUILD-INDEX\|BUILD-INDEX" /tmp/*.sh 2>/dev/null | head
echo '--- DS-M6-VERIFY ---'
ls /home/arm/junyang/carbon/data/ds_m6/ 2>&1
du -sh /home/arm/junyang/carbon/data/ds_m6/ 2>&1
ls /home/arm/junyang/carbon/data/ds_m6/shards/ 2>/dev/null | wc -l
/home/arm/junyang/carbon/.venv/bin/python -c "
import json
d=json.load(open('/home/arm/junyang/carbon/data/ds_m6/meta.json'))
print('meta-type', type(d).__name__)
if isinstance(d,dict):
    print('meta-keys', sorted(d.keys()))
    print({k:(len(v) if isinstance(v,list) else v) for k,v in d.items()})
else:
    print('meta-len', len(d))
    import os
    missing=[e for e in d if not os.path.exists('/home/arm/junyang/carbon/data/ds_m6/shards/'+e['shard'].split('/')[-1])]
    print('entries', len(d), 'missing-shards', len(missing), 'total-n', sum(e.get('n',0) for e in d))
    print('val-count', sum(1 for e in d if e.get('val')))
" 2>&1
echo '--- CORPUS-STATE ---'
ls /home/arm/junyang/carbon/data/m6_corpus/ | wc -l
ls /home/arm/junyang/carbon/data/m6_corpus/index.json 2>&1
ls -d /home/arm/junyang/carbon/data/m6_corpus/replays 2>&1
ls /home/arm/junyang/carbon/data/m6_corpus/replays/ 2>/dev/null | wc -l
echo '--- INDEX-BUILDER-FILE ---'
grep -rn "def build_index\|def rebuild_index\|INDEX-BUILT" /home/arm/junyang/carbon/code/*.py 2>/dev/null | head -5
echo "M6PRB41855K3 DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"

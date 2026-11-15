#!/bin/bash
# M320 step A (runs ON DGX): pull repo to tip, report m9xj320 counts + diff,
# inspect replica_chain.sh interface and one real ds_m9xj index entry shape.
set -u
REPO=/home/arm/junyang/carbon-repo
WORK=/home/arm/junyang/carbon
cd "$REPO" || { echo NO-REPO; exit 1; }
echo "PRE-HEAD $(git rev-parse --short HEAD)"
timeout 60 git fetch origin cnet2-search-v12 || { echo FETCH-FAIL; exit 1; }
git checkout cnet2-search-v12 2>&1 | tail -1
timeout 60 git pull --ff-only || { echo PULL-FAIL; exit 1; }
echo "HEAD $(git rev-parse HEAD)"
python3 - <<'EOF'
import json, os
m=json.load(open('data/m9xj320_matches.json'))
print('MATCHES', len(m))
files=[f for f in os.listdir('data/m9xj320_corpus') if f.endswith('.json.gz')]
print('CORPUS', len(files))
print('ENTRY0', json.dumps(m[0], ensure_ascii=False)[:500])
ids={e['id'] for e in m}
fids={f.rpartition('-')[0] for f in files}
print('DIFF files-not-in-matches', len(fids-ids), 'matches-not-in-files', len(ids-fids))
names=set()
for e in m: names.add(e['name_a']); names.add(e['name_b'])
print('NAMES', sorted(names))
EOF
echo '--- replica_chain.sh ---'
wc -l "$REPO/code/replica_chain.sh"; md5sum "$REPO/code/replica_chain.sh"
echo '--- existing index entry shape (ds_m9xj build) ---'
python3 -c "import json; ix=json.load(open('$WORK/data/m9xj_corpus/index.json')); print(json.dumps(ix[0], ensure_ascii=False)[:500])"
echo '--- build_m9xj_w.py head ---'
head -12 "$WORK/code/build_m9xj_w.py"
echo STEPA-DONE

#!/bin/bash
# M320 poll2 (runs ON DGX): fetch+pull, report counts/diff, dump replica_chain.sh 61-114.
set -u
REPO=/home/arm/junyang/carbon-repo
cd "$REPO" || { echo NO-REPO; exit 1; }
timeout 60 git fetch origin cnet2-search-v12 || { echo FETCH-FAIL; exit 1; }
timeout 60 git pull --ff-only || { echo PULL-FAIL; exit 1; }
echo "HEAD $(git rev-parse --short HEAD)"
python3 - <<'EOF'
import json, os
m=json.load(open('data/m9xj320_matches.json'))
files=[f for f in os.listdir('data/m9xj320_corpus') if f.endswith('.json.gz')]
ids={e['id'] for e in m}
fids={f.rpartition('-')[0] for f in files}
print('MATCHES', len(m), 'CORPUS', len(files), 'DIFF', len(ids-fids), len(fids-ids))
EOF
echo '--- replica_chain.sh 61-114 ---'
sed -n '61,114p' "$REPO/code/replica_chain.sh"
echo POLL2-DONE

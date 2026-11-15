#!/bin/bash
# M320 poll (runs ON DGX): fetch+pull, report m9xj320 counts/diff, dump replica_chain.sh part 1.
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
echo '--- replica_chain.sh 1-60 ---'
sed -n '1,60p' "$REPO/code/replica_chain.sh"
echo POLL-DONE

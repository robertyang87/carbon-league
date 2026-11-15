#!/bin/bash
# K4v2 DGX staging (runs ON DGX, detached): pull repo, sync corpus, build
# index (nested strategies shape), validate via cnet2_build.scan, build
# ds_k4v2. Does NOT train (Studio is the trainer). Marker: K4V2STG1
MARK=K4V2STG1
D=/home/arm/junyang/carbon/data
C=/home/arm/junyang/carbon/code
R=/home/arm/junyang/carbon-repo
exec > $D/dq_logs/k4v2_stage.log 2>&1
echo "$MARK STAGE-BEGIN $(date '+%F %T %Z')"
cd $C || { echo "$MARK FAIL cd code"; exit 1; }

echo '--- SWEEP-GIT-RESIDUE ---'
ps -eo pid,etime,args | grep -E 'git (fetch|pull)' | grep -v grep || echo NO-GIT-RESIDUE

echo '--- PULL-REPO ---'
echo "repo HEAD before: $(git -C $R rev-parse --short HEAD 2>/dev/null)"
timeout 120 git -C $R pull --ff-only 2>&1 | tail -3
echo "repo HEAD after: $(git -C $R rev-parse --short HEAD 2>/dev/null)"

echo '--- REPO-COUNTS ---'
REPO_MATCHES=$(../.venv/bin/python -c "import json;print(len(json.load(open('$R/data/k4v2_matches.json'))))" 2>&1)
REPO_CORPUS=$(ls $R/data/k4v2_corpus/*.json.gz 2>/dev/null | wc -l)
REPO_CORPUS_ALL=$(ls $R/data/k4v2_corpus 2>/dev/null | wc -l)
echo "repo matches=$REPO_MATCHES corpus_gz=$REPO_CORPUS corpus_all_files=$REPO_CORPUS_ALL"

echo '--- SYNC-CORPUS ---'
mkdir -p $D/k4v2_corpus/replays
cp -n $R/data/k4v2_corpus/*.json.gz $D/k4v2_corpus/replays/ 2>/dev/null
cp $R/data/k4v2_matches.json $D/k4v2_matches.json
echo -n 'local replays='; ls $D/k4v2_corpus/replays/*.json.gz | wc -l

echo '--- BUILD-INDEX ---'
../.venv/bin/python - <<'PYEOF'
import json, gzip, os, glob
D = '/home/arm/junyang/carbon/data'
TNAME = 'stephen-K4TacSearch-v2'
TID = 'aaa02959b15d57e4e28f80d665ae3482'
matches = json.load(open(f'{D}/k4v2_matches.json'))
print('matches entries:', len(matches))
names = set()
for m in matches:
    names.add(m.get('name_a')); names.add(m.get('name_b'))
assert TNAME in names, f'teacher name not found verbatim in matches: {sorted(str(x) for x in names)[:8]}'
# teacher appearance check: wins/losses + side counts from task trigger caliber
tw = sum(1 for m in matches if (m['name_a']==TNAME and m.get('winner')=='a') or (m['name_b']==TNAME and m.get('winner')=='b'))
print('teacher wins in matches list:', tw)
entries, missing, skipped = [], 0, 0
for m in matches:
    fn = f"{m['id']}-{m['game']}.json.gz"
    path = f'{D}/k4v2_corpus/replays/{fn}'
    if not os.path.exists(path):
        missing += 1; print('missing replay for', fn); continue
    if m.get('order') is None:
        skipped += 1; print('no order for', fn); continue
    with gzip.open(path, 'rt') as f:
        rep = json.load(f)
    def side(name):
        return {'id': TID if name == TNAME else '', 'name': name}
    e = {'id': m['id'], 'game': m['game'], 'order': m['order'],
         'scores': m['scores'], 'winner': m.get('winner'),
         'steps': len(rep['steps']) - 1, 'file': f'replays/{fn}',
         'created': m.get('created'),
         'strategies': {'a': side(m['name_a']), 'b': side(m['name_b'])}}
    entries.append(e)
have = {(e['id'], e['game']) for e in entries}
orphans = []
for p in glob.glob(f'{D}/k4v2_corpus/replays/*.json.gz'):
    fn = os.path.basename(p)
    mid, g = fn[:-len('.json.gz')].rsplit('-', 1)
    try:
        if (mid, int(g)) not in have:
            orphans.append(fn)
    except ValueError:
        orphans.append(fn)
print('orphan replays:', orphans)
json.dump(entries, open(f'{D}/k4v2_corpus/index.json', 'w'), ensure_ascii=False)
print('INDEX-BUILT entries:', len(entries), 'missing:', missing, 'skipped:', skipped)
# shape check on first entry, verbatim vs a real m7bs entry is done by scan below
print('first entry strategies:', json.dumps(entries[0]['strategies'], ensure_ascii=False))
assert isinstance(entries[0]['strategies']['a'], dict) and 'id' in entries[0]['strategies']['a']
PYEOF
IDX_N=$(../.venv/bin/python -c "import json;print(len(json.load(open('$D/k4v2_corpus/index.json'))))" 2>&1)
echo "index entries: $IDX_N"
REPLAY_N=$(ls $D/k4v2_corpus/replays/*.json.gz | wc -l)
echo "THREE-COUNT matches=$REPO_MATCHES replays=$REPLAY_N index=$IDX_N"

echo '--- SCAN-VALIDATE (real consumer cnet2_build.scan) ---'
SCAN_N=$(CNET2_TOP_TEACHERS="stephen-K4TacSearch-v2@2" ../.venv/bin/python -c "
import build_top
import cnet2_build as B
games = B.scan(['$D/k4v2_corpus'])
print(len(games))
" 2>$D/dq_logs/k4v2_scan.err | tail -1)
echo "scan games: $SCAN_N"
tail -3 $D/dq_logs/k4v2_scan.err 2>/dev/null | cut -c1-200
if [ -z "$SCAN_N" ] || [ "$SCAN_N" = "0" ]; then echo "$MARK FAIL scan returned $SCAN_N"; exit 1; fi

echo '--- BUILD-DS-K4V2 ---'
rm -rf $D/ds_k4v2
CNET2_TOP_TEACHERS="stephen-K4TacSearch-v2@2" ../.venv/bin/python build_top.py $D/ds_k4v2 $D/k4v2_corpus > $D/dq_logs/build_k4v2.log 2>&1; echo "build_k4v2_rc=$?"
../.venv/bin/python set_uw.py $D/ds_k4v2 1 1 1 >> $D/dq_logs/build_k4v2.log 2>&1; echo "set_uw_rc=$?"
tail -3 $D/dq_logs/build_k4v2.log | cut -c1-300
../.venv/bin/python -c 'import json,sys; d=json.load(open(sys.argv[1])); print("DS-META games:", len(d["games"]) if isinstance(d.get("games"), list) else d.get("games"), "total:", d.get("total"), "teachers:", d.get("teachers"))' "$D/ds_k4v2/meta.json" || { echo "$MARK FAIL no ds meta"; exit 1; }
echo '--- DS-AGGREGATE ---'
du -sh $D/ds_k4v2
find $D/ds_k4v2 -type f | wc -l
cd $D/ds_k4v2 && find . -type f | sort | xargs md5sum | md5sum
echo "$MARK STAGE-DONE scan_games=$SCAN_N index=$IDX_N $(date '+%F %T %Z')"

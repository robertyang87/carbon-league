#!/bin/bash
# M6 DGX chain (runs ON DGX, detached): pull repo, sync corpus, build index
# (nested strategies), scan-validate, rebuild ds_m6 (191 games, replaces the
# stale 135-game ds_m6), dual-arm train, attest launch trio.
# Marker: M6STG1
MARK=M6STG1
D=/home/arm/junyang/carbon/data
C=/home/arm/junyang/carbon/code
R=/home/arm/junyang/carbon-repo
exec > $D/dq_logs/m6_stage.log 2>&1
echo "$MARK STAGE-BEGIN $(date '+%F %T %Z')"
cd $C || { echo "$MARK FAIL cd code"; exit 1; }

echo '--- 0) GPU-FREE-GATE (wait up to 5 min) ---'
for i in $(seq 1 5); do
  if ! pgrep -f 'cnet2_train[.]py' >/dev/null 2>&1; then echo "gpu free at wait-min $i"; break; fi
  echo "train proc present at min $i, waiting..."
  [ "$i" = 5 ] && { echo "$MARK FAIL gpu busy after 5min"; pgrep -af 'cnet2_train[.]py'; exit 1; }
  sleep 60
done
nvidia-smi --query-gpu=utilization.gpu,temperature.gpu --format=csv,noheader

echo '--- SWEEP-GIT-RESIDUE ---'
ps -eo pid,etime,args | grep -E 'git (fetch|pull)' | grep -v grep || echo NO-GIT-RESIDUE

echo '--- PULL-REPO ---'
echo "repo HEAD before: $(git -C $R rev-parse --short HEAD 2>/dev/null)"
timeout 120 git -C $R pull --ff-only 2>&1 | tail -3
echo "repo HEAD after: $(git -C $R rev-parse --short HEAD 2>/dev/null)"

echo '--- REPO-COUNTS ---'
REPO_MATCHES=$(../.venv/bin/python -c "import json;print(len(json.load(open('$R/data/m6_matches.json'))))" 2>&1)
REPO_CORPUS=$(ls $R/data/m6_corpus/*.json.gz 2>/dev/null | wc -l)
echo "repo matches=$REPO_MATCHES corpus_gz=$REPO_CORPUS"
[ "$REPO_MATCHES" = 191 ] && [ "$REPO_CORPUS" = 191 ] || { echo "$MARK FAIL repo counts matches=$REPO_MATCHES corpus=$REPO_CORPUS (need 191/191)"; exit 1; }

echo '--- SYNC-CORPUS ---'
mkdir -p $D/m6_corpus/replays
TOP_N=$(ls $D/m6_corpus/*.json.gz 2>/dev/null | wc -l)
echo "top-level gz before move: $TOP_N"
if [ "$TOP_N" = 191 ]; then
  mv $D/m6_corpus/*.json.gz $D/m6_corpus/replays/ || { echo "$MARK FAIL mv replays"; exit 1; }
elif [ "$TOP_N" != 0 ]; then
  echo "$MARK FAIL top-level gz=$TOP_N (need 191 or 0)"; exit 1
fi
cp $R/data/m6_matches.json $D/m6_matches.json
echo -n 'local replays='; ls $D/m6_corpus/replays/*.json.gz | wc -l
echo -n 'local top-level gz='; ls $D/m6_corpus/*.json.gz 2>/dev/null | wc -l
LOCAL_MATCHES=$(../.venv/bin/python -c "import json;print(len(json.load(open('$D/m6_matches.json'))))" 2>&1)
echo "local matches=$LOCAL_MATCHES"

echo '--- BUILD-INDEX ---'
../.venv/bin/python - <<'PYEOF'
import json, gzip, os, glob
D = '/home/arm/junyang/carbon/data'
TNAME = 'RyanX-AIPowercore-M6'
# NOTE: M6's ladder strategy id is not recoverable from any live source:
# M6 vanished from the board 2026-11-12 17:28, the old index.json was deleted
# in the m6 closure, and the DGX leaderboard snapshots (11-09/11-10) contain
# no M6 entry. The real consumer cnet2_build.scan reads ONLY ["name"]
# (cnet2_build.py:225), never ["id"], so the teacher id is recorded as ''
# (same as every opponent side). Training is unaffected; re-stamp the id
# later if it resurfaces, without rebuilding the dataset.
TID = ''
matches = json.load(open(f'{D}/m6_matches.json'))
print('matches entries:', len(matches))
names = set()
for m in matches:
    names.add(m.get('name_a')); names.add(m.get('name_b'))
assert TNAME in names, f'teacher name not found verbatim in matches: {sorted(str(x) for x in names)[:8]}'
tw = sum(1 for m in matches if (m['name_a']==TNAME and m.get('winner')=='a') or (m['name_b']==TNAME and m.get('winner')=='b'))
print('teacher wins in matches list:', tw)
entries, missing, skipped = [], 0, 0
for m in matches:
    fn = f"{m['id']}-{m['game']}.json.gz"
    path = f'{D}/m6_corpus/replays/{fn}'
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
for p in glob.glob(f'{D}/m6_corpus/replays/*.json.gz'):
    fn = os.path.basename(p)
    mid, g = fn[:-len('.json.gz')].rsplit('-', 1)
    try:
        if (mid, int(g)) not in have:
            orphans.append(fn)
    except ValueError:
        orphans.append(fn)
print('orphan replays:', orphans)
json.dump(entries, open(f'{D}/m6_corpus/index.json', 'w'), ensure_ascii=False)
print('INDEX-BUILT entries:', len(entries), 'missing:', missing, 'skipped:', skipped)
print('first entry strategies:', json.dumps(entries[0]['strategies'], ensure_ascii=False))
assert isinstance(entries[0]['strategies']['a'], dict) and 'id' in entries[0]['strategies']['a']
PYEOF
[ $? -ne 0 ] && { echo "$MARK FAIL index build rc=$?"; exit 1; }
IDX_N=$(../.venv/bin/python -c "import json;print(len(json.load(open('$D/m6_corpus/index.json'))))" 2>&1)
echo "index entries: $IDX_N"
REPLAY_N=$(ls $D/m6_corpus/replays/*.json.gz | wc -l)
echo "THREE-COUNT matches=$LOCAL_MATCHES replays=$REPLAY_N index=$IDX_N"
[ "$LOCAL_MATCHES" = 191 ] && [ "$REPLAY_N" = 191 ] && [ "$IDX_N" = 191 ] || { echo "$MARK FAIL three-count"; exit 1; }

echo '--- SCAN-VALIDATE (real consumer cnet2_build.scan) ---'
SCAN_N=$(CNET2_TOP_TEACHERS="RyanX-AIPowercore-M6@2" ../.venv/bin/python -c "
import build_top
import cnet2_build as B
games = B.scan(['$D/m6_corpus'])
print(len(games))
" 2>$D/dq_logs/m6_scan.err | tail -1)
echo "scan games: $SCAN_N"
tail -3 $D/dq_logs/m6_scan.err 2>/dev/null | cut -c1-200
if [ -z "$SCAN_N" ] || [ "$SCAN_N" = "0" ]; then echo "$MARK FAIL scan returned $SCAN_N"; exit 1; fi

echo '--- BUILD-DS-M6 (replaces stale 135-game ds_m6) ---'
rm -rf $D/ds_m6
CNET2_TOP_TEACHERS="RyanX-AIPowercore-M6@2" ../.venv/bin/python build_top.py $D/ds_m6 $D/m6_corpus > $D/dq_logs/build_m6.log 2>&1; echo "build_m6_rc=$?"
../.venv/bin/python set_uw.py $D/ds_m6 1 1 1 >> $D/dq_logs/build_m6.log 2>&1; echo "set_uw_rc=$?"
tail -3 $D/dq_logs/build_m6.log | cut -c1-300
../.venv/bin/python -c 'import json,sys; d=json.load(open(sys.argv[1])); g=d.get("games"); print("DS-META games:", len(g) if isinstance(g, list) else g, "total:", d.get("total"), "teachers:", d.get("teachers"))' "$D/ds_m6/meta.json" || { echo "$MARK FAIL no ds meta"; exit 1; }

echo '--- DS-AGGREGATE ---'
du -sh $D/ds_m6
find $D/ds_m6 -type f | wc -l
cd $D/ds_m6 && find . -type f | sort | xargs md5sum | md5sum
cd $C

echo '--- LAUNCH-M6-TRAIN ---'
export BC_DEVICE=cuda BC_RAM=1 BC_USE_W=1 BC_USE_UW=1 BC_BS=256 BC_SAVE_EPOCHS=1 BC_PCT=0.1 BC_LR=3e-4
BC_INIT=$D/weights/cnet2_arch_n1_last.npz nohup ../.venv/bin/python cnet2_train.py "$D/ds_m6" "$D/weights/cnet2_m6r1.npz" 20 '{"C":128,"NB":8,"D":128,"NH":4,"L":2,"E":32,"G":8,"ZI":1,"TA":4,"TNH":4}' > "$D/dq_logs/m6r1.log" 2>&1 &
BC_INIT=$D/weights/cnet2_b128m2t3.npz nohup ../.venv/bin/python cnet2_train.py "$D/ds_m6" "$D/weights/cnet2_m6g1.npz" 20 '{"C":128,"NB":8,"D":128,"NH":4,"L":2,"E":32,"G":8}' > "$D/dq_logs/m6g1.log" 2>&1 &
sleep 5; echo "train pids: $(pgrep -f 'cnet2_train[.]py' | tr '\n' ' ')"

echo '--- ATTESTATION (sleep 180 for ep0) ---'
sleep 180
echo "=== ATTESTATION M6 $(date '+%F %T %Z') ==="
echo "processes:"; ps -eo pid,etime,args | grep 'cnet2_train' | grep -v grep || echo NO-TRAIN-PROC
echo "gpu:"; nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader
echo "logs:"; ls -l --time-style=+%F_%T "$D/dq_logs/m6r1.log" "$D/dq_logs/m6g1.log"
echo "ep0 r1:"; grep -h '"ep": 0' "$D/dq_logs/m6r1.log" 2>/dev/null | tail -1
echo "ep0 g1:"; grep -h '"ep": 0' "$D/dq_logs/m6g1.log" 2>/dev/null | tail -1
EP0R=$(grep -c '"ep": 0' "$D/dq_logs/m6r1.log" 2>/dev/null || true)
EP0G=$(grep -c '"ep": 0' "$D/dq_logs/m6g1.log" 2>/dev/null || true)
echo "ep0 lines: r1=$EP0R g1=$EP0G"
echo "counts: matches=$LOCAL_MATCHES replays=$REPLAY_N index=$IDX_N scan=$SCAN_N"
if [ "${EP0R:-0}" -lt 1 ] || [ "${EP0G:-0}" -lt 1 ] || ! pgrep -f 'cnet2_train[.]py' >/dev/null 2>&1; then
  echo "$MARK FAIL attestation trio not live (ep0 r1=$EP0R g1=$EP0G)"; exit 1
fi
echo "$MARK LAUNCH-OK + ATTESTED $(date '+%F %T %Z')"

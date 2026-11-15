#!/bin/bash
# M7Bs v2 retrain (runs ON DGX, detached): wait Carago r1 -> sync 142-game
# corpus -> rebuild index (ladder_corpus nested shape) -> ds_m7bs_v2 ->
# dual-arm train -> wait completion -> report final numbers.
# Marker: M7V2K8R3
MARK=M7V2K8R3
D=/home/arm/junyang/carbon/data
C=/home/arm/junyang/carbon/code
R=/home/arm/junyang/carbon-repo
echo "$MARK STAGE-BEGIN $(date '+%F %T %Z')"
cd $C || { echo "$MARK FAIL cd code"; exit 1; }

echo '--- WAIT-CARAGO-R1 ---'
CARAGO_OK=0
for i in $(seq 1 45); do
  if ! pgrep -f 'cnet2_train[.]py' >/dev/null 2>&1 && [ -f $D/weights/cnet2_caragor1_ep19.npz ]; then
    CARAGO_OK=1; echo "carago clear at wait-min $i"; break
  fi
  sleep 60
done
if [ "$CARAGO_OK" != 1 ]; then
  echo "$MARK FAIL carago not clear after 45min"
  pgrep -af 'cnet2_train[.]py' || echo NO-TRAIN-PROC
  ls -la $D/weights/cnet2_caragor1_ep19.npz 2>&1
  exit 1
fi
nvidia-smi --query-gpu=utilization.gpu,temperature.gpu --format=csv,noheader
echo "carago r1 ep19: $(grep -h '"ep": 19' $D/dq_logs/caragor1.log 2>/dev/null | tail -1)"
echo "carago r1 best: $(grep -h '^best' $D/dq_logs/caragor1.log 2>/dev/null | tail -1)"
echo "carago g1 best: $(grep -h '^best' $D/dq_logs/caragog1.log 2>/dev/null | tail -1)"
md5sum $D/weights/cnet2_caragor1_ep19.npz $D/weights/cnet2_caragog1_ep19.npz 2>&1

echo '--- SYNC-CORPUS ---'
mkdir -p $D/m7bs_corpus/replays
echo "repo HEAD: $(git -C $R rev-parse --short HEAD 2>/dev/null)"
REPO_N=$(ls $R/data/m7bs_corpus/*.json.gz 2>/dev/null | wc -l)
echo "repo m7bs files: $REPO_N"
if [ "$REPO_N" -lt 142 ]; then
  echo "repo has $REPO_N < 142, pulling..."
  timeout 120 git -C $R pull --ff-only 2>&1 | tail -3
  REPO_N=$(ls $R/data/m7bs_corpus/*.json.gz 2>/dev/null | wc -l)
  echo "repo after pull: $REPO_N"
fi
echo -n 'local replays before='; ls $D/m7bs_corpus/replays/*.json.gz 2>/dev/null | wc -l
cp -n $R/data/m7bs_corpus/*.json.gz $D/m7bs_corpus/replays/ 2>/dev/null
echo -n 'local replays after='; ls $D/m7bs_corpus/replays/*.json.gz | wc -l

echo '--- REBUILD-INDEX (ladder_corpus nested shape) ---'
../.venv/bin/python - <<'PYEOF'
import json, gzip, os, glob
D = '/home/arm/junyang/carbon/data'
R = '/home/arm/junyang/carbon-repo'
TNAME = 'RyanX-AIPowercore-M7Bs-b010'
def load(p):
    try:
        return json.load(open(p))
    except Exception as e:
        print('load fail', p, e); return []
lm = load(f'{D}/m7bs_matches.json')
rm = load(f'{R}/data/m7bs_matches.json')
print('local matches:', len(lm), 'repo matches:', len(rm))
matches = lm if len(lm) >= len(rm) else rm
print('using matches:', len(matches))
base = load(f'{D}/m7bs_corpus/index.json')
print('base index entries:', len(base))
assert isinstance(base[0]['strategies'], dict), 'base strategies not nested'
tid = ''
for e in base:
    for s in ('a', 'b'):
        if e['strategies'][s]['name'] == TNAME:
            tid = e['strategies'][s]['id']
            break
    if tid:
        break
print('teacher id:', repr(tid))
have = {(e['id'], e['game']) for e in base}
entries = list(base)
added, missing, skipped = 0, 0, 0
for m in matches:
    key = (m['id'], m['game'])
    if key in have:
        continue
    fn = f"{m['id']}-{m['game']}.json.gz"
    path = f'{D}/m7bs_corpus/replays/{fn}'
    if not os.path.exists(path):
        missing += 1
        continue
    if m.get('order') is None:
        skipped += 1
        continue
    with gzip.open(path, 'rt') as f:
        rep = json.load(f)
    def side(name):
        return {'id': tid if name == TNAME else '', 'name': name}
    e = {'id': m['id'], 'game': m['game'], 'order': m['order'],
         'scores': m['scores'], 'winner': m.get('winner'),
         'steps': len(rep['steps']) - 1, 'file': f'replays/{fn}',
         'created': m.get('created'),
         'strategies': {'a': side(m['name_a']), 'b': side(m['name_b'])}}
    entries.append(e)
    have.add(key)
    added += 1
orphans = 0
for p in glob.glob(f'{D}/m7bs_corpus/replays/*.json.gz'):
    fn = os.path.basename(p)
    mid, g = fn[:-len('.json.gz')].rsplit('-', 1)
    if (mid, int(g)) not in have:
        orphans += 1
        if orphans <= 5:
            print('orphan replay:', fn)
json.dump(entries, open(f'{D}/m7bs_corpus/index.json', 'w'), ensure_ascii=False)
print('index merged: base', len(base), 'added', added, 'missing replays', missing, 'skipped no-order', skipped, 'orphans', orphans, 'FINAL', len(entries))
PYEOF
FINAL_N=$(../.venv/bin/python -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$D/m7bs_corpus/index.json")
echo "index final entries: $FINAL_N"
if [ "$FINAL_N" -lt 140 ]; then echo "$MARK FAIL index only $FINAL_N entries (<140)"; exit 1; fi

echo '--- BUILD-DS-M7BSV2 ---'
rm -rf $D/ds_m7bs_v2
CNET2_TOP_TEACHERS="RyanX-AIPowercore-M7Bs-b010@2" ../.venv/bin/python build_top.py $D/ds_m7bs_v2 $D/m7bs_corpus > $D/dq_logs/build_m7bsv2.log 2>&1; echo "build_m7bsv2_rc=$?"
../.venv/bin/python set_uw.py $D/ds_m7bs_v2 1 1 1 >> $D/dq_logs/build_m7bsv2.log 2>&1
tail -4 $D/dq_logs/build_m7bsv2.log
cat $D/ds_m7bs_v2/meta.json 2>/dev/null || echo "$MARK FAIL no meta.json, training NOT launched"

if [ -f $D/ds_m7bs_v2/meta.json ]; then
  echo '--- LAUNCH-M7BSV2-TRAIN ---'
  nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader
  export BC_DEVICE=cuda BC_RAM=1 BC_THREADS=5 BC_USE_W=1 BC_USE_UW=1 BC_BS=256 BC_SAVE_EPOCHS=1 BC_PCT=0.1 BC_LR=3e-4
  unset BC_AMP BC_CLIP BC_TRAIN_PREFIX BC_CACHE_TRUNK BC_SELF_KL BC_EXIT_W BC_AUX_W BC_COMPILE BC_CAR_KL
  BC_INIT=$D/weights/cnet2_arch_n1_last.npz nohup ../.venv/bin/python cnet2_train.py $D/ds_m7bs_v2 $D/weights/cnet2_m7bsv2r1.npz 20 '{"C":128,"NB":8,"D":128,"ZI":1,"TA":4,"TNH":4}' > $D/dq_logs/m7bsv2r1.log 2>&1 &
  BC_INIT=$D/weights/cnet2_b128m2t3.npz nohup ../.venv/bin/python cnet2_train.py $D/ds_m7bs_v2 $D/weights/cnet2_m7bsv2g1.npz 20 '{"C":128,"NB":8,"D":128}' > $D/dq_logs/m7bsv2g1.log 2>&1 &
  sleep 5
  echo "train pids: $(pgrep -f 'cnet2_train[.]py' | tr '\n' ' ')"
else
  echo "$MARK FAIL build failed, exiting"; exit 1
fi

sleep 300
echo "ep0 r1: $(grep -h '"ep": 0' $D/dq_logs/m7bsv2r1.log 2>/dev/null | tail -1)"
echo "ep0 g1: $(grep -h '"ep": 0' $D/dq_logs/m7bsv2g1.log 2>/dev/null | tail -1)"
if ! grep -q '"ep": 0' $D/dq_logs/m7bsv2r1.log 2>/dev/null || ! grep -q '"ep": 0' $D/dq_logs/m7bsv2g1.log 2>/dev/null; then
  echo "$MARK FAIL ep0 lines missing after 300s"
  tail -5 $D/dq_logs/m7bsv2r1.log; tail -5 $D/dq_logs/m7bsv2g1.log
  exit 1
fi
echo "$MARK LAUNCH-OK $(date '+%F %T %Z')"

echo '--- WAIT-TRAINING ---'
for i in $(seq 1 90); do
  sleep 60
  if [ -f $D/dq_logs/m7bsv2r1.done ] && [ -f $D/dq_logs/m7bsv2g1.done ]; then echo "both done at wait-min $i"; break; fi
  if ! pgrep -f 'cnet2_train[.]py' >/dev/null 2>&1; then
    echo "$MARK FAIL trainers gone, done missing at min $i"
    tail -5 $D/dq_logs/m7bsv2r1.log; tail -5 $D/dq_logs/m7bsv2g1.log
    exit 1
  fi
  if [ $((i % 15)) = 0 ]; then
    echo "wait-min $i r1: $(tail -1 $D/dq_logs/m7bsv2r1.log | cut -c1-160)"
    echo "wait-min $i g1: $(tail -1 $D/dq_logs/m7bsv2g1.log | cut -c1-160)"
  fi
done
if [ ! -f $D/dq_logs/m7bsv2r1.done ] || [ ! -f $D/dq_logs/m7bsv2g1.done ]; then
  echo "$MARK FAIL timeout 90min, done missing"; exit 1
fi

echo '--- M7V2 FINAL ---'
echo "r1 ep19: $(grep -h '"ep": 19' $D/dq_logs/m7bsv2r1.log | tail -1)"
echo "r1 best: $(grep -h '^best' $D/dq_logs/m7bsv2r1.log | tail -1)"
echo "g1 ep19: $(grep -h '"ep": 19' $D/dq_logs/m7bsv2g1.log | tail -1)"
echo "g1 best: $(grep -h '^best' $D/dq_logs/m7bsv2g1.log | tail -1)"
md5sum $D/weights/cnet2_m7bsv2r1_ep19.npz $D/weights/cnet2_m7bsv2g1_ep19.npz
ls -la $D/weights/cnet2_m7bsv2r1_ep19.npz $D/weights/cnet2_m7bsv2g1_ep19.npz
cat $D/ds_m7bs_v2/meta.json
echo "$MARK DONE $(date '+%F %T %Z')"

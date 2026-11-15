echo "M320WCHK2225A L20_BEGIN $(date '+%H:%M:%S')"
cat > /tmp/m320_wchk.sh <<'STEOF'
echo "M320WCHK2225A STUDIO_BEGIN $(date '+%F %T %Z')"
D=$HOME/junyang/carbon/data
$HOME/junyang/carbon/.venv/bin/python - <<'PYEOF'
import json, os
import numpy as np
D = os.path.expanduser('~/junyang/carbon/data')
SRC = os.path.join(D, 'ds_m9xj320'); DST = os.path.join(D, 'ds_m9xj320_w')
meta = json.load(open(os.path.join(SRC, 'meta.json'))); games = meta['games']
meta2 = json.load(open(os.path.join(DST, 'meta.json')))
gid = np.load(os.path.join(SRC, 'gid.npy')); step = np.load(os.path.join(SRC, 'step.npy'))
N = len(gid); factors = np.ones(N, np.int64)
rep = {'end_only': 0, 'lc_only': 0, 'both': 0, 'plain': 0, 'val_rows': 0, 'train_rows': 0,
       'end_rows_train': 0, 'lost_games': 0, 'close_games': 0, 'val_games': 0}
is_val = np.zeros(N, bool)
for gi, g in enumerate(games):
    idx = np.nonzero(gid == gi)[0]
    if len(idx) == 0: continue
    if g['val']:
        is_val[idx] = True; rep['val_rows'] += len(idx); rep['val_games'] += 1; continue
    rep['train_rows'] += len(idx)
    st = step[idx].astype(np.int64)
    seg = np.concatenate([[0], np.cumsum(np.diff(st) < 0)])
    sides = g['sides']; scores = g['scores']
    gmargin = abs(float(scores[0]) - float(scores[1]))
    side_margins = {s: float(scores[s]) - float(scores[1 - s]) for s in sides}
    if any(m < 0 for m in side_margins.values()): rep['lost_games'] += 1
    if gmargin <= 500: rep['close_games'] += 1
    for j, i in enumerate(idx):
        s = sides[int(seg[j])]; end = bool(st[j] >= 270)
        lc = (side_margins[s] < 0) or (gmargin <= 500)
        if end: rep['end_rows_train'] += 1
        f = (3 if end else 1) * (2 if lc else 1); factors[i] = f
        if end and lc: rep['both'] += 1
        elif end: rep['end_only'] += 1
        elif lc: rep['lc_only'] += 1
        else: rep['plain'] += 1
repidx = np.repeat(np.arange(N), factors); NN = len(repidx)
print('WEIGHTED-REPORT-REPLICA', json.dumps(rep))
print('expected total:', NN, 'ds_w meta total:', meta2['total'])
ok = True
def chk(name, cond, detail=''):
    global ok
    print(('PASS ' if cond else 'FAIL ') + name + (' ' + str(detail) if detail != '' else ''))
    ok = ok and bool(cond)
chk('total matches', NN == meta2['total'] == 34232, NN)
gid2 = np.load(os.path.join(DST, 'gid.npy'))
newn = np.bincount(gid2, minlength=len(games))
chk('per-game n matches', all(int(meta2['games'][gi]['n']) == int(newn[gi]) for gi in range(len(games))))
fields = sorted(f[:-4] for f in os.listdir(SRC) if f.endswith('.npy'))
bad_fields = []
for k in fields:
    a = np.load(os.path.join(SRC, k + '.npy'))[repidx]
    b = np.load(os.path.join(DST, k + '.npy'))
    if not (a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b)):
        bad_fields.append(k)
print('fields compared:', len(fields), 'mismatched:', bad_fields)
chk('all fields row-identical via repidx', not bad_fields)
print('WEIGHT-RECHECK-' + ('PASS' if ok else 'FAIL'))
PYEOF
echo "M320WCHK2225A STUDIO_END"
STEOF
timeout -k 30 600 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.68 'bash -s' < /tmp/m320_wchk.sh
echo "wchk_rc=$?"
echo "M320WCHK2225A L20_END"

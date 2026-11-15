echo "M320XFER2157A L20_BEGIN $(date '+%H:%M:%S')"
cat > /tmp/m320_xfer.sh <<'STEOF'
echo "M320XFER2157A STUDIO_BEGIN $(date '+%F %T %Z')"
D=$HOME/junyang/carbon/data
DGX=arm@10.188.0.23
SSHOPT="-n -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new"
DGXD=/home/arm/junyang/carbon/data
echo '--- small files pull ---'
ssh $SSHOPT $DGX "cat $DGXD/m9xj320_matches.json" > $D/m9xj320_matches.json && echo MATCHES-PULLED
ssh $SSHOPT $DGX "cat $DGXD/m9xj320_manifest.json" > $D/m9xj320_manifest.json && echo MANIFEST-PULLED
ssh $SSHOPT $DGX "cat $DGXD/M9XJ320_READY.txt" > $D/M9XJ320_READY.txt && echo READY-PULLED
echo '--- corpus pull ---'
if [ -d $D/m9xj320_corpus ]; then echo CORPUS-PRESENT; else
  mkdir -p $D/incoming_m320
  ssh $SSHOPT $DGX "tar -C $DGXD -cf - m9xj320_corpus" | tar -C $D/incoming_m320 -xf - && mv $D/incoming_m320/m9xj320_corpus $D/m9xj320_corpus && echo CORPUS-MOVED-INTO-PLACE
fi
echo '--- datasets pull ---'
for ds in ds_m9xj320 ds_m9xj320_w; do
  if [ -d $D/$ds ]; then echo "$ds PRESENT"; else
    mkdir -p $D/incoming_m320
    ssh $SSHOPT $DGX "tar -C $DGXD -cf - $ds" | tar -C $D/incoming_m320 -xf - && mv $D/incoming_m320/$ds $D/$ds && echo "$ds MOVED-INTO-PLACE"
  fi
done
echo '--- assertions ---'
$HOME/junyang/carbon/.venv/bin/python - <<'PYEOF'
import json, os, hashlib
import numpy as np
D = os.path.expanduser('~/junyang/carbon/data')
fails = []
def chk(name, ok, detail=''):
    print(('PASS ' if ok else 'FAIL ') + name + (' ' + str(detail) if detail != '' else ''))
    if not ok: fails.append(name)

man = json.load(open(os.path.join(D, 'm9xj320_manifest.json')))
print('manifest:', {k: man[k] for k in ('freeze_games', 'ds_rows', 'ds_games', 'ds_val_games', 'ds_w_rows')})
matches = json.load(open(os.path.join(D, 'm9xj320_matches.json')))
replays = []
croot = os.path.join(D, 'm9xj320_corpus')
for dirpath, dirs, files in os.walk(croot):
    replays += [f for f in files if f.endswith('.json.gz')]
print('matches entries:', len(matches), 'corpus replays:', len(replays))
chk('matches==corpus==91', len(matches) == 91 and len(replays) == 91)

meta = json.load(open(os.path.join(D, 'ds_m9xj320_w', 'meta.json')))
meta0 = json.load(open(os.path.join(D, 'ds_m9xj320', 'meta.json')))
gw = meta.get('games', []); g0 = meta0.get('games', [])
print('ds_w total:', meta.get('total'), 'games:', len(gw), 'val games:', sum(1 for x in gw if x.get('val')))
print('ds_w weighted:', meta.get('weighted'))
chk('ds_w total==manifest 34232', meta.get('total') == man['ds_w_rows'] == 34232, meta.get('total'))
chk('ds total==manifest 27209', meta0.get('total') == man['ds_rows'] == 27209, meta0.get('total'))
chk('games==91 both', len(gw) == 91 and len(g0) == 91)
chk('val games==13 both', sum(1 for x in gw if x.get('val')) == 13 and sum(1 for x in g0 if x.get('val')) == 13)
chk('uw.npy present', os.path.exists(os.path.join(D, 'ds_m9xj320_w', 'uw.npy')))
step = np.load(os.path.join(D, 'ds_m9xj320_w', 'step.npy'))
chk('step.npy rows==34232', int(step.shape[0]) == 34232, int(step.shape[0]))
smd5 = hashlib.md5(open(os.path.join(D, 'ds_m9xj320_w', 'step.npy'), 'rb').read()).hexdigest()
chk('step.npy md5==manifest', smd5 == man['ds_w_step_md5'], smd5)

# aggregate md5, manifest method: os.walk sorted, md5(relpath + content chunks 1MiB)
root = os.path.join(D, 'ds_m9xj320_w')
h = hashlib.md5(); nfiles = 0
for dirpath, dirs, files in os.walk(root):
    dirs.sort()
    for fn in sorted(files):
        p = os.path.join(dirpath, fn)
        rel = os.path.relpath(p, root)
        h.update(rel.encode())
        with open(p, 'rb') as f:
            for chunk in iter(lambda: f.read(1 << 20), b''):
                h.update(chunk)
        nfiles += 1
print('agg files:', nfiles, 'agg md5 (relpath+chunks):', h.hexdigest())
chk('agg md5==manifest', h.hexdigest() == man['ds_w_agg_md5'], h.hexdigest())

# val byte-identity between ds and ds_w (row-shaped npy files)
chk('game keys order identical', [x['key'] for x in g0] == [x['key'] for x in gw])
chk('val flags identical', [bool(x.get('val')) for x in g0] == [bool(x.get('val')) for x in gw])
def offsets(games):
    off, o = [], 0
    for x in games:
        off.append((o, o + int(x['n']))); o += int(x['n'])
    return off
off0, offw = offsets(g0), offsets(gw)
val_idx = [i for i, x in enumerate(g0) if x.get('val')]
npy_files = sorted(f for f in os.listdir(root) if f.endswith('.npy'))
all_val_ok = True
for fn in npy_files:
    a = np.load(os.path.join(D, 'ds_m9xj320', fn)); b = np.load(os.path.join(root, fn))
    if a.ndim == 0 or b.ndim == 0 or a.shape[0] != meta0['total'] or b.shape[0] != meta['total']:
        print('SKIP non-row file', fn, a.shape, b.shape); continue
    ok = all(np.array_equal(a[off0[i][0]:off0[i][1]], b[offw[i][0]:offw[i][1]]) for i in val_idx)
    print(('PASS ' if ok else 'FAIL ') + 'val bytes ' + fn)
    all_val_ok = all_val_ok and ok
chk('val byte-identical all row files', all_val_ok)

# weighting arithmetic: per train game expected n_w from ds steps + matches margins
step0 = np.load(os.path.join(D, 'ds_m9xj320', 'step.npy'))
by_id = {e['id']: e for e in matches}
TEACHER = 'RyanX-AIPowercore-M9XJ-it320'
bad = []
for i, x in enumerate(g0):
    if x.get('val'):
        if int(gw[i]['n']) != int(x['n']): bad.append((x['key'], 'val-n-changed'))
        continue
    s, e_ = off0[i]
    seg = step0[s:e_]
    e270 = int((seg >= 270).sum())
    m = by_id.get(x['key'])
    if m is None:
        bad.append((x['key'], 'no-match-entry')); continue
    tk = 'a' if m['name_a'] == TEACHER else ('b' if m['name_b'] == TEACHER else None)
    if tk is None:
        bad.append((x['key'], 'teacher-absent')); continue
    ok_ = 'b' if tk == 'a' else 'a'
    margin = m['scores'][tk] - m['scores'][ok_]
    mult = 2 if (m.get('winner') != tk or abs(margin) <= 500) else 1
    expect = (int(x['n']) - e270) * mult + e270 * 3 * mult
    if expect != int(gw[i]['n']):
        bad.append((x['key'], f"expect {expect} got {gw[i]['n']}"))
print('weighting per-game mismatches:', len(bad), bad[:5])
chk('weighting arithmetic per-game', not bad)
print('ALL_CHECKS_PASS' if not fails else 'FAILURES: ' + ','.join(fails))
PYEOF
echo '--- init weight + trainer ---'
ls -la $D/weights/cnet2_arch_n1_last.npz 2>/dev/null || echo INIT-MISSING
ls $HOME/junyang/carbon/code/cnet2_train.py >/dev/null && echo TRAINER-SCRIPT-OK
echo "M320XFER2157A STUDIO_END"
STEOF
timeout -k 30 1500 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.68 'bash -s' < /tmp/m320_xfer.sh
echo "xfer_rc=$?"
echo "M320XFER2157A L20_END"

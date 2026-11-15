echo "__MARK__ L20_BEGIN $(date '+%H:%M:%S')"
cat > /tmp/m9wr1_xfer2.sh <<'STEOF'
echo "__MARK__ STUDIO_BEGIN $(date '+%T')"
D=$HOME/junyang/carbon/data
DGX=arm@10.188.0.23
SSHOPT="-n -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new"
echo '--- matches pull ---'
if ! $HOME/junyang/carbon/.venv/bin/python -c "import json;d=json.load(open('$D/m9xj_matches.json'));assert len(d)==97" 2>/dev/null; then
  ssh $SSHOPT $DGX 'cat /home/arm/junyang/carbon/data/m9xj_matches.json' > $D/m9xj_matches.json && echo MATCHES-PULLED
else
  echo MATCHES-ALREADY-OK
fi
echo '--- dataset pull (if still absent) ---'
if [ -d $D/ds_m9xj_w ]; then
  echo DS_W_PRESENT
else
  mkdir -p $D/incoming_m9wr1
  ssh $SSHOPT $DGX 'tar -C /home/arm/junyang/carbon/data -cf - ds_m9xj_w' | tar -C $D/incoming_m9wr1 -xf - && echo DS-PULLED-TO-STAGING
  if [ -d $D/ds_m9xj_w ]; then
    echo PUSH-ARRIVED-DURING-PULL-KEEPING-THEIRS
  else
    mv $D/incoming_m9wr1/ds_m9xj_w $D/ds_m9xj_w && echo DS-MOVED-INTO-PLACE
  fi
fi
echo '--- assertions ---'
$HOME/junyang/carbon/.venv/bin/python - <<'PYEOF'
import json, os, hashlib
import numpy as np
D = os.path.expanduser('~/junyang/carbon/data')
matches = json.load(open(os.path.join(D, 'm9xj_matches.json')))
replays = []
croot = os.path.join(D, 'm9xj_corpus')
for dirpath, dirs, files in os.walk(croot):
    replays += [f for f in files if f.endswith('.json.gz')]
print('matches entries:', len(matches))
print('corpus replays:', len(replays))
print('ASSERT replays==matches==97:', len(matches) == 97 and len(replays) == 97)
meta = json.load(open(os.path.join(D, 'ds_m9xj_w', 'meta.json')))
print('ds_w total:', meta.get('total'), 'games:', len(meta.get('games', [])), 'val games:', sum(1 for x in meta.get('games', []) if x.get('val')))
print('ds_w weighted:', meta.get('weighted'))
print('uw.npy present:', os.path.exists(os.path.join(D, 'ds_m9xj_w', 'uw.npy')))
step = np.load(os.path.join(D, 'ds_m9xj_w', 'step.npy'))
print('step.npy rows:', int(step.shape[0]))
root = os.path.join(D, 'ds_m9xj_w')
h = hashlib.md5(); n = 0
for dirpath, dirs, files in os.walk(root):
    dirs.sort()
    for fn in sorted(files):
        p = os.path.join(dirpath, fn)
        rel = os.path.relpath(p, root)
        fh = hashlib.md5()
        with open(p, 'rb') as f:
            for chunk in iter(lambda: f.read(1 << 20), b''):
                fh.update(chunk)
        h.update((rel + ' ' + fh.hexdigest() + '\n').encode()); n += 1
print('STUDIO_DS_W_FILES:', n)
print('STUDIO_DS_W_AGG_MD5:', h.hexdigest())
PYEOF
echo '--- init weight check ---'
ls -la $D/weights/cnet2_arch_n1_last.npz 2>/dev/null || echo INIT-MISSING
ls $HOME/junyang/carbon/code/cnet2_train.py >/dev/null && echo TRAINER-SCRIPT-OK
echo "__MARK__ STUDIO_END"
STEOF
timeout -k 30 600 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.68 'bash -s' < /tmp/m9wr1_xfer2.sh
echo "xfer_rc=$?"
echo "__MARK__ L20_END"

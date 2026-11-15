#!/bin/bash
# M9XJ-it320 single chain v2 (runs ON DGX, detached): sweep ->
# WAIT-FREEZE poll (repo pull loop; freeze iff diff==0 AND games>=90 AND echo2_h2h==6;
# if diff==0 AND games>=90 AND echo2!=6 -> NEEDS-RULING state + exit, no freeze) ->
# sync+index -> replica_chain.sh (build ds_m9xj320 + dual-arm launch + attestation) ->
# build ds_m9xj320_w (same weighting as it260) -> smoke ->
# manifest + READY marker for Studio LAN pull -> final attestation.
# Marker: M320N90A
MARK=M320N90A
TEACHER='RyanX-AIPowercore-M9XJ-it320'
REPO=/home/arm/junyang/carbon-repo
WORK=/home/arm/junyang/carbon
DATA=$WORK/data
LOGD=$DATA/dq_logs; mkdir -p "$LOGD"
STATE=$DATA/m9xj320_chain.state
echo "$MARK CHAIN-BEGIN $(date '+%F %T %Z') host=$(hostname)"
set -u
fail(){ echo "$MARK FAIL $1 $(date '+%F %T %Z')"; echo "FAIL $1" > "$STATE"; exit 1; }
stage(){ echo "$MARK STAGE $1 $(date '+%F %T %Z')"; echo "$1" > "$STATE"; }

stage SWEEP
ps -eo pid,etimes,args | grep -E 'git (fetch|pull)' | grep -v grep || echo NO-GIT-LEFTOVER
ps -eo pid,etimes,args | grep -E 'git (fetch|pull)' | grep -v grep | awk '$2>600{print $1}' | while read p; do echo "killing stale git pid $p"; kill "$p" 2>/dev/null || true; done
pgrep -f 'cnet2_train[.]py' && fail train-already-running || echo NO-TRAIN-PROC
pgrep -f 'm320n_chain[.]sh' | grep -v $$ || echo NO-SELF-DUP
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader || true

stage WAIT-FREEZE
cd "$REPO" || fail no-repo
git checkout cnet2-search-v12 2>&1 | tail -1
DEADLINE=$(( $(date +%s) + 6000 ))
FREEZE_N=
while :; do
  timeout 60 git fetch origin cnet2-search-v12 || echo FETCH-FAIL-CONTINUE
  timeout 60 git pull --ff-only || echo PULL-FAIL-CONTINUE
  read DIFF N E2 <<< $(python3 - <<'PYEOF'
import json, os
try:
    m=json.load(open('data/m9xj320_matches.json'))
    files=[f for f in os.listdir('data/m9xj320_corpus') if f.endswith('.json.gz')]
    ids={e['id'] for e in m}; fids={f.rpartition('-')[0] for f in files}
    e2=sum(1 for e in m if 'Echo2' in (e.get('name_a') or '') or 'Echo2' in (e.get('name_b') or ''))
    print(len(ids-fids)+len(fids-ids), len(m), e2)
except Exception as ex:
    print('ERR', ex)
    print(-1, -1, -1)
PYEOF
)
  echo "$MARK CHECK $(date '+%F %T %Z') diff=$DIFF games=$N echo2=$E2 head=$(git rev-parse --short HEAD 2>/dev/null)"
  if [ "$DIFF" = "0" ] && [ "$N" -ge 90 ] 2>/dev/null; then
    if [ "$E2" = "6" ]; then FREEZE_N=$N; break; fi
    echo "$MARK NEEDS-RULING echo2=$E2 games=$N diff=0 $(date '+%F %T %Z')"
    echo "NEEDS-RULING echo2=$E2 games=$N" > "$STATE"
    exit 3
  fi
  [ $(date +%s) -ge $DEADLINE ] && fail wait-freeze-timeout
  sleep 120
done
HEADFULL=$(git rev-parse HEAD); HEADSHORT=$(git rev-parse --short HEAD)
echo "HEAD $HEADFULL"
echo "$MARK FREEZE games=$FREEZE_N echo2=6 commit=$HEADSHORT"
python3 - <<'PYEOF'
import json
m=json.load(open('data/m9xj320_matches.json'))
for e in m:
    if 'Echo2' in (e.get('name_a') or '') or 'Echo2' in (e.get('name_b') or ''):
        print('ECHO2-GAME', e['id'], 'game', e.get('game'), e.get('name_a'), 'vs', e.get('name_b'), 'scores', e.get('scores'), 'winner', e.get('winner'))
PYEOF

stage SYNC-INDEX
rm -rf "$DATA/m9xj320_corpus"
mkdir -p "$DATA/m9xj320_corpus/replays"
cp "$REPO/data/m9xj320_matches.json" "$DATA/m9xj320_matches.json" || fail cp-matches
cp "$REPO"/data/m9xj320_corpus/*.json.gz "$DATA/m9xj320_corpus/replays/" || fail cp-corpus
python3 - <<'PYEOF' || fail index-build
import json, gzip, os
DATA='/home/arm/junyang/carbon/data'
matches=json.load(open(f'{DATA}/m9xj320_matches.json'))
by_id={}
for e in matches:
    assert e['id'] not in by_id, 'dup id '+e['id']
    by_id[e['id']]=e
files=sorted(os.listdir(f'{DATA}/m9xj320_corpus/replays'))
assert len(files)==len(matches), f'files {len(files)} != matches {len(matches)}'
index=[]; seen=set(); short=[]
for fn in files:
    stem=fn[:-len('.json.gz')]; gid,_,gnum=stem.rpartition('-')
    e=by_id.get(gid); assert e is not None, 'orphan replay '+fn
    assert int(gnum)==int(e['game']), 'game mismatch '+fn
    seen.add(gid)
    steps=len(json.load(gzip.open(f'{DATA}/m9xj320_corpus/replays/{fn}'))['steps'])-1
    if steps<200: short.append((gid,steps))
    index.append({'id':e['id'],'game':e['game'],'order':e['order'],'scores':e['scores'],
      'winner':e.get('winner'),'steps':steps,'file':f'replays/{fn}','created':e.get('created'),
      'strategies':{'a':{'id':'','name':e['name_a']},'b':{'id':'','name':e['name_b']}}})
assert seen==set(by_id), 'matches without replay remain'
json.dump(index, open(f'{DATA}/m9xj320_corpus/index.json','w'), ensure_ascii=False)
print('INDEX-BUILT entries=', len(index))
print('SHORT-GAMES steps<200:', short)
print('ENTRY0-STRATEGIES', json.dumps(index[0]['strategies'], ensure_ascii=False))
PYEOF
RP=$(ls "$DATA"/m9xj320_corpus/replays/*.json.gz | wc -l)
MT=$(python3 -c "import json;print(len(json.load(open('$DATA/m9xj320_matches.json'))))")
IX=$(python3 -c "import json;print(len(json.load(open('$DATA/m9xj320_corpus/index.json'))))")
echo "COUNTS replays=$RP matches=$MT index=$IX freeze=$FREEZE_N"
[ "$RP" = "$FREEZE_N" ] && [ "$MT" = "$FREEZE_N" ] && [ "$IX" = "$FREEZE_N" ] || fail "counts-$RP-$MT-$IX"

if [ -d "$DATA/ds_m9xj320" ]; then mv "$DATA/ds_m9xj320" "$DATA/ds_m9xj320_prescrap_$(date +%H%M)"; echo MOVED-OLD-DS-ASIDE; fi

stage REPLICA-CHAIN
cp "$REPO/code/replica_chain.sh" "$WORK/code/replica_chain.sh" || fail cp-chain
md5sum "$WORK/code/replica_chain.sh"
cd "$WORK/code" || fail no-code
PREFIX=m9xj320 TEACHER="$TEACHER" CORPUS=$DATA/m9xj320_corpus MATCHES=$DATA/m9xj320_matches.json DS=$DATA/ds_m9xj320 EXPECTED=$FREEZE_N bash ./replica_chain.sh > "$LOGD/replica_chain_m9xj320.log" 2>&1
RC=$?
tail -30 "$LOGD/replica_chain_m9xj320.log"
[ "$RC" = 0 ] || fail "replica-chain-rc$RC"
python3 - <<'PYEOF'
import json
DATA='/home/arm/junyang/carbon/data'
meta=json.load(open(f'{DATA}/ds_m9xj320/meta.json'))
g=meta['games']
print("DS-M9XJ320 games=", len(g), "samples=", meta['total'], "val_games=", sum(1 for x in g if x['val']))
print("DS-GAME0-KEYS", sorted(g[0].keys()))
PYEOF

stage BUILD-W
python3 - <<'PYEOF' || fail buildw-transform
src=open('/home/arm/junyang/carbon/code/build_m9xj_w.py').read()
assert "data/ds_m9xj'" in src and "data/ds_m9xj_w'" in src
src=src.replace("data/ds_m9xj_w'","data/ds_m9xj320_w'").replace("data/ds_m9xj'","data/ds_m9xj320'")
src=src.replace("'from':'ds_m9xj'","'from':'ds_m9xj320'")
open('/home/arm/junyang/carbon/code/build_m9xj320_w.py','w').write(src)
print('BUILD-W-SCRIPT-WRITTEN')
PYEOF
grep -E '^(SRC|DST)=' "$WORK/code/build_m9xj320_w.py"
"$WORK/.venv/bin/python" "$WORK/code/build_m9xj320_w.py" || fail build-w

stage SMOKE
cd "$WORK/code" || fail no-code2
BC_DEVICE=cuda BC_RAM=1 BC_THREADS=5 BC_USE_W=1 BC_USE_UW=1 BC_BS=256 BC_SAVE_EPOCHS=1 BC_PCT=0.1 BC_LR=3e-4 BC_SUBSET=1024 BC_INIT=$DATA/weights/cnet2_arch_n1_last.npz timeout 900 ../.venv/bin/python cnet2_train.py "$DATA/ds_m9xj320_w" "$DATA/weights/smoke_m9xj320_w.npz" 1 '{"C":128,"NB":8,"D":128,"ZI":1,"TA":4,"TNH":4}' > "$LOGD/smoke_m9xj320_w.log" 2>&1
if grep -qm1 '"ep": 0' "$LOGD/smoke_m9xj320_w.log"; then echo SMOKE-OK; else echo SMOKE-FAILED; tail -15 "$LOGD/smoke_m9xj320_w.log"; fi
rm -f "$DATA"/weights/smoke_m9xj320_w*.npz

stage MANIFEST-READY
python3 - <<PYEOF
import json, os, hashlib, time
DATA='$DATA'
def agg(root):
    h=hashlib.md5()
    for dp,dn,fn in os.walk(root):
        dn.sort()
        for f in sorted(fn):
            p=os.path.join(dp,f); h.update(os.path.relpath(p,root).encode())
            with open(p,'rb') as fh:
                for chunk in iter(lambda: fh.read(1<<20), b''): h.update(chunk)
    return h.hexdigest()
def rows(ds):
    meta=json.load(open(os.path.join(ds,'meta.json')))
    return meta['total'], sum(1 for g in meta['games'] if g['val']), len(meta['games'])
t,games_val,games_n = rows(f'{DATA}/ds_m9xj320')
tw,_,_ = rows(f'{DATA}/ds_m9xj320_w')
man={'teacher':"$TEACHER",'freeze_games':$FREEZE_N,'freeze_commit':'$HEADFULL',
     'ds':'/home/arm/junyang/carbon/data/ds_m9xj320','ds_rows':t,'ds_games':games_n,'ds_val_games':games_val,
     'ds_w':'/home/arm/junyang/carbon/data/ds_m9xj320_w','ds_w_rows':tw,
     'ds_w_agg_md5':agg(f'{DATA}/ds_m9xj320_w'),
     'ds_w_step_md5':hashlib.md5(open(f'{DATA}/ds_m9xj320_w/step.npy','rb').read()).hexdigest(),
     'agg_method':'os.walk sorted, md5(relpath+content chunks 1MiB)',
     'ready_at':time.strftime('%F %T %Z')}
json.dump(man, open(f'{DATA}/m9xj320_manifest.json','w'), ensure_ascii=False, indent=1)
with open(f'{DATA}/M9XJ320_READY.txt','w') as f:
    f.write('M9XJ320 READY for Studio LAN pull\n')
    for k,v in man.items(): f.write(f'{k}: {v}\n')
print('MANIFEST', json.dumps(man, ensure_ascii=False))
PYEOF
ls -l "$DATA/M9XJ320_READY.txt" "$DATA/m9xj320_manifest.json" || fail manifest-missing

stage ATTEST
echo '--- processes ---'; ps -eo pid,etime,args | grep 'cnet2_train' | grep -v grep || echo NO-TRAIN-PROC
echo '--- gpu ---'; nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader
echo '--- train logs ---'; ls -l --time-style=+%F_%T "$LOGD/m9xj320r1.log" "$LOGD/m9xj320g1.log" 2>/dev/null
tail -2 "$LOGD/m9xj320r1.log" 2>/dev/null; tail -2 "$LOGD/m9xj320g1.log" 2>/dev/null
echo '--- weights ---'; ls -lt --time-style=+%F_%T "$DATA"/weights/cnet2_m9xj320*.npz 2>/dev/null | head -6
echo "$MARK CHAIN-DONE freeze_games=$FREEZE_N commit=$HEADSHORT $(date '+%F %T %Z')"
echo DONE > "$STATE"

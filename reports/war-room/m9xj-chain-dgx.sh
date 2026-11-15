#!/bin/bash
# M9XJ-it260 single-chain on DGX: sweep -> pull -> sync+index -> profile ->
# replica_chain.sh (build ds_m9xj + dual-arm launch + attestation) ->
# build ds_m9xj_w (Line A formal weighting) + validate + smoke ->
# push ds_m9xj_w to Studio -> final attestation. Marker: M9XJCHN2026A
MARK=M9XJCHN2026A
REPO=/home/arm/junyang/carbon-repo
WORK=/home/arm/junyang/carbon
DATA=$WORK/data
LOGD=$DATA/dq_logs; mkdir -p "$LOGD"
STATE=$DATA/m9xj_chain.state
echo "$MARK CHAIN-BEGIN $(date '+%F %T %Z') host=$(hostname)"
set -u
fail(){ echo "$MARK FAIL $1 $(date '+%F %T %Z')"; echo "FAIL $1" > "$STATE"; exit 1; }
stage(){ echo "$MARK STAGE $1 $(date '+%F %T %Z')"; echo "$1" > "$STATE"; }

stage SWEEP
echo '--- leftover git fetch/pull (own line) ---'
ps -eo pid,etimes,args | grep -E 'git (fetch|pull)' | grep -v grep || echo NO-GIT-LEFTOVER
ps -eo pid,etimes,args | grep -E 'git (fetch|pull)' | grep -v grep | awk '$2>600{print $1}' | while read p; do echo "killing stale git pid $p"; kill "$p" 2>/dev/null || true; done
pgrep -af 'cnet2_train' || echo NO-TRAIN-PROC
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader || true

stage PULL
cd "$REPO" || fail no-repo
timeout 60 git fetch origin cnet2-search-v12 || fail git-fetch
git checkout cnet2-search-v12 2>&1 | tail -1
timeout 60 git pull --ff-only || fail git-pull
echo "HEAD $(git rev-parse HEAD)"
cp "$REPO/code/replica_chain.sh" "$WORK/code/replica_chain.sh" || fail cp-chain

stage SYNC
mkdir -p "$DATA/m9xj_corpus/replays"
cp "$REPO/data/m9xj_matches.json" "$DATA/m9xj_matches.json" || fail cp-matches
cp "$REPO"/data/m9xj_corpus/*.json.gz "$DATA/m9xj_corpus/replays/" || fail cp-corpus
python3 - <<'PYEOF' || fail index-build
import json, gzip, os
DATA='/home/arm/junyang/carbon/data'
matches=json.load(open(f'{DATA}/m9xj_matches.json'))
by_id={}
for e in matches:
    assert e['id'] not in by_id, 'dup id '+e['id']
    by_id[e['id']]=e
files=sorted(os.listdir(f'{DATA}/m9xj_corpus/replays'))
assert len(files)==len(matches), f'files {len(files)} != matches {len(matches)}'
index=[]; seen=set()
for fn in files:
    stem=fn[:-len('.json.gz')]; gid,_,gnum=stem.rpartition('-')
    e=by_id.get(gid); assert e is not None, 'orphan replay '+fn
    assert int(gnum)==int(e['game']), 'game mismatch '+fn
    seen.add(gid)
    steps=len(json.load(gzip.open(f'{DATA}/m9xj_corpus/replays/{fn}'))['steps'])-1
    index.append({'id':e['id'],'game':e['game'],'order':e['order'],'scores':e['scores'],
      'winner':e.get('winner'),'steps':steps,'file':f'replays/{fn}','created':e.get('created'),
      'strategies':{'a':{'id':'','name':e['name_a']},'b':{'id':'','name':e['name_b']}}})
assert seen==set(by_id), 'matches without replay remain'
json.dump(index, open(f'{DATA}/m9xj_corpus/index.json','w'), ensure_ascii=False)
print('INDEX-BUILT entries=', len(index))
PYEOF
RP=$(ls "$DATA"/m9xj_corpus/replays/*.json.gz | wc -l)
MT=$(python3 -c "import json;print(len(json.load(open('$DATA/m9xj_matches.json'))))")
IX=$(python3 -c "import json;print(len(json.load(open('$DATA/m9xj_corpus/index.json'))))")
echo "COUNTS replays=$RP matches=$MT index=$IX expected=97"
[ "$RP" = 97 ] && [ "$MT" = 97 ] && [ "$IX" = 97 ] || fail "counts-$RP-$MT-$IX"

stage PROFILE
python3 - <<'PYEOF' || fail profile
import json, gzip, glob, os, math
from statistics import mean, median, pstdev
TEACHER="RyanX-AIPowercore-M9XJ-it260"
REPO='/home/arm/junyang/carbon-repo'; DATA='/home/arm/junyang/carbon/data'
def pct(vals,q):
    s=sorted(vals); k=(len(s)-1)*q/100.0; f=math.floor(k); c=math.ceil(k)
    return s[int(k)] if f==c else s[int(f)]*(c-k)+s[int(c)]*(k-f)
def desc(vals):
    if not vals: return {}
    return {"n":len(vals),"mean":round(mean(vals),2),"median":round(median(vals),2),
            "sd":round(pstdev(vals),2) if len(vals)>1 else 0.0,
            "p25":round(pct(vals,25),1),"p75":round(pct(vals,75),1),"min":min(vals),"max":max(vals)}
matches=json.load(open(f'{REPO}/data/m9xj_matches.json'))
by_id={e["id"]:e for e in matches}
files=sorted(glob.glob(f'{REPO}/data/m9xj_corpus/*.json.gz'))
rows=[]; unmatched=[]; ok=bad=skip=0
for fp in files:
    stem=os.path.basename(fp)[:-len(".json.gz")]; gid,_,_=stem.rpartition("-")
    e=by_id.get(gid)
    if e is None: unmatched.append(stem); continue
    tkey="a" if e["name_a"]==TEACHER else ("b" if e["name_b"]==TEACHER else None)
    if tkey is None: unmatched.append(stem+":noteacher"); continue
    slot=list(e["order"]).index(tkey)
    d=json.load(gzip.open(fp)); st=d["steps"]
    def tset(t,s):
        c=st[t][0]["observation"]["players"][s][3]
        return set(c.keys()) if isinstance(c,dict) else set()
    A=[tset(t,slot) for t in range(len(st))]; B=[tset(t,1-slot) for t in range(len(st))]
    plants=seized=0
    for t in range(len(st)-1):
        newA=A[t+1]-A[t]; plants+=len(newA-B[t]); seized+=len(newA & B[t])
    sc=e["scores"]; margin=round(sc[tkey]-sc["b" if tkey=="a" else "a"],1)
    won=e.get("winner")==tkey
    rw=d.get("rewards")
    if isinstance(rw,(list,tuple)) and len(rw)==2 and all(isinstance(x,(int,float)) for x in rw):
        rs=(rw[slot]>rw[1-slot])-(rw[slot]<rw[1-slot]); ms=(margin>0)-(margin<0)
        if rs==0 or ms==0: skip+=1
        elif rs==ms: ok+=1
        else: bad+=1
    else: skip+=1
    rows.append({"id":gid[:8],"slot":slot,"plants":plants,"seized":seized,
                 "residual":len(A[-1]),"treesteps":sum(len(a) for a in A),"margin":margin,"won":won})
out={"games":len(rows),"unmatched":unmatched,"reward_sign_ok_bad_skip":[ok,bad,skip],
     "plants_all":desc([r["plants"] for r in rows]),"residual_all":desc([r["residual"] for r in rows]),
     "seized_all":desc([r["seized"] for r in rows]),"treesteps_all":desc([r["treesteps"] for r in rows]),
     "margin_all":desc([r["margin"] for r in rows]),
     "teacher_record":{"W":sum(1 for r in rows if r["won"]),"L":sum(1 for r in rows if not r["won"])}}
for s in (0,1):
    sub=[r for r in rows if r["slot"]==s]
    out[f"slot{s}"]={"n":len(sub),"plants":desc([r["plants"] for r in sub]),"residual":desc([r["residual"] for r in sub]),
                     "winrate":round(sum(1 for r in sub if r["won"])/len(sub),3) if sub else None}
json.dump(out, open(f'{DATA}/m9xj_profile97.json','w'), ensure_ascii=False)
print("PROFILE-RESULT", json.dumps(out, ensure_ascii=False))
PYEOF

stage REPLICA-CHAIN
cd "$WORK/code" || fail no-code
PREFIX=m9xj260 TEACHER='RyanX-AIPowercore-M9XJ-it260' CORPUS=$DATA/m9xj_corpus MATCHES=$DATA/m9xj_matches.json DS=$DATA/ds_m9xj EXPECTED=97 bash ./replica_chain.sh > "$LOGD/replica_chain_m9xj260.log" 2>&1
RC=$?
tail -25 "$LOGD/replica_chain_m9xj260.log"
[ "$RC" = 0 ] || fail "replica-chain-rc$RC"
python3 - <<'PYEOF'
import json
DATA='/home/arm/junyang/carbon/data'
idx=json.load(open(f'{DATA}/m9xj_corpus/index.json'))
dropped=[(g['id'],g['steps']) for g in idx if g.get('steps',0)<200]
print("SCAN-DROPPED steps<200:", dropped)
meta=json.load(open(f'{DATA}/ds_m9xj/meta.json'))
print("DS-M9XJ games=", len(meta['games']), "samples=", meta['total'], "val_games=", sum(1 for g in meta['games'] if g['val']))
PYEOF

stage BUILD-W
cat > "$WORK/code/build_m9xj_w.py" <<'PYEOF'
import json, os, sys, hashlib
import numpy as np
SRC='/home/arm/junyang/carbon/data/ds_m9xj'; DST='/home/arm/junyang/carbon/data/ds_m9xj_w'
# Stacking rule (stated): factors MULTIPLY. Train row factor = 3 if step>=270,
# times 2 if its side lost the game or the game margin (either side) is <=500.
# A row hit by both rules appears 6 times. Validation games: factor 1, untouched.
meta=json.load(open(os.path.join(SRC,'meta.json'))); games=meta['games']
gid=np.load(os.path.join(SRC,'gid.npy')); step=np.load(os.path.join(SRC,'step.npy'))
N=len(gid); factors=np.ones(N,np.int64)
rep={'end_only':0,'lc_only':0,'both':0,'plain':0,'val_rows':0,'train_rows':0,
     'end_rows_train':0,'lost_games':0,'close_games':0,'val_games':0}
is_val=np.zeros(N,bool)
for gi,g in enumerate(games):
    idx=np.nonzero(gid==gi)[0]
    if len(idx)==0: continue
    if g['val']:
        is_val[idx]=True; rep['val_rows']+=len(idx); rep['val_games']+=1; continue
    rep['train_rows']+=len(idx)
    st=step[idx].astype(np.int64)
    seg=np.concatenate([[0],np.cumsum(np.diff(st)<0)])
    sides=g['sides']; scores=g['scores']
    gmargin=abs(float(scores[0])-float(scores[1]))
    side_margins={s: float(scores[s])-float(scores[1-s]) for s in sides}
    if any(m<0 for m in side_margins.values()): rep['lost_games']+=1
    if gmargin<=500: rep['close_games']+=1
    for j,i in enumerate(idx):
        s=sides[int(seg[j])]; end=bool(st[j]>=270)
        lc=(side_margins[s]<0) or (gmargin<=500)
        if end: rep['end_rows_train']+=1
        f=(3 if end else 1)*(2 if lc else 1); factors[i]=f
        if end and lc: rep['both']+=1
        elif end: rep['end_only']+=1
        elif lc: rep['lc_only']+=1
        else: rep['plain']+=1
repidx=np.repeat(np.arange(N),factors); NN=len(repidx)
os.makedirs(DST,exist_ok=True)
fields=sorted(f[:-4] for f in os.listdir(SRC) if f.endswith('.npy'))
for k in fields:
    a=np.load(os.path.join(SRC,k+'.npy')); np.save(os.path.join(DST,k+'.npy'), a[repidx])
# meta: same games/val/teachers; per-game n and total updated to expanded counts
newn=np.bincount(gid[repidx], minlength=len(games))
meta2=json.loads(json.dumps(meta)); meta2['total']=int(NN)
for gi,g in enumerate(meta2['games']): g['n']=int(newn[gi])
meta2['weighted']={'from':'ds_m9xj','rule':'train rows: x3 if step>=270, x2 if side lost or |game margin|<=500, multiplicative (both=x6); val games untouched'}
json.dump(meta2, open(os.path.join(DST,'meta.json'),'w'))
# validation: val rows byte-identical (same order, same bytes) in every field
gid2=np.load(os.path.join(DST,'gid.npy'))
valg=np.array([g['val'] for g in games]); is_val2=valg[gid2]
assert int(is_val2.sum())==int(is_val.sum()), 'val row count changed'
for k in fields:
    a=np.load(os.path.join(SRC,k+'.npy'))[is_val]; b=np.load(os.path.join(DST,k+'.npy'))[is_val2]
    assert a.shape==b.shape and a.dtype==b.dtype and np.array_equal(a,b), 'val mismatch field '+k
    h=hashlib.md5(b.tobytes()).hexdigest()
    if k in ('y','step','gid'): rep[f'val_md5_{k}']=h
rep['rows_before']=int(N); rep['rows_after']=int(NN)
rep['train_rows_after']=int(NN-rep['val_rows'])
rep['end_rows_train_after']=int(rep['end_only']*3+rep['both']*6)
print('WEIGHTED-REPORT', json.dumps(rep))
print('VAL-IDENTICAL-OK fields=', len(fields))
PYEOF
"$WORK/.venv/bin/python" "$WORK/code/build_m9xj_w.py" || fail build-w

stage SMOKE
cd "$WORK/code" || fail no-code2
BC_DEVICE=cuda BC_RAM=1 BC_THREADS=5 BC_USE_W=1 BC_USE_UW=1 BC_BS=256 BC_SAVE_EPOCHS=1 BC_PCT=0.1 BC_LR=3e-4 BC_SUBSET=1024 BC_INIT=$DATA/weights/cnet2_arch_n1_last.npz timeout 900 ../.venv/bin/python cnet2_train.py "$DATA/ds_m9xj_w" "$DATA/weights/smoke_m9xj_w.npz" 1 '{"C":128,"NB":8,"D":128,"ZI":1,"TA":4,"TNH":4}' > "$LOGD/smoke_m9xj_w.log" 2>&1
grep -m1 '"ep": 0' "$LOGD/smoke_m9xj_w.log" || { tail -15 "$LOGD/smoke_m9xj_w.log"; fail smoke; }
echo SMOKE-OK
rm -f "$DATA"/weights/smoke_m9xj_w*.npz

stage PUSH-STUDIO
AGG='import hashlib,os,sys
root=sys.argv[1]; h=hashlib.md5()
for dp,dn,fn in os.walk(root):
    dn.sort()
    for f in sorted(fn):
        p=os.path.join(dp,f); h.update(os.path.relpath(p,root).encode())
        with open(p,"rb") as fh:
            for chunk in iter(lambda: fh.read(1<<20), b""): h.update(chunk)
print(h.hexdigest())'
DGXAGG=$("$WORK/.venv/bin/python" -c "$AGG" "$DATA/ds_m9xj_w")
echo "DSW-DGX-AGGMD5 $DGXAGG"
PUSHOK=0
for H in 10.211.43.68; do
  if timeout -k 10 60 ssh -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new arm@$H 'echo STUDIO-REACHABLE $(hostname)' 2>&1; then
    tar cf - -C "$DATA" ds_m9xj_w | timeout -k 30 600 ssh -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new arm@$H 'mkdir -p ~/junyang/carbon/data && tar xf - -C ~/junyang/carbon/data' && PUSHOK=1 && echo "PUSH-OK via $H"
    if [ "$PUSHOK" = 1 ]; then
      STAGG=$(timeout -k 10 120 ssh -o BatchMode=yes -o ConnectTimeout=15 arm@$H "~/junyang/carbon/.venv/bin/python -c \"$AGG\" ~/junyang/carbon/data/ds_m9xj_w" 2>&1 | tail -1)
      echo "DSW-STUDIO-AGGMD5 $STAGG"
      [ "$STAGG" = "$DGXAGG" ] && echo DSW-MD5-MATCH || echo DSW-MD5-MISMATCH
    fi
    break
  fi
done
[ "$PUSHOK" = 1 ] || echo "$MARK PUSH-FAILED ds_m9xj_w NOT delivered to Studio (needs fallback local build)"

stage ATTEST
echo '--- processes ---'; ps -eo pid,etime,args | grep 'cnet2_train' | grep -v grep || echo NO-TRAIN-PROC
echo '--- gpu ---'; nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader
echo '--- train logs ---'; ls -l --time-style=+%F_%T "$LOGD/m9xj260r1.log" "$LOGD/m9xj260g1.log" 2>/dev/null
tail -2 "$LOGD/m9xj260r1.log" 2>/dev/null; tail -2 "$LOGD/m9xj260g1.log" 2>/dev/null
echo '--- weights ---'; ls -lt --time-style=+%F_%T "$DATA"/weights/cnet2_m9xj260*.npz 2>/dev/null | head -6
echo "$MARK CHAIN-DONE $(date '+%F %T %Z')"
echo DONE > "$STATE"

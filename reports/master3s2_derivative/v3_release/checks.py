"""Root-public replay of the frozen v2/v3 candidates. Never supply the opponent's current actual order."""
import copy
import gzip
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
D=HERE.parent
sys.path.insert(0,str(D.parent/"sapling_1200k/all_top5"))
from context import Game

SOURCES=[
    ("Fable v7", "bdbe59b04f71e43e53696874c8fd5958",1,[152]),
    ("doubao placement","4c3bc230e7cc588cf3dc70dbf1dfa8ae",0,[215,218]),
    ("winning Hajimi rematch","cd651c5cbf982d68df4fe2f5633398d6",1,[127,269])
]
FOCAL={"Fable v7":("player-1-worker-28","player-0-worker-27"),"doubao placement":("player-0-worker-44","player-1-worker-44")}


def sha(raw):return hashlib.sha256(raw).hexdigest()
def h(obj):return sha(json.dumps(obj,sort_keys=True,separators=(",",":"),allow_nan=False).encode())
def load(name,p):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    m._torch.set_num_threads(1);return m
def save(p,x):
    raw=(json.dumps(x,ensure_ascii=False,allow_nan=False,sort_keys=True,separators=(",",":"))+"\n").encode()
    p.write_bytes(gzip.compress(raw,mtime=0) if p.suffix==".gz" else raw)


def run():
    receipt=json.loads((HERE/'manifest.json').read_bytes())
    old_path=D/"ladder/prediction_feedback/package/candidate.py"
    assert sha(old_path.read_bytes())==receipt['parent_sha256']
    assert sha((HERE/"submission.py").read_bytes())==receipt['candidate_sha256']
    parent=load("old_v2",old_path);v3=load("new_v3",HERE/"submission.py")
    orig=v3._v3_intercept; details=[]
    def capture(players,me,previous,step,base,protected,logits,pred):
        outcome=orig(players,me,previous,step,base,protected,logits,pred)
        details.append(dict(v2_predicted=dict(pred),protected=list(protected),native_A0=dict(base),
                          raw_tactical=copy.deepcopy(outcome),public_enemies={u:dict(state=w,v2_probs=[round(float(x),5) for x in v3._probs(logits[w[0]])]) for u,w in players[1-me][2].items()}))
        return outcome
    v3._v3_intercept=capture
    rows=[]
    for name,gid,me,horizon in SOURCES:
        file=D/'ladder/replays'/(gid+'-1.json.gz')
        if not file.exists():file=D/'ladder/placement_refresh/replays'/(gid+'-1.json.gz')
        raw=file.read_bytes();game=json.loads(gzip.decompress(raw));tape=game['steps'];cfg=game['configuration'];visible=Game(game)
        track=parent.BeliefTracker(parent._Bel,me);prev=None
        for step in range(max(horizon)+1):
            obs=visible._obs(step,me); assert 'full_carbon' not in obs
            if step in horizon:
                seed=int.from_bytes(hashlib.sha256((gid+str(step)).encode()).digest()[:4],"little")
                out=dict(name=name,id=gid,seat=me,step=step,root_seed=seed,legacy_source=sha(raw),public_sha=h(obs),runs=[])
                for tag in ("v2","v3","v3","v2"):
                    m=parent if tag=='v2' else v3
                    m._ST=dict(tr=copy.deepcopy(track),prev=copy.deepcopy(prev),rng=np.random.RandomState(seed))
                    m._S_COST[:]=[.001,.002,.001,.002];m._S_STATS.pop('last',None);m._S_REC=[];details.clear()
                    if tag=="v3":m._V3_AUDIT=[]
                    wt=time.perf_counter();ct=time.process_time();issued=m.agent(obs,cfg)
                    wall=time.perf_counter()-wt;cpu=time.process_time()-ct
                    records=m._S_REC;m._S_REC=None
                    if records:
                        _,_,menu,depths,vals,chosen=records[0]; assert menu[chosen]==issued
                    else:menu=[issued];depths=[];vals=[];chosen=0
                    if tag=='v3':
                        audit=copy.deepcopy(m._V3_AUDIT);m._V3_AUDIT=None
                        for z in audit:
                            if z['index'] is not None:
                                assert len(menu)<=z['limit'] and menu[z['index']]==z['added'] and menu[0]==z['original']
                                assert {u for u in set(z['added'])|set(z['original']) if z['added'].get(u)!=z['original'].get(u)}=={z['worker']}
                    else:audit=[]
                    out['runs'].append(dict(version=tag,wall=wall,cpu=cpu,issued=issued,menu=menu,depths=depths,values=vals,best=chosen,audit=audit,diagnostic=copy.deepcopy(details)))
                rows.append(out)
                print(name,step,[(r['version'],len(r['menu']),r['best'],[(v['worker'],v['enemy'],v['direction'],v['mode'],v['index']) for v in r['audit']]) for r in out['runs']],flush=True)
            own_order=tape[step+1][me]['action'] or {}
            track.step(obs['players'],step,obs['carbon']);track.set_actions(obs['players'],own_order);prev=obs['players']
    save(HERE/"checks.json.gz",dict(candidate=receipt['candidate_sha256'],parent=receipt['parent_sha256'],affinity=sorted(os.sched_getaffinity(0)),rows=rows))
if __name__=="__main__":run()

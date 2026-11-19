"""Rebuild signed inputs independently of fitting and test that hidden carbon cannot affect them."""
from collections import Counter,defaultdict
import hashlib
import numpy as np
import model
from fit import replay
from sources import HERE,read,save,sha

def run():
    manifest=read(HERE/'sources.json')['records']
    rows=read(HERE/'decisions.json.gz')
    by=defaultdict(dict)
    for i,r in enumerate(rows):
        key=(r['t'],r['uid'])
        assert key not in by[r['game']]
        by[r['game']][key]=(i,r)
    inputs=np.empty((len(rows),5,len(model.FEATURES)),dtype=np.float32)
    found=set()
    tests=Counter()
    for g,rec in enumerate(manifest):
        sim,pairs,end,_=replay(rec);side=rec['seat'];prev={}
        for t,orders in enumerate(pairs):
            if t>=200:break
            public=sim.obs(side);workers=public['players'][side][2]
            if t>=100:
                visible=set()
                for pos,load,kind in workers.values():
                    visible.update([pos] if kind=='COLLECTOR' else model.DEST[pos,1:])
                forged=dict(public,carbon=[v if k in visible else 9342.125 for k,v in enumerate(public['carbon'])])
                for uid,(pos,load,kind) in workers.items():
                    if kind!='PLANTER':continue
                    i,r=by[g][t,uid]
                    actual=model.features(public,uid,sim.config,prev.get(uid))
                    assert hashlib.sha256(actual.tobytes()).hexdigest()==r['feature_sha256']
                    assert r['a']==model.MOVES.index(orders[side].get(uid) or 'NONE') and r['cell']==pos
                    inputs[i]=actual
                    assert i not in found;found.add(i)
                    if t%11==0:
                        with_forgery=model.features(forged,uid,sim.config,prev.get(uid))
                        assert np.array_equal(actual,with_forgery)
                        tests['hidden_perturbations_tested']+=1
            prev={k:v[0] for k,v in workers.items()}
            sim.step(*orders)
    assert len(found)==len(rows) and tests['hidden_perturbations_tested']>0
    np.savez_compressed(HERE/'inputs.npz',features=inputs)
    buy=set();tags=0
    for r in rows:
        if r['target']!=2:continue
        lot=r['lot']
        assert lot['kind']=='plant' and lot['net']>0 and abs(lot['net']-(lot['absorption']+lot['collision']-lot['cost']))<1e-6
        buy.add((r['game'],lot['t'],lot['uid'],lot['tree_id']));tags+=1
    save(HERE/'verification.json',dict(rows=len(rows),action_names=model.MOVES,columns=model.FEATURES,full_inputs_sha256=sha(HERE/'inputs.npz'),
        exact_match_input_hashes=len(found),**tests,positive_label_rows=tags,unique_positive_purchases_linked= len(buy),
        decision_order_sha256=sha(HERE/'decisions.json.gz'),script_sha256=sha(HERE/'verify_inputs.py')))
if __name__=='__main__':run()

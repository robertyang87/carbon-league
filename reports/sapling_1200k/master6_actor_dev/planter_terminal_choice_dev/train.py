"""Train a conservative two-action selector on actual terminal-margin differences."""
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
TREE=HERE.parent/'tree_response_training'
FEATURES=['stay','payment','edge_carbon','tree_life','exposure_1','exposure_2',
          'own_cargo','enemy_cargo','own_depot_return','enemy_depot_return','remaining_x_pay','cash_x_pay']

def read(p):
    data=p.read_bytes()
    return json.loads(gzip.decompress(data) if p.suffix=='.gz' else data)

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def dist(a,b):
    y,x=abs(a//15-b//15),abs(a%15-b%15)
    return min(y,15-y)+min(x,15-x)

def features(root,cmd):
    ob=root['public_at_root']; me=ob['player'];players=ob['players']
    a=root['candidate']['actions'][cmd];cell=a['cell']
    pay=int(a['can_pay_now']);tree=a['live_tree_now'];remaining=a['remaining_transitions']
    enemy_distance=min((dist(cell,w[0]) for w in players[1-me][2].values()),default=99)
    cargo=[];returns=[]
    for seat in (me,1-me):
        bases=list(players[seat][1].values()); nearby=0.; inbound=0.
        for pos,load,kind in players[seat][2].values():
            if kind!='COLLECTOR':continue
            weight=max(0,1-dist(cell,pos)/4)
            nearby+=weight*load
            inbound+=weight*load/(1+min(dist(pos,b) for b in bases))
        cargo.append(nearby);returns.append(inbound)
    # Count only cells known to this seat in the pre-action observation.
    visible=sum((ob['carbon'][c] or 0) for c in range(225) if dist(c,cell)==1)
    return np.array([cmd=='NONE',(a['price'] if pay else 0),visible,(50-tree['age']) if tree else 0,
                     enemy_distance<=1,enemy_distance==2,*cargo,*returns,remaining*pay,
                     np.log1p(players[me][0])*pay],dtype=float)

def dataset():
    groups=[TREE,*[TREE/f'batches/intake_{i:02}' for i in range(2,6)],
            *[TREE/f'supplement_01/batches/intake_{i:02}' for i in range(6,9)]]
    rows=[]
    for i,group in enumerate(groups,1):
        cases=sorted(group.glob('case_*/root.json.gz'));assert len(cases)==8
        for p in cases:
            root=read(p);case=p.parent;cfg=read(case/'live_protocol.json')
            scores={};joint={};stamps=[]
            for arm in ('parent','alternative'):
                rec=read(case/f'live_{arm}.json');tape=case/rec['action_tape']
                assert sha(tape)==rec['action_sha256'] and rec['statuses']==['DONE','DONE']
                assert rec['protocol_sha256']==sha(case/'live_protocol.json')
                assert rec['source_sha256']=='65f268b2919ce30fe19355115a928eada032a6beaa018d801662d434c083f2b5'
                joint[arm]=read(tape)[0];s=cfg['seat'];scores[arm]=rec['score'][s]-rec['score'][1-s]
                stamps.append(rec['action_sha256'])
            s=cfg['seat']; uid=cfg['root']['worker']
            assert joint['parent'][1-s]==joint['alternative'][1-s]
            a,b=[joint[arm][s] for arm in ('parent','alternative')]
            assert {k for k in a.keys()|b.keys() if a.get(k)!=b.get(k)}=={uid}
            assert (a.get(uid,'NONE'),b.get(uid,'NONE'))==(root['old'],root['new'])
            rows.append(dict(batch=i,seed=cfg['seed'],seat=s,step=cfg['root_t'],case=str(case.relative_to(TREE)),
                             x=(features(root,root['new'])-features(root,root['old'])).tolist(),
                             observed=scores['alternative']-scores['parent'],tapes=stamps))
    return rows

def main():
    rows=dataset()
    X=np.array([r['x'] for r in rows]); y=np.array([r['observed'] for r in rows])
    seeds=np.array([r['seed'] for r in rows]); train=np.array([r['batch']<=6 for r in rows]); valid=~train
    train_maps=np.unique(seeds[train]); valid_maps=np.unique(seeds[valid])
    assert len(train_maps)==12 and len(valid_maps)==4 and not set(train_maps)&set(valid_maps)
    models=[];means=[];scales=[]
    for excluded in [None,*train_maps]:
        keep=train if excluded is None else train & (seeds!=excluded)
        mu=X[keep].mean(0); sd=X[keep].std(0); sd[sd<1e-8]=1
        z=np.column_stack((np.ones(sum(keep)),(X[keep]-mu)/sd))
        penalty=np.diag([0]+[12]*X.shape[1])
        models.append(np.linalg.solve(z.T@z+penalty,z.T@y[keep]));means.append(mu);scales.append(sd)
    weights=np.array(models);means=np.array(means);scales=np.array(scales)
    np.savez_compressed(HERE/'selector.npz',weights=weights,centers=means,scales=scales,features=FEATURES,
                        excluded=np.array([-1,*train_maps]))
    with np.load(HERE/'selector.npz') as w:
        votes=((X[:,None,:]-w['centers'])/w['scales']*w['weights'][:,1:]).sum(2)+w['weights'][:,0]
    use=votes.min(axis=1)>0
    per_map={str(s):float(np.mean((use*y)[seeds==s])) for s in valid_maps}
    result=dict(per_map=per_map,held_mean=float(np.mean(list(per_map.values()))),held_selected=int(use[valid].sum()),
                held_positive_selected=int(((y>0)&use&valid).sum()),weight_sha256=sha(HERE/'selector.npz'))
    result['gate']=result['held_mean']>0 and sum(v>0 for v in per_map.values())>=3 and result['held_selected']>=2
    for i,r in enumerate(rows):
        r.update(vote_min=float(votes[i].min()),vote_max=float(votes[i].max()),selected=bool(use[i]))
    (HERE/'rows.json.gz').write_bytes(gzip.compress(json.dumps(rows,allow_nan=False).encode(),mtime=0))
    (HERE/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result)
if __name__=='__main__':main()

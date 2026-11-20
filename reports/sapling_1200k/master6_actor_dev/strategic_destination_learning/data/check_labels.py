"""Re-compute every scored worker horizon from an independent simulator and stored physical UIDs."""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
STRATEGIC=HERE.parent
ROOT=next(q for q in HERE.parents if (q/'.git').exists())
sys.path.insert(0,str(ROOT/'code'))
from fastsim import FastCarbon


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def deltas(old,new):
    sy,sx=divmod(old,15);ty,tx=divmod(new,15)
    def short(s,t):
        return min((t-s,t+15-s,t-15-s),key=abs)
    return short(sx,tx),short(sy,ty)


def main():
    info=json.loads((STRATEGIC/'data.json').read_bytes())
    src=json.loads((STRATEGIC/'sources.json').read_bytes())
    cards={c['sha256']:c for c in src['cards']}
    counts={};daily=Counter();group=Counter()
    for s in info['shards']:
        c=cards[s['game']];seat=c['expert_seat'];raw=ROOT/c['path'];npz=STRATEGIC/s['file']
        if sha(raw)!=s['game'] or sha(npz)!=s['sha256']:raise ValueError('Changed input')
        tape=json.load(gzip.open(raw));sim=FastCarbon(tape['game_configuration']).reset(seed=c['seed'],full_carbon=tape.get('initial_full_carbon'))
        if hashlib.sha256(bytes(map(int,sim.carbon))).hexdigest()!=c['terrain']:raise ValueError('Physical terrain mismatch')
        positions=[]
        for t in range(len(tape['actions'])+1):
            workers=sim.obs(seat)['players'][seat][2]
            positions.append({u:(v[0],v[2]) for u,v in workers.items()})
            if t<len(tape['actions']):sim.step(*tape['actions'][t])
        if not np.allclose(sim.cash,tape['last_post_cash'],rtol=0,atol=1e-7):raise ValueError('Terminal cash')
        rows=json.loads(npz.with_suffix('.json').read_bytes())['root_manifest']
        with np.load(npz,allow_pickle=False) as z:
            if not np.array_equal(z['root_step'],c['registered_times']):raise ValueError('Incorrect time or seat')
            for i,row in enumerate(rows):
                t=row['t'];end=t+12;daily['roots']+=1
                if bool(z['window_complete'][i])!=(end<len(positions)):raise ValueError('Censoring mask')
                for j,uid in enumerate(row['uid']):
                    dst=alive=-1
                    if j<row['cut'] and uid in positions[t] and end<len(positions):
                        alive=int(uid in positions[end])
                        if alive:
                            a=positions[t][uid][0];b=positions[end][uid][0];x,y=deltas(a,b)
                            if x==y==0:dst=0
                            elif abs(y)>abs(x):dst=3 if y>0 else 1
                            else:dst=2 if x>0 else 4
                            daily['nonzero_horizontal_ties']+=abs(x)==abs(y) and abs(x)>0
                            daily['wrapped_endpoint_difference']+=abs(a%15-b%15)>7 or abs(a//15-b//15)>7
                        daily['dead_labels' if not alive else 'survivor_labels']+=1
                        group[f"{s['split']}_{seat}_{positions[t][uid][1]}"]+=1
                    if (int(z['survive12'][i,j]), int(z['dst12'][i,j]))!=(alive,dst):
                        raise ValueError((s['game'],t,uid,j,alive,dst))
                    daily['slots_checked']+=1
                daily['fully_censored_roots']+=end>=len(positions)
        counts[s['game']]=len(rows)
    out=dict(schema='sapling-strategic-destination-independent-physical-replay/1',maps=len(src['maps']),verified_tapes=len(counts),
             counts=daily,worker_labels_by_split_seat_role=group,
             sources_sha256=sha(STRATEGIC/'sources.json'),cache_receipt_sha256=sha(STRATEGIC/'data.json'),
             fastsim_sha256=sha(ROOT/'code/fastsim.py'),checker_sha256=sha(Path(__file__)),checks='Both survival and physical direction on all slots, including all masked bases, unrepresentable suffixes, incomplete horizons and disappearing workers. Terminal cash and physical terrain matched independently.')
    (HERE/'label_audit.json').write_text(json.dumps(out,sort_keys=True,indent=2,allow_nan=False)+'\n')
    print(dict(daily))


if __name__=='__main__':main()

"""Screen for covering every forward exit with two light collectors while preserving friendly reservations."""
from itertools import product
from coverage import HERE,sr
from route import _TO,_D,_DIRECTION

def choices(obs,orders):
    s=obs['player']; own,enemy=obs['players'][s],obs['players'][1-s];homes=list(enemy[1].values())
    current={k:int(_TO[w[0],_DIRECTION.index(orders.get(k))]) for k,w in own[2].items()}
    recruits={p for k,p in own[1].items() if orders.get(k)};best=None
    for target,(p,load,kind) in enemy[2].items():
        if kind!='COLLECTOR' or load<100:continue
        d=min(_D[p,b] for b in homes)
        if not 2<=d<=5:continue
        exits=[int(z) for z in _TO[p] if min(_D[z,b] for b in homes)<d]
        if len(exits)!=2 or any(z in recruits for z in exits):continue
        defenders=[]
        for z in exits:
            defenders.append([k for k,(pos,cargo,typ) in own[2].items() if typ=='COLLECTOR' and cargo<min(load,26) and _D[pos,z]<=1
                              and all(w[2]!='COLLECTOR' or w[1]>cargo or _D[w[0],z]>1 for w in enemy[2].values())])
        for a,b in product(*defenders):
            if a==b or any(v in exits for k,v in current.items() if k not in (a,b)):continue
            moves={k:(_DIRECTION[list(_TO[own[2][k][0]]).index(z)] or 'NONE') for k,z in zip((a,b),exits)}
            edits={k:v for k,v in moves.items() if v!=(orders.get(k) or 'NONE')}
            if len(edits)!=2:continue
            r=dict(t=obs['step'],target=target,load=load,exits=exits,edits=edits)
            if best is None or load>best['load']:best=r
    return best
def main():
    allcases=[]
    for c in sr.load(HERE/'coverage.json.gz'):
        raw=sr.load(sr.ROOT/c['source']);sim=sr.FastCarbon(raw['game_configuration']).reset(seed=c['seed']);result=[]
        for t,pair in enumerate(raw['actions'][:201]):
            q=choices(sim.obs(c['seat']),pair[c['seat']]) if t>=100 else None
            if q:result.append(q)
            sim.step(*pair)
        allcases.append(dict(seed=c['seed'],seat=c['seat'],eligible=result))
    sr.retain(HERE/'double_exit.json',allcases);print([len(c['eligible']) for c in allcases])
if __name__=='__main__':main()

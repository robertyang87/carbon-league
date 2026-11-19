"""Measure visible harvest remaining, nearby taxation, and affordable stationary collection after entry."""
import copy
from scan import sr,HERE
from evade import _D
def study():
    rows=[]
    for c in sr.load(HERE/'coverage.json.gz'):
        by_t={a['root']['t']:a['root'] for a in c['opportunities']};raw=sr.load(sr.ROOT/c['source']);s=c['seat']
        sim=sr.FastCarbon(raw['game_configuration']).reset(seed=c['seed'])
        for t,pair in enumerate(raw['actions'][:201]):
            if t in by_t:
                r=by_t[t];uid=r['worker'];alt=copy.deepcopy(sim);cmds=list(pair);cmds[s]=dict(pair[s],**{uid:r['alternative']})
                alt.step(*cmds);ob=alt.obs(s);own,enemy=ob['players'][s],ob['players'][1-s];w=own[2].get(uid)
                if w:
                    p,load,_=w
                    risk=any(k=='COLLECTOR' and v<=load and _D[p,q]<=1 for q,v,k in enemy[2].values())
                    tree=any(a[0]==p for v in ob['players'] for a in v[3].values())
                    visible=ob['carbon'][p]
                else:risk=tree=visible=None
                rows.append(dict(seed=c['seed'],seat=s,t=t,alive=w is not None,visible=visible,tree=tree,risk=risk))
            sim.step(*pair)
    sr.retain(HERE/'patch_yield.json',rows)
    print(len(rows),sum(r['alive'] for r in rows),sum(r['alive'] and not r['tree'] and not r['risk'] and r['visible']>=15 for r in rows))
if __name__=='__main__':study()

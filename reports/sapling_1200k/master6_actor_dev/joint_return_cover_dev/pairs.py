"""Cover both shortest return exits simultaneously using light collectors and pre-action state."""
from itertools import product
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'freight_toll_route_dev'))
from measure import sr,_D,_DIRECTION,_TO

def choose(obs,cmd):
    s=obs['player'];own,foe=obs['players'][s],obs['players'][1-s]
    dest={u:int(_TO[v[0],_DIRECTION.index(cmd.get(u))]) for u,v in own[2].items()}
    hires={cell for k,cell in own[1].items() if cmd.get(k)}
    base=set(own[1].values())|set(foe[1].values());opts=[]
    for target,(p,load,kind) in foe[2].items():
        if kind!='COLLECTOR' or load<100:continue
        home=list(foe[1].values());d=min(_D[p,b] for b in home)
        if not 2<=d<=5:continue
        exits=[int(c) for c in _TO[p] if min(_D[c,b] for b in home)==d-1]
        if len(exits)<2 or any(c in base|hires for c in exits):continue
        available=[[u for u,(z,cargo,k) in own[2].items() if k=='COLLECTOR' and cargo<load and cargo<=30 and c in _TO[z]
                    and not (cargo>0 and dest[u] in own[1].values())] for c in exits]
        if not all(available):continue
        if all(any(dest[u]==c for u in units) for units,c in zip(available,exits)):continue
        for team in product(*available):
            if len(set(team))!=len(team):continue
            if any(dest[u] in exits for u in dest if u not in team):continue
            edits={u:_DIRECTION[list(_TO[own[2][u][0]]).index(c)] for u,c in zip(team,exits) if dest[u]!=c}
            if not edits:continue
            value=(len(edits),-load,sum(own[2][u][1] for u in team),target,team)
            opts.append((value,dict(target=target,load=load,exits=exits,team=list(team),edits=edits)))
    return min(opts,key=lambda v:v[0])[1] if opts else None

def run():
    games=sr.load(HERE.parent/'tactical_exchange_dev/paired.json.gz')
    latest=HERE.parent/'freight_toll_route_dev/games/candidate'
    for path in sorted(latest.glob('raw_*.json.gz')):
        g=sr.load(path);games.append(dict(path=str(path.relative_to(sr.ROOT)),seat=g['candidate_seat'],seed=g['seed']))
    starts=[];rows=[]
    for g in games:
        raw=sr.load(sr.ROOT/g['path']);cfg=raw['game_configuration'];s=g['seat'];sim=sr.FastCarbon(cfg).reset(seed=g['seed']);alt=sr.FastCarbon(cfg).reset(seed=g['seed'])
        for t,pair in enumerate(raw['actions']):
            obs=sim.obs(s);z=choose(obs,pair[s])
            if z:
                joint=[dict(a) for a in pair]
                for u,act in z['edits'].items():
                    if act is None:joint[s].pop(u,None)
                    else:joint[s][u]=act
                alt.set_state(sim.get_state());alt.step(*joint)
                pos=obs['players'][1-s][2][z['target']][0];land=int(_TO[pos,_DIRECTION.index(pair[1-s].get(z['target']))])
            sim.step(*pair)
            if z:
                def state(v):
                    return dict(cash=list(v.cash),target_alive=z['target'] in v.workers,
                           freight=[v.workers[u].carbon if u in v.workers else None for u in z['team']])
                rows.append(dict(seed=g['seed'],seat=s,t=t,source=g['path'],choice=z,actual_landing=land,parent=state(sim),changed=state(alt)))
        assert list(sim.cash)==raw['official_score_by_seat']
        starts.append(dict(path=g['path'],seed=g['seed'],seat=s,sha256=sr.sha(sr.ROOT/g['path'])))
    sr.retain(HERE/'sources.json',starts);sr.retain(HERE/'positions.json.gz',rows)
    print('games',len(starts),'positions',len(rows),'actual shortest return',sum(r['actual_landing'] in r['choice']['exits'] for r in rows),'extra kills',sum(r['parent']['target_alive'] and not r['changed']['target_alive'] for r in rows))
if __name__=='__main__':run()

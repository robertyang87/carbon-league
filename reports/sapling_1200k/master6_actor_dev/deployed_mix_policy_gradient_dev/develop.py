"""Matched complete from-opening official-return comparison on held boards with the same frozen opponent."""
import math,time
from study import HERE,PARENT,OPP,PLAN,arena,policy,World
def evaluate():
    trained=arena.read(HERE/'update.json');rows=[]
    if not trained['accepted']:
        arena.write(HERE/'development.json',dict(eligible=False,formal=False,reason='no calibrated learned pack'));return
    candidate=HERE/'candidate.py'
    if arena.sha(candidate)!=trained['agent_sha256']:raise ValueError("candidate bytes changed")
    for j,seed in enumerate(PLAN['development_seeds']):
        for seat in (0,1):
            for label in (('candidate','parent') if (j+seat)%2 else ('parent','candidate')):
                path=candidate if label=='candidate' else PARENT
                out=HERE/'games'/f'{seed}_{seat}_{label}.json.gz'
                if out.exists():
                    r=arena.read(out)
                    if r['own_sha']!=arena.sha(path) or r['opponent_sha']!=arena.sha(OPP):raise ValueError('stale game')
                    rows.append({k:v for k,v in r.items() if k!='actions'});continue
                w=World(seed); own=policy.load(path);other=policy.load(OPP);zones=[own,other] if seat==0 else [other,own]
                t0=time.monotonic();tape=[]
                while not w.terminal:
                    commands=[m.agent(w.obs(s),w.config) for s,m in enumerate(zones)]
                    tape.append(commands);w.advance(commands)
                official=arena.verify(seed,[],tape,w);te=time.monotonic()-t0;sig=official['margin_by_seat'][seat]
                record=dict(seed=seed,seat=seat,arm=label,own_sha=arena.sha(path),opponent_sha=arena.sha(OPP),U=official['u_by_seat'][seat],margin=sig,
                            result=official['result_by_seat'][seat],score=official['scores'],steps=official['steps'],extinct=official['extinct_by_seat'][seat],
                            terrain=w.terrain_sha256,wall_seconds=te)
                arena.write(out,dict(**record,actions=tape));rows.append(record);print('development',seed,seat,label,record['score'],round(te,2),flush=True)
    deltas=[]
    for seed in PLAN['development_seeds']:
        diffs=[]
        for s in (0,1):
            a={v['arm']:v for v in rows if v['seed']==seed and v['seat']==s}
            diffs.append(dict(seat=s,delta_u=a['candidate']['U']-a['parent']['U'],delta_margin=a['candidate']['margin']-a['parent']['margin']))
        deltas.append(dict(seed=seed,pairs=diffs,mean_delta_u=sum(x['delta_u'] for x in diffs)/2,mean_delta_margin=sum(x['delta_margin'] for x in diffs)/2))
    lost=lambda arm:sum(x['extinct'] for x in rows if x['arm']==arm)
    passed=all(v['mean_delta_u']>0 for v in deltas) and lost('candidate')<=lost('parent')
    arena.write(HERE/'development.json',dict(rows=rows,per_terrain=deltas,eligible=passed,formal=False,examiner_games=0))
    print('result',passed,deltas,flush=True)
if __name__=='__main__':evaluate()

"""Observe all 77 surveyed choices for one joint step with the taped simultaneous opponent command."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import runpy
trial=runpy.run_path(str(Path(__file__).resolve().with_name("trial.py")))
HERE,sr,W=(trial[k] for k in ("HERE","sr","W"))
from decision import _D,_TO,update_memory,values,predicted,quote

def inspect():
    records=[]
    for g in sr.load(HERE/'approaches.json.gz'):
        raw=sr.load(sr.ROOT/g['path']);cfg=raw['game_configuration']
        sim=sr.FastCarbon(cfg).reset(seed=g['seed']);s=g['seat'];mem={};wanted={r['t']:r for r in g['hits']}
        for t,pair in enumerate(raw['actions'][:200]):
            obs=sim.obs(s);update_memory(obs,mem)
            if t in wanted:
                r=wanted[t];uid=r['worker'];target=r['target']
                root_d=min((int(_D[target,w[0]]) for w in obs['players'][1-s][2].values()),default=30)
                branch=deepcopy(sim);a=list(pair);a[s]=dict(pair[s],**{uid:r['alternative']});branch.step(*a)
                post=branch.obs(s);postmem=mem.copy();update_memory(post,postmem)
                w=post['players'][s][2].get(uid)
                arrived=w is not None and w[0]==target
                d=min((int(_D[target,w[0]]) for w in post['players'][1-s][2].values()),default=30)
                visible=sum(post['carbon'][int(j)] for j in _TO[target][1:]) if arrived else None
                v=values(post,postmem,uid,target,cfg);net=(predicted(v,W)-quote(post,cfg)) if v else None
                records.append(dict(seed=g['seed'],seat=s,t=t,worker=uid,root_enemy_distance=root_d,
                        root_predicted_soil=r['remembered_total'],arrived=arrived,next_enemy_distance=d,
                        next_visible_soil=visible,next_positive_net=net is not None and net>0,
                        only_preconditions=bool(arrived and d>=2 and visible>=120 and net is not None and net>0)))
            sim.step(*pair)
    summary={}
    for key,select in [('all',lambda r:True),('distance_two',lambda r:r['root_enemy_distance']==2),
                      ('distance_at_least_three',lambda r:r['root_enemy_distance']>=3)]:
        subset=[r for r in records if select(r)]
        stats=Counter(eligible_roots=len(subset),unique_maps=len({r['seed'] for r in subset}))
        for r in subset:
            stats['arrive']+=r['arrived'];stats['not_safe_after_move']+=r['next_enemy_distance']<2
            stats['arrived_below_soil_threshold']+=r['arrived'] and r['next_visible_soil']<120
            stats['arrived_safe_soil_positive_net']+=r['only_preconditions']
        summary[key]=dict(stats)
    sr.retain(HERE/'all_site_arrivals.json',dict(summary=summary,per_root=records))
    print(summary)
if __name__=='__main__':inspect()

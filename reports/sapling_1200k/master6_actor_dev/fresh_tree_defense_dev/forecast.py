"""Public five-action warning for an already held fresh tree."""
from math import exp,log1p
from policy import DIST,near

FEATURES=('threat_distance','life_left','second_threat_distance','enemy_planters_within_two',
          'enemy_planters_within_five','nearest_approach_one','nearest_approach_three',
          'nearest_stayed','rival_can_pay','own_collector_distance','low_cargo_defender_distance',
          'nearest_cargo','own_base_distance','enemy_base_distance','defender_advantage','previous_tree_income')


def vector(obs, tree, history, cfg):
    s=obs['player'];me,enemy=obs['players'][s],obs['players'][1-s];p=tree['pos'];tid=tree['tid']
    rank=sorted((DIST[p][loc],uid,loc) for uid,(loc,_,role) in enemy[2].items() if role=='PLANTER')
    d,u,loc=(rank or [(15,None,None)])[0]
    old1=history[-2].get(u,loc) if len(history)>=2 else loc
    old3=history[-4].get(u,loc) if len(history)>=4 else loc
    cols=sorted((DIST[p][pos],cargo,uid) for uid,(pos,cargo,kind) in me[2].items() if kind=='COLLECTOR')
    dc,cargo,_=(cols or [(15,0,None)])[0]
    cheap=min((v[0] for v in cols if v[1]<=25),default=15)
    remain=max(0,cfg['treeLifespan']-me[3][tid][1])
    return (min(d,8),min(remain,20),min(8,rank[1][0] if len(rank)>1 else 15),sum(r[0]<=2 for r in rank),
            sum(r[0]<=5 for r in rank),DIST[p][old1]-d if loc is not None else 0,DIST[p][old3]-d if loc is not None else 0,
            int(loc is not None and old1==loc),int(enemy[0]>=cfg['seizeCost']),min(dc,8),min(cheap,8),log1p(cargo),
            min(8,near(p,me[1].values())),min(8,near(p,enemy[1].values())),max(-5,min(5,d-cheap)),
            log1p(max(0,obs['trees'][tid][1])))


def predict(net,values):
    z=net['bias']+sum(w*(values[i]-mu)/sd for i,mu,sd,w in zip(net['columns'],net['center'],net['scale'],net['weights']))
    return 1/(1+exp(-min(35,max(-35,z))))

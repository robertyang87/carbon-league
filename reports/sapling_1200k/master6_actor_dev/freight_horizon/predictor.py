"""Predict an enemy courier's same-ID ten-step home proceeds from public pre-action state only."""
import json
import math
from pathlib import Path


def distance(a,b):
    d=abs(a//15-b//15);e=abs(a%15-b%15)
    return min(d,15-d)+min(e,15-e)

DIST=[[distance(i,j) for j in range(225)] for i in range(225)]
NEXT=[[(i//15+dy)%15*15+(i%15+dx)%15 for dy,dx in ((1,0),(-1,0),(0,1),(0,-1))] for i in range(225)]
NAMES=('cargo','own_base_distance','remaining','nearest_lighter_blocker','nearby_lighter_blockers','nearest_blocker_to_base','nearest_support','safe_shortest_first_steps','shortest_first_steps','taxed_shortest_first_steps','local_enemy_tree_count','route_low_threat_squares','route_blocked_squares','other_enemy_couriers','total_observer_collectors','cargo_delta_3','home_progress_3','observed_3','cargo_delta_5','home_progress_5','observed_5','unchanged_cell_3','unchanged_cell_5','other_nearby_loaded_couriers','nearby_defending_tree_count')


def public_features(obs,history):
    """Physical, uncanonicalized 15x15 board. History excludes the supplied current public step."""
    t=int(obs['step']);s=int(obs['player']);own,enemy=obs['players'][s],obs['players'][1-s]
    hist={int(x['step']):x for x in history}
    if any(h>=t or int(x['player'])!=s for h,x in hist.items()):
        raise ValueError('only strictly earlier same-seat public history is allowed')
    homes=list(enemy[1].values());guards=[(pos,load) for pos,load,kind in own[2].values() if kind=='COLLECTOR']
    trees=[pos for pos,age in own[3].values()]; own_trees=[pos for pos,age in enemy[3].values()]
    couriers={k:v for k,v in enemy[2].items() if v[2]=='COLLECTOR'}
    def adjacent_trees(p,tree):
        return sum(DIST[p][q]<=2 and min(abs(p//15-q//15),15-abs(p//15-q//15))<=1 and min(abs(p%15-q%15),15-abs(p%15-q%15))<=1 for q in tree)
    result={}
    for uid,(pos,cargo,_) in couriers.items():
        if cargo<=0:continue
        hdist=min(DIST[pos][b] for b in homes)
        closest=[b for b in homes if DIST[pos][b]==hdist]
        blockers=[p for p,c in guards if c<=cargo]
        near=lambda p,points:min((DIST[p][q] for q in points),default=15)
        forward=[q for q in NEXT[pos] if min(DIST[q][b] for b in closest)<hdist]
        reach=[q for q in range(225) if any(DIST[pos][q]+DIST[q][b]==hdist for b in closest) and q!=pos]
        allies=[w[0] for k,w in couriers.items() if k!=uid]
        trip=[];stays=[]
        for lag in (3,5):
            earlier=hist.get(t-lag);v=None if earlier is None else earlier['players'][1-s][2].get(uid)
            trip+=([cargo-v[1],min(DIST[v[0]][b] for b in homes)-hdist,1] if v is not None else [0,0,0]);stays.append(int(v is not None and v[0]==pos))
        row=(cargo,hdist,299-t,near(pos,blockers),sum(DIST[pos][p]<=3 for p in blockers),min((near(b,blockers) for b in closest),default=15),near(pos,allies),
             sum(near(q,blockers)>1 for q in forward),len(forward),sum(adjacent_trees(q,trees)>0 for q in forward),adjacent_trees(pos,trees),
             sum(near(q,blockers)>1 for q in reach),sum(near(q,blockers)<=1 for q in reach),len(couriers)-1,len(guards),*trip,*stays,
             sum(DIST[pos][v[0]]<=3 and v[1]>0 for k,v in couriers.items() if k!=uid),adjacent_trees(pos,own_trees))
        result[uid]=row
    return result


def ensemble(tree_set,row):
    val=tree_set['offset']
    for tree in tree_set['trees']:
        i=0
        while True:
            feat,cut,left,right,leaf=tree[i]
            if feat<0:
                val+=leaf;break
            i=left if row[feat]<=cut else right
    return val


class FreightPredictor:
    def __init__(self,model_path=None):
        self.model=json.loads(Path(model_path or Path(__file__).with_name('model.json')).read_bytes())
        if tuple(self.model['features'])!=NAMES:raise ValueError('incompatible input fields')
    def predict(self,obs,history):
        out={}
        for uid,row in public_features(obs,history).items():
            frac=min(1,max(0,ensemble(self.model['principal'],row)));p=1/(1+math.exp(-ensemble(self.model['any'],row)))
            out[uid]=dict(cargo=float(row[0]),home_distance=int(row[1]),p_deposit_10=p,expected_root_fraction_10=frac,expected_root_cash_10=frac*row[0])
        return out

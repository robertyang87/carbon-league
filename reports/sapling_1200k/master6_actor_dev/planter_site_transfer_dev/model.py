"""Five physical planting routes using only the acting seat's public observation."""
from pathlib import Path
from functools import lru_cache
import numpy as np

HERE=Path(__file__).resolve().parent
MOVES=('NONE','UP','RIGHT','DOWN','LEFT')
OFFSETS=((0,0),(-1,0),(0,1),(1,0),(0,-1))
CELLS=np.arange(225)
DEST=np.asarray([[((p//15+i)%15)*15+(p%15+j)%15 for i,j in OFFSETS] for p in CELLS])
DIST=np.asarray([[min(abs(p//15-q//15),15-abs(p//15-q//15))+min(abs(p%15-q%15),15-abs(p%15-q%15)) for q in CELLS] for p in CELLS])
FEATURES=('stay','continue_heading','retreat_heading','unknown_heading','visible_carbon_sum_over_four','known_fraction','visible_max_neighbor','unclaimed_visible_carbon','covered_by_own_tree','covered_by_enemy_tree','open','friendly_tree','enemy_tree','remaining_tree_life','base','other_friendly_occupancy','opponent_occupancy','nearest_enemy_collector','nearest_enemy_planter','nearest_ally_collector','nearest_ally_planter','friendly_home_distance','enemy_home_distance','cash','cash_vs_price','plant_price','own_alive_trees','opponent_alive_trees','own_collectors','own_planters','enemy_collectors','enemy_planters','t','public_advantage','remaining_until_cutoff','adjacent_enemy_cargo','nearby_enemy_cargo','capture_price','enemy_can_contest','our_nearby_planters')
OUTCOMES=('other','new_nonpositive','productive_new','seize')
def features(obs,uid,cfg,previous_cell=None):
    side=obs['player'];me=obs['players'][side];opp=obs['players'][1-side]
    center,_,kind=me[2][uid]
    assert kind=='PLANTER' and cfg['size']==15
    seen=set(); ally=[[],[]]; enemy=[[],[]]
    for a,(pos,v,k) in me[2].items():
        seen.update([pos] if k=='COLLECTOR' else DEST[pos,1:])
        if a!=uid:ally[int(k=='PLANTER')].append(pos)
    for pos,v,k in opp[2].values():enemy[int(k=='PLANTER')].append(pos)
    life=cfg['treeLifespan']
    ours={p:a for p,a in me[3].values() if a<life}
    theirs={p:a for p,a in opp[3].values() if a<life}
    homes=set(me[1].values()); away=set(opp[1].values()); bases=homes|away
    alive=len(ours)+len(theirs)
    quote=cfg['plantCost']+(cfg['plantCostInflationRatio']*cfg['plantCostInflationBase']**alive if len(me[3])>cfg['playerTreeProtectiveNumber'] else 0)
    ids={a:p for a,(p,v,k) in me[2].items() if a!=uid}; op_positions={w[0] for w in opp[2].values()}
    first=next((i for i in range(1,5) if previous_cell is not None and DEST[previous_cell,i]==center),None)
    edges=[{u for p in tree for u in DEST[p,1:]} for tree in (ours,theirs)]
    heavy=[(p,v) for p,v,k in opp[2].values() if k=='COLLECTOR']
    def near(q,g):return min((int(DIST[q,p]) for p in g),default=15)/15
    rows=[]
    for a,q in enumerate(DEST[center]):
        hood=DEST[q,1:];vis=[v for v in hood if v in seen];levels=[obs['carbon'][v] for v in vis]
        overlap=[sum(v in e for v in hood)/4 for e in edges]
        them=min(near(q,enemy[0]),near(q,enemy[1]))
        cash=me[0];now=obs['step']
        rows.append([a==0,a==first,previous_cell is not None and q==previous_cell and a>0,first is None,
            sum(levels)/400,len(vis)/4,max(levels,default=0)/100,
            sum(obs['carbon'][v] for v in vis if v not in edges[0]|edges[1])/400,*overlap,
            q not in ours and q not in theirs and q not in bases,q in ours,q in theirs,
            (life-ours.get(q,theirs.get(q,life)))/life,q in bases,q in ids.values(),q in op_positions,
            near(q,enemy[0]),near(q,enemy[1]),near(q,ally[0]),near(q,ally[1]),near(q,homes),near(q,away),
            np.log1p(max(cash,0))/10,np.clip((cash-quote)/200,-3,3),np.log1p(quote)/5,len(ours)/15,len(theirs)/15,
            len(ally[0])/10,(len(ally[1])+1)/10,len(enemy[0])/10,len(enemy[1])/10,now/300,
            np.clip((cash-opp[0])/1500,-4,4),(cfg['episodeSteps']-1-now)/300,
            sum(v for p,v in heavy if DIST[p,q]<=1)/500,sum(v for p,v in heavy if DIST[p,q]<=3)/1000,
            cfg['seizeCost']/50,them<=1/15,sum(DIST[p,q]<=2 for p in ally[1])/5])
    x=np.array(rows,dtype='float32')
    assert x.shape==(5,len(FEATURES)) and np.isfinite(x).all()
    return x

@lru_cache(None)
def weights(name='primary'):
    with np.load(HERE/(name+'.npz'),allow_pickle=False) as f:return dict(f)
def score(x,w):
    h=np.tanh(x @ w['hidden']+w['hidden_bias'])
    a=(h @ w['action']).squeeze(-1); z=h @ w['settlement']+w['settlement_bias']
    def soft(v):
        p=np.exp(v-np.max(v,axis=-1,keepdims=True))
        return p/p.sum(axis=-1,keepdims=True)
    p,settle=soft(a),soft(z)
    combined=p*(1+settle[...,2])
    return p,settle,combined
def decide(obs,uid,cfg,previous_cell=None):
    p,settle,combined=score(features(obs,uid,cfg,previous_cell),weights())
    return dict(action=MOVES[int(combined.argmax())],bc_action=MOVES[int(p.argmax())],
                probabilities=p.tolist(),productive_probability=settle[:,2].tolist())

"""Shared action utility. Inputs are exactly one player observation and one complete own command."""
from collections import Counter
from functools import lru_cache
import numpy as np

NAMES = [
    "time_left", "cash_gap", "tree_count_gap", "collector_cargo_gap",
    "collector_stays", "stationary_cargo", "cargo_home_progress", "cargo_reaching_base",
    "collector_visible_ore_stayed", "collector_exposed_cargo", "collector_under_enemy_tree",
    "planter_stays_bare", "planter_stays_enemy_tree", "planter_potential_soil",
    "planter_exposed", "planter_approaches_enemy_tree", "friendly_destination_pairs",
    "recruit_collector", "recruit_planter"
]
MOVES = {"NONE":(0,0), "UP":(-1,0), "RIGHT":(0,1), "DOWN":(1,0), "LEFT":(0,-1)}

@lru_cache(maxsize=8)
def geometry(n):
    cells = np.arange(n*n)
    def axis(z):
        diff = abs(z[:,None]-z[None,:])
        return np.minimum(diff,n-diff)
    distance = axis(cells//n) + axis(cells%n)
    moves = {cmd:((cells//n + dr)%n)*n + (cells%n+dc)%n for cmd,(dr,dc) in MOVES.items()}
    return distance, moves

def context(obs):
    seat = obs['player']
    me, enemy = obs['players'][seat], obs['players'][1-seat]
    visible = np.asarray(obs['carbon'])
    n = int(len(visible)**.5)
    distance,moves = geometry(n)
    def nearest(pos, fallback):
        return distance[:,list(pos)].min(axis=1) if pos else np.full(len(visible),fallback)
    own_trees={v[0] for v in me[3].values()}
    enemy_trees={v[0] for v in enemy[3].values()}
    all_bases=set(me[1].values())|set(enemy[1].values())
    home=nearest(set(me[1].values()), 2*n)
    nearest_enemy_tree=nearest(enemy_trees, 2*n)
    enemy_workers=list(enemy[2].values())
    enemy_col=[(p,c) for p,c,k in enemy_workers if k=='COLLECTOR']
    enemy_min=np.min([np.where(distance[p]<=1,c,np.inf) for p,c in enemy_col],axis=0) if enemy_col else np.full(n*n,np.inf)
    enemy_near=nearest({p for p,c,k in enemy_workers},2*n)<=1
    adjacent=visible[[moves[cmd] for cmd in ("UP","RIGHT","DOWN","LEFT")]].sum(axis=0)/200
    state=[(299-obs['step'])/299, np.tanh((me[0]-enemy[0])/1500), (len(own_trees)-len(enemy_trees))/10,
           (sum(w[1] for w in me[2].values())-sum(w[1] for w in enemy_workers))/500]
    def represent(orders):
        a=np.zeros(len(NAMES))
        a[:4]=state
        dest=Counter()
        for uid,(p,cargo,role) in me[2].items():
            cmd=orders.get(uid,'NONE')
            d=moves[cmd][p]
            dest[d]+=1
            stay=cmd=='NONE'
            if role=='COLLECTOR':
                load=cargo/200
                a[4]+=stay
                a[5]+=load*stay
                a[6]+=load*(home[p]-home[d])
                a[7]+=load*(home[d]==0)
                a[8]+=(visible[d]/100)*stay
                a[9]+=load*(enemy_min[d]<=cargo)
                a[10]+=load*(nearest_enemy_tree[d]<=1)
            else:
                bare=d not in own_trees and d not in enemy_trees and d not in all_bases
                a[11]+=stay*bare
                a[12]+=stay*(d in enemy_trees)
                a[13]+=adjacent[d]*stay*bare
                a[14]+=enemy_near[d]
                a[15]+=nearest_enemy_tree[p]-nearest_enemy_tree[d]
        for uid,p in me[1].items():
            cmd=orders.get(uid)
            if cmd in ('RECCOLLECTOR','RECPLANTER'):
                dest[p]+=1
                a[17 if cmd=='RECCOLLECTOR' else 18]+=1
        a[16]=sum(k*(k-1)//2 for k in dest.values())
        return a
    return represent

def choose(scores):
    delta=scores-scores[0]
    allowed=np.all(delta>0,axis=1)
    if not np.any(allowed):
        return 0
    return int(np.argmax(np.where(allowed,delta.mean(axis=1),-np.inf)))

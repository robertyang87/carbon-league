"""Predict a seat-relative change in cash margin over 30 steps from public history."""
from collections import deque
from pathlib import Path
import numpy as np

HORIZON=30
SIZE=15
OFFSETS=((0,0),(0,-1),(1,0),(0,1),(-1,0))
def neighbor(pos,dx,dy):
    return ((pos%SIZE+dx)%SIZE)+SIZE*((pos//SIZE+dy)%SIZE)
DEST=tuple(tuple(neighbor(p,dx,dy) for dx,dy in OFFSETS) for p in range(SIZE*SIZE))
RING=tuple(tuple(neighbor(p,dx,dy) for dx in (-1,0,1) for dy in (-1,0,1)) for p in range(SIZE*SIZE))
def dist(p,q):
    dx,dy=abs(p%SIZE-q%SIZE),abs(p//SIZE-q//SIZE)
    return min(dx,SIZE-dx)+min(dy,SIZE-dy)
DIST=tuple(tuple(dist(p,q) for q in range(SIZE*SIZE)) for p in range(SIZE*SIZE))
SIDE_FIELDS=(
    'cash','log_cash','can_recruit','collectors','planters','available_bases',
    'light_collectors','loaded_collectors','cargo','cargo_home_1','cargo_home_2_3',
    'cargo_home_4_6','cargo_home_far','cargo_nearest_return_discount','cargo_homeward_route_threatened',
    'cargo_no_safe_homeward','light_intercept_exposures','cargo_stay_exposed','cargo_enemy_tree_tax_now',
    'friendly_bank_traffic','collector_known_deposit_area','trees','productive_life_next30',
    'trees_expiring_next10','trees_income_last_step','trees_life_weighted_income','trees_near_enemy_cargo',
    'known_tree_neighborhood_carbon','tree_neighborhood_visibility','planters_near_opponent_trees',
    'known_collector_ground','collector_ground_visibility','recent_margin_3','recent_margin_10','recent_margin_30')
FEATURE_NAMES=tuple(name+suffix for suffix in ('','_early','_late') for name in SIDE_FIELDS)


class History:
    def __init__(self):
        self.seat=None
        self.step=-1
        self.known=[0.]*(SIZE*SIZE)
        self.seen=[-100]*(SIZE*SIZE)
        self.cash=deque(maxlen=31)

    def update(self,obs):
        s=obs['player'];t=obs['step']
        if t!=self.step+1 or (self.seat is not None and s!=self.seat):
            raise ValueError('requires consecutive observations from one seat starting at step zero')
        self.step=t;self.seat=s
        players=obs['players']
        self.cash.append((float(players[0][0]),float(players[1][0])))
        for pos,_,kind in players[s][2].values():
            for p in ((pos,) if kind=='COLLECTOR' else DEST[pos][1:]):
                self.known[p]=float(obs['carbon'][p]);self.seen[p]=t

    def trend(self,k):
        n=min(k,len(self.cash)-1)
        if not n:return 0.
        side=self.seat
        start,end=self.cash[-n-1],self.cash[-1]
        return ((end[side]-end[1-side])-(start[side]-start[1-side]))*HORIZON/n

    def encode(self,obs,cfg):
        if obs['player']!=self.seat or obs['step']!=self.step:
            raise ValueError('update history with this observation first')
        pl=obs['players']
        if cfg.get('size',SIZE)!=SIZE:
            raise ValueError('value head requires a 15 by 15 board')
        tree_life=cfg['treeLifespan'];tax_rate=cfg['collectorAbsorptionRate']
        trees=[{p for p,_ in side[3].values()} for side in pl]
        workers=[list(side[2].values()) for side in pl]
        coll=[[(p,c) for p,c,kind in ws if kind=='COLLECTOR'] for ws in workers]
        bases=[tuple(side[1].values()) for side in pl]
        available=[max(0.,1-(self.step-last)/12.) for last in self.seen]
        readings=[self.known[i]*q for i,q in enumerate(available)]
        rows=[]
        for s in (0,1):
            other=1-s;cash=pl[s][0];own=coll[s];enemy=coll[other]
            visited={p for p,_,_ in workers[s]}
            f=[cash,np.log1p(max(cash,0)),float(cash>=cfg['recCollectorCost']),len(own),len(workers[s])-len(own),
               sum(p not in visited for p in bases[s])]
            light=sum(c<=15 for _,c in own);loaded=sum(c>=50 for _,c in own)
            csum=0.;buckets=[0.]*4; discounted=exposed_path=blocked=weak_opportunities=stay_risk=tolls=near_desk=local_yield=ground=seen=0.
            for p,c in own:
                to_home=min(DIST[p][b] for b in bases[s]);csum+=c
                buckets[0 if to_home<=1 else 1 if to_home<=3 else 2 if to_home<=6 else 3]+=c
                discounted+=c/(1+to_home)
                nearby=[(q,ec) for q,ec in enemy if DIST[p][q]<=2]
                threats={d for q,ec in nearby if ec<=c for d in DEST[q]}
                homeward=[d for d in DEST[p][1:] if min(DIST[d][b] for b in bases[s])<to_home]
                if homeward:
                    count=sum(q in threats for q in homeward)
                    exposed_path+=c*count/len(homeward)
                    if count==len(homeward):blocked+=c
                if c<=15:weak_opportunities+=sum(ec>=50 and c<ec for _,ec in nearby)
                if p in threats:stay_risk+=c
                tolls+=c*tax_rate*sum(q in trees[other] for q in RING[p])
                near_desk+=int(to_home<=2)
                local_yield+=sum(readings[d] for d in DEST[p])/(1+to_home)
                ground+=readings[p];seen+=available[p]
            life30=expiring=last_income=weighted_income=enemy_stock=near_carbon=coverage=0.
            for tid,(p,age) in pl[s][3].items():
                remaining=max(0,tree_life-age)
                life30+=min(HORIZON,remaining);expiring+=int(remaining<=10)
                income=float(obs['trees'].get(tid,[None,0])[1] or 0)
                last_income+=income;weighted_income+=income*min(HORIZON,remaining)/HORIZON
                enemy_stock+=sum(c for q,c in enemy if q in RING[p])
                near_carbon+=sum(readings[q] for q in RING[p]);coverage+=sum(available[q] for q in RING[p])
            planters_threat=sum(any(DIST[p][q]<=2 for q in trees[other]) for p,_,k in workers[s] if k=='PLANTER')
            side_trends=[]
            for back in (3,10,30):
                n=min(back,len(self.cash)-1)
                side_trends.append((self.cash[-1][s]-self.cash[-n-1][s])/n if n else 0.)
            f.extend((light,loaded,csum,*buckets,discounted,exposed_path,blocked,weak_opportunities,stay_risk,tolls,
                      near_desk,local_yield,len(trees[s]),life30,expiring,last_income,weighted_income,enemy_stock,
                      near_carbon,coverage,planters_threat,ground,seen,*side_trends))
            rows.append(np.asarray(f,dtype=np.float64))
        diff=rows[self.seat]-rows[1-self.seat]
        phase=min(1.,self.step/max(1,cfg['episodeSteps']-1))
        return np.concatenate((diff,diff*max(0,1-2*phase),diff*max(0,2*phase-1)))


class CashMargin30:
    def __init__(self,path=None):
        path=Path(path) if path else Path(__file__).with_name('cash_margin_30.npz')
        with np.load(path,allow_pickle=False) as d:
            if tuple(d['names'])!=FEATURE_NAMES or int(d['horizon'])!=HORIZON:
                raise ValueError('incompatible prediction checkpoint')
            self.coef=d['coef'].copy()
            self.scale=d['scale'].copy()
        if self.coef.shape!=self.scale.shape or len(self.coef)!=len(FEATURE_NAMES) or not np.isfinite(self.coef).all() or not np.isfinite(self.scale).all() or min(self.scale)<=0:
            raise ValueError('invalid weights or scale')

    def predict(self,history,obs,cfg):
        return float((history.encode(obs,cfg)/self.scale)@self.coef)

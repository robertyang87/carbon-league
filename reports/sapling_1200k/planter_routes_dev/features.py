"""What a planter can know before choosing its next cell, including partial carbon visibility."""
import math
import numpy as np

_ROUTE_CELLS=np.arange(225)
_rr,_cc=np.divmod(_ROUTE_CELLS,15)
_ROUTE_DEST=np.stack([_ROUTE_CELLS,((_rr-1)%15)*15+_cc,_rr*15+(_cc+1)%15,((_rr+1)%15)*15+_cc,
                      _rr*15+(_cc-1)%15],axis=1)
_dr=abs(_rr[:,None]-_rr);_dc=abs(_cc[:,None]-_cc)
_ROUTE_DIST=np.minimum(_dr,15-_dr)+np.minimum(_dc,15-_dc)
_ROUTE_FIELDS=("seize","remaining_tenure","estimated_resource","visible_fraction","recent_fraction","previous_income",
               "near_enemy_planter","near_enemy_collector","near_own_collector","crowded_tree_edge","price",
               "cash_after_price","income_deficit","own_home_distance","enemy_home_distance","remaining_x_resource")


class PlanterRouteState:
    def __init__(self):
        self.last=np.full(225,-1000,dtype=int)
        self.reading=np.full(225,12.,dtype=float)

    def observe(self,board,seat,visible,cfg,step):
        self.step=step;self.cfg=cfg;self.seat=seat
        own,opp=board["players"][seat],board["players"][1-seat]
        self.cash=own[0];self.own_bases=list(own[1].values());self.opp_bases=list(opp[1].values())
        self.own_trees={p:(k,a) for k,(p,a) in own[3].items()}
        self.opp_trees={p:(k,a) for k,(p,a) in opp[3].items()}
        self.income=board.get("trees",{})
        self.own_c=[];self.opp_c=[];self.opp_p=[]
        vis=set()
        for pos,_,typ in own[2].values():
            if typ=="COLLECTOR":self.own_c.append(pos);vis.add(pos)
            else:vis.update(map(int,_ROUTE_DEST[pos,1:]))
        for pos,_,typ in opp[2].values():
            (self.opp_c if typ=="COLLECTOR" else self.opp_p).append(pos)
        ix=list(vis)
        if ix:
            self.last[ix]=step;self.reading[ix]=np.asarray(visible)[ix]
        total=sum(a<cfg.get("treeLifespan",50) for _,a in list(self.own_trees.values())+list(self.opp_trees.values()))
        self.new_price=cfg.get("plantCost",20)
        if len(own[3])>cfg.get("playerTreeProtectiveNumber",0):
            self.new_price+=cfg.get("plantCostInflationRatio",5)*cfg.get("plantCostInflationBase",1.235)**total
        def flow(z):
            return sum(self.income.get(tid,[None,0])[1] for tid in z)
        self.deficit=np.clip((flow(opp[3])-flow(own[3]))/60.,-2.,2.)

    def features(self,pos):
        if pos in self.own_trees or pos in self.own_bases or pos in self.opp_bases:return None
        life=self.cfg.get("treeLifespan",50);enemy=self.opp_trees.get(pos)
        if enemy and enemy[1]>=life-1:return None
        left=min(life-(enemy[1]+1 if enemy else 0),self.cfg.get("episodeSteps",300)-2-self.step)
        if left<=0:return None
        around=_ROUTE_DEST[pos,1:];ages=self.step-self.last[around]
        available=np.array([c not in self.own_trees and c not in self.opp_trees for c in around])
        recent=ages<=6
        observed=np.where(recent,self.reading[around],12.)
        resource=float(observed@available)/180.
        prev=float(self.income.get(enemy[0],[None,0])[1]) if enemy else 0.
        price=self.cfg.get("seizeCost",20) if enemy else self.new_price
        dist=lambda cells:min((_ROUTE_DIST[pos,int(x)] for x in cells),default=14)
        z=[bool(enemy),left/50.,resource,float(np.mean(ages==0)),float(recent.mean()),np.clip(prev/8.,0.,3.),
           1./(1.+dist(self.opp_p)),1./(1.+dist(self.opp_c)),1./(1.+dist(self.own_c)),
           1.-float(available.mean()),np.clip(price/120.,0.,3.),np.clip((self.cash-price)/200.,-1.,3.),
           self.deficit,dist(self.own_bases)/12.,dist(self.opp_bases)/12.,left/50.*resource]
        return np.asarray(z,dtype=float),float(price)


def route_value(model,x):
    a=(x-np.asarray(model["mean"]))/np.asarray(model["scale"])
    prob=1./(1.+np.exp(-np.clip(a@model["purchase_weights"]+model["purchase_bias"],-30.,30.)))
    net=np.clip(a@model["net_weights"]+model["net_bias"],-150.,250.)
    return float(prob*net),float(prob),float(net)

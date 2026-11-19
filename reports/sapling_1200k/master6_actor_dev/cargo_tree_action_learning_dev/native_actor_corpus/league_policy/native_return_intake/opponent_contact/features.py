"""Causal physical-coordinate features for forecasting an opposing collector."""
from collections import deque

import numpy as np

SIDE=15
N=SIDE**2
MOVES=("NONE","UP","RIGHT","DOWN","LEFT")
cells=np.arange(N)
r,c=divmod(cells,SIDE)
DEST=np.stack([cells,((r-1)%SIDE)*SIDE+c,r*SIDE+(c+1)%SIDE,
               ((r+1)%SIDE)*SIDE+c,r*SIDE+(c-1)%SIDE],axis=1)
dr=np.abs(r[:,None]-r);dc=np.abs(c[:,None]-c)
DIST=np.minimum(dr,SIDE-dr)+np.minimum(dc,SIDE-dc)
STEP={(int(p),int(q)):a for p in cells for a,q in enumerate(DEST[p])}
MISSING=5


class Tracks:
    def __init__(self):
        self.step=-1
        self.seat=None
        self.frames=deque(maxlen=6)

    def observe(self,public):
        t,seat=int(public["step"]),int(public["player"])
        if t!=self.step+1 or (self.seat is not None and seat!=self.seat):
            raise ValueError("history must start at step zero and be consecutive for one seat")
        side=public["players"][1-seat]
        self.frames.append({uid:(int(w[0]),float(w[1])) for uid,w in side[2].items() if w[2]=="COLLECTOR"})
        self.step,self.seat=t,seat

    def history(self,uid):
        hist=[frame.get(uid) for frame in reversed(self.frames)]
        values=[]
        for back in range(3):
            a=MISSING
            if len(hist)>back+1 and hist[back] is not None and hist[back+1] is not None:
                a=STEP.get((hist[back+1][0],hist[back][0]),MISSING)
            values.extend(float(a==k) for k in range(6))
        for back in (3,5):
            ok=len(hist)>back and all(v is not None for v in hist[:back+1])
            values.extend((float(ok),np.clip((hist[0][1]-hist[back][1])/200,-2,2) if ok else 0.,
                           DIST[hist[0][0],hist[back][0]]/7 if ok else 0.))
        return values


GLOBAL=("step","own_cash","opp_cash","own_c","own_p","opp_c","opp_p","own_t","opp_t","cargo")
LOCAL=("dist_own_base","dist_opp_base","dist_own_tree","dist_opp_tree",
       "dist_own_collector","dist_own_planter","dist_opp_planter","dist_other_opp_collector",
       "own_collectors_1","own_planters_1","other_opp_collectors_1","own_lighter_c_1",
       "own_heavier_c_1","on_own_planter","on_own_collector","on_opp_base","on_own_base",
       "on_own_tree","on_opp_tree","visible_carbon","carbon_visible")
NAMES=list(GLOBAL)+[move+"_"+f for move in MOVES for f in LOCAL]
HISTORY_OFFSET=len(NAMES)
NAMES+=["ago_"+str(lag)+"_"+a for lag in (1,2,3) for a in (*MOVES,"unknown")]
NAMES+=[f"{key}_{lag}" for lag in (3,5) for key in ("known","cargo_change","displacement")]
HISTORY_COLS=tuple(range(HISTORY_OFFSET,len(NAMES)))
CONTINUOUS_COLS=tuple(i for i,f in enumerate(NAMES) if not (f.startswith("ago_") or f.startswith("known_")
                     or "_on_" in f or f.endswith("carbon_visible")))


def _distance(positions):
    return DIST[:,positions].min(1) if positions else np.full(N,15)


def board(public,tracks):
    """Collect features without consuming current commands. Targets are separate."""
    seat=int(public["player"])
    if (tracks.seat,tracks.step)!=(seat,int(public["step"])):
        raise ValueError("features require current one-seat public history")
    own,opp=(public["players"][p] for p in (seat,1-seat))
    groups=[[w for w in side[2].values() if w[2]==role] for side in (own,opp)
            for role in ("COLLECTOR","PLANTER")]
    oc,op,ec,ep=([int(w[0]) for w in group] for group in groups)
    ot,et=([int(w[0]) for w in side[3].values()] for side in (own,opp))
    ob,eb=(list(side[1].values()) for side in (own,opp))
    foes=[uid for uid,w in opp[2].items() if w[2]=="COLLECTOR"]
    own_cargo=np.asarray([float(w[1]) for w in groups[0]])
    pubcarbon=np.asarray(public["carbon"])
    if pubcarbon.shape!=(N,):
        raise ValueError("unsupported board size")
    seen=np.zeros(N,bool)
    seen[oc]=True
    seen[DEST[op,1:].ravel()]=True
    basegrid=np.stack([_distance(positions) for positions in (ob,eb,ot,et,oc,op,ep)],axis=1)
    rows=[]
    for uid in foes:
        position,cargo,_=opp[2][uid]
        position,cargo=int(position),float(cargo)
        targets=DEST[position]
        other=[int(w[0]) for name,w in opp[2].items() if name!=uid and w[2]=="COLLECTOR"]
        geom=basegrid[targets]
        own_close=DIST[np.ix_(targets,oc)]<=1
        locals_=np.column_stack([geom/15,_distance(other)[targets]/15,own_close.sum(1)/10,
            (DIST[np.ix_(targets,op)]<=1).sum(1)/10,(DIST[np.ix_(targets,other)]<=1).sum(1)/10,
            (own_close&(own_cargo<cargo)).sum(1)/10,(own_close&(own_cargo>cargo)).sum(1)/10,
            geom[:,5]==0,geom[:,4]==0,geom[:,1]==0,geom[:,0]==0,geom[:,2]==0,geom[:,3]==0,
            np.where(seen[targets],pubcarbon[targets],0)/100,seen[targets]])
        pre=[tracks.step/299,np.log1p(max(0,float(own[0])))/10,np.log1p(max(0,float(opp[0])))/10,
             len(oc)/10,len(op)/10,len(ec)/10,len(ep)/10,len(ot)/25,len(et)/25,np.log1p(max(cargo,0))/7]
        rows.append(np.concatenate([pre,locals_.ravel(),tracks.history(uid)]))
    return foes,np.asarray(rows,np.float32).reshape(len(foes),len(NAMES))


def target_links(public,ids):
    """Yield own planter's five intents and possible enemy arrivals on each cell."""
    seat=public["player"]; own,opp=(public["players"][p] for p in (seat,1-seat))
    reaches={}
    for i,uid in enumerate(ids):
        for act,target in enumerate(DEST[int(opp[2][uid][0])]):
            reaches.setdefault(int(target),[]).append((i,act))
    for uid,w in own[2].items():
        if w[2]!="PLANTER":
            continue
        for intent,target in enumerate(DEST[int(w[0])]):
            links=reaches.get(int(target),[])
            yield uid,intent,int(target),tuple(i for i,_ in links),tuple(a for _,a in links)

"""Public geometric conditions of a loaded opposing collector's possible detour."""
import numpy as np

_EV_DELTAS=((0,0),(-1,0),(0,1),(1,0),(0,-1))
_EV_M=[[((p//15+dy)%15)*15+(p%15+dx)%15 for dy,dx in _EV_DELTAS] for p in range(225)]
_EV_D=[[min(abs(p//15-q//15),15-abs(p//15-q//15))+min(abs(p%15-q%15),15-abs(p%15-q%15)) for q in range(225)] for p in range(225)]

def _ev_units(players,opp,previous):
    side=players[opp]; our=players[1-opp][2]
    homes=list(side[1].values()); others=previous[1-opp][2] if previous else {};old=previous[opp][2] if previous else {}
    result=[]
    if not homes:return result
    for uid,(pos,cargo,role) in side[2].items():
        if role!='COLLECTOR' or cargo<100 or pos in homes:continue
        foes=[(key,w[0],w[1]) for key,w in our.items() if w[2]=='COLLECTOR' and w[1]<cargo]
        if not foes or min(_EV_D[pos][p] for _,p,_ in foes)>3:continue
        dest=_EV_M[pos];d0=min(_EV_D[pos][b] for b in homes)
        dist=[min(_EV_D[q][b] for b in homes) for q in dest]
        close=[min(_EV_D[q][p] for _,p,_ in foes) for q in dest]
        if not any(d<d0 and k<=1 for d,k in zip(dist,close)) or max(close)<=1:continue
        mineprev=old.get(uid,[pos])[0]
        def ahead(start,current):
            for i,p in enumerate(_EV_M[start]):
                if p==current:return _EV_M[current][i]
            return current
        incoming=[ahead(others.get(key,[p])[0],p) for key,p,_ in foes]
        coords=[]
        for q,d,f in zip(dest,dist,close):
            nearest=min(foes,key=lambda enemy:_EV_D[q][enemy[1]])
            eprev=others.get(nearest[0],[nearest[1]])[0]
            coords.append((d0-d,min(d0,10)/10.,float(q==pos),float(q in homes),
                min(f,4)/4.,float(f<=1),float(f==0),min(3,sum(_EV_D[q][p]<=1 for _,p,_ in foes))/3.,
                min(4,min(_EV_D[q][p] for p in incoming))/4.,min(1.,max(-1.,(_EV_D[pos][eprev]-_EV_D[pos][nearest[1]])/2)),
                float(ahead(mineprev,pos)==q and mineprev!=pos),float(q==mineprev and mineprev!=pos),
                min(cargo/500.,2.),min((cargo-nearest[2])/500.,2.),
                float(any(w[0]==q for u,w in side[2].items() if u!=uid)), min(_EV_D[q][b] for b in homes)/12.))
        avoid=np.array([d>=d0 and f>1 for d,f in zip(dist,close)],dtype=bool)
        result.append((uid,pos,np.array(coords,dtype=np.float32),avoid))
    return result


def _ev_adjust(players,opp,previous,worker_logits,params):
    units=_ev_units(players,opp,previous)
    if not units:return worker_logits
    result=worker_logits.copy()
    beta,W,b,O=params
    for _,pos,x,_ in units:
        correction=2.*np.tanh(np.tanh(x@W+b)@O)
        result[pos]=beta*worker_logits[pos]+correction
    return result

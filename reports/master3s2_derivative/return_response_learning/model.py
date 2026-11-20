"""Value of present own cargo reaching the same carrier's friendly bank within twelve public moves."""
import numpy as np

_RR_ACTIONS = ('NONE','UP','RIGHT','DOWN','LEFT')
_RR_DELTAS = ((0,0),(-1,0),(0,1),(1,0),(0,-1))
_RR_MOVE = [[((p//15+r)%15)*15+(p%15+c)%15 for r,c in _RR_DELTAS] for p in range(225)]
_RR_DIST = [[min(abs(p//15-q//15),15-abs(p//15-q//15))+min(abs(p%15-q%15),15-abs(p%15-q%15)) for q in range(225)] for p in range(225)]


def rr_features(players, me, step, order):
    _,bases,workers,_=players[me];foe=players[1-me];homes=set(bases.values());enemy_home=set(foe[1].values())
    out=[];seats=[];remaining=min(12,299-step)
    destinations={u:_RR_MOVE[w[0]][_RR_ACTIONS.index(order.get(u,'NONE'))] for u,w in workers.items()}
    hired={v for u,v in bases.items() if order.get(u,'NONE')!='NONE'}
    foe_collectors=[(p,w) for p,w,role in foe[2].values() if role=='COLLECTOR']
    own_collectors=[(u,p,w) for u,(p,w,role) in workers.items() if role=='COLLECTOR']
    for uid,pos,load in own_collectors:
        if load<=0:continue
        dest=destinations[uid];d0=min((_RR_DIST[pos][b] for b in homes),default=15);d=min((_RR_DIST[dest][b] for b in homes),default=15)
        lighter=[p for p,w in foe_collectors if w<=load];closest0=min((_RR_DIST[pos][p] for p in lighter),default=15)
        near=min((_RR_DIST[dest][p] for p in lighter),default=15);home_dist=min((_RR_DIST[dest][b] for b in enemy_home),default=15)
        allies=sum(_RR_DIST[pos][p]<=2 for u,p,w in own_collectors if u!=uid and w<=load)
        next_allies=sum(_RR_DIST[dest][nextpos]<=1 for u,nextpos in destinations.items() if u!=uid)
        row=(min(load/300.,3),d0/8.,min(closest0,5)/5.,min(len(lighter)/10.,1),min(allies,3)/3.,remaining/12.,step/299.,float(d0<=remaining),
             float(dest==pos),d/8.,float(d0-d),float(dest in homes),float(dest in enemy_home),min(near,5)/5.,float(near<=1),
             min(sum(_RR_DIST[dest][p]<=2 for p in lighter),3)/3.,float(dest in hired or any(q==dest for u,q in destinations.items() if u!=uid)),
             min(next_allies,3)/3.,min(home_dist,5)/5.,max(-1.,min(1.,(remaining-1-d)/8.)),float(d<=max(remaining-1,0)))
        out.append(row);seats.append((uid,float(load)))
    return np.asarray(out,dtype=float).reshape((-1,21)),seats


def rr_predict(x, head):
    cols=np.asarray(head['columns'],dtype=int);x=x[:,cols]
    hidden=np.tanh(x@np.asarray(head['w0'])+np.asarray(head['b0']))
    response=hidden@np.asarray(head['w1'])+head['b1']
    return 1./(1.+np.exp(-np.clip(response,-35,35)))


def rr_scores(players, me, step, commands, weights):
    values=[]
    for command in commands:
        feats,items=rr_features(players,me,step,command)
        if items:values.append(float(rr_predict(feats,weights['enhanced'])@np.asarray([load for _,load in items])))
        else:values.append(0.)
    return np.array(values)

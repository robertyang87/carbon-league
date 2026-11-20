"""Action-conditioned forecasts of actual credited own-base deposits in the next ten transitions."""
from collections import Counter
import numpy as np

ACTIONS=('NONE','UP','RIGHT','DOWN','LEFT')
DELTA=((0,0),(-1,0),(0,1),(1,0),(0,-1))
M=[[((p//15+y)%15)*15+(p%15+x)%15 for y,x in DELTA] for p in range(225)]
D=[[min(abs(a//15-b//15),15-abs(a//15-b//15))+min(abs(a%15-b%15),15-abs(a%15-b%15)) for b in range(225)] for a in range(225)]
def closest(p,targets):
    return min((D[p][q] for q in targets),default=16)

def features(players,seat,step,own_order):
    sides=(players[seat],players[1-seat])
    homes=[set(s[1].values()) for s in sides]
    workers=[list(s[2].items()) for s in sides]
    collectors=[[(u,int(p),float(c)) for u,(p,c,k) in ws if k=='COLLECTOR'] for ws in workers]
    totals=[sum(w[2] for w in ws) for ws in collectors]
    state=[step/299.,min(20,299-step)/20.]
    for side in (0,1):
        s=sides[side];ws=collectors[side];en=collectors[1-side];bases=homes[side]
        a=np.zeros(14)
        a[:4]=[len(ws),len(s[2])-len(ws),len(s[3]),s[0]/1000.]
        ds=[];heavy=[]
        for uid,p,c in ws:
            dist=closest(p,bases);strong=[q for _,q,ec in en if ec<=c]
            ds.append(dist)
            a[4]+=c/500.;a[5]+=c/500.*(dist<=2);a[6]+=c/500.*(dist<=5)
            a[7]+=c/500./(1+dist);a[8]+=c/500.*(closest(p,strong)<=2)
            a[9]+=c/500.*(closest(p,strong)<=4)
            a[10]+=(c>0 and dist<=3);a[11]+=c/500.*(dist<=min(10,299-step))
            if c>0 and dist<=5:heavy.append(c)
        a[12]=max(heavy,default=0)/500.;a[13]=sorted(heavy,reverse=True)[1]/500. if len(heavy)>1 else 0.
        state+=list(a)
    # Proposed destinations only for the controlled side.
    after={uid:M[w[0]][ACTIONS.index(own_order.get(uid,'NONE') or 'NONE')] for uid,w in workers[0]}
    recruited=[b for uid,b in sides[0][1].items() if own_order.get(uid) in ('RECCOLLECTOR','RECPLANTER')]
    allies=[(uid,p,c,after[uid]) for uid,p,c in collectors[0]]
    occ=Counter(after.values());occ.update(recruited)
    action=np.zeros(21)
    for uid,p,c,dst in allies:
        d0=closest(p,homes[0]);d1=closest(dst,homes[0]);strong=[q for _,q,ec in collectors[1] if ec<=c]
        weight=c/500.
        action[0]+=weight*(d0-d1);action[1]+=weight*(d1==0)
        action[2]+=weight*(p==dst);action[3]+=weight*(d1<=2);action[4]+=weight*(d1<=5)
        action[5]+=weight*(closest(dst,strong)<=1); action[6]+=weight*(occ[dst]>1)
        action[7]+=weight*(closest(dst,strong)<=2)
        guards=[y for u,_,load,y in allies if u!=uid and load<c]
        action[8]+=weight*(closest(dst,guards)<=1)
        action[9]+=(d1<d0 and c>0);action[10]+=(c==0 and p==dst)
    # The opponent's possible homeward moves are a geometric support set, with no actual enemy order supplied.
    for _,p,c in collectors[1]:
        if c<=0:continue
        weight=c/500.;d0=closest(p,homes[1])
        contest_before=[x for _,x,v,_ in allies if v<c]
        contest_after=[y for _,_,v,y in allies if v<c]
        routes=[q for q in M[p][1:] if closest(q,homes[1])<d0]
        support=routes or [p]
        action[11]+=weight*sum(q in contest_after for q in support)/len(support)
        action[12]+=weight*sum(closest(q,contest_after)<=1 for q in support)/len(support)
        action[13]+=weight*(p in contest_after)
        action[14]+=weight*(closest(p,contest_before)-closest(p,contest_after))
        action[15]+=weight*sum(q in contest_after for q in M[p])/5.
        action[16]+=weight*sum(closest(q,contest_after)<=1 for q in homes[1])/max(1,len(homes[1]))
        action[17]+=weight*all(closest(q,contest_after)<=1 for q in support)
    action[18]=len(recruited);action[19]=sum(n-1 for n in occ.values() if n>1)
    action[20]=sum(w[0]!=after[uid] for uid,w in workers[0] if w[2]=='PLANTER')
    return np.array(totals)/500.,np.array(state),action

def predict(players,seat,step,order,weights):
    cargo,state,action=features(players,seat,step,order)
    results={}
    for name,data in [('cargo',cargo),('state',state),('action',np.r_[state,action])]:
        spec=weights[name]
        x=(data-np.array(spec['mean']))/np.array(spec['scale'])
        if name=='cargo':
            y=np.maximum(0,x@np.array(spec['w'])+np.array(spec['b']))
        else:
            z=np.tanh(x@np.array(spec['w'])+np.array(spec['b']))
            y=np.logaddexp(0,z@np.array(spec['out'])+np.array(spec['bias']))
        results[name]=(y*500.).tolist()
    return results

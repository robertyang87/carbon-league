"""Price defending a tree and diverting the acting courier over the same next 16 turns."""
from math import exp
import numpy as np
from guard_tree import _D,_DIRECTION,_TO

HORIZON=16

def freight(obs,cfg,uid,action):
    s=obs['player'];own,foe=obs['players'][s],obs['players'][1-s]
    p,load,_=own[2][uid];z=int(_TO[p,_DIRECTION.index(action)])
    home=min(int(_D[z,b]) for b in own[1].values())
    enemies=[(q,v) for q,v,role in foe[2].values() if role=='COLLECTOR']
    tax=0
    # Trees collect within the wrapped 3x3 neighborhood, including diagonal cells.
    for cell,age in foe[3].values():
        dy=abs(cell//15-z//15);dx=abs(cell%15-z%15)
        if min(dy,15-dy)<=1 and min(dx,15-dx)<=1 and age<cfg['treeLifespan']:tax+=1
    threat=sum(_D[z,q]<=1 and v<=load for q,v in enemies)
    c=load/100.
    return [1.,c,home,c*home,home<=1,c*(home<=1),min(tax,3)*c,threat>0,c*(threat>0),
            not action,c*(not action),min(299-obs['step'],HORIZON)/HORIZON]

def tree_features(obs,tree,cfg):
    s=obs['player'];cell,age=obs['players'][s][3][tree]
    pay=float((obs['trees'].get(tree) or [None,0])[1] or 0)
    left=min(HORIZON,cfg['treeLifespan']-age,299-obs['step'])/HORIZON
    return [1.,left,pay,pay*left,age/50.,min(pay,4.),min(pay,12.)]

def threat(obs,cmd,cfg,root,prev,w):
    seat=obs['player'];own,enemy=obs['players'][seat],obs['players'][1-seat]
    cell=root['cell'];age=root['age'];t=obs['step'];foe=root['enemy'];tree=root['tree']
    oc=[_D[cell,v[0]] for v in own[2].values() if v[2]=='COLLECTOR']
    ec=[_D[cell,v[0]] for v in enemy[2].values() if v[2]=='COLLECTOR']
    ep=[_D[cell,v[0]] for v in enemy[2].values() if v[2]=='PLANTER']
    lands=[_D[int(_TO[v[0],_DIRECTION.index(cmd.get(k))]),cell] for k,v in own[2].items() if v[2]=='COLLECTOR']
    x=[1,age/50,t/299,min(299-t,50-age)/50,(obs['trees'].get(tree) or [None,0])[1]/5,
       min(enemy[0]/100,4),sum(v<=2 for v in oc),sum(v<=2 for v in ec),min(min(oc,default=15),8)/4,
       sum(v<=2 for v in ep),foe in prev and _D[prev[foe][0],cell]>1,min(min(lands,default=15),8)/4,
       sum(v<=1 for v in lands),sum(v<=2 for v in lands)]
    return 1/(1+exp(-float(np.dot(x,w))))

def price(obs,cmd,cfg,root,prev,weights):
    altered=dict(cmd);uid=root['uid'];altered[uid]=root['after']
    p0=threat(obs,cmd,cfg,root,prev,weights['threat'])
    p1=threat(obs,altered,cfg,root,prev,weights['threat'])
    rent=max(0.,float(np.dot(tree_features(obs,root['tree'],cfg),weights['tree'])))
    # A successful seizure costs the opponent cash; preventing it leaves that cash with the opponent.
    margin=(p0-p1)*(2*rent-cfg['seizeCost'])
    receipt=lambda a:max(0.,float(np.dot(freight(obs,cfg,uid,a),weights['freight'])))
    base,changed=receipt(cmd.get(uid)),receipt(root['after'])
    gain=margin-max(0.,base-changed)
    return dict(p_parent=p0,p_screen=p1,tree_receipts=rent,courier_parent=base,courier_screen=changed,
                expected_tree_margin=margin,net=gain,accept=gain>0)

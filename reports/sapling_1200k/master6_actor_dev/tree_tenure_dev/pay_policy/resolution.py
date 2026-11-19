"""Offline with the submitted simultaneous enemy order; never used by the live package."""
import copy
from opportunities import CashEvents
import cnet2_feat as F


def focal(sim,joint,s,uid,cell):
    obs=sim.obs(s); arrivals=[]; public_dist=[]
    size=sim.size
    for side,pl in enumerate(obs['players']):
        for worker,(p,cargo,role) in pl[2].items():
            if worker==uid:continue
            cmd=joint[side].get(worker,'NONE'); follows=int(F.DEST[p,F.MOVES.index(cmd)])
            if side!=s:
                dr=abs(p//size-cell//size);dc=abs(p%size-cell%size)
                public_dist.append(min(dr,size-dr)+min(dc,size-dc))
            if follows==cell:
                arrivals.append(dict(side=side,uid=worker,role=role,origin=p,command=cmd,cargo=float(cargo)))
    branch=copy.deepcopy(sim);branch.cash=CashEvents(list(branch.cash));branch.step(*joint)
    after=branch.obs(s); worker=after['players'][s][2].get(uid)
    pay=sum(-v for k,who,v,*a in branch.cash.events if k in ('plant','seize') and who==s and a[0]==uid)
    return dict(alive_after=worker is not None,cell_after=worker[0] if worker is not None else None,
                focus_debit=pay,other_endpoints=arrivals,nearest_enemy_before=min(public_dist,default=None))


def both(sim,joint,s,trace):
    uid=trace['uid'];center=trace['cell']
    move=[dict(z) for z in joint];pay=[dict(z) for z in joint]
    move[s]=dict(trace['base']);pay[s]=dict(trace['base']);pay[s].pop(uid,None)
    a=focal(sim,move,s,uid,center);b=focal(sim,pay,s,uid,center)
    return dict(parent=a,pay=b,survival_flip=int(b['alive_after'])-int(a['alive_after']))

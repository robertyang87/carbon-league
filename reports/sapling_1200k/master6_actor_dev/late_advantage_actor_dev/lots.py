"""Measure strictly new mining and what its original worker actually delivered to its own base."""
from collections import defaultdict
from math import isclose
from pathlib import Path
import sys
TOP=Path(__file__).resolve().parents[2]/"all_top5"
sys.path.insert(0,str(TOP))
from defense import DEST,MOVE
from economy_extract import visible_ledger

FATES=("original_worker_home","tax","capture","terminal_inventory","other_loss")


def measure(game,side,lo=225):
    frames=game["steps"]; cfg=game["configuration"]; T=len(frames)-1
    if cfg["moveCost"] or cfg["smeltCost"] or cfg["size"]!=15:raise ValueError("unsupported rules")
    state=[r[0]["observation"]["players"] for r in frames]
    bankable={u:(0.,0.,0.,1.,0.) for u,w in state[-1][side][2].items() if w[2]=="COLLECTOR"}
    result={}; actual_deposit=0
    for t in reversed(range(lo,T)):
        pre,post=state[t:t+2]; acts=[z["action"] or {} for z in frames[t+1]]
        old_trees={tid for p in pre for tid in p[3]}
        taxed=[pos for tid,(pos,_) in post[1-side][3].items() if tid in old_trees]
        old=sum((list(z[1].values()) for z in pre),[]);ourbases=set(pre[side][1].values()); bases=set(old)
        landing=defaultdict(float)
        survivor_by_cell={v[0]:(p,u) for p,player in enumerate(post) for u,v in player[2].items() if v[2]=="COLLECTOR"}
        for p,player in enumerate(pre):
            for uid,(pos,cargo,kind) in player[2].items():
                if kind=="COLLECTOR":landing[DEST[pos][MOVE[acts[p].get(uid)]]]+=cargo
        back={}
        for uid,(pos,principal,role) in pre[side][2].items():
            if role!="COLLECTOR":continue
            cmd=acts[side].get(uid) or "NONE"; cell=DEST[pos][MOVE[cmd]]
            ending=post[side][2].get(uid)
            fresh=0.; raw=0.; recovered=0.
            if ending is None:
                captured=cell in survivor_by_cell and survivor_by_cell[cell][0]!=side
                back[uid]=(0.,0.,float(captured),0.,float(not captured))
            elif cell in bases:
                if cell in ourbases:
                    actual_deposit+=landing[cell]
                    back[uid]=(1.,0.,0.,0.,0.)
                else: back[uid]=(0.,0.,0.,0.,1.)
            else:
                if ending[0]!=cell:raise ValueError("worker destination")
                y,x=divmod(cell,15)
                def near(z):
                    yy,xx=divmod(z,15);return min(abs(y-yy),15-abs(y-yy))<=1 and min(abs(x-xx),15-abs(x-xx))<=1
                rate=1-cfg["collectorAbsorptionRate"]*sum(map(near,taxed))
                if not 0<rate<=1:raise ValueError("unsupported tax")
                # A surviving collector receives all pre-existing cargo at its collision cell.
                # The residual after accounting for original and collision cargo is fresh mining after tax.
                fresh=ending[1]-landing[cell]*rate
                if fresh < -1e-5 or cmd!="NONE" and abs(fresh)>1e-5:raise ValueError(("movement mined",t,uid,fresh,cmd))
                fresh=max(0.,fresh);raw=fresh/rate
                later=bankable[uid]
                recovered=fresh*later[0]
                back[uid]=tuple(rate*v+(1-rate if i==1 else 0) for i,v in enumerate(later))
            if abs(sum(back[uid])-1)>1e-8:raise ValueError("principal mass not conserved")
            # Current-step tree tax on new mining precedes forward delivery.
            factors=bankable.get(uid,(0.,0.,0.,0.,1.))
            newfate=[fresh*x for x in factors];newfate[1]+=raw-fresh
            if abs(sum(newfate)-raw)>1e-5:raise ValueError("new-lot mass not conserved")
            result[t,uid]={"gross":raw,"direct":recovered,"new_fates":newfate,"existing_direct":principal*back[uid][0]}
        bankable=back
    # A unit newly born on a base can also receive collision cargo; deposit check therefore uses a one-sided bound.
    flows=visible_ledger(frames,cfg)[0]
    bank=sum(row[side][0] for row in flows[lo:])
    whole=sum(v["direct"] for v in result.values())
    if whole>bank+.001 or actual_deposit>bank+.001:raise ValueError(("collected capital exceeds recorded deposits",whole,bank))
    return result,{"unique_worker_transitions":len(result),"new_gross":sum(v["gross"] for v in result.values()),
                   "new_fates":[sum(v["new_fates"][j] for v in result.values()) for j in range(5)],
                   "official_own_worker_deposit":bank}

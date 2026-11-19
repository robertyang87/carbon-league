"""One optional late-collector residual. Public spatial inputs only, in physical coordinates."""
from math import isfinite, log1p

MOVES=("NONE","UP","RIGHT","DOWN","LEFT")
OFFSETS=((0,0),(-1,0),(0,1),(1,0),(0,-1))
FEATURES=("bias","new_mining","principal_progress","principal_reachable","principal_at_base","lighter_attack","tree_tax","time_slack")


def step(pos,cmd,size):
    y,x=divmod(pos,size);dy,dx=OFFSETS[MOVES.index(cmd or "NONE")]
    return ((y+dy)%size)*size+(x+dx)%size


def distance(a,b,size):
    ay,ax=divmod(a,size);by,bx=divmod(b,size);dy,dx=abs(ay-by),abs(ax-bx)
    return min(dy,size-dy)+min(dx,size-dx)


def nearby(a,b,size):
    ay,ax=divmod(a,size);by,bx=divmod(b,size);dy,dx=abs(ay-by),abs(ax-bx)
    return max(min(dy,size-dy),min(dx,size-dx))<=1


def variants(obs,cfg,parent_actions):
    """At most one fixed public alternative, common to the training and deployment entry points."""
    t=obs["step"]; n=cfg.get("size",15);left=cfg.get("episodeSteps",300)-1-t
    if n!=15 or not 286<=t<=296 or left<2:return []
    s=obs["player"];own,enemy=obs["players"][s],obs["players"][1-s]
    bases=set(own[1].values()); theirs=set(enemy[1].values())
    if not bases:return []
    trees={v[0] for p in (own,enemy) for v in p[3].values()}
    enemy_trees=[v[0] for v in enemy[3].values() if v[1]<cfg.get("treeLifespan",50)]
    foes=[v for v in enemy[2].values() if v[2]=="COLLECTOR"]
    destinations={wid:step(w[0],parent_actions.get(wid,"NONE"),n) for wid,w in own[2].items()}
    hiring={p for bid,p in own[1].items() if parent_actions.get(bid) in ("RECPLANTER","RECCOLLECTOR")}
    proposals=[]
    for uid,(pos,cargo,role) in own[2].items():
        if role!="COLLECTOR":continue
        cmd=parent_actions.get(uid) or "NONE"
        if cmd not in MOVES:continue
        base_distance=min(distance(pos,b,n) for b in bases)
        if base_distance>left+1 or base_distance==0:continue
        obstacles=hiring|{p for other,p in destinations.items() if other!=uid}
        def attributes(a):
            dest=step(pos,a,n);d=min(distance(dest,b,n) for b in bases)
            mining=obs["carbon"][pos]*cfg.get("initialCollectRate",.3) if a=="NONE" and pos not in trees and pos not in bases|theirs else 0.
            return [mining/30 if d<=left-1 else 0.,max(-1.,min(1.,base_distance-d))*min(cargo/100,4),
                    float(d<=left-1)*min(cargo/100,4),float(dest in bases)*min(cargo/100,4),
                    float(any(w[1]<=cargo and distance(w[0],dest,n)<=1 for w in foes))*min(1+cargo/100,4),
                    sum(nearby(dest,z,n) for z in enemy_trees)*min(cargo/100,4),
                    min(max(left-1-d,-5),5)/5]
        origin=attributes(cmd)
        if cmd=="NONE":
            alternatives=[a for a in MOVES[1:] if min(distance(step(pos,a,n),b,n) for b in bases)<base_distance]
        else:
            alternatives=["NONE"] if min(distance(step(pos,cmd,n),b,n) for b in bases)<base_distance else []
        for alt in alternatives:
            dest=step(pos,alt,n)
            if dest in obstacles or dest in theirs:continue
            new=attributes(alt)
            f=[1.]+[v-u for v,u in zip(new,origin)]
            # Restrict to real mining versus a viable one-step return direction; no hypothetical unseen ground yield.
            if max(origin[0],new[0])<=0 or max(origin[2],new[2])==0 and cargo>0:continue
            urgency=1+max(0,min(1,(base_distance+3-left)/3))
            sw=1.7*f[0]+(.45*urgency)*f[1]+2*f[2]+1.4*f[3]-.6*f[4]-.15*f[5]
            priority=sw+.22*max(origin[0],new[0])+.05*min(cargo/100,4)
            proposals.append({"uid":uid,"original":cmd,"action":alt,"features":f,"priority":priority})
    return [max(proposals,key=lambda p:(p["priority"],p["uid"],-MOVES.index(p["action"])))] if proposals else []


def predict(proposal,weights):
    x=proposal["features"]
    if tuple(weights["features"])!=FEATURES or not all(isfinite(v) for v in x):raise ValueError("unexpected residual features")
    return {k:sum(a*b for a,b in zip(x,coef,strict=True)) for k,coef in weights["heads"].items()}


def residual_action(obs,cfg,parent_actions,weights):
    if not weights or not weights.get("enabled"):return {}
    choices=variants(obs,cfg,parent_actions)
    if not choices:return {}
    p=choices[0];z=predict(p,weights)
    if not(z["margin"]>10 and z["principal"]>=-5 and z["own_delivery"]>=-5):return {}
    return {p["uid"]:p["action"]}

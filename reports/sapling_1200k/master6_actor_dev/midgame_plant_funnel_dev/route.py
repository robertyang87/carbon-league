"""Adjacent site approach with a single conditional payment on the next turn."""
from decision import _D, _DIRECTION, _TO, predicted, quote, values

def mode(obs, joint, uid, target, cfg):
    seat=obs["player"];own=obs["players"][seat];enemy=obs["players"][1-seat]
    sites={p:s for s,pl in enumerate(obs["players"]) for p,_ in pl[3].values()}
    homes={p for pl in obs["players"] for p in pl[1].values()}
    if target in homes or target in sites:return False
    if any(v[0]==target for pl in obs["players"] for k,v in pl[2].items() if k!=uid):return False
    if any(_D[target,v[0]]<=1 for v in enemy[2].values()):return False
    if any(_TO[v[0],_DIRECTION.index(joint.get(k))]==target for k,v in own[2].items() if k!=uid):return False
    fee=quote(obs,cfg); reserve=0
    for k,(p,_,role) in own[2].items():
        if k!=uid and role=="PLANTER" and not joint.get(k) and p not in homes:
            reserve+=fee if p not in sites else (cfg["seizeCost"] if sites[p]!=seat else 0)
    for base in own[1]:
        order=joint.get(base)
        if order:
            role=order[3:].title()
            count=sum(w[2]==role.upper() for pl in obs["players"] for w in pl[2].values())
            reserve+=cfg["rec"+role+"Cost"]+count*cfg["rec"+role+"IncreaseCost"]
    return own[0]>=reserve+fee

def approach(obs,mem,joint,cfg,weights):
    t=obs["step"];s=obs["player"];own=obs["players"][s];enemy=obs["players"][1-s]
    if not 100<=t<200:return None
    enemytrees={p for p,age in enemy[3].values()}
    fee=quote(obs,cfg);best=None;best_key=None
    for uid,(cell,_,role) in own[2].items():
        old=joint.get(uid)
        if role!="PLANTER" or old not in _DIRECTION[1:]:continue
        before=int(_TO[cell,_DIRECTION.index(old)])
        if cell in enemytrees or before in enemytrees:continue
        original=values(obs,mem,uid,before,cfg)
        net0=max(0.,predicted(original,weights)-fee) if original is not None else 0.
        for i,target in enumerate(_TO[cell][1:],1):
            target=int(target)
            if target==before or not mode(obs,joint,uid,target,cfg):continue
            v=values(obs,mem,uid,target,cfg)
            if v is None or v[3]<.5 or v[2]<1.2:continue
            net=predicted(v,weights)-fee;gain=net-net0
            if gain<=10:continue
            key=(gain,uid,-i)
            if best_key is None or key>best_key:
                best_key=key
                best=dict(t=t,worker=uid,old=old,alternative=_DIRECTION[i],tree=None,cell=cell,
                          target=target,prior_destination=before,known_fraction=v[3],remembered_total=v[2]*100,
                          quoted_fee=fee,target_net=net,alternative_prior_net=net0,predicted_advantage=gain)
    return best

def pay(obs,mem,joint,cfg,root,weights):
    s=obs["player"];t=obs["step"];uid=root["worker"];target=root["target"]
    w=obs["players"][s][2].get(uid)
    if t!=root["t"]+1 or w is None or w[2]!="PLANTER" or w[0]!=target:
        return dict(joint),"did_not_arrive"
    if not mode(obs,joint,uid,target,cfg):return dict(joint),"unsafe_or_unfunded"
    # At the target the planter can see all four neighbors, independently of its earlier estimate.
    visible=sum(obs["carbon"][int(q)] for q in _TO[target][1:])
    v=values(obs,mem,uid,target,cfg)
    if visible<120 or v is None or predicted(v,weights)<=quote(obs,cfg):
        return dict(joint),"real_site_below_threshold"
    revised=dict(joint);revised.pop(uid,None)
    return revised,"pay"

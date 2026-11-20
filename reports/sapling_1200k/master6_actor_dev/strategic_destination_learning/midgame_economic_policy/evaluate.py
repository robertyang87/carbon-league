"""Replay the public decision entry for frozen releases. Historical returns are labels only."""
from collections import Counter, defaultdict
import numpy as np
import torch

import learn as L

MOVES={"NONE":(0,0),"UP":(-1,0),"RIGHT":(0,1),"DOWN":(1,0),"LEFT":(0,-1)}
ARMS=("parent","uniform","weighted")


def move(p,act):
    y,x=divmod(p,15);dy,dx=MOVES[act];return ((y+dy)%15)*15+(x+dx)%15


def actual_geometry(obs,cmd):
    s=obs["player"];our=obs["players"][s];their=obs["players"][1-s]
    enemy=[(p,w) for p,w,k in their[2].values() if k=="COLLECTOR"]
    own_heavy=[(p,w) for p,w,k in our[2].values() if k=="COLLECTOR" and w>=75 and
               any(b<w and L.distance(p,q)<=3 for q,b in enemy)]
    events=defaultdict(list)
    for uid,(p,w,k) in our[2].items():
        if k!="COLLECTOR":continue
        q=move(p,cmd.get(uid,"NONE"))
        if q==p:continue
        if w>=75 and min(L.distance(q,b) for b in our[1].values())<min(L.distance(p,b) for b in our[1].values()):
            events["loaded_toward_base"].append(uid)
        if w>=75 and any(b<w and L.distance(p,e)<=3 for e,b in enemy):
            risk=[e for e,b in enemy if b<w and L.distance(e,p)<=3]
            if min(L.distance(q,e) for e in risk)>min(L.distance(p,e) for e in risk):
                events["withdraw_from_lighter"].append(uid)
        if any(w<hw and L.distance(q,h)<=1<L.distance(p,h) for h,hw in own_heavy):
            events["approach_at_risk_teammate"].append(uid)
        if any(w<hw and hw>=75 and L.distance(e,q)==1 and q not in their[1].values()
               and min(L.distance(q,b) for b in their[1].values())<min(L.distance(e,b) for b in their[1].values())
               for e,hw in enemy):
            events["enter_possible_return_lane"].append(uid)
    return dict(events)


def tally(rows):
    stats={a:Counter() for a in ARMS}
    geo={a:Counter() for a in ARMS};changed=Counter()
    supported=[r for r in rows if r["supported"]]
    for r in rows:
        for a in ARMS:
            metrics=stats[a]
            metrics["rows"]+=1;metrics["eligible_full_demonstration"]+=r["supported"]; metrics["guard_rejection"]+=r[a]["guard_rejected"]
            same=r["expert"]==r[a]["actions"];metrics["joint_hit"]+=same
            for u,role in r["roles"].items():
                ok=r[a]["actions"].get(u,"NONE")==r["expert"].get(u,"NONE")
                metrics[role+"_correct"]+=ok;metrics[role+"_total"]+=1
            for name,units in r[a]["action_geometry"].items():geo[a][name]+=len(units)
        for lft,rgt in (("uniform","weighted"),("parent","uniform"),("parent","weighted")):
            changed[lft+"_vs_"+rgt]+=r[lft]["actions"]!=r[rgt]["actions"]
    def map_mean(key, subset):
        by=defaultdict(list)
        for r in subset:by[r["terrain"]].append(float(r[key]["actions"]==r["expert"]))
        return np.mean([np.mean(v) for v in by.values()]).item() if by else None
    return dict(roots=len(rows),maps=len({r["terrain"] for r in rows}),stats=stats,changed=changed,
                map_equal_full_joint_agreement={a:map_mean(a,supported) for a in ARMS},action_geometry=geo)


def main():
    fit=L.c.read(L.HERE/"fit.json");require=L.require
    require(fit["source"]==L.c.sha(L.HERE/"learn.py") and fit["registered"]==L.c.sha(L.HERE/"protocol.json"),"fit source")
    paths={"parent":L.PARENT,**{k:L.HERE/(k+".py") for k in L.ARMS}}
    expected={"parent":L.RULE["parent"],**{k:fit["arms"][k]["package"] for k in L.ARMS}}
    require(all(L.c.sha(p)==expected[k] for k,p in paths.items()),"released identity")
    require(fit["arms"]["weighted"]["updates"]==fit["arms"]["uniform"]["updates"],"unequal budgets")
    models={k:L.prefix.load("midgame_release_"+k,p) for k,p in paths.items()}
    probers={k:L.c.package(p) for k,p in paths.items()}
    targets=[];spot=set()
    for card in L.cards():
        if card["split"]=="train":continue
        game=L.archive(card);selected=dict(L.source_rows(card,game));seat=card["seat"]
        if not selected:continue
        spot_check=seat not in spot
        if spot_check:
            spot.add(seat);cold={k:L.prefix.load("trajectory_check_"+k,p) for k,p in paths.items()}
        F=L.sys.modules["cnet2_feat"];feature=F.Feat(seat)
        for t in range(max(selected)+1):
            obs=game._obs(t,seat);cfg={k:v for k,v in game._data["configuration"].items() if k!="randomSeed"}
            if spot_check:
                truth={k:m.agent(obs,cfg) for k,m in cold.items()}
            x=feature.observe(obs)
            # Between sampled roots the only persisting actor state is its observable feature history.
            # The two fully continuous cold comparisons (one per physical seat) exercise that equivalence.
            if t==0 or t in selected:
                actions={k:m.agent(obs,cfg) for k,m in models.items()}
                if spot_check:require(truth==actions,"sparse and continuous public calls differ")
            else:
                for m in models.values():m._FEAT.observe(obs)
            if t not in selected:continue
            r=selected[t];expert=game._frames[t+1][seat]["action"] or {};r["expert"]=expert
            r["roles"]={u:k.lower() for u,(_,_,k) in obs["players"][seat][2].items()}
            r["roles"].update({u:"base" for u in obs["players"][seat][1]})
            gold=L.c.actual_features(feature,x,expert,probers["parent"])
            r["supported"]=bool(not np.any(x["tok_mask"][gold["cut"]:F.OWN]) and gold["eligible"].any())
            for k,m in probers.items():
                sent,rej,_=L.prefix.free(m,x,feature)
                require(feature.to_actions(sent)==actions[k],"full normal commands differ from independently decoded own prefix")
                checked=L.c.actual_features(feature,x,actions[k],m)
                require(checked["cut"]==F.OWN and set(actions[k])<=r["roles"].keys(),"infeasible student complete action")
                r[k]=dict(actions=actions[k],guard_rejected=bool(rej),action_geometry=actual_geometry(obs,actions[k]))
            targets.append(r)
        print("checked",card["split"],card["id"],len(targets),flush=True)
    require(spot=={0,1},"missing continuous seat checks")
    rmodel=L.c.read(L.HERE/"ridge.json")
    for r,p in zip(targets,L.predict(rmodel["train_all"],targets)):
        r["baseline"]=p;r["residual"]=r["target"]-p
    L.c.save(L.HERE/"held_commands.json.gz",targets)
    combined=tally(targets)
    split={s:tally([r for r in targets if r["split"]==s]) for s in ("val","holdout")}
    def by(key):
        return {str(k):tally([r for r in targets if r[key]==k]) for k in sorted({r[key] for r in targets})}
    changed=combined["changed"]["uniform_vs_weighted"]
    L.c.save(L.HERE/"entry.json",dict(registered=fit["registered"],fit=L.c.sha(L.HERE/"fit.json"),export_sha256=expected,
                                      trace=L.c.sha(L.HERE/"held_commands.json.gz"),evaluation_source=L.c.sha(L.HERE/"evaluate.py"),
                                      eligible=changed>0,continuous_cold_control_seats=sorted(spot),all_commands_valid=True,
                                      max_evaluated_t=max(r["t"] for r in targets),
                                      pooled=combined,split=split,by_physical_seat=by("seat"),by_opportunity=by("opportunity"),
                                      by_t=by("t"),ahead=tally([r for r in targets if r["cash_margin"]>0]),
                                      behind=tally([r for r in targets if r["cash_margin"]<0]),
                                      high_realized=tally([r for r in targets if r["residual"]>0]),
                                      low_realized=tally([r for r in targets if r["residual"]<=0]),
                                      by_map=by("terrain")))


if __name__=="__main__":
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    main()

"""Two legal simultaneous own joint orders, independently valued from the same public history."""
from functools import lru_cache
import importlib.util
import math
from pathlib import Path

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("existing_planter_geometry",HERE.parent/"tree_response_training/planter_survival/model.py")
geom=importlib.util.module_from_spec(spec)
spec.loader.exec_module(geom)
EXTRA=["root_kind","destination_kind","destination_quote","own_cash","cash_above_price",
       "remaining","tree_age","visible_neighbor_carbon","visible_neighbor_count","global_live_trees","own_trees",
       "adjacent_enemy_trees","planter_rehire","own_planters","own_collectors","opp_planters","opp_collectors"]
NAMES=geom.NAMES+EXTRA

def public(turn,seat):
    actors=turn[0]["observation"]
    # In archived multi-seat format only board-common values fall back to the first seat.
    view=turn[seat]["observation"]
    return {"player":seat,"step":actors["step"],"players":actors["players"],"trees":actors["trees"],
            "carbon":view["carbon"]}

def features(obs,history,joint,uid,cfg):
    cmd=joint.get(uid) or "NONE"
    x,reach=geom.features(obs,history,joint,uid,cmd)
    players=obs["players"]; seat=obs["player"];own,opp=players[seat],players[1-seat]
    origin=own[2][uid][0];size=int(cfg.get("size",15));dest=geom.destination(origin,cmd,size)
    trees={p:(s,age) for s,v in enumerate(players) for p,age in v[3].values()}
    lifespan=cfg.get("treeLifespan",50)
    live=sum(v[1]<lifespan for v in trees.values())
    bases={p for side in players for p in side[1].values()}
    def kind(p):
        if p in bases:return 3
        if p not in trees or trees[p][1]>=lifespan:return 0
        return 2 if trees[p][0]==seat else 1
    typ=kind(dest)
    price=cfg.get("seizeCost",20) if typ==1 else cfg.get("plantCost",20)
    if typ==0 and len(own[3])>cfg.get("playerTreeProtectiveNumber",0):
        price+=cfg.get("plantCostInflationRatio",5)*cfg.get("plantCostInflationBase",1.235)**live
    if typ>=2:price=0.
    neighbors=[geom.destination(dest,a,size) for a in ("UP","RIGHT","DOWN","LEFT")]
    visible=[obs["carbon"][i] for i in neighbors if i not in trees and obs["carbon"][i] is not None]
    counts=[sum(w[2]==k for w in p[2].values()) for p in (own,opp) for k in ("PLANTER","COLLECTOR")]
    rehire=cfg.get("recPlanterCost",30)+(counts[0]+counts[2])*cfg.get("recPlanterIncreaseCost",0)
    x.extend([kind(origin),typ,price,own[0],own[0]-price,cfg.get("episodeSteps",300)-1-obs["step"],
              trees.get(dest,(None,-1))[1],sum(visible),len(visible),live,len(own[3]),
              sum(kind(i)==1 for i in neighbors),rehire,*counts])
    return x,rehire

def _predict(ensemble,x):
    raw=ensemble["initial"]
    for tree in ensemble["trees"]:
        i=0
        while tree["l"][i]>=0:
            i=tree["l"][i] if x[tree["f"][i]] <= tree["t"][i] else tree["r"][i]
        raw+=tree["v"][i]
    return raw

def scores(model,obs,history,joint,uid,cfg):
    if model["features"]!=NAMES:raise ValueError("two-arm feature schema changed")
    x,rehire=features(obs,history,joint,uid,cfg)
    arm="stay" if not joint.get(uid) or joint.get(uid)=="NONE" else "move"
    color=model["arms"][arm]
    d=_predict(color["death"],x)
    logistic_slope,logistic_intercept=color["death_cal"]
    p=1/(1+math.exp(-max(-35.,min(35.,d*logistic_slope+logistic_intercept))))
    econ=_predict(color["net"],x)
    slope,intercept=color["net_cal"]
    net=econ*slope+intercept
    return {"p_death":p,"net_tree":net,"utility":net-rehire*p,"rehire_quote":rehire}

def decide(model,obs,history,stay_joint,move_joint,uid,cfg):
    if (stay_joint.get(uid) or "NONE")!="NONE" or (move_joint.get(uid) or "NONE")=="NONE":
        raise ValueError("distinct stay/move orders required")
    if any((stay_joint.get(k) or "NONE") != (move_joint.get(k) or "NONE") for k in stay_joint.keys()|move_joint.keys() if k!=uid):
        raise ValueError("nonfocal order differs")
    a=scores(model,obs,history,stay_joint,uid,cfg);b=scores(model,obs,history,move_joint,uid,cfg)
    return {"stay":a,"move":b,"pick":"stay" if a["utility"]>b["utility"] else "move"}

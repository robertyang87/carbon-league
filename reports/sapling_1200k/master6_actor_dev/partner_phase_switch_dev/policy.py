"""One economic assessment at the declared turn, followed by a persistent partner choice."""
from statistics import mean


def distance(a,b):
    r,c=divmod(a,15);y,x=divmod(b,15)
    return min(abs(r-y),15-abs(r-y))+min(abs(c-x),15-abs(c-x))


def accounts(obs,side):
    """All positions, cargo, ages and both players' cash are public."""
    own,other=obs["players"][side],obs["players"][1-side]
    homes=list(own[1].values())
    rivals=[w for w in other[2].values() if w[2]=="COLLECTOR"]
    collectors=[w for w in own[2].values() if w[2]=="COLLECTOR"]
    bankable=risk=0.; exposed=0
    for pos,load,_ in collectors:
        home_dist=min((distance(pos,h) for h in homes),default=99)
        charged=any(rival_load<load and distance(pos,rival_pos)<=2 for rival_pos,rival_load,_ in rivals)
        if charged and load>0:risk+=load;exposed+=1
        if home_dist<=4 and not charged:bankable+=load
    return {"cash":own[0],"near_safe_cargo":bankable,"contested_cargo":risk,"at_risk_collectors":exposed,
            "collectors":len(collectors),"planters":len(own[2])-len(collectors),
            "tree_remaining_fraction":sum(max(0.,50-age)/50. for _,age in own[3].values())}


def snapshot(obs,history):
    if "full_carbon" in obs or any("full_carbon" in o for o in history):
        raise ValueError("per-seat public observations only")
    s=obs["player"];t=obs["step"]
    if len(history)<10 or history[-10]['step']!=t-10 or history[-1]['step']!=t-1:
        raise ValueError("ten consecutive prior observations are required")
    if any(o["player"]!=s or o["step"]!=t-len(history)+i for i,o in enumerate(history)):
        raise ValueError("discontinuous/cross-seat history")
    a,b=accounts(obs,s),accounts(obs,1-s)
    cashflow=(a["cash"]-history[-10]["players"][s][0])-(b["cash"]-history[-10]["players"][1-s][0])
    freight=b["near_safe_cargo"]-a["near_safe_cargo"]+a["contested_cargo"]-b["contested_cargo"]
    output=b["collectors"]-a["collectors"]+.25*(b["tree_remaining_fraction"]-a["tree_remaining_fraction"])
    flags=[freight>=50, output>=1, cashflow<=-30]
    return {"us":a,"them":b,"relative_ten_step_cash_change":cashflow,"freight_exposure":freight,
            "productive_capacity_gap":output,"pressure_flags":flags,
            "regime":"pressure" if sum(flags)>=2 else "stable"}


def choose(model,obs,history):
    return model["regime_to_partner"][snapshot(obs,history)["regime"]]


def estimate(rows):
    """Fit exactly two values from independent-map primary responses. No map/seat as model input."""
    out={}
    for g in ("pressure","stable"):
        r=[v["delta"] for v in rows if v["regime"]==g]
        if len(r)<2:raise ValueError("insufficient distinct training maps")
        out[g]="it20" if mean(r)>0 and sum(d>0 for d in r)>=2 else "it50"
    return {"regime_to_partner":out}

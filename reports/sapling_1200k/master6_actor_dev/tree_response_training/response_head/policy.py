"""Runtime operations need only the current public observation, both moves and frozen v1's own commands."""
import math

FEATURE_NAMES=("share_of_available_cash","visible_adjacent_carbon_pay_minus_defer",
               "effective_remaining_tenure","unsupported_contact_pay_minus_defer",
               "enemy_cargo_within_four_of_own_base")
_DELTA={"NONE":(0,0),"UP":(-1,0),"RIGHT":(0,1),"DOWN":(1,0),"LEFT":(0,-1)}


def _cell(p,cmd,n):
    dr,dc=_DELTA[cmd]
    return n*((p//n+dr)%n)+(p%n+dc)%n


def _dist(p,q,n):
    dr=abs(p//n-q//n);dc=abs(p%n-q%n)
    return min(dr,n-dr)+min(dc,n-dc)


def features(obs,cfg,uid,pay_cmd,defer_cmd,v1_commands):
    if (pay_cmd or "NONE")!="NONE" or defer_cmd not in _DELTA or defer_cmd=="NONE":
        raise ValueError("pay must stay; defer must name an exact physical move")
    n=cfg["size"];s=obs["player"];t=obs["step"]
    if n!=15 or s not in (0,1):raise ValueError("unsupported board")
    own,opp=obs["players"][s],obs["players"][1-s]
    if uid not in own[2] or own[2][uid][2]!="PLANTER":raise ValueError("not an acting planter")
    old=v1_commands.get(uid) or "NONE"
    if old not in ("NONE",defer_cmd):raise ValueError("v1 moved differently from tested defer action")
    at=own[2][uid][0];there=_cell(at,defer_cmd,n)
    trees={p:(seat,age) for seat,pl in enumerate(obs["players"]) for p,age in pl[3].values()}
    bases={p for pl in (own,opp) for p in pl[1].values()}
    life=cfg["treeLifespan"];live=sum(a<life for _,a in trees.values())
    new_price=float(cfg["plantCost"])
    if len(own[3])>cfg["playerTreeProtectiveNumber"]:new_price+=cfg["plantCostInflationRatio"]*cfg["plantCostInflationBase"]**live
    def quote(p):
        if p in bases:return None
        tr=trees.get(p)
        if tr and tr[1]<life:return float(cfg["seizeCost"]) if tr[0]!=s else None
        return new_price
    price=quote(at)
    if price is None or price<=0:raise ValueError("no current payable tree/soil")
    population={w:sum(r[2]==w for pl in (own,opp) for r in pl[2].values()) for w in ("COLLECTOR","PLANTER")}
    spare=max(0,cfg["workerLimit"]-len(own[2]));cash=float(own[0])
    hire={"RECCOLLECTOR":"Collector","RECPLANTER":"Planter"}
    for bid in own[1]:
        kind=hire.get(v1_commands.get(bid))
        if kind and spare>0:
            amt=cfg["rec"+kind+"Cost"]+population[kind.upper()]*cfg["rec"+kind+"IncreaseCost"]
            if cash>=amt:cash-=amt;spare-=1
    for other,(p,_,kind) in own[2].items():
        if other!=uid and kind=="PLANTER" and (v1_commands.get(other) or "NONE")=="NONE":
            cash-=quote(p) or 0.
    def neighbors(p):return [_cell(p,k,n) for k in ("UP","RIGHT","DOWN","LEFT")]
    visible=set()
    for p,_,kind in own[2].values():
        if kind=="COLLECTOR":visible.add(p)
        else:visible.update(neighbors(p))
    def adjacent(p):
        return sum(float(obs["carbon"][q]) for q in neighbors(p) if q in visible and q not in bases and q not in trees)/400.
    tr=trees.get(at)
    life_remaining=(life-tr[1]-1) if tr and tr[1]<life else life
    turns=max(0,min(life_remaining,cfg["episodeSteps"]-2-t))
    guards=[p for p,_,kind in own[2].values() if kind=="COLLECTOR"]
    hostile=[p for p,_,_ in opp[2].values()]
    def unsupported(p):
        ed=min((_dist(p,q,n) for q in hostile),default=15);gd=min((_dist(p,q,n) for q in guards),default=15)
        return max(0.,1./(1+ed)-1./(1+gd))
    freight=sum(float(c) for p,c,kind in opp[2].values() if kind=="COLLECTOR" and any(_dist(p,b,n)<=4 for b in opp[1].values()))
    vals=(min(1.,price/(max(cash,0.)+price)),adjacent(at)-adjacent(there),turns/max(1,life),
          unsupported(at)-unsupported(there),min(1.,freight/400.))
    if len(vals)!=len(FEATURE_NAMES) or not all(math.isfinite(x) for x in vals):
        raise ValueError("invalid public features")
    return tuple(float(x) for x in vals)


def predict(weights,x):
    if len(x)!=len(FEATURE_NAMES) or tuple(weights["features"])!=FEATURE_NAMES or len(weights["beta"])!=1+len(x):
        raise ValueError("wrong feature schema")
    y=weights["beta"][0]+sum(a*b for a,b in zip(weights["beta"][1:],x))
    if not math.isfinite(y):raise ValueError("nonfinite predicted pay-minus-defer margin")
    return y


def choose(weights,x,pay_is_v1):
    parent="pay" if pay_is_v1 else "defer"
    if len(x)!=len(FEATURE_NAMES) or len(weights["low"])!=len(x) or len(weights["high"])!=len(x):return parent
    if any(not math.isfinite(v) or v<a-weights["support_pad"] or v>b+weights["support_pad"] for v,a,b in zip(x,weights["low"],weights["high"])):return parent
    y=predict(weights,x)
    if pay_is_v1 and -y>=weights["minimum_predicted_departure"]:return "defer"
    if not pay_is_v1 and y>=weights["minimum_predicted_departure"]:return "pay"
    return parent

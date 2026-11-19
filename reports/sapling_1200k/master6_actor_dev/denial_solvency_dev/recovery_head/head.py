"""Dependency-free conditional recovery estimate from the current actor's last eleven observations."""
from math import exp, isfinite, log1p

FEATURES = (
    "rival_has_tree","rival_cash_shortfall_in_fees","rival_trees","rival_tree_remaining_10",
    "rival_recent_tree_income_in_fees","rival_recent_net_cash_in_fees","rival_visible_return_load_in_fees",
    "rival_planters","rival_bases","rival_recent_collector_mean","rival_recent_can_afford_rate",
    "rival_planters_near_our_trees","our_collectors","our_planters","our_cargo_log",
    "our_empty_within_two","our_nearest_base_pressure","step_fraction","near_term_rehire_gap"
)
BASE_FEATURES = FEATURES[:2]


def distance(a,b,n):
    dy,dx = abs(a//n-b//n),abs(a%n-b%n)
    return min(dy,n-dy)+min(dx,n-dx)


def features(window, actor, cfg):
    """Pass public views for consecutive frames max(0,t-10)..t, projected for actor."""
    o = window[-1]
    t = o["step"]
    if [z["step"] for z in window] != list(range(max(0,t-10),t+1)) or any(z["player"] != actor for z in window):
        raise ValueError("expected eleven consecutive views of the same physical seat (fewer only near game start)")
    rival=1-actor
    players=o["players"]
    own,other = players[actor],players[rival]
    collectors=lambda p: [w for w in p[2].values() if w[2]=="COLLECTOR"]
    planters=lambda p: [w for w in p[2].values() if w[2]=="PLANTER"]
    def price(view):
        return cfg["recCollectorCost"]+cfg["recCollectorIncreaseCost"]*sum(len(collectors(z)) for z in view["players"])
    fee = price(o)
    other_c = collectors(other)
    if fee<=0 or len(other_c)!=1 or other_c[0][1]!=0 or other[0]>=fee:
        raise ValueError("head was trained only for a cash-poor rival's single empty collector")
    if len(other[1])==0:
        raise ValueError("training cohort requires at least one rival base")
    n=cfg["size"]
    target=other_c[0][0]
    own_c=collectors(own)
    target_bases=set(other[1].values())
    own_tree_cells={p for p,age in own[3].values()}
    rival_p=planters(other)
    incomes=[]
    arrivals=[]
    for idx,z in enumerate(window):
        p=z["players"][rival]
        # Public tree entries belong to the current owner. Ownership can change between observations.
        incomes.append(sum(z["trees"][tid][1] for tid in p[3]))
        if idx==0:
            continue
        prev=window[idx-1]["players"][rival]
        load=sum(w[1] for uid,w in prev[2].items() if w[2]=="COLLECTOR" and w[1]>0 and uid in p[2]
                 and p[2][uid][0] in target_bases and p[2][uid][1]==0)
        arrivals.append(load)
    recent=sum(incomes[-5:])/min(5,len(incomes))
    missing=(fee-other[0])/fee
    return dict(zip(FEATURES, (
        float(bool(other[3])),missing,log1p(len(other[3])),
        sum(min(10,max(0,cfg["treeLifespan"]-age))/10 for _,age in other[3].values()),
        sum(incomes)/fee,(other[0]-window[0]["players"][rival][0])/fee,log1p(sum(arrivals)/fee),
        len(rival_p),len(target_bases),sum(len(collectors(z["players"][rival])) for z in window)/len(window),
        sum(z["players"][rival][0]>=price(z) for z in window)/len(window),
        sum(any(distance(w[0],c,n)<=1 for c in own_tree_cells) for w in rival_p),
        len(own_c),len(planters(own)),log1p(sum(w[1] for w in own_c)),
        sum(w[1]==0 and distance(w[0],target,n)<=2 for w in own_c),
        sum(any(w[1]==0 and distance(w[0],b,n)<=1 for w in own_c) for b in target_bases),
        t/(cfg["episodeSteps"]-1),missing-3*recent/fee,
    ),strict=True))


def predict(values, weights):
    if weights["target"]!="no_collector_in_every_next_30_states_given_trade":
        raise ValueError("unsupported target")
    z=weights["intercept"]
    for k,mu,scale,b in zip(weights["features"],weights["mean"],weights["scale"],weights["coef"],strict=True):
        x=values[k]
        if not isfinite(x):
            raise ValueError("nonfinite observed feature")
        z+=b*max(-3.,min(3.,(x-mu)/scale))
    return 1/(1+exp(-max(-40,min(40,z))))

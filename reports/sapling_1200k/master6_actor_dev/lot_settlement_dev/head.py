"""Portable, inference-only estimator. Uses public state and an optional proposed friendly order."""
from math import isfinite
HORIZONS=(1,3,6,10,16,25,40,64,100,300)
CATEGORIES=("own_direct","own_relay","lost_to_enemy","tax","ground_or_cost","terminal_inventory")
MOVE=("NONE","UP","RIGHT","DOWN","LEFT")
D=tuple(tuple(min(abs(i%15-j%15),15-abs(i%15-j%15))+min(abs(i//15-j//15),15-abs(i//15-j//15)) for j in range(225)) for i in range(225))
N=tuple(tuple((i%15+dx)%15+15*((i//15+dy)%15) for dx,dy in ((0,0),(0,-1),(1,0),(0,1),(-1,0))) for i in range(225))
A=tuple(frozenset((i%15+dx)%15+15*((i//15+dy)%15) for dx in (-1,0,1) for dy in (-1,0,1)) for i in range(225))
BASE_NAMES=("cargo","home_distance","remaining")
NAMES=BASE_NAMES+("home_heading_1","home_heading_3","home_heading_5","cargo_growth_1","cargo_growth_3","cargo_growth_5","visible_carbon_at_worker","other_distance","other_count_2","support_distance","lighter_distance","lighter_near_2","lighter_near_4","heavy_enemy_near_2","enemy_tree_count_here","own_tree_count_here","enemy_tree_density_near_home","safe_forward_steps","own_cash","enemy_cash","own_workers","enemy_workers","own_trees","enemy_trees","home_is_within_remaining","action_known","action_stay","action_banks","home_progress_action","next_lighter_distance","next_enemy_tree_count","next_ally_collision","next_foe_base","next_harvest_if_staying","action_repeats_prior_direction")
def near(cell,places):
    return min((D[cell][p] for p in places),default=15)

def features(obs, history, seat, uid, action=None, own_orders=None):
    if seat!=obs["player"] or len(obs["carbon"])!=225:raise ValueError("expected own public observation")
    t=obs["step"]; history_by_t={}
    for h in history or ():
        if h["step"]>=t or h["player"]!=seat or h["step"] in history_by_t:raise ValueError("mixed-seat, future, or duplicate history")
        history_by_t[h["step"]]=h
    own,enemy=obs["players"][seat],obs["players"][1-seat]
    start,cargo,role=own[2][uid]
    if role!="COLLECTOR" or cargo<=0:raise ValueError("requires a positive principal")
    bases=tuple(own[1].values()); dist=near(start,bases);rem=299-t
    if not bases or rem<0:raise ValueError("unsupported rules")
    allies=[p for u,(p,c,k) in own[2].items() if u!=uid]
    foes=[p for p,c,k in enemy[2].values()]
    light=[p for p,c,k in enemy[2].values() if k=="COLLECTOR" and c<=cargo]
    heavy=[p for p,c,k in enemy[2].values() if k=="COLLECTOR" and c>cargo]
    ft={p for p,age in enemy[3].values() if age<50}; at={p for p,age in own[3].values() if age<50}
    closest=[b for b in bases if D[start][b]==dist]
    homehead=[]; growth=[];oldcell=None
    for lag in (1,3,5):
        old=history_by_t.get(t-lag); ref=old["players"][seat][2].get(uid) if old is not None else None
        homehead.append(near(ref[0],bases)-dist if ref else -16)
        growth.append(cargo-ref[1] if ref else -999)
        if lag==1 and ref:oldcell=ref[0]
    visible=float(obs["carbon"][start])
    forward=[c for c in N[start][1:] if near(c,bases)<dist]
    x=[cargo,dist,rem,*homehead,*growth,visible,near(start,foes),sum(D[start][p]<=2 for p in foes),near(start,allies),near(start,light),
       sum(D[start][p]<=2 for p in light),sum(D[start][p]<=4 for p in light),sum(D[start][p]<=2 for p in heavy),len(A[start]&ft),
       len(A[start]&at),sum(any(D[p][b]<=2 for b in closest) for p in ft),sum(near(c,light)>1 for c in forward),
       own[0],enemy[0],len(own[2]),len(enemy[2]),len(at),len(ft),int(rem>=dist)]
    if action is None:
        x += [0]*9
    else:
        if action not in MOVE:raise ValueError("invalid action")
        q=N[start][MOVE.index(action)]
        orders=own_orders or {}
        reserved=[N[p][MOVE.index(orders.get(u) or "NONE")] for u,(p,c,k) in own[2].items() if u!=uid]
        reserved += [p for u,p in own[1].items() if orders.get(u) in ("RECCOLLECTOR","RECPLANTER")]
        oldmove=None if oldcell is None or start not in N[oldcell] else MOVE[N[oldcell].index(start)]
        x+=[1,int(action=="NONE"),int(q in bases),dist-near(q,bases),near(q,light),len(A[q]&ft),
            int(q in reserved),int(q in enemy[1].values()),visible if action=="NONE" and q not in at|ft and q not in bases else 0]
        x.append(int(action==oldmove))
    if action is None:x.append(0)
    if len(x)!=len(NAMES) or not all(isfinite(v) for v in x):raise ValueError("feature mismatch")
    return x

def from_features(weights,x):
    if weights["version"]!=1 or weights["features"]!=list(NAMES):raise ValueError("unknown predictor format")
    return _from_rows(weights,x)
def _from_rows(weights,x):
    # Each tree stores a terminal node ID as its leaf index and a dense mixture at that node.
    output=[0.]*(len(CATEGORIES)+len(HORIZONS)*2)
    for tree in weights["trees"]:
        nodes=tree["nodes"]; leaves=tree["values"]; i=0
        while nodes[i][0]>=0:
            feature,cut,left,right=nodes[i]
            i=left if x[feature]<=cut else right
        vec=leaves[-nodes[i][0]-1]
        for k,v in enumerate(vec):output[k]+=v
    out=[max(0.,v/len(weights["trees"])) for v in output];n=len(CATEGORIES);m=len(HORIZONS)
    causes=dict(zip(CATEGORIES,out[:n]));total=sum(causes.values())
    if abs(total-1)>1e-5:raise ValueError("cause mixture does not allocate the full initial lot")
    recovered=causes["own_direct"]+causes["own_relay"]
    masses=out[n:n+m]; moments=out[n+m:];cash_by={};cum=0.;duration=0.
    if abs(sum(masses)-recovered)>1e-5:raise ValueError("cash-time and competing-cause outputs disagree")
    for i,dt in enumerate(HORIZONS):
        if masses[i]>0: duration+=masses[i]*min(x[2],moments[i]/masses[i])
        cum+=masses[i]
        if dt!=300:cash_by[str(dt)]=x[0]*(recovered if x[2]<=dt else cum)
    cash_by["terminal"]=x[0]*recovered
    return dict(expected_cash=x[0]*recovered,expected_cash_fraction=recovered,expected_time_if_cash=(duration/recovered if recovered>0 else None),
                cash_by_deadline=cash_by,outcome_fraction=causes)
def predict(weights,obs,history,seat,uid,action=None,own_orders=None):
    return from_features(weights,features(obs,history,seat,uid,action,own_orders))

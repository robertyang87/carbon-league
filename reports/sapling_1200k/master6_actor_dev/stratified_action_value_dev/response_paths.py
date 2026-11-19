"""Reconstruct public lead-ups and trace signed native response costs; this program never calls either policy."""
from collections import Counter, defaultdict
import sys

from core import HERE, OLD, read, save, sha, value_hash
import actions
sys.path.insert(0,str(HERE/"native"))
import run as native
sys.path.insert(0,str(native.prior.BASE/"all_top5"))
from context import visibility

NATIVE=HERE/"native"
settled=native.account.settled
ARMS=("base","alt")
LAGS=(10,5,3,1,0)


def equal(a,b):
    settled.equal(a,b,"turn or terminal reconciliation")


def xyz(x):
    return round(float(x),7)


def move(pos,a,n):
    return actions.step(pos,a or "NONE",n)


def public_context(obs,s,uid,tracked):
    n=15
    own,enemy=obs["players"][s],obs["players"][1-s]
    own_workers,foes=own[2],enemy[2]
    visible=visibility(own_workers,n)
    def worker(wid,roster):
        if not wid or wid not in roster:return None
        p,c,k=roster[wid]
        return {"id":wid,"pos":p,"cargo":xyz(c),"role":k,
                "visible_ground":obs["carbon"][p] if visible[p] else None}
    focal=worker(uid,own_workers)
    threats=[]
    if focal:
        for wid,(p,c,k) in foes.items():
            if k=="COLLECTOR" and c<=own_workers[uid][1]:
                dist=actions.distance(p,focal["pos"],n)
                threats.append((dist,c,wid,p))
    threats.sort()
    near=[{"id":wid,"pos":p,"cargo":xyz(c),"distance":d} for d,c,wid,p in threats if d<=2]
    return {"focal":focal,"nearest_base_distance":min((actions.distance(focal["pos"],b,n) for b in own[1].values()),default=None) if focal else None,
            "cash_physical":[xyz(x[0]) for x in obs["players"]],"roster_sizes":[len(x[2]) for x in obs["players"]],
            "closest_lighter_or_tied_enemy_id":threats[0][2] if threats else None,
            "closest_lighter_or_tied_distance":threats[0][0] if threats else None,
            "lighter_or_tied_within_two":near,"tracked_root_enemy":worker(tracked,foes)}


def prefix(tape,card):
    t0,s,uid=card["t"],card["seat"],card["proposal"]["uid"]
    sim=settled.FastCarbon(tape["game_configuration"]).reset(seed=card["seed"])
    saved=[]
    for t,j in enumerate(tape["actions"][:t0]):
        if t>=t0-10:saved.append((t,sim.obs(s),j))
        sim.step(*j)
    root=sim.obs(s)
    # Candidate observations are selected only from the source prefix.
    tracked=public_context(root,s,uid,None)["closest_lighter_or_tied_enemy_id"]
    items=[]
    for t,obs,j in saved:
        public=public_context(obs,s,uid,tracked)
        items.append({"t":t,"lag_to_root":t0-t,"public_pre":public,
                      "observed_after_issue":{"focal":j[s].get(uid,"NONE") if public["focal"] else None,
                                              "root_enemy":j[1-s].get(tracked,"NONE") if public["tracked_root_enemy"] else None}})
    return sim.get_state(),tracked,items


def before_orders(obs,j,s,uid,tracked):
    owner=obs["players"][s][2]; foe=obs["players"][1-s][2]
    if uid in owner:
        dest=move(owner[uid][0],j[s].get(uid),15)
        arrivals=[[p,w,v[0],v[2],xyz(v[1]),j[p].get(w,"NONE")] for p,team in enumerate(obs["players"])
                  for w,v in team[2].items() if move(v[0],j[p].get(w),15)==dest]
        recruit=[[p,bid,j[p][bid]] for p,team in enumerate(obs["players"]) for bid,pos in team[1].items()
                 if pos==dest and j[p].get(bid) in ("RECCOLLECTOR","RECPLANTER")]
    else:
        dest=None;arrivals=[];recruit=[]
    return {"focal_command":j[s].get(uid,"NONE") if uid in owner else None,
            "root_enemy_command":j[1-s].get(tracked,"NONE") if tracked in foe else None,
            "focal_destination":dest,"same_destination_existing_workers":arrivals,
            "same_destination_base_orders":recruit}


def propagate(q, held, amount, relayed, fates):
    if amount<=1e-12 or held is None:return [],None,0.,relayed
    rows,next_uid,remaining=q[held]
    detail=[]
    for kind,reason,payer,recipient,share in rows:
        if kind=="relay":relayed=True
        elif kind=="own_direct" and relayed:kind="own_relay"
        v=amount*share
        detail.append({"kind":kind,"reason":reason,"payer":payer,"recipient":recipient,"amount":xyz(v)})
        if kind!="relay":fates[kind]+=v
    return detail,next_uid,amount*remaining,relayed


def branch(card,line,arm,snap,tracked,tape):
    s=card["seat"];uid=card["proposal"]["uid"];folder=NATIVE/card["id"]
    pp=folder/f"{arm}.json.gz"; ap=folder/f"{arm}_account.json.gz"
    receipt=read(folder/"pair.json")
    assert receipt["card_sha256"]==value_hash(card) and receipt["files"][arm]==sha(pp)
    assert line["account_sha256"][arm]==sha(ap)
    native_arm=read(pp); full=read(ap)
    assert full["arm_sha256"]==sha(pp) and full["measures"]==line["arms"][arm]
    ledger=full["account"];replies=native_arm["history"];joints=[native_arm["joint0"]]+[q["joint"] for q in replies]
    assert joints[0][s]==card[arm] and joints[0][1-s]==tape["actions"][card["t"]][1-s] and len(joints)-1==native_arm["completed_original_calls"]
    sim=settled.FastCarbon(tape["game_configuration"]);sim.set_state(snap)
    interest=ledger["per_worker"][uid]["principal"]
    held=uid; linked=False;fates=Counter()
    accounts=[Counter(),Counter()]; turns=[]; workers_deposits=Counter()
    for i,j in enumerate(joints):
        obs=sim.obs(s);t=sim.step_count; assert t==card["t"]+i
        before=public_context(obs,s,uid,tracked)
        joint_info=before_orders(obs,j,s,uid,tracked)
        _,flow,_=settled.one_step(sim,j,s,verify=True)
        post=sim.obs(s); after=public_context(post,s,uid,tracked)
        cash_deltas=[Counter(),Counter()]; supplies=[]; tax_components=[]
        for event in sim.cash.events:
            kind,p,amount=event[:3];accounts[p][kind]+=amount;cash_deltas[p][kind]+=amount
            if kind=="own_delivery":
                worker=event[3];workers_deposits[p,worker]+=amount
                supplies.append([p,worker,xyz(amount),event[4]])
            elif kind=="tree_absorption" and event[5]:
                # Official verified aggregate enemy carried cargo absorbed by this tree,
                # distinct from the focal root-principal contribution in principal_events.
                tax_components.append([p,event[3],xyz(event[5])])
        actual_landings=[]
        for p,who,start_pos,role,cargo,cmd in joint_info["same_destination_existing_workers"]:
            closing=post["players"][p][2].get(who)
            actual_landings.append({"seat":p,"worker":who,"landed":joint_info["focal_destination"],
                                    "observed_after":None if closing is None else [closing[0],xyz(closing[1]),closing[2]],
                                    "lost_in_collision":any(ev[0]==who for ev in sim.last_collisions)})
        flow_steps,held,interest,linked=propagate(flow,held,interest,linked,fates)
        for p in (0,1):equal(sum(cash_deltas[p].values()),post["players"][p][0]-obs["players"][p][0])
        turns.append({"t":t,"pre":before,"issued_joint_effect":joint_info,"realized_landings_at_focal_cell":actual_landings,"post":after,
                      "collision_log":[[w,p,xyz(m),owner,winner] for w,p,m,owner,winner in sim.last_collisions],
                      "base_deposits":supplies,"enemy_cargo_absorbed_by_tree":tax_components,"principal_events":flow_steps,
                      "remaining_root_principal":{"holder":held,"amount":xyz(interest)} if held else None,
                      "cash_flow_physical":[{k:xyz(v) for k,v in side.items()} for side in cash_deltas]})
    if held:fates["terminal_inventory"]+=interest
    w=ledger["per_worker"][uid]
    for typ,v in w["principal_fates"].items():equal(fates[typ],v)
    for p in (0,1):
        tot=ledger["team"]["treasury_flows_physical"][p]
        for k in set(tot)|set(accounts[p]):equal(accounts[p][k],tot.get(k,0.))
    equal(workers_deposits[s,uid],w["total_deposited_by_worker"])
    assert ledger["official_scores_physical"]==native_arm["official_scores"] and list(sim.cash)==native_arm["cash"]
    return {"arm_sha256":sha(pp),"account_sha256":sha(ap),"original_calls":len(replies),"turns":turns,
            "terminal_t":sim.step_count,"cash_physical":list(sim.cash),"official_scores_physical":ledger["official_scores_physical"],
            "terminal_adjustment_physical":[xyz(q-c) for q,c in zip(native_arm["official_scores"],sim.cash)],
            "focal_strict_principal_deposit":w["strict_principal_delivered"],"focal_strict_fresh_deposit":w["strict_fresh_delivered"],
            "principal_fates":{k:xyz(v) for k,v in fates.items()}}


def paired_curve(case):
    arms=case["arms"]; s=case["seat"]; t0=case["t"]
    times={a:{q["t"]:q for q in arm["turns"]} for a,arm in arms.items()}
    common={x["t"]:x for x in case["root_antecedents"]}
    total=0.; points=[]
    for t in range(t0,max(a["terminal_t"] for a in arms.values())):
        costs={}
        for a,arm in arms.items():
            q=times[a].get(t);delta=Counter()
            if q:
                cash=q["cash_flow_physical"];delta.update(cash[s]);delta.subtract(cash[1-s])
            if t+1==arm["terminal_t"]:
                settle=arm["terminal_adjustment_physical"]
                delta["terminal_adjustment"]=settle[s]-settle[1-s]
            costs[a]=delta
        delta={k:xyz(costs["alt"][k]-costs["base"][k]) for k in costs["alt"].keys() | costs["base"].keys()}
        amount=sum(delta.values());total+=amount
        points.append({"t":t,"paired_margin_change":xyz(amount),"paired_cumulative_change":xyz(total),
                       "channels":{k:v for k,v in delta.items() if abs(v)>1e-7}})
    equal(round(total,4),round(case["delta"]["margin"],4))
    peak=max(points,key=lambda q:abs(round(q["paired_margin_change"]*100)))
    best=max(points,key=lambda q:round(q["paired_margin_change"]*100))
    worst=min(points,key=lambda q:round(q["paired_margin_change"]*100))
    # References point into the same trace, leaving archived native actions in their original format.
    def reference(point):
        back=[]
        for n in LAGS:
            t=point["t"]-n
            if t in common:
                source="common_real_prefix"
                assert common[t]["public_pre"]
            elif t>=t0:
                source="real_arm_responses"
            else:continue
            back.append({"lag":n,"t":t,"lookup":source})
        return back
    return {"largest_absolute_single_turn":peak,"before_peak":reference(peak),
            "best_single_turn":best,"before_best":reference(best),"worst_single_turn":worst,"before_worst":reference(worst), "per_turn":[[p["t"],p["paired_margin_change"],p["paired_cumulative_change"]] for p in points]}


def build_case(card,line):
    tape=native.source(card);snap,tracked,prefix_rows=prefix(tape,card)
    case={k:line[k] for k in ("id","state_id","seed","seat","t","map","stream","diagnostic","primary_id","delta")}
    case.update(raw_sha256=card["raw_sha256"],source_official_sha256=card["official_sha256"],focal_uid=card["proposal"]["uid"],
                proposal={k:card["proposal"][k] for k in ("kind","original","action")},tracked_root_enemy=tracked,root_antecedents=prefix_rows,
                arms={a:branch(card,line,a,snap,tracked,tape) for a in ARMS})
    case["paired_turning_window"]=paired_curve(case)
    return case


def main():
    study=read(NATIVE/"train_labels.json")
    assert study["frozen_sha256"]==sha(NATIVE/"frozen.json") and study["split"]=="train"
    assert study["runner_sha256"]==sha(NATIVE/"run.py") and study["verifier_sha256"]==sha(NATIVE/"labels.py")
    cards={q["id"]:q for q in read(HERE/"cards.json")["cases"] if q["split"]=="train"}
    assert set(cards)=={r["id"] for r in study["cases"]}
    cases=[build_case(cards[r["id"]],r) for r in study["cases"]]
    primary=[q for q in cases if not q["diagnostic"]]; keyed={q["id"]:q for q in primary}
    def summary(q):
        return {"id":q["id"],"t":q["t"],"uid":q["focal_uid"],"commands":q["proposal"],
                "official_margin_delta":q["delta"]["margin"],"principal_delta":q["delta"]["principal"],
                "own_delivery_delta":q["delta"]["own_delivery"],"enemy_delivery_delta":q["delta"]["enemy_delivery"],
                "peak":q["paired_turning_window"]["largest_absolute_single_turn"]}
    repeats=[]
    for q in cases:
        if not q["diagnostic"]:continue
        p=keyed[q["primary_id"]]
        assert p["state_id"]==q["state_id"] and p["proposal"]==q["proposal"] and p["root_antecedents"]==q["root_antecedents"]
        repeats.append({"primary":summary(p),"new_stream":summary(q),
                        "opposite_sign":p["delta"]["margin"]*q["delta"]["margin"]<0})
    doc={"schema":1,"scope":"Certified training/source roots only. Paired one-action conditional responses. Extremes are descriptive. Future opponent orders, adjudicated collisions and tree tax were excluded from model inputs.",
         "directions":"All monetary paired changes are alternative less parent from the focal physical seat. Arrays with physical index give seats 0 then 1.",
         "observability":"Focal worker and root-nearest lighter-or-tied rival are tracked by native ID. Missing entity, unavailable tile carbon or absent qualifying rival is null. Identical-landing existing worker rows contain seat, id, starting position, role, starting cargo, and actual issued order. Collision rows contain dead id, seat, lost cargo, credited seat or null, and surviving collector seat or null. The simultaneous opponent order is retrospective.",
         "accounting":"Principal is only opening inventory. A relay row transfers its claim and does not credit income. Per-tree records report all absorbed enemy worker cargo, and are already part of tree treasury receipts. Post-root board transitions match official Board, each seat's cash writes reconcile, and surviving inventory plus realized dispositions reconcile exactly with the independent principal certificate.",
         "labels_sha256":sha(NATIVE/"train_labels.json"),"source_protocol_sha256":study["source_protocol_sha256"],
         "script_sha256":sha(HERE/"response_paths.py"),"counts":{"primary":len(primary),"alternate_stream":len(repeats)},
         "extremes":{"lowest":summary(min(primary,key=lambda q:q["delta"]["margin"])),
                     "highest":summary(max(primary,key=lambda q:q["delta"]["margin"]))},
         "stream_replications":repeats,"cases":cases}
    out=HERE/"response_paths.json.gz";save(out,doc);print("verified",len(cases),sha(out),flush=True)

if __name__=="__main__":main()

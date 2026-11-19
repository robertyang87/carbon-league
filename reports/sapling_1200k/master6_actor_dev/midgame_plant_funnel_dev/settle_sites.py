"""Attribute actual A/B tree fees, first-tenure ground rentals, cargo taxes and terminal cash."""
from funnel import HERE,ROOT,load,save,FastCarbon,CashEvents
from hashlib import sha256

def arm(q,name):
    p=HERE/f"seat{q['seat']}";rec=load(p/f"live_{name}.json");f=p/rec["action_tape"]
    if sha256(f.read_bytes()).hexdigest()!=rec["action_sha256"]:raise ValueError("action tape identity changed")
    tape=load(f);raw=load(ROOT/q["record"]["source"]);s=q["seat"];cfg=raw["game_configuration"];t=q["root_t"]
    sim=FastCarbon(cfg).reset(seed=q["seed"])
    for pair in raw["actions"][:t]:sim.step(*pair)
    if tape[0][1-s]!=raw["actions"][t][1-s]:raise ValueError("simultaneous enemy command changed")
    tid=None;alive=False;ledger=[];soil=0.;fees=0.;origin_soil=0.;origin_tax=0.;fee=0.;end=None
    for h,pair in enumerate(tape):
        begin=list(sim.cash);sim.cash=CashEvents(sim.cash);sim.step(*pair)
        for kind,owner,v,*a in sim.cash.events:
            if owner==s and kind=="plant":
                fees+=-v
                if h==1 and a[0]==q["root"]["worker"] and a[1]==q["root"]["target"]:
                    born=[tr for tr in sim.trees.values() if tr.pos==a[1] and tr.player==s and tr.age==1]
                    if len(born)!=1:raise ValueError("new seedling identity")
                    tid=born[0].id;alive=True;fee=-v
                    ledger.append(dict(step=t+h,kind="plant",amount=-v))
            elif kind=="tree_absorption" and owner==s:
                ground=v-a[2];soil+=ground
                if a[0]==tid and alive:
                    origin_soil+=ground;origin_tax+=a[2]
                    ledger.append(dict(step=t+h,kind="income",ground=ground,tax=a[2]))
        if alive and (tid not in sim.trees or sim.trees[tid].player!=s):
            alive=False;end=t+h+1
    extinct=[not sim.player_workers[k] and not sim.player_trees[k] and
             (not sim.bases[k] or sim.cash[k]<min(cfg["recPlanterCost"],cfg["recCollectorCost"])) for k in (0,1)]
    official=[sim.step_count-cfg["episodeSteps"]-1 if extinct[k] else begin[k] if any(extinct) else sim.cash[k] for k in (0,1)]
    if max(abs(a-b) for a,b in zip(official,rec["score"]))>.011:raise ValueError("official terminal score differs")
    return dict(target=tid,fee=fee,first_tenure_ground=origin_soil,first_tenure_tax=origin_tax,target_net=origin_soil-fee,
                end_of_tenure=end,all_own_ground=soil,all_own_fees=fees,all_own_ground_minus_fees=soil-fees,
                target_cash_writes=ledger,score=official,margin=rec["margin"],search_calls=rec["search_calls"])

def verify():
    rows=[]
    for q in load(HERE/"selected_sites.json"):
        parent=arm(q,"parent");candidate=arm(q,"approach")
        rows.append(dict(seed=q["seed"],seat=q["seat"],root=q["root"],parent=parent,candidate=candidate,
                         paid_additional=candidate["target"] is not None and parent["target"] is None,
                         delta_margin=candidate["margin"]-parent["margin"],
                         delta_team_ground_net=candidate["all_own_ground_minus_fees"]-parent["all_own_ground_minus_fees"]))
    passed=(len(rows)==2 and all(r["paid_additional"] and r["candidate"]["target_net"]>0 for r in rows)
            and sum(r["delta_margin"] for r in rows)>0 and sum(r["delta_team_ground_net"] for r in rows)>0)
    save(HERE/"native_site_settlement.json",dict(results=rows,eligible_for_fresh_map=passed))
    print([(r["paid_additional"],r["candidate"]["target_net"],r["delta_team_ground_net"],r["delta_margin"]) for r in rows],passed)

if __name__=="__main__":verify()

"""Replay the complete observed opponent order, verify cash writes and follow freshly harvested carbon."""
from collections import Counter
import hashlib
import sys
from fit import HERE, ROOT, FastCarbon, load, save
sys.path.insert(0, str(HERE.parent/"lot_settlement_dev"))
from lots import one_step

HORIZON=30

def balance(ledger):
    return sum(ledger.get(k,0) for k in ("own_direct","own_relay","ground_or_cost","lost_to_enemy","tax","remaining"))

def settle(sim,cfg,seat,uid,tape,verify=True):
    n=cfg["size"]; origin=sim.workers[uid].pos
    obs=sim.obs(seat);prev_load=sim.workers[uid].carbon
    patch=obs["carbon"][origin]
    running=0.;holder=uid;gross=0.;tracked=Counter();credits=[];first=None;relayed=False
    origin_alt=(tape[0][seat].get(uid) or "NONE")=="NONE"
    focal_banked=0.
    for h,cmds in enumerate(tape):
        live_before=set(sim.trees) if h==0 else ()
        previous_cash=list(sim.cash)
        initial, transitions, _=one_step(sim,cmds,seat,verify)
        for ev in sim.cash.events:
            if ev[0] in ("own_delivery","base_collision_cash","tree_collision_cash") and ev[1]==seat and ev[3]==uid:
                focal_banked+=ev[2]
        if h==0:
            survivor=sim.workers.get(uid); born_here=origin_alt and survivor is not None
            if born_here:
                # The root guard excludes trees, bases and friendly arrival. The collector survives;
                # physical collection precedes tree tax. Do not tag its preexisting load or collision loot.
                n_workers=sum(w.player==seat and w.is_collector for w in sim.workers.values())
                gross=patch*max(cfg["initialCollectRate"]-cfg["collectDecreaseRate"]*n_workers,0)
                r,c=divmod(origin,n)
                neigh={((r+a)%n)*n+(c+b)%n for a in (-1,0,1) for b in (-1,0,1)}
                taxes=sum(tr.player==1-seat and tid in live_before and tr.pos in neigh
                          for tid,tr in sim.trees.items())
                tracked["tax"]+=gross*taxes*cfg["collectorAbsorptionRate"]
                running=gross-tracked["tax"]
                if running>survivor.carbon+1e-5: raise ValueError("newly tagged lot exceeds visible surviving load")
            first=dict(alive=survivor is not None,load=survivor.carbon if survivor else 0.,bank=list(sim.cash),gross_harvest=gross)
        elif running>1e-9:
            if holder not in transitions:raise ValueError("tag holder missing")
            effects,next_holder,fraction=transitions[holder]
            for kind,reason,actor,receiver,f in effects:
                if kind=="relay":
                    relayed=True
                    continue
                cash_kind="own_relay" if kind=="own_direct" and relayed else kind
                amount=running*f
                tracked[cash_kind]+=amount
                if cash_kind in ("own_direct","own_relay"):
                    credits.append(dict(offset=h,actor=actor,receiver=receiver,reason=reason,amount=amount))
            holder=next_holder;running*=fraction
    tracked["remaining"]=running
    if abs(balance(tracked)-gross)>1e-5:raise ValueError("tag not conserved")
    empty=[not sim.player_workers[p] and not sim.player_trees[p] and
           (not sim.bases[p] or sim.cash[p]<min(cfg["recCollectorCost"],cfg["recPlanterCost"])) for p in (0,1)]
    official=[sim.step_count-cfg["episodeSteps"]-1 if empty[p] else
              previous_cash[p] if any(empty) else sim.cash[p] for p in (0,1)]
    return dict(first=first,ending_cash=list(sim.cash),ending_margin=sim.cash[seat]-sim.cash[1-seat],
                final_score_rule=official,
                focal_cash=focal_banked,new_carbon=gross,tagged=dict(tracked),tag_credits=credits,
                focal_and_external_relays=focal_banked+sum(e['amount'] for e in credits if e["actor"]!=uid))

def restart(raw,t):
    sim=FastCarbon(raw["game_configuration"]).reset(seed=raw["seed"])
    for joint in raw["actions"][:t]:sim.step(*joint)
    return sim

def diagnostic():
    out=[]
    for c in load(HERE/"disagreements.json.gz"):
        raw=load(ROOT/c["source"]);cfg=raw["game_configuration"];s=c["seat"]
        for r in c["roots"]:
            t=r["t"];uid=r["worker"];orig=raw["actions"][t:t+HORIZON];arms={}
            for name in ("parent","hold"):
                tape=[[dict(a),dict(b)] for a,b in orig]
                if name=="hold":tape[0][s].pop(uid)
                arms[name]=settle(restart(raw,t),cfg,s,uid,tape)
            out.append(dict(seed=c["seed"],seat=s,t=t,uid=uid,horizon=len(orig),**arms))
    save(HERE/"taped_settlement.json.gz",out)
    h=sum(r["hold"]["new_carbon"] for r in out)
    credited=sum(r["hold"]["tagged"].get("own_direct",0)+r["hold"]["tagged"].get("own_relay",0) for r in out)
    dm=sum(r["hold"]["ending_margin"]-r["parent"]["ending_margin"] for r in out)
    df=sum(r["hold"]["focal_and_external_relays"]-r["parent"]["focal_and_external_relays"] for r in out)
    save(HERE/"taped_summary.json",dict(roots=len(out),gross=h,verified_tagged_own_cash=credited,
                                        horizon=HORIZON,delta_focal_cash_sum=df,delta_margin_sum=dm,
                                        scope="All held-map parent disagreements; fixed recorded joint commands after root. Overlapping paths are not independent games."))
    print("tape",len(out),h,credited,df,dm)

def native():
    pairs=[]
    for q in load(HERE/"selected.json"):
        s=q["seat"];uid=q["root"]["worker"];root_t=q["root_t"]
        raw=load(ROOT/q["record"]["source"]);d=HERE/f"seat{s}";arms={}
        for arm in ("parent","hold"):
            rec=load(d/f"live_{arm}.json");tape_path=d/rec["action_tape"]
            assert hashlib.sha256(tape_path.read_bytes()).hexdigest()==rec["action_sha256"]
            tape=load(tape_path)
            assert tape[0][1-s]==raw["actions"][root_t][1-s]
            arms[arm]=settle(restart(raw,root_t),raw["game_configuration"],s,uid,tape)
            if max(abs(x-y) for x,y in zip(arms[arm]["final_score_rule"],rec["score"]))>.011:raise ValueError("official final cash mismatch")
            arms[arm]["official_score"]=rec["score"]
        dmargin=(arms["hold"]["official_score"][s]-arms["hold"]["official_score"][1-s])-(arms["parent"]["official_score"][s]-arms["parent"]["official_score"][1-s])
        ddeposit=arms["hold"]["focal_and_external_relays"]-arms["parent"]["focal_and_external_relays"]
        tag=arms["hold"]["tagged"]
        pairs.append(dict(seed=q["seed"],seat=s,t=root_t,worker=uid,arms=arms,delta_margin=dmargin,
                          delta_focal_receipt=ddeposit,verified_tag_cash=tag.get("own_direct",0)+tag.get("own_relay",0)))
    eligible=all(r["delta_focal_receipt"]>0 and r["verified_tag_cash"]>0 for r in pairs) and sum(r["delta_margin"] for r in pairs)>0
    save(HERE/"native_settlement.json",dict(pairs=pairs,passes_receipt_and_margin_gate=eligible))
    print("native",[(r["delta_margin"],r["delta_focal_receipt"],r["verified_tag_cash"]) for r in pairs],eligible)

if __name__=="__main__":
    if sys.argv[1:]==["native"]:native()
    else:diagnostic()

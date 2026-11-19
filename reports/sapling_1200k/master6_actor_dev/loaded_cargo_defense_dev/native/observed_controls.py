"""Audit original, unedited actions in the eight public triggers and follow three actually stolen lots to settlement."""
from collections import Counter, defaultdict
from math import isclose
import run, labels

S=run.prior.account.settled
A=run.prior.account

def locate(players,uid):
    for p,team in enumerate(players):
        if uid in team[2]:return p,team[2][uid]
    return None,None

def first_lot_fate(snap,cfg,orders,side,uid,expected=None):
    """Follow a tagged share of initial inventory, including ownership reversals. Ground identity thereafter is unresolved."""
    sim=S.FastCarbon(cfg);sim.set_state(snap)
    assert cfg["moveCost"]==cfg["smeltCost"]==0 and cfg["size"]==15
    start=sim.obs(side)["players"][side][2][uid][1]
    amount=start;holder=uid;ledger=[];positions=[];destinations=Counter()
    carrier_stats={}; first_cross=None
    def record(t,kind,qty,worker,other=None,seat=None,**extra):
        ledger.append(dict(t=t,event=kind,amount=qty,worker=worker,to=other,beneficiary=seat,**extra))
    root_t=sim.step_count
    for j in orders:
        t=sim.step_count;pre=sim.full_observation();who,w=locate(pre["players"],holder)
        if amount>1e-10:
            assert w and w[2]=="COLLECTOR" and amount<=w[1]+1e-5
            old_holder=holder
            positions.append({"t":t,"holder":holder,"seat":who,"pos":w[0],"amount":amount,"command":j[who].get(holder,"NONE")})
            dest=sim._move_nbr[j[who][holder]][w[0]] if j[who].get(holder) in sim._move_nbr else w[0]
        _,tr,post=S.one_step(sim,j,None,verify=True)
        if amount<=1e-10:continue
        deaths={row[0]:row for row in sim.last_collisions}
        cash=sim.cash.events
        if holder in deaths:
            deceased,loser,total,beneficiary,winner_seat=deaths[holder]
            assert loser==who and amount<=total+1e-5
            if winner_seat is None:
                tx=[ev for ev in cash if ev[0] in ("tree_collision_cash","base_collision_cash") and ev[3]==holder]
                if beneficiary is None:
                    assert not tx;kind="unattributed_after_ground_drop"
                    end_to=None
                else:
                    event,=tx;S.equal(event[2],total,"collision treasury payment")
                    kind=event[0];end_to=event[5] if len(event)>5 else event[4]
                record(t,kind,amount,holder,end_to,beneficiary)
                destinations[(kind,beneficiary)]+=amount;amount=0;holder=None;continue
            winner,=[u for u,z in post["players"][winner_seat][2].items() if z[0]==dest and z[2]=="COLLECTOR"]
            kind="friendly_relay" if winner_seat==who else "cross_team_capture"
            record(t,kind,amount,holder,winner,winner_seat,landing=dest)
            if kind=="cross_team_capture" and first_cross is None:first_cross=ledger[-1].copy()
            holder=winner;who=winner_seat
        bases={cell:(p,bid) for p,team in enumerate(pre["players"]) for bid,cell in team[1].items()}
        if dest in bases:
            recipient,bid=bases[dest]
            receipt,=[e for e in cash if e[0] in ("own_delivery","opponent_worker_delivery_to_this_base") and e[3]==holder]
            assert receipt[1]==recipient and receipt[2]+1e-5>=amount
            record(t,receipt[0],amount,holder,bid,recipient,whole_deposit=receipt[2])
            destinations[(receipt[0],recipient)]+=amount;amount=0;holder=None;continue
        old_live={tree for team in pre["players"] for tree,(_,age) in team[3].items() if age<cfg["treeLifespan"]}
        trees=[(p,tree) for p,team in enumerate(post["players"]) if p!=who for tree,(tree_pos,_) in team[3].items()
               if tree in old_live and dest in (tree_pos,*sim._nbr8[tree_pos])]
        chunk=amount*cfg["collectorAbsorptionRate"]
        assert len(trees)*cfg["collectorAbsorptionRate"]<1
        for p,tree in trees:
            ev,=[e for e in cash if e[0]=="tree_absorption" and e[3]==tree]
            assert ev[5]+1e-5>=chunk
            record(t,"carried_cargo_tree_tax",chunk,holder,tree,p,tree_total_worker_tax=ev[5])
            destinations[("carried_cargo_tree_tax",p)]+=chunk
        amount-=len(trees)*chunk
        assert holder in post["players"][who][2] and amount<=post["players"][who][2][holder][1]+1e-5
    if amount>1e-10:
        who,w=locate(sim.obs(side)["players"],holder)
        assert w and w[1]+1e-5>=amount
        record(sim.step_count,"terminal_inventory",amount,holder,None,who)
        destinations[("terminal_inventory",who)]+=amount
    S.equal(sum(destinations.values()),start,"tagged principal conservation")
    if expected:
        assert first_cross and first_cross["t"]==root_t==expected["t"] and first_cross["worker"]==uid==expected["uid"]
        assert first_cross["to"]==expected["winner"] and first_cross["beneficiary"]==1-side
        S.equal(first_cross["amount"],expected["actual_transferred"],"actual victim-cargo transfer")
    summary=[{"kind":k,"recipient_seat":s,"amount":v} for (k,s),v in destinations.items()]
    return {"root":root_t,"uid":uid,"starting_principal":start,"first_worker_capture":first_cross,
            "holdings_before_orders":positions,"observed_money_and_transfers":ledger,"terminal_fates":summary}


def enemy_cashout(trace,event,seat):
    enemy=1-seat;winner=event["winner"]
    realised=[e for e in trace["observed_money_and_transfers"] if e["t"]>=event["t"]]
    direct=[e for e in realised if e["event"]=="own_delivery" and e["worker"]==winner and e["beneficiary"]==enemy]
    by_team=[e for e in realised if e["event"]=="own_delivery" and e["beneficiary"]==enemy]
    # The tag is followed until cash, tax, ground, or horizon even if ownership changes again.
    return {"captor_id":winner,"captor_directly_banked_tag":sum(e["amount"] for e in direct),
            "captor_team_banked_tag_including_relays":sum(e["amount"] for e in by_team),
            "captor_team_deposit_basis":by_team,"unresolved_after_ground":sum(e["amount"] for e in realised if e["event"]=="unattributed_after_ground_drop"),
            "still_on_captor_team_at_horizon":sum(e["amount"] for e in realised if e["event"]=="terminal_inventory" and e["beneficiary"]==enemy)}


def main():
    coverage=run.read(run.DEV/"coverage.json"); assert not coverage["native_diagnostic_possible"]
    panel=run.read(run.DEV/"all_opportunities.json.gz");history=run.read(run.DEV/"historical_train_losses.json.gz")["cases"]
    controls=defaultdict(list); thefts=defaultdict(list)
    for g in panel["groups"]:
        if g["split"]!="train":continue
        for c in g["cards"]:
            if not c["excluded"]:controls[c["seed"],c["seat"]].append(c)
    for g in history:
        thefts[g["seed"],g["seat"]].extend(g["events"])
    inventory={(g["seed"],g["seat"]):g for g in run.read(run.DEV/"source/inventory.json")["games"] if g["split"]=="train"}
    ctrl=[];stolen=[];identities={}
    for key,row in inventory.items():
        group=controls[key];events=thefts[key]
        if not (group or events):continue
        tape=run.prior.source(row); cfg=tape["game_configuration"]; s=row["seat"];seed=row["seed"]
        official_file=row["official"];official=run.read(run.ROOT/official_file)
        cert,=[g for z in official["runs"].values() for g in z["games"] if g["transcript_sha256"]==row["raw_sha256"]]
        ident={"seed":seed,"seat":s,"raw":row["raw"],"raw_sha256":row["raw_sha256"],"official":official_file,
               "official_sha256":row["official_sha256"],"map":row["map"],"official_rewards":cert["official_reward_by_seat"],"final_t":cert["steps"]}
        identities[row["raw_sha256"]]=ident
        sim=S.FastCarbon(cfg).reset(seed=seed)
        required={c["t"] for c in group}|{e["t"] for e in events}
        snaps={}
        for t,j in enumerate(tape["actions"]):
            if t in required:snaps[t]=sim.get_state()
            sim.step(*j)
        assert set(snaps)==required and sim.step_count==cert["steps"]
        for c in group:
            t=c["t"];wid=c["proposal"]["uid"];orders=tape["actions"][t:]
            assert c["base"]==orders[0][s] and c["root_enemy"]==orders[0][1-s] and c["raw_sha256"]==row["raw_sha256"]
            proof=labels.augment(A,snaps[t],cfg,orders,s,[wid])
            assert proof["official_scores_physical"]==cert["official_reward_by_seat"]
            trace=first_lot_fate(snaps[t],cfg,orders,s,wid)
            old=proof["per_worker"][wid]; defense=proof["defense"]["unit_accounts"][wid]
            S.equal(trace["starting_principal"],old["principal"],"control root principal")
            if trace["first_worker_capture"] is None:S.equal(defense["measures"]["principal_to_enemy_worker"],0.,"no control transfer")
            else:S.equal(trace["first_worker_capture"]["amount"],defense["measures"]["principal_to_enemy_worker"],"control transfer")
            strict=[e for e in trace["observed_money_and_transfers"] if e["event"]=="own_delivery" and e["worker"]==wid and e["beneficiary"]==s]
            S.equal(sum(x["amount"] for x in strict),old["strict_principal_delivered"],"strict original bank receipt")
            ctrl.append({"id":c["id"],"seed":seed,"seat":s,"t":t,"raw_sha256":row["raw_sha256"],"root_action":orders[0][s].get(wid,"NONE"),
                         "held_back_alternative":c["proposal"]["action"],"worker":wid,"original_principal_fates":old["principal_fates"],
                         "strict_direct_banked":old["strict_principal_delivered"],"original_worker_actual_total_deposits":old["total_deposited_by_worker"],
                         "reconciled_defense_measures":defense["measures"],"tagged_trace":trace})
        for e in events:
            t=e["t"];wid=e["uid"];orders=tape["actions"][t:]
            trace=first_lot_fate(snaps[t],cfg,orders,s,wid,expected=e)
            proof=labels.augment(A,snaps[t],cfg,orders,s,[wid])
            assert proof["official_scores_physical"]==cert["official_reward_by_seat"]
            S.equal(proof["defense"]["measures"]["principal_to_enemy_worker_at_root"],e["actual_transferred"],"independent original-worker robbery")
            stolen.append({"seed":seed,"seat":s,"t":t,"raw_sha256":row["raw_sha256"],"actual_transfer":e["actual_transferred"],
                           "victim":wid,"captor":e["winner"],"physical_landing":e["destination"],"captor_prior_cargo":e["enemy_pre_collision_cargo"],
                           "traced_cashout":enemy_cashout(trace,e,s),"tagged_trace":trace})
    assert len(ctrl)==coverage["public_train_options"]==8 and len(stolen)==3
    overlapping=[]
    for earlier in ctrl:
        for later in ctrl:
            if later["raw_sha256"]!=earlier["raw_sha256"] or later["t"]<=earlier["t"]:continue
            for record in earlier["tagged_trace"]["holdings_before_orders"]:
                if record["t"]==later["t"] and record["holder"]==later["worker"]:
                    overlapping.append({"earlier":earlier["id"],"later":later["id"],"repeated_capital":record["amount"]})
    assert len({c["id"] for c in ctrl})==len(ctrl)
    result={"verdict_sha256":run.sha(run.DEV/"verdict.json"),"coverage_sha256":run.sha(run.DEV/"coverage.json"),
            "opportunities_sha256":run.sha(run.DEV/"all_opportunities.json.gz"),"historical_losses_sha256":run.sha(run.DEV/"historical_train_losses.json.gz"),
            "script_sha256":run.sha(run.HERE/"observed_controls.py"),"lot_verifier_sha256":run.sha(run.HERE/"labels.py"),
            "sources":identities,"new_original_calls":0,"alternative_orders_applied":0,"repeated_capital_between_trigger_snapshots":overlapping,
            "accounting":"Each root counts only the cargo already held by its worker. An ownership transfer records carriage, not a bank receipt. The eventual tagged share is attributed from official recorded moves and rule-proportional tax. Treasury payments in whole_deposit are receipts for the whole carrier and cannot be added to the traced shares. Unresolved ground lots remain unassigned. Roots within a game are correlated; no cross-trigger capital sum or treatment gain is reported.",
            "observed_control_roots":sorted(ctrl,key=lambda c:(c["seed"],c["seat"],c["t"])),
            "actual_robberies":sorted(stolen,key=lambda e:(e["seed"],e["seat"],e["t"]))}
    run.freeze(run.HERE/"observed_controls.json",result)
    print("verified",len(ctrl),len(stolen),run.sha(run.HERE/"observed_controls.json"),flush=True)
if __name__=="__main__":main()

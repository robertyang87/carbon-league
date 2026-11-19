"""Independently replay registered economic interventions and separate valid paired labels from raw outcomes."""
import argparse
from collections import Counter
import gzip
import hashlib
import importlib.util
import json
import math
import subprocess
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
LIVE = HERE.parents[1]/"live_curriculum"
PAIRS = LIVE/"terminal_pairs"
COMMON = LIVE/"shared_enemy_root"
PLAN = "02374a25676aeb83ff9eb1a9fbe7184b3e63eafd49c720dd1baef637dc3c1c77"
sys.path[:0] = [str(LIVE),str(PAIRS),str(COMMON),str(HERE)]
import play
import settle_pairs as native
import select_root
import proposals

def load(name,path):
    spec = importlib.util.spec_from_file_location(name,path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

econ = load("signed_semantic_pair_economy",PAIRS/"economy.py")
shared_audit = load("signed_single_root_evidence",COMMON/"audit.py")
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = play.read
need = play.need

def checked_student(actor,obs,cfg,issued,category=None):
    """Independently enumerate the actual published action probabilities and witness on this public input."""
    capture = []
    mod = actor.mod
    normal = mod._mixed_actions
    def tap(x,aux):
        capture.append((x,aux))
        return normal(x,aux)
    if category:
        mod._mixed_actions = tap
    try:
        event = native.served(actor,obs,cfg,issued)
    finally:
        mod._mixed_actions = normal
    if not category:
        return event,None
    need(len(capture)==1 and "full_carbon" not in obs,"Missing causal features or private input")
    cached = [net.trunk for net in mod._NETS]
    try:
        with play.audit_menu.collect.attached(actor.modules):
            built = proposals.build(mod,*capture[0],obs)
    finally:
        for net,trunk in zip(mod._NETS,cached):
            net.trunk = trunk
    old = None
    if event and event["supported"]:
        old = dict(p=event["probabilities"],pick=event["choice"])
    proposed = select_root.pick(built,obs,cfg,issued,category,old)
    if proposed and old is None:
        proposed["native_A_probability"]=1.0
    return event,proposed

def compare_natural(actual,stored):
    if not actual or not actual["supported"]:
        need(stored is None,"Unexpected recorded native draw on singleton or ordinary step")
        return
    need(stored is not None and stored["pick"] == stored["unforced_choice"] == actual["choice"] and not stored["forced"],
         "Unforced natural draw differs")
    for key,other in (("options","options"),("p","probabilities"),("actions","actions"),("prefix","prefix"),
                      ("legal","legal"),("route","route"),("native_logits","native_logits"),("logp","logp"),("slot","slot")):
        need(stored[key] == actual[other],"Changed native menu field "+key)

def focal_cash(seed,seat,terrain,prefix,root_frames,menu):
    """Census all signed cash writes on the focal UID; other workers on the same seat cannot satisfy it."""
    out = []
    expect = {"capture":"seize","plant":"plant","deposit":"own_delivery"}[menu["kind"]]
    for root in root_frames:
        cfg = econ.cash.config()
        sim = econ.cash.FastCarbon(cfg).reset(seed=seed)
        need(econ.cash.sha(bytes(map(int,sim.carbon)))==terrain,"Different cash simulator terrain")
        for row in prefix:
            sim.step(*row["orders"])
        uid = menu["focal_uid"]
        sim.cash = econ.cash.CashEvents(sim.cash)
        before = list(sim.cash)
        pos = sim.workers[uid].pos
        # base_at maps physical position to owning seat and its UID.
        home = {cell for cell,(owner,_) in sim.base_at.items() if owner==seat}
        sim.step(*root["orders"])
        events = [ev for ev in sim.cash.events if ev[0] in ("plant","seize","own_delivery") and
                  ev[1]==seat and ev[3]==uid]
        for kind,owner,amount,who,tile,*_ in events:
            if kind=="own_delivery":
                need(tile in home and amount>0,"Non-own-base credit or nonpositive delivery")
            else:
                need(tile==pos and amount<0,"Nonfocal paid planting/seizure")
        for i,v in enumerate(econ.cash.cash_flow(sim.cash.events)):
            econ.cash.check_equal(sum(v.values()),sim.cash[i]-before[i],"cash-event closure")
        out.append(dict(actual_focal_events=events,all_first_transition_cash_changes_by_seat=
                        [round(sim.cash[s]-before[s],8) for s in (0,1)]))
    expected = menu["effect_arm"]
    relevant = {"own_delivery"} if menu["category"]=="freight" else {"plant","seize"}
    matches = [any(ev[0]==expect for ev in arm["actual_focal_events"]) for arm in out]
    others = any(ev[0] in relevant for ev in out[1-expected]["actual_focal_events"])
    return dict(focal_uid=menu["focal_uid"],kind=expect,predicted_event_arm="AB"[expected],
                A=out[0],B=out[1],verified_focal_contrast=matches[expected] and not others)

def verify_request(r,row,side,obs,live):
    need(r["step"]==row["step"] and r["order"]==row["orders"][1-side] and
         r["observation_sha256"]==row["public_before_by_seat"][1-side],"Wrong observed original request or action")
    public = {k:v for k,v in obs.items() if k!="remainingOverageTime"}
    need(play.sha(live.encoded(public))==r["game_observation_sha256"],"Original observer saw different game state")
    for key in ("wall","cpu"):
        need(0<r[key]<=row["original"][key]+2e-6,"Nested actual native request clock mismatch")
    shared_audit.check_stats(row["original"])

def replay(f,plan,pre,tail,reqs,menu,common,idx):
    seed,seat,terrain=f["seed"],f["seat"],f["board_sha256"]
    env,cfg,live=play.official(seed,terrain,seat)
    need(play.sha(bytes(env.states[0].observation.full_carbon))==terrain,"Wrong physical terrain")
    actor=play.audit_menu.collect.Player(play.student_history.PACK)
    bank=[5.,5.];nprobes=0
    for t,row in enumerate(pre+tail):
        is_root = bool(tail) and t==len(pre)
        is_probe = t in plan["probe_turns"] and (t<len(pre) or is_root)
        obs,hashes=shared_audit.public(live,env,bank)
        need(row["step"]==t==env.states[0].observation.step and not env.done and row["public_before_by_seat"]==hashes,
             "Public observations, seating or phase diverged")
        need(row["orders"]==([row["original"]["order"],row["student"]["order"]] if seat else
                            [row["student"]["order"],row["original"]["order"]]),"Wrong physical command or seat")
        for side in (0,1):
            shared_audit.legal(obs[side],row["orders"][side])
        rid=row["native_request_id"]
        verify_request(reqs[rid],row,seat,obs[1-seat],live)
        issued=menu["native_A"] if is_root else row["orders"][seat]
        event,selected=checked_student(actor,obs[seat],cfg,issued,f["category"] if is_probe else None)
        if is_probe:
            nprobes+=1
        if is_root:
            need(is_probe and selected is not None and row["shared_root"] and rid==common["request_id"] and
                 row["original"]==common["root_native_call"] and row["orders"][seat]==menu["options"][idx],
                 "Wrong root issuance or missing proposed economic contrast")
            need(common["pre_board_sha256"]==play.board_hash(env,terrain) and common["pre_public_by_seat"]==hashes and
                 common["original_observation_sha256"]==hashes[1-seat],"Root checkpoint/public physical state changed")
            for key in selected.keys()-{"native_choice"}:
                need(selected[key]==menu[key],"Changed registered selection/witness: "+key)
            compare_natural(event,menu["native_choice"])
            need(menu["public_observation_sha256"]==hashes[seat],"Menu features and public observation mismatch")
        else:
            need(not row["shared_root"],"A later or prefix enemy order was reused")
            if is_probe:
                need(selected is None,"Earlier admissible public economic contrast skipped")
            if row.get("student_event") is not None or not is_probe:
                compare_natural(event,row.get("student_event"))
        for s,name in ((seat,"student"),(1-seat,"original")):
            bank[s]-=max(row[name]["wall"]-1.,0.)
        need(bank==row["bank_after"],"Incorrect native or student time debit")
        env.step(row["orders"])
        need(play.board_hash(env,terrain)==row["board_after_sha256"] and
             shared_audit.public(live,env,bank)[1]==row["public_after_by_seat"],"Official board/IDs/observation replay mismatch")
    need(env.done and all(s.status=="DONE" for s in env.states),"Missing healthy official terminal")
    scores=[s.reward for s in env.states]
    diff=scores[seat]-scores[1-seat]
    wdl=1. if diff>0 else 0. if diff<0 else .5
    return dict(official_scores_by_seat=scores,terminal_step=env.states[0].observation.step,margin=diff,
                u=.5*wdl+.25*(1+math.tanh(diff/2000.)),wdl=wdl,public_semantic_probes=nprobes)

def native_census(pre,arms,reqs):
    # Verify physical invocation order independently from the per-arm clock ledgers.
    ordered=pre+arms[0]+(arms[1][1:] if len(arms)>1 else [])
    ids=[row["native_request_id"] for row in ordered]
    need(ids==list(range(len(reqs))) and [r["id"] for r in reqs]==ids,"Missing, repeated or shuffled actual native requests")
    return dict(physical_requests=len(ids),natural_prefix=len(pre),actual_root_requests=int(bool(arms[0])),
                single_shared_native_root=bool(arms[0]),physical_wall=sum(z["wall"] for z in reqs),
                physical_cpu=sum(z["cpu"] for z in reqs))

def settle(f,plan):
    out=HERE/"outputs"/f"{f['seed']}_{f['seat']}"
    root=dict(seed=f["seed"],seat=f["seat"],terrain=f["board_sha256"],category=f["category"],arm_order=f["arm_order"])
    # Keep any registered ticket even when no branch was established.
    if not (out/"finish.json").exists():
        reason=read(out/"failure.json") if (out/"failure.json").exists() else None
        return {**root,"eligible":False,"status":"failed" if reason else "pending","failure":reason}
    claim=read(out/"claim.json");finish=read(out/"finish.json")
    gates=read(HERE/"preflight.json")
    if finish["status"]=="preflight_hold":
        need(not gates["permitted"][f["category"]] and not (out/"prefix.jsonl.gz").exists(),"False preflight hold")
        return {**root,"eligible":False,"status":"preflight_hold","claim_sha256":sha(out/"claim.json")}
    need(gates["permitted"][f["category"]],"Unauthorized historical preflight")
    prefix=read(out/"prefix.json")
    for item in (prefix,claim,finish):
        need(item["seed"]==f["seed"] and item["seat"]==f["seat"],"Different registered map or seat")
    need(claim["protocol_sha256"]==PLAN,"Receipt references different protocol")
    need(claim["terrain"]==prefix["terrain"]==f["board_sha256"] and claim["category"]==f["category"] and
         claim["arm_order"]==f["arm_order"] and prefix["sources"]==claim["source_sha256"] and prefix["release"]==claim["release"],
         "Wrong actual source tree or registration")
    required={str(p.relative_to(play.ROOT)) for p in (HERE/"protocol.json",HERE/"select_root.py",HERE/"preflight.json",
              HERE/"run.py",COMMON/"external_menu.py",HERE.parent/"proposals.py",PAIRS/"run.py")}
    need(required<=set(prefix["sources"]),"Incomplete frozen runtime source identities")
    for name,expected in prefix["sources"].items():
        sealed=subprocess.check_output(["git","-C",str(play.ROOT),"show",claim["release"]+":"+name])
        need(hashlib.sha256(sealed).hexdigest()==expected and sha(play.ROOT/name)==expected,"Unbound executed source: "+name)
    machine=prefix["environment"]
    need(machine["local_diagnostic"] and machine["affinity"]==[0,1] and
         all(a==[0,1] for a in machine["all_thread_affinities"].values()) and machine["torch_threads"]==2 and
         machine["blas"]==machine["omp"]=="1","Different local examiner CPU availability")
    req_file=out/"native_requests.json.gz"
    reqs=json.loads(gzip.decompress(req_file.read_bytes()))
    pre=native.read_tape(out,"prefix",prefix)
    root.update(runtime_release=claim["release"],resources=machine,receipt_hashes={path.name:sha(path) for path in (out/"prefix.json",out/"claim.json",out/"finish.json",req_file)},
                prefix_search=native.search_stats(pre),prefix_tape_sha256=prefix["turns_sha256"])
    if not (out/"shared.json").exists():
        need(prefix["root_step"] is None,"Unreported selected root")
        truth=replay(f,plan,pre,[],reqs,None,None,None)
        need(truth["official_scores_by_seat"]==prefix["official_scores_by_seat"],"Unverified default prefix-only terminal")
        return {**root,**native_census(pre,[[]],reqs),**truth,"eligible":False,"status":"no_supported_root"}
    shared=read(out/"shared.json");menu=prefix["root_menu"]
    need(shared["pre_fingerprints"]==prefix["root_fingerprints"] and
         shared["pre_board_sha256"]==prefix["root_board_sha256"] and shared["pre_public_by_seat"]==prefix["root_public_by_seat"]
         and shared["pre_fingerprints"]["original"]!=shared["post_original_fingerprint"],"Absent genuine private-state carry")
    t=len(pre);seat=f["seat"]
    need(t==prefix["root_step"] and t in plan["probe_turns"] and len(menu["options"])==2 and
         menu["options"][0]==menu["native_A"]!=menu["options"][1] and menu["category"]==f["category"],"Unsigned root/order")
    need(0<menu["native_A_probability"]<=1 and 0<menu["catalog_probabilities"][1]<=1 and
         menu["conditional_selection_probability_given_A_and_menu"]==1.,"Wrong probability provenance")
    branches={}; rows={}
    for idx,name in enumerate(("A","B")):
        card=read(out/(name+".json"));tail=native.read_tape(out,name,card);rows[name]=tail
        need(card["arm"]==name and card["index"]==idx and card["root_step"]==t and
             card["stats"]["steps"]==len(tail) and card["root_fingerprints"]==prefix["root_fingerprints"] and
             card["sources"]==prefix["sources"] and card["release"]==claim["release"] and
             card["pid"]==claim["pid"],"Wrong arm, process, runtime or suffix size")
        proof=card["root_proof"]
        need(proof["issuer"]=="published_external_menu" and proof["pre_original_fingerprint"]==shared["pre_fingerprints"]["original"] and
             proof["restored_post_original_fingerprint"]==shared["post_original_fingerprint"] and
             proof["student_after_sha256"]==menu["parent_after_sha256"] and
             proof["student_rng_after_sha256"]==menu["parent_rng_after_sha256"] and
             proof["root_own_public_sha256"]==prefix["root_public_by_seat"][seat],"Branch changed signed private carry")
        ev={**menu,"pick":idx}
        need(proof["root_event"]==tail[0]["student_event"]==ev and card["root_intervention"]=={"event":ev,"orders":tail[0]["orders"]},"Published external event differs")
        outcome=replay(f,plan,pre,tail,reqs,menu,shared,idx)
        need(outcome["official_scores_by_seat"]==card["official_scores_by_seat"] and
             outcome["terminal_step"]==card["terminal_step"] and abs(outcome["margin"]-card["margin"])<1e-8
             and abs(outcome["u"]-card["u"])<1e-12,"Official terminal reward/utility mismatch")
        timing=native.combine_clocks(pre,tail)
        depth=native.search_stats(tail)
        branches[name]=dict(**outcome,clocks=timing,suffix_search=depth,sha256=sha(out/(name+".json")),
                            tape_sha256=card["turns_sha256"])
    need(finish["status"]=="paired_raw" and finish["root_step"]==t and finish["arm_order"]==f["arm_order"] and
         finish["final_root_restored"] and finish["root_examiner_commands_equal"] and
         finish["shared_request_id"]==shared["request_id"] and finish["physical_requests"]==len(reqs),
         "Completion card drift or failed restoration")
    money=econ.compare(f,out)
    need(money["status"]=="verified_official_terminal_accounts" and money["same_original_order_at_root"] and
         money["distinct_student_order_at_root"],"Income accounts or common original root action failed")
    event=focal_cash(f["seed"],seat,f["board_sha256"],pre,[rows["A"][0],rows["B"][0]],menu)
    scores=[branches[name]["margin"] for name in ("A","B")]
    diff=scores[1]-scores[0]
    econ.cash.check_equal(money["official_margin_B_minus_A"],diff,"Terminal accounting crosscheck")
    for name,b in branches.items():
        need(money["arms"][name]["official_scores_by_seat"]==b["official_scores_by_seat"],"Terminal cash replay disagrees")
    health = [root["prefix_search"]["meets_minimum_80_percent"] is True]
    health.extend(b["suffix_search"]["meets_minimum_80_percent"] is True for b in branches.values())
    cold=prefix["cold"];need(set(cold)=={"student","original"},"Missing cold timing");cold_ok=all(0<=t<=20 for t in cold.values())
    warm_ok=all(clock["healthy"] for b in branches.values() for clock in b["clocks"].values())
    eligible=event["verified_focal_contrast"] and all(health) and cold_ok and warm_ok
    return {**root,**native_census(pre,[rows[name] for name in f["arm_order"]],reqs),"status":"verified_pair",
            "root_step":t,"shared_sha256":sha(out/"shared.json"),"root_menu":menu,"root_native_call":shared["root_native_call"],
            "branches":branches,"economic_cash_reconciliation":money,"focal_event":event,"official_margin_B_minus_A":diff,
            "delta_u_B_minus_A":branches["B"]["u"]-branches["A"]["u"],"eligible":eligible,"cold":cold,
            "eligibility":{"cold":cold_ok,"warm_clock":warm_ok,"depth_prefix_A_B":health,"realized_semantic_event":event["verified_focal_contrast"]}}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-seed",type=int,choices=(621758462,621758463))
    args=parser.parse_args()
    need(sha(HERE/"protocol.json")==PLAN,"Unregistered experiment")
    plan=read(HERE/"protocol.json")
    paths={"proposal_protocol":HERE.parent/"protocol.json","proposal_code":HERE.parent/"proposals.py",
           "reported_48_roots":HERE.parent/"roots.json.gz","coverage":HERE.parent/"coverage.json",
           "shared_enemy_mechanism":COMMON/"receipt.json","shared_enemy_independent_audit":COMMON/"audit.json"}
    need({k:sha(p) for k,p in paths.items()}==plan["static_inputs"],"Qualification source drift")
    identity=read(LIVE/"protocol.json")["actors"]
    need(sha(play.ROOT/identity["student"]["file"])==plan["student_sha256"] and
         sha(play.ROOT/identity["original"]["file"])==plan["original_sha256"],"Policy changed")
    result=[]
    for f in plan["fixtures"]:
        if args.check_seed and args.check_seed!=f["seed"]:continue
        ticket=settle(f,plan)
        ticket.update(verification_sha256=sha(Path(__file__)),protocol_sha256=PLAN)
        if ticket["status"]!="pending":
            card_path=HERE/"outputs"/f"{f['seed']}_{f['seat']}"/"verified.json"
            if card_path.exists():
                need(read(card_path)==ticket,"Different existing independently verified ticket")
            else:
                play.save(card_path,ticket)
        result.append(ticket)
    if args.check_seed:
        print(json.dumps(result,sort_keys=True));return
    need(not any(r["status"]=="pending" for r in result),"Registered experiment not finished")
    doc=dict(schema="sapling-audited-real-economic-conditional-returns/1",protocol_sha256=PLAN,verification_sha256=sha(Path(__file__)),
             requested_roots=len(plan["fixtures"]),eligible_pairs=sum(r["eligible"] for r in result),fixtures=result,
             formal_games=0,new_games_by_auditor=0,scope="Causal root interventions conditional on one real sampled opponent response and one natural continuation per arm. Separate economic eligibility and official returns; both exposed training boards. No new inference about win rate, other terrains or ladder certification.")
    dest=HERE/"verify.json"
    if dest.exists():
        need(read(dest)==doc,"Different existing independent verification")
    else:
        play.save(dest,doc)
    print(sha(dest),[(r["seed"],r["status"],r["eligible"]) for r in result])

if __name__=="__main__":
    main()

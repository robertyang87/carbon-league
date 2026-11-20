"""Recompute public choices, actual examiner requests, official outcomes and economic transfers."""
import argparse
import gzip
import hashlib
import subprocess
import json
import math
from run import HERE, INTAKE, TERMINAL, check, load, menu, need, play, sha

evidence = load("income_interception_existing_verifier", TERMINAL/"verify.py")
native = evidence.native
cash = evidence.econ.cash


def verify_arm(f, pre, tail, requests, root_menu=None, shared=None, idx=None):
    seat, terrain = f["seat"], f["board_sha256"]
    env,cfg,live = play.official(f["seed"],terrain,seat)
    cash.np.random.seed(f["seed"])
    sim = cash.FastCarbon(cash.config()).reset(seed=f["seed"])
    sim.cash = cash.CashEvents(sim.cash)
    actor = play.audit_menu.collect.Player(play.student_history.PACK)
    bank=[5.,5.]
    target_deposits=[]; target_collisions=[]; root_effect=None
    for t,row in enumerate(pre+tail):
        own = row["orders"][seat]
        is_root = idx is not None and t==len(pre)
        obs,digests=evidence.shared_audit.public(live,env,bank)
        need(row["step"]==t==sim.step_count==env.states[0].observation.step and row["public_before_by_seat"]==digests, "public input hash")
        need(own==row["student"]["order"] and row["orders"][1-seat]==row["original"]["order"], "physical seats")
        for s in (0,1):
            evidence.shared_audit.legal(obs[s],row["orders"][s])
        evidence.verify_request(requests[row["native_request_id"]],row,seat,obs[1-seat],live)
        logical=root_menu["native_A"] if is_root else own
        actual=native.served(actor,obs[seat],cfg,logical)
        if (40<=t<=240) and (t<len(pre) or is_root):
            offered=menu.select(obs[seat],logical)
            if is_root:
                need(offered and all(root_menu[k]==v for k,v in offered.items()), "public offer differs")
                need(row["orders"][seat]==offered["options"][idx] and row["shared_root"],"wrong chosen intervention")
                need(row["original"]==shared["root_native_call"] and row["native_request_id"]==shared["request_id"],"different physical response at simultaneous root")
                need(digests==shared["pre_public_by_seat"] and row["student_event"]==dict(root_menu,pick=idx), "shared state or selected menu")
                need(root_menu["public_observation_sha256"]==play.sha(play.encode(obs[seat])) and
                     shared["pre_public_by_seat"][seat]==play.sha(live.encoded(obs[seat])),
                     "own visible hash in canonical and native-wire serializations")
                evidence.compare_natural(actual,root_menu["native_choice"])
            else:
                need(offered is None and not row["shared_root"],"earlier qualifying root was skipped")
        elif not is_root:
            need(not row["shared_root"],"reused later native response")
            evidence.compare_natural(actual,row["student_event"])
        sim.cash.events=[]
        before=list(sim.cash)
        sim.step(*row["orders"])
        env.step(row["orders"])
        for s,role in ((seat,"student"),(1-seat,"original")):
            bank[s]-=max(0.,row[role]["wall"]-1.)
        next_obs=env.states[0].observation
        expected=sim.full_observation()
        need(expected["players"]==next_obs.players and expected["trees"]==next_obs.trees and sim.carbon==next_obs.full_carbon, "double-engine state")
        need(bank==row["bank_after"] and row["board_after_sha256"]==play.board_hash(env,terrain) and
             row["public_after_by_seat"]==evidence.shared_audit.public(live,env,bank)[1],"poststate or bank mismatch")
        for s,flows in enumerate(cash.cash_flow(sim.cash.events)):
            cash.check_equal(sum(flows.values()),sim.cash[s]-before[s], "official flow closure")
        if idx is not None and t>=len(pre):
            uid,focal=root_menu["target_uid"],root_menu["focal_uid"]
            deposits=[e for e in sim.cash.events if e[0]=="own_delivery" and e[1]==1-seat and e[3]==uid]
            target_deposits.extend({"decision":t,"amount":e[2],"home":e[4]} for e in deposits)
            target_collisions.extend({"decision":t,"collision":list(event)} for event in sim.last_collisions if event[0]==uid)
            if is_root:
                def state(side,worker):
                    return next_obs.players[side][2].get(worker)
                root_effect=dict(target_command=row["orders"][1-seat].get(uid,"NONE"),actual_enemy_at_contact_cell=(
                    menu.board(15)[0][root_menu["target_pos"]][row["orders"][1-seat].get(uid,"NONE")]==root_menu["cell"]),
                    focal_after=state(seat,focal),target_after=state(1-seat,uid),
                    root_collisions=list(sim.last_collisions),direct_target_normal_deposit=deposits)
    need(env.done and all(s.status=="DONE" for s in env.states),"wrong official terminal")
    scores=[s.reward for s in env.states];margin=scores[seat]-scores[1-seat]
    u=.5*(1. if margin>0 else .5 if margin==0 else 0.)+.25*(1+math.tanh(margin/2000.))
    spans=[3,5,10,20,30]
    return dict(official_scores=scores,margin=margin,u=u,transitions=len(pre)+len(tail),clocks=native.combine_clocks(pre,tail),
                root_effect=root_effect,target_uid_all_later_own_warehouse_payments=target_deposits,
                original_laden_target_payments_by_horizon={str(h):round(sum(v["amount"] for v in target_deposits if v["decision"]<len(pre)+h),6) for h in spans},
                target_uid_collision_events=target_collisions,original_suffix_search=native.search_stats(tail) if tail else None)


def settle(plan,f):
    folder=HERE/"outputs"/f"{f['seed']}_{f['seat']}"
    claim=play.read(folder/"claim.json")
    n=lambda stem:native.read_tape(folder,stem,play.read(folder/(stem+".json")))
    prefix=play.read(folder/"prefix.json"); pre=n("prefix"); end=play.read(folder/"finish.json")
    rev=plan["frozen_source_commit"]
    canonical="sources/master3s2/master3s2.py"
    original=subprocess.check_output(["git","-C",str(play.ROOT),"show",rev+":"+canonical])
    need(hashlib.sha256(original).hexdigest()==plan["original_sha256"]==sha(play.ROOT/"data/master3s2_source.py"),
         "Examiner package does not match repository original")
    reqs=json.loads(gzip.decompress((folder/"native_requests.json.gz").read_bytes()))
    need(claim["protocol_sha256"]==sha(HERE/"protocol.json") and prefix["sources"]==plan["sources"] and
         (claim["terrain"],claim["seat"],claim["seed"])==(f["board_sha256"],f["seat"],f["seed"]),"registration mismatch")
    result=dict(terrain=f["board_sha256"],seed=f["seed"],seat=f["seat"],natural_observed=len(pre),status=end["status"],
                original_sha256=plan["original_sha256"],source_original=canonical,original_source_commit=rev,
                student_sha256=plan["student_sha256"],
                native_request_sha256=sha(folder/"native_requests.json.gz"),per_turn_raw_sha256=prefix["turns_sha256"],
                cold=prefix["cold"],environment=prefix["environment"],prefix_native_search=native.search_stats(pre))
    if end["status"]=="no_registered_supported_root":
        result.update(evidence.native_census(pre,[[]],reqs))
        result["natural_game"]=verify_arm(f,pre,[],reqs)
        need(prefix["official_scores_by_seat"]==result["natural_game"]["official_scores"],"natural game score mismatch")
        return result
    need(end["status"]=="paired_raw" and end["final_root_restored"] and end["root_examiner_commands_equal"],"incomplete A/B")
    offer=prefix["root_menu"]; shared=play.read(folder/"shared.json"); branch={}; tails={}
    for idx,label in enumerate(("A","B")):
        card=play.read(folder/(label+".json"));tail=n(label);tails[label]=tail
        proof=card["root_proof"]
        need(proof["pre_original_fingerprint"]==shared["pre_fingerprints"]["original"] and
             proof["restored_post_original_fingerprint"]==shared["post_original_fingerprint"] and
             proof["student_after_sha256"]==offer["parent_after_sha256"] and proof["student_rng_after_sha256"]==offer["parent_rng_after_sha256"],
             "native hidden state or private sampling restoration")
        branch[label]=verify_arm(f,pre,tail,reqs,offer,shared,idx)
        r=branch[label]
        need(r["official_scores"]==card["official_scores_by_seat"] and abs(r["u"]-card["u"])<1e-10,"terminal mismatch")
    result.update(evidence.native_census(pre,[tails[label] for label in f["arm_order"]],reqs))
    money=evidence.econ.compare(f,folder)
    need(money["status"]=="verified_official_terminal_accounts","missing double-engine economy")
    intercept=offer["intercept_arm"];other="B" if intercept=="A" else "A"
    result.update(root_step=len(pre),offer=offer,branch=branch,intercept_minus_avoid_margin=round(branch[intercept]["margin"]-branch[other]["margin"],2),
                  intercept_minus_avoid_u=branch[intercept]["u"]-branch[other]["u"],cash_account=money,
                  elapsed_wall_seconds=end["elapsed_seconds"])
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed",type=int)
    args=parser.parse_args();plan,_=check()
    for f in plan["fixtures"]:
        if args.seed is not None and args.seed!=f["seed"]:continue
        result=settle(plan,f);folder=HERE/"outputs"/f"{f['seed']}_{f['seat']}"
        play.save(folder/"verified.json",result)
        print(f["seed"],result["status"],result.get("intercept_minus_avoid_margin"),flush=True)


if __name__=="__main__":main()

"""Offline paired policy ablation over the complete registered maps, both seats and draws."""
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
CONTACT=HERE.parent
INTAKE=CONTACT.parent
ECON=INTAKE/"economy_proposals"
PLAN="6af8a912aa2c1f4880f4d99422191efc26002bd1d6b392a81a22de695f57f520"

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod

qa=load("registered_actual_own_menu",ECON/"terminal_intake/verify.py")
physical=load("archived_single_transition",ECON/"verification/verify.py")
source=physical.audit
features=load("signed_enemy_public_features",CONTACT/"features.py")
model=load("signed_enemy_classifier",CONTACT/"model.py")
play=qa.play
need=play.need

def distribution(base,risks):
    masses=[p*(1.-r) for p,r in zip(base,risks)]
    if sum(masses)==0:
        masses=base
    return float(masses[1]/math.fsum(masses))

def paired_draw(root_sha):
    prefix=b"sapling-contact-comparison-v1:"
    return int.from_bytes(hashlib.sha256(prefix+root_sha.encode()).digest()[:8],"big")/2**64

def policies(public,tracks,menu,weights):
    ids,x=features.board(public,tracks)
    seat=public["player"];uid=menu["focal_uid"];own=public["players"][seat][2];foe=public["players"][1-seat][2]
    tiles=[int(features.DEST[own[uid][0],features.MOVES.index(cmd.get(uid,"NONE"))]) for cmd in menu["options"]]
    contributing=[[(i,int(move)) for i,other in enumerate(ids)
                   for move in np.flatnonzero(features.DEST[foe[other][0]]==target)] for target in tiles]
    pred={}
    for name,w in weights.items():
        q=model.predict(w,x)
        risks=[1.-math.prod(1.-float(q[i,move]) for i,move in reachable) for reachable in contributing]
        pred[name]=dict(contact=risks,p_B=distribution(menu["catalog_probabilities"],risks))
    return dict(focal_uid=uid,focus_position=own[uid][0],effect_arm=menu["effect_arm"],
                destinations=tiles,nominal_kind=menu["kind"],options=menu["options"],
                catalog_masses=menu["catalog_probabilities"],heads=pred)

def check_immediate(env,sim,both,seat,pol):
    result=[]
    state=sim.get_state()
    uid=pol["focal_uid"]
    foe=env.states[0].observation.players[1-seat][2]
    foe_targets={int(features.DEST[v[0],features.MOVES.index(both[1-seat].get(u,"NONE"))])
                 for u,v in foe.items() if v[2]=="COLLECTOR"}
    for i,cmd in enumerate(pol["options"]):
        orders=list(both);orders[seat]=cmd
        after=physical.branch(env,state,orders);cash=after["account"]
        result.append(dict(
            named_planter_lost=any(v[0]==uid and v[1]==seat for v in cash["collisions"]),
            enemy_collector_intends_destination=pol["destinations"][i] in foe_targets,
            named_paid_events=[ev for ev in cash["cash_events"] if ev[0] in ("plant","seize") and ev[1]==seat and ev[3]==uid],
            own_delivered=cash["by_seat"][seat]["warehouse_deposit"],
            original_delivered=cash["by_seat"][1-seat]["warehouse_deposit"],
            cash_after=after["official_post_cash_by_seat"]))
    return result

def run_episode(ep,weights):
    seat,seed=ep["seat"],ep["seed"]
    tracks=features.Tracks();bank=[5.,5.];rows=[];digest=hashlib.sha256();prefix=hashlib.sha256()
    with source.trajectory(ep,play.student_history.PACK,ep["policy_sha256"]) as (live,env,cfg,tape,actor):
        sim=physical.mechanical.FastCarbon(cfg).reset(seed=seed)
        for t,line in enumerate(tape):
            digest.update(line);frame=json.loads(line);views,hashes=qa.shared_audit.public(live,env,bank)
            need(t==frame["step"]==sim.step_count and hashes==frame["public_observation_sha256"],"Diverged recorded prefix")
            obs=views[seat];tracks.observe(obs)
            need(all(c["status"]=="ok" for c in frame["calls"]),"Unhealthy original action")
            proposed=frame["calls"][seat]["response"]["action"]
            opportunity=t in qa.proposals.STEPS
            _,menu=qa.checked_student(actor,obs,cfg,proposed,"plant_or_capture" if opportunity else None)
            item=None
            if opportunity:
                item=dict(seed=seed,seat=seat,draw=ep["arm"],step=t,prefix_actions_sha256=prefix.hexdigest(),
                          public_sha256=hashes[seat],board_sha256=play.board_hash(env,ep["terrain"]),terrain=ep["terrain"],
                          natural_proposal=proposed,status="no_distinct_semantic_order")
                if menu:
                    if min(menu["catalog_probabilities"])>0:
                        pol=policies(obs,tracks,menu,weights)
                        item.update(status="probabilities",**pol)
                        delta=pol["heads"]["full"]["p_B"]-pol["heads"]["geometry"]["p_B"]
                        u=paired_draw(hashes[seat])
                        item.update(delta_p_B=delta,expected_shared_draw_change=abs(delta),shared_uniform=u,
                                    modal_command_changed=(pol["heads"]["full"]["p_B"]>.5)!=(pol["heads"]["geometry"]["p_B"]>.5),
                                    illustrative_common_draw_changed=(u<pol["heads"]["full"]["p_B"])!=(u<pol["heads"]["geometry"]["p_B"]))
                    else:
                        item["status"]="missing_positive_pair_mass"
            # The present enemy order and all its successor effects are joined only after probabilities were computed.
            both=[v["response"]["action"] for v in frame["calls"]]
            if item:
                item["observed_foe_order"]=both[1-seat]
                if item["status"]=="probabilities":
                    need(item["options"][0]==both[seat],"Changed historical natural A")
                    physical.mechanical.check_board(sim,env)
                    item["conditional_one_step"]=check_immediate(env,sim,both,seat,item)
                rows.append(item)
            sim.step(*both);env.step(both);bank=frame["overage_remaining_seconds"]
            prefix.update(play.encode(both))
            if item and item["status"]=="probabilities":
                a=item["conditional_one_step"][0]
                need(a["cash_after"]==[v[0] for v in env.states[0].observation.players] and
                     a["named_planter_lost"]==(item["focal_uid"] not in env.states[0].observation.players[seat][2]),
                     "One-step A differs from the actual archived transition")
        physical.mechanical.check_board(sim,env)
        need(env.done and [v.reward for v in env.states]==ep["official_scores_by_seat"] and digest.hexdigest()==ep["tape_sha256"],
             "Wrong official terminal or incomplete real trajectory")
    return dict(seed=seed,seat=seat,draw=ep["arm"],terrain=ep["terrain"],tape_sha256=ep["tape_sha256"],
                official_scores_by_seat=ep["official_scores_by_seat"],steps=sim.step_count,opportunities=rows)

def sum_group(items):
    paired=[it for it in items if it["status"]=="probabilities"]
    def delta(fn,row):return fn(row["conditional_one_step"][1])-fn(row["conditional_one_step"][0])
    fn={"named_planter_loss":lambda a:int(a["named_planter_lost"]),
        "named_paid_plant_or_seize":lambda a:int(bool(a["named_paid_events"])),
        "own_ordinary_delivery":lambda a:a["own_delivered"],
        "opponent_ordinary_delivery":lambda a:a["original_delivered"]}
    diffs={name:sum(r["delta_p_B"]*delta(cost,r) for r in paired) for name,cost in fn.items()}
    return dict(scheduled=len(items),semantic_support=len(paired),fallback_counts=dict(Counter(r["status"] for r in items if r["status"]!="probabilities")),
                positive_contact_opportunities=sum(any(r["heads"]["full"]["contact"]) for r in paired),
                any_probability_shift=sum(r["expected_shared_draw_change"]>1e-12 for r in paired),
                modal_command_changes=sum(r["modal_command_changed"] for r in paired),
                illustrative_real_order_differences_if_same_draw=sum(r["illustrative_common_draw_changed"] for r in paired),
                expected_different_joint_commands=sum(r["expected_shared_draw_change"] for r in paired),
                factual_A_named_losses=sum(r["conditional_one_step"][0]["named_planter_lost"] for r in paired),
                conditional_B_named_losses=sum(r["conditional_one_step"][1]["named_planter_lost"] for r in paired),
                actual_A_paid=sum(bool(r["conditional_one_step"][0]["named_paid_events"]) for r in paired),
                simulated_B_paid=sum(bool(r["conditional_one_step"][1]["named_paid_events"]) for r in paired),
                expected_one_step_difference_full_minus_geometry_summed_over_registered_roots=diffs)

def main():
    plan=play.read(HERE/"protocol.json");manifest=INTAKE/"hosted/training_manifest.json"
    need(qa.sha(HERE/"protocol.json")==PLAN and qa.sha(manifest)==plan["source_manifest"],"Registration or telemetry source changed")
    ws={}
    for k,h in plan["weights"].items():
        f=CONTACT/(k+".npz");need(qa.sha(f)==h,"Wrong frozen classifier")
        with np.load(f,allow_pickle=False) as pack:ws[k]={v:pack[v] for v in pack.files}
    eps=[ep for ep in play.read(manifest)["episodes"] if ep["seed"] in (621758462,621758463)]
    need(len(eps)==8 and len({(e["seed"],e["seat"],e["arm"]) for e in eps})==8 and
         all(e["episode_verified"] and e["policy_sha256"]==plan["actor"] and e["examiner_sha256"]==plan["original"] for e in eps),"Missing or wrong original episodes")
    games=[]
    for ep in eps:
        g=run_episode(ep,ws);games.append(g)
        print(ep["seed"],ep["seat"],ep["arm"],flush=True)
    rows=[r for g in games for r in g["opportunities"]]
    need(len(rows)==96,"Missing registered opportunities")
    idx=defaultdict(list)
    def key(r):
        return (r["terrain"],r["seat"],r["step"],r["board_sha256"],r["prefix_actions_sha256"],r["public_sha256"])
    for r in rows:idx[key(r)].append(r)
    possible=[]
    for r in rows:
        if r["status"]=="probabilities":
            possible.extend((r["public_sha256"],s["draw"]) for s in idx[key(r)] if s["natural_proposal"]==r["options"][1] and s["observed_foe_order"]==r["observed_foe_order"])
    oldpair=INTAKE/"economy_proposals/terminal_intake/verify.json"
    ev=play.read(oldpair)
    need(qa.sha(oldpair)=="348eba159ea80121ec9d888550fb8e14e83b2f1223c7acc04160f1efd21bd175","Different live pair")
    eligible=[r for r in ev["fixtures"] if r["eligible"]]
    need(not eligible and not possible,"Additional terminal evidence requires explicit label/continuation identity adjudication")
    source_root=ev["fixtures"][0];pub=source_root["root_menu"]["public_observation_sha256"]
    desc=play.read(CONTACT/"example_prediction.json")
    need(desc["public_input_sha256"]==pub and desc["step"]==source_root["root_step"],"Incorrect exposed example")
    situation={}
    for name,h in desc["models"].items():
        need(h["weights_sha256"]==plan["weights"][name],"Example uses different network")
        risks=[h["own_planter_targets"][a]["contact_probability"] for a in source_root["root_menu"]["physical_actions"]]
        situation[name]={"contact":risks,"p_B":distribution(source_root["root_menu"]["catalog_probabilities"],risks)}
    delta=situation["full"]["p_B"]-situation["geometry"]["p_B"]
    result=dict(schema="sapling-contact-conditioning-ablation/1",protocol_sha256=PLAN,implementation_sha256=qa.sha(Path(__file__)),
                frozen_weights_sha256=plan["weights"],games=games,overall=sum_group(rows),by_map={str(seed):sum_group([r for r in rows if r["seed"]==seed]) for seed in (621758462,621758463)},
                unique_registered_pre_histories=len(idx),possible_terminal_B_matches_within_eight_tapes=possible,
                ticket_verified_sha256=qa.sha(oldpair),eligible_shared_original_terminal_pairs=0,
                eligible_pairs_with_a_modal_action_change=0,new_original_games=0,new_weights=0,deploy=False,
                exposed_example=dict(root_public_hash=pub,registered_8_tape_roots_with_same_public_hash=sum(r["public_sha256"]==pub for r in rows),
                    same_physical_root_dual_terminal=True,paid_semantic_event=False,heads=situation,delta_p_B=delta,
                    factual_B_minus_A_margin=source_root["official_margin_B_minus_A"],factual_B_minus_A_u=source_root["delta_u_B_minus_A"],
                    raw_expected_u_change_on_these_two_exposed_outcomes=delta*source_root["delta_u_B_minus_A"]))
    play.save(HERE/"results.json",result)
    print(json.dumps(result["overall"],sort_keys=True),flush=True)
if __name__=="__main__":main()

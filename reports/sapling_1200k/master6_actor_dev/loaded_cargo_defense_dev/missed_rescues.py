"""Six retrospective checkpoints for every registered real enemy-carrier loss on the TRAINING maps."""
from collections import Counter
from copy import deepcopy
import json
import sys

import actions
from opportunities import old
from shared import HERE,ROOT,ACTOR,PARENT,read,save,sha,source

sys.path.insert(0,str(ROOT/"code"))
from fastsim import FastCarbon
sys.path.insert(0,str(ACTOR/"lot_settlement_dev"))
import lots

LEADS=(10,5,3,2,1,0)


def chk(ok,msg):
    if not ok:
        raise ValueError(msg)


def endpoints(own,base):
    return {u:actions.pos_after(v[0],base.get(u,"NONE"),15) for u,v in own[2].items()}


def ban_reasons(obs,cfg,base,scores,uid,cmd):
    seat=obs["player"]; own,enemy=obs["players"][seat],obs["players"][1-seat]
    pos,cargo,_=own[2][uid]
    homes=set(own[1].values());hostile=set(enemy[1].values())
    end=endpoints(own,base);recruit={p for b,p in own[1].items() if base.get(b) in ("RECCOLLECTOR","RECPLANTER")}
    reach=actions.common_enemy_reach(obs,cfg,15);factor=1-cfg.get("moveCost",0.)
    orig=base.get(uid,"NONE"); first=end[uid];dest=actions.pos_after(pos,cmd,15)
    amt=cargo if cmd=="NONE" else cargo*factor
    attack=actions.exposure(reach,dest,amt)
    parent_attack=actions.exposure(reach,first,cargo if orig=="NONE" else cargo*factor)
    blocked_ids=[u for u,p in end.items() if u!=uid and p==dest]
    blocked=recruit|{p for u,p in end.items() if u!=uid}
    d=min(actions.distance(dest,b,15) for b in homes); parent_d=min(actions.distance(first,b,15) for b in homes)
    clear=sorted(q for q in actions._downhill(dest,homes,15) if q not in blocked|hostile
                 and not actions.exposure(reach,q,amt*factor)["commitments"])
    routes=[];cost=0.
    for u,(fp,fc,fr) in own[2].items():
        ep=end[u]
        if u==uid or fr!="COLLECTOR" or fc<=0 or ep in homes:
            continue
        others=recruit|{p for v,p in end.items() if v not in (uid,u)}
        safe={q for q in actions._downhill(ep,homes,15) if q not in others|hostile
              and not actions.exposure(reach,q,fc*factor)["commitments"]}
        before=safe-{first}; after=safe-{dest}
        if len(after)<len(before):
            cost+=fc/max(1,len(before))
            routes.append({"uid":u,"cargo":fc,"before":sorted(before),"after":sorted(after)})
    ranking=sorted(actions.MOVES,key=lambda a:(a!=orig,-scores[a],actions.MOVES.index(a)))
    reasons=[]
    local=[]
    if cmd==orig:local.append("original_command")
    if cargo<actions.MIN_CARGO:local.append("principal_below_80")
    if not parent_attack["commitments"]:local.append("no_lighter_or_equal_can_reach_parent")
    if blocked_ids:local.append("teammate_final_destination")
    if dest in recruit:local.append("recruit_landing")
    if dest in hostile:local.append("enemy_base")
    if attack["strict"]:local.append("strict_lighter_can_reach")
    if attack["equal"]:local.append("equal_collector_can_reach")
    left=cfg["episodeSteps"]-1-obs["step"]
    if d>left-1:local.append("cannot_bank_by_end")
    if d>parent_d+2:local.append("return_detour")
    if d and not clear:local.append("no_static_safe_downhill")
    if any(r["cargo"]>=actions.MIN_CARGO and not r["after"] for r in routes):local.append("blocks_loaded_friends_last_downhill")
    if cost>cargo:local.append("teammate_opportunity_cost_exceeds_principal")
    if not 180<=obs["step"]<=295:reasons.append("outside_180_295")
    if not parent_attack["strict"]:reasons.append("parent_has_no_strict_lighter_threat")
    return {"action":cmd,"parent_order":orig,"rank":ranking.index(cmd)+1,"logit":scores[cmd],"destination":dest,
            "strict":attack["strict"],"equal":attack["equal"],"friendly":blocked_ids,"hiring":dest in recruit,
            "distance_home":d,"safe_downhill":clear,"teammate_route_loss":routes,"teammate_cost":cost,
            "local_rejections":local,"policy_rejections":reasons}


def next_safe(post, cfg, uid, actor, valid, ctx, seat):
    if post["step"]>=cfg["episodeSteps"]-1 or any(not team[2] and not team[3] and
                (not team[1] or team[0]<min(cfg["recCollectorCost"],cfg["recPlanterCost"])) for team in post["players"]):
        return None
    if uid not in post["players"][seat][2]:
        return []
    feat,count,trace=actor._FEAT,actor._COUNTS,actor._loaded_trace
    old_ctx=list(ctx)
    actor._FEAT=deepcopy(feat); actor._COUNTS=deepcopy(count)
    try:
        baseline=actor.agent(actions.public(post),cfg)
        x,aux,_,_=actor._loaded_trace
        own,enemy=post["players"][seat],post["players"][1-seat]
        end=endpoints(own,baseline)
        blocked={p for u,p in end.items() if u!=uid}|{p for b,p in own[1].items() if baseline.get(b) in ("RECCOLLECTOR","RECPLANTER")}
        reach=actions.common_enemy_reach(post,cfg,15)
        p,c,_=own[2][uid]
        safe=[]
        for cmd in actions.MOVES:
            q=actions.pos_after(p,cmd,15)
            if q in blocked or q in enemy[1].values() or actions.exposure(reach,q,c if cmd=="NONE" else c*(1-cfg.get("moveCost",0.)))["commitments"]:
                continue
            _,reason=valid(post,cfg,baseline,{uid:cmd},x,aux,actor._FEAT)
            if reason in ("accepted","already_selected"):
                safe.append([cmd,q])
        return safe
    finally:
        actor._FEAT,actor._COUNTS,actor._loaded_trace=feat,count,trace
        ctx[:]=old_ctx


def study(row,event,used):
    tape=source(row);s=row["seat"];uid=event["uid"];cfg=tape["game_configuration"]
    actor,ctx,valid=old.fresh_actor()
    actions.install_parent_trace(actor,PARENT.read_bytes(),valid)
    env=FastCarbon(cfg).reset(seed=row["seed"])
    result=[]
    for t,joint in enumerate(tape["actions"]):
        if t>event["t"]:
            break
        obs=actions.public(env.obs(s));base=actor.agent(obs,cfg)
        chk(base==joint[s],("wrong full parent prefix",row["seed"],s,t))
        if event["t"]-t in LEADS:
            pre=obs["players"][s][2].get(uid)
            checkpoint={"lead":event["t"]-t,"step":t,"focal":pre,"previous_real_guard_t":used if used is not None and used<t else None}
            if pre is None:
                checkpoint["directions"]=[{"action":cmd,"local_rejections":["focal_not_yet_alive"]} for cmd in actions.MOVES]
            else:
                x,aux,labels,raw=actor._loaded_trace
                feat=sys.modules["cnet2_feat"]
                slot=actor._FEAT.own_uid.index(uid)
                scores={a:raw[slot][int(feat.CANON_MOVE[i]) if s else i] for i,a in enumerate(actions.MOVES)}
                current_cfg={**cfg}
                geometry=actions.ranked_options(obs,current_cfg,base,{uid:scores})
                qualified={p["action"] for p in geometry}
                deploy=next((p for p in actions.candidates(obs,cfg,base,actor,valid) if p["threat"]["strict"]),None)
                checkpoint.update(original=base.get(uid,"NONE"),current_choice=deploy and [deploy["uid"],deploy["action"]],
                                  fixed_enemy_root=dict(joint[1-s]),directions=[])
                for cmd in actions.MOVES:
                    d=ban_reasons(obs,cfg,base,scores,uid,cmd)
                    chk((not d["local_rejections"])==(cmd in qualified),("frozen guard reason drift",row["seed"],t,cmd,d,qualified))
                    _,reason=valid(obs,cfg,base,{uid:cmd},x,aux,actor._FEAT)
                    d["full_prefix_verdict"]=reason
                    if reason not in ("accepted","already_selected"):
                        d["policy_rejections"].append("prefix:"+reason)
                    if used is not None and used<t:
                        d["policy_rejections"].append("prior_intervention_budget_spent")
                    if not d["local_rejections"] and not d["policy_rejections"] and (deploy is None or deploy["uid"]!=uid or deploy["action"]!=cmd):
                        d["policy_rejections"].append("another_legal_alternative_ranked_above")
                    branch=FastCarbon(cfg)
                    branch.set_state(env.get_state())
                    changed=[dict(j) for j in joint]
                    if cmd=="NONE":
                        changed[s].pop(uid,None)
                    else:
                        changed[s][uid]=cmd
                    initial,flows,after=lots.one_step(branch,changed,s,verify=True)
                    live=after["players"][s][2].get(uid)
                    booked=Counter()
                    if uid in initial:
                        chunks,agent,part=flows[uid]
                        for tag,why,_,destination,fraction in chunks:
                            if tag!="relay":
                                booked[tag]+=fraction*initial[uid]
                        booked["still_with_team"]+=part*initial[uid]
                    next_options=next_safe(branch.obs(s),cfg,uid,actor,valid,ctx,s) if live else []
                    d["fixed_recorded_response"]={"survived":bool(live),"worker_after":live,"principal_disposition":dict(booked),
                            "next_safe_actions_given_branch_parent_teammates":next_options}
                    checkpoint["directions"].append(d)
                if t==event["t"]:
                    origin=next(r for r in checkpoint["directions"] if r["action"]==checkpoint["original"])
                    chk(not origin["fixed_recorded_response"]["survived"],"recorded theft was not reproduced")
                    chk(abs(origin["fixed_recorded_response"]["principal_disposition"].get("lost_to_enemy",0.)-event["actual_transferred"])<1e-4,"wrong robbery amount")
            result.append(checkpoint)
        # Verify each actual original transition against the bundled official Board and maintain exact ID counters.
        lots.one_step(env,joint,s,verify=True)
    return {"map":row["map"],"seed":row["seed"],"seat":s,"uid":uid,"event_t":event["t"],"victim_amount":event["actual_transferred"],
            "raw_sha256":row["raw_sha256"],"oracle_receiver":event["winner"],"checks":result}


def main():
    inv=read(HERE/"source/inventory.json")
    train={(r["seed"],r["seat"]):r for r in inv["games"] if r["split"]=="train"}
    history=read(HERE/"historical_train_losses.json.gz")
    result=[]
    for case in history["cases"]:
        row=train[case["seed"],case["seat"]]
        if case["events"]:
            panel=read(HERE/"public_sources"/f'{row["seed"]}_{row["seat"]}.json.gz')["panel"]
            used=panel["cards"][0]["t"] if panel["cards"] else None
        for e in case["events"]:
            result.append(study(row,e,used))
    chk(len(result)==3 and all({c["lead"] for c in r["checks"]}==set(LEADS) for r in result),"incomplete event panel")
    out={"schema":"all-trained-theft-forensic-v1","source_history_sha256":sha(HERE/"historical_train_losses.json.gz"),
         "action_method_sha256":sha(HERE/"actions.py"),"driver_sha256":sha(HERE/"missed_rescues.py"),
         "official_one_step_accounting_sha256":sha(ACTOR/"lot_settlement_dev/lots.py"),"fixed_response_scope":
         "At each branch all other workers and the simultaneous opponent root commands are those actually logged at the original root. Verify each first transition against the official Board. On its resulting public board, enumerate next safe alternatives using the genuine branch parent joint commands for the other own workers, against all individually possible light/equal arrivals. No adaptive opponent continuation, terminal outcome or benefit estimate.",
         "events":result}
    save(HERE/"missed_rescues.json",out)
    for e in result:
        at=next(r for r in e["checks"] if r["lead"]==0)
        print(e["seed"],e["seat"],e["event_t"],[(p["action"],p["rank"],p["fixed_recorded_response"]["survived"],
                 len(p["fixed_recorded_response"]["next_safe_actions_given_branch_parent_teammates"] or [])) for p in at["directions"]])


if __name__=="__main__":
    main()

"""Replay the registered two physical orders and compute their separate tree and wallet accounts."""
from collections import Counter
from pathlib import Path
import json

from study import (HERE, REG, DIRS, advance, check, dump, freight, read, setup, sha, source, token,
                   tree, views)


def target(m):
    return max(-2.5,min(2.5,m/3000))+(0.5 if m>0 else -0.5 if m<0 else 0.)


def calculate(selection):
    ticket=selection["ticket"];s=ticket["seat"];t=ticket["step"]
    info=selection["intervention"];uid=info["info"]["worker"];cell=info["info"]["cell"]
    weights=HERE/"rent/weights.json"
    weight_sha=sha(weights.read_bytes())
    check(weight_sha=="0a9fb5f2bcafae18cf1a25fafc72e9fe9d6b84ac7e75c27155a93e3282ea42ab","Wrong rent predictor")
    head=read(weights);results={};logs={};root_crossings={};root_reply=None
    for arm in ("plant","leave"):
        path=HERE/f"{ticket['seed']}_{s}_{t}_{arm}.json.gz";data=read(path)
        check(sha(path.read_bytes())==read(HERE/"response.json")["arms"][arm]["evidence_sha256"],"Changed branch")
        tape,actions,cfg,env,sim=setup(ticket,source(ticket));history={}
        for i,a in enumerate(actions[:t]):
            if t-i in (3,5):history[t-i]=views(env)[s]["players"]
            advance(env,sim,a)
        obs=views(env)[s]
        X=tree.features(obs,uid,cfg,history)
        raw=head["intercept"]+sum(b*(x-m)/z for x,b,m,z in zip(X,head["beta"],head["center"],head["scale"],strict=True))
        prediction=max(0.,raw) if cfg["episodeSteps"]-1-t>8 else 0.
        if root_reply is None:root_reply=data["root_shared_original_order"]
        check(root_reply==data["rows"][0]["orders"][1-s],"Different root opponent")
        rid=None; active=False;end=None;cash_components=[[0.]*3 for _ in range(2)]
        focal_cost=0.; signed_h8=[Counter(),Counter()];signed_after=[Counter(),Counter()]
        worker_log=[]; first_own_changes=[];bank=[o["remainingOverageTime"] for o in views(env)]
        for row in data["rows"]:
            i=row["t"];ps=views(env)[s]["players"];pre=dict(ps[s][3])
            # Official state excludes the wire client's private bank. Reconstruct its request too.
            fresh=[freight.public(env,k,bank) for k in (0,1)]
            check([freight.digest(o) for o in fresh]==row["input_sha256"],"Changed public request")
            check(fresh[s]["step"]==i,"Wrong step index")
            bank=[b-max(0,c["wall"]-1) for b,c in zip(bank,row["clocks"])]
            cash=advance(env,sim,row["orders"])
            check(cash==row["end_cash"],"Official cash changed")
            after=views(env)[s]["players"]
            new=[k for k,v in after[s][3].items() if k not in pre and v[0]==cell]
            if i==t:
                entrants=[]
                for seat,side in enumerate(ps):
                    for worker,(pos,load,role) in side[2].items():
                        order=row["orders"][seat].get(worker)
                        dst=tree.neighbors(pos,cfg["size"])[DIRS.index(order)] if order else pos
                        if dst==cell:
                            entrants.append(dict(seat=seat,uid=worker,role=role,from_cell=pos,carried=load,command=order or "NONE"))
                root_crossings[arm]=dict(entrants=entrants,focused_after=after[s][2].get(uid),official_cash=cash,
                                          events=[v for v in sim.cash.events if len(v)>4 and v[4]==cell])
                if new:
                    check(len(new)==1,"Conflicting focal trees")
                    rid=new[0];active=True
            holdings={k:seat for seat,p in enumerate(after) for k in p[3]}
            econ=freight.econ.cash_flow(sim.cash.events)
            for seat in (0,1):(signed_h8 if i<t+8 else signed_after)[seat].update(econ[seat])
            funds=cash_components[int(i>=t+8)]
            prev_active=active
            for typ,side,val,*detail in sim.cash.events:
                if i==t and typ=="plant" and side==s and detail[:2]==[uid,cell]:focal_cost-=val
                if typ=="tree_absorption" and rid and detail[0]==rid and side==s and active and holdings.get(rid)==s:
                    funds[0]+=val-detail[2];funds[1]+=detail[2]
                if typ=="tree_collision_cash" and rid and len(detail)>2 and detail[2]==rid and side==s and prev_active:
                    funds[2]+=val
                if typ in ("plant","seize") and side==s and detail[0]==uid:
                    first_own_changes.append([i,typ,*detail,float(val)])
            if active and holdings.get(rid)!=s:
                active=False;end={"step":i,"reason":"seized" if rid in holdings else "withered"}
            worker_log.append((i,ps[s][2].get(uid),row["orders"][s].get(uid,"NONE"),after[s][2].get(uid)))
        score=[float(st.reward) for st in env.states]
        check(score==data["outcome"]["scores"] and [st.status for st in env.states]==["DONE","DONE"],"Wrong official terminal")
        results[arm]=dict(scores=score,margin=score[s]-score[1-s],native_target=target(score[s]-score[1-s]),
                          focal_root_cost=focal_cost,first_tenure_id=rid,
                          first_tenure_end=end or ({"step":data["rows"][-1]["t"],"reason":"horizon"} if active else None),
                          first_8_trees_receipts=cash_components[0],later_first_tenure_trees_receipts=cash_components[1],
                          first_8_wallet_by_seat=signed_h8,post_8_wallet_by_seat=signed_after,
                          focus_worker_new_plant_or_seize=first_own_changes,worker_history=worker_log,
                          preaction_rent_if_plant_succeeds=prediction,source_sha256=sha(path.read_bytes()))
        logs[arm]=data["rows"]
    a,b=results["plant"],results["leave"]
    check(a["preaction_rent_if_plant_succeeds"]==b["preaction_rent_if_plant_succeeds"],"Post-treatment leakage in rent predictor")
    p,l=logs["plant"],logs["leave"]
    def divergent(seat):
        return [r["t"] for r,q in zip(p[1:],l[1:]) if r["orders"][seat]!=q["orders"][seat]]
    dump("root_interaction.json",dict(root=[ticket["seed"],s,t],cell=cell,worker=uid,arms=root_crossings))
    winner=selection["scans"][-1]["own"]
    native=info["plant_value"]-info["leave_value"];truth=a["native_target"]-b["native_target"]
    return dict(root=[ticket["seed"],s,t],worker=uid,cell=cell,source_parent=REG["parent"],
                opponent=REG["examiner"],policy_selected="plant" if winner==info["plant"] else "leave" if winner==info["leave"] else "third",
                true_plant_and_no_plant=a["focal_root_cost"]>0 and b["focal_root_cost"]==0,
                native_plant_minus_leave=native,official_plant_minus_leave=a["margin"]-b["margin"],
                official_mapped_plant_minus_leave=truth,paired_residual_native_units=truth-native,
                next_own_unequal_command_turns=divergent(s),next_enemy_unequal_command_turns=divergent(1-s),
                cash_order=["ground","adjacent_enemy_cargo","collision_on_tree"],arms=results,
                rent_weight_sha256=weight_sha,root_shared_opponent_sha256=token(root_reply))


if __name__=="__main__":
    s=read(HERE/"selection.json")
    if s["status"]=="selected_before_original_reply":
        dump("verified_economy.json",calculate(s))

"""Reconcile the exact one-planter intervention, its tree cash, and per-map held-out predictions."""
import gzip,json,hashlib
from pathlib import Path
import numpy as np
from register import HERE,D,sha
from order import Anchor,destination
from labels import bind_terminal_returns

def read(path):
    b=path.read_bytes()
    return json.loads(gzip.decompress(b) if path.suffix==".gz" else b)

def summary_arm(tape,ticket):
    s=ticket["seat"];cell=ticket["cell"];first=tape["rows"][0]
    created=[f for f in first["flows"] if f[0]=="plant" and f[1]==s and f[3]==ticket["uid"]]
    tree=next((tr["tree"] for tr in first["tree_on_root_cell"] if tr["seat"]==s),None)
    if bool(created)!=bool(tree):raise ValueError("Tree identity/payment mismatch")
    paid=-sum(e[2] for e in created);captured=0;held=0;broken=False
    for step in tape["rows"]:
        if not tree or broken:break
        # Collisions resolve before the stationary planters take trees. Credit the actual
        # owner named in the event even on its last turn before losing this exact tree ID.
        captured+=sum(e[2] for e in step["flows"] if e[0]=="tree_collision_cash" and e[1]==s and len(e)>5 and e[5]==tree)
        captured+=sum(e[2] for e in step["flows"] if e[0]=="tree_absorption" and e[1]==s and e[3]==tree)
        if any(v["tree"]==tree and v["seat"]==s for v in step["tree_on_root_cell"]):held+=1
        else:broken=True
    all_call=[v for row in tape["rows"] for v in row["clocks"]]
    return dict(scores=tape["scores"],margin=tape["official_margin"],root_actor_state=first["focal_planter"],first_ownership_steps=held,
                planted=bool(tree),root_tree_cost=round(paid,4),root_tree_gross=round(captured,4),root_tree_net=round(captured-paid,4),
                terminal_step=tape["final_t"],max_wall=max(v["wall"] for v in all_call),slow_calls=sum(v["wall"]>1 for v in all_call),
                score_by_category=tape["flows_by_seat"],owned_steps_at_cell=sum(any(v["seat"]==s for v in row["tree_on_root_cell"]) for row in tape["rows"]))

def main():
    protocol=read(HERE/"protocol.json");pr_sha=sha((HERE/"protocol.json").read_bytes())
    cfg=read(D.parents[1]/"code/env_src/carbon_challenge-master/zerosum_env/envs/carbon/carbon.json")["configuration"]
    cfg={k:v["default"] for k,v in cfg.items() if isinstance(v,dict) and "default" in v};cfg.setdefault("episodeSteps",300)
    rows=[]
    for ticket in protocol["tickets"]:
        seed=ticket["seed"];s=ticket["seat"];t=ticket["step"];base=read(HERE/f"{seed}_root.json.gz");rr=read(HERE/f"{seed}_result.json")
        assert base["protocol"]==rr["protocol"]==pr_sha and base["ticket"]==ticket and rr["root_sha256"]==sha((HERE/f"{seed}_root.json.gz").read_bytes())
        parent="stay" if ticket["historical"]=="NONE" else "leave";alt="leave" if parent=="stay" else "stay"
        anchor=Anchor(base["public_root_by_seat"][s],ticket[parent],ticket["uid"],cfg);traces={};arms={}
        for name in ("stay","leave"):
            f=HERE/f"{seed}_{name}.json.gz";r=read(f)
            assert rr["arms"][name]["sha256"]==sha(f.read_bytes()) and r["root_enemy"]==base["local_root_calls"][1-s][0]
            assert r["rows"][0]["actions"][s]==ticket[name] and [q["step"] for q in r["rows"]]==list(range(t,r["final_t"]))
            assert list(np.round(np.subtract(r["rows"][-1]["wallets"],[v[0] for v in base["public_root_by_seat"][s]["players"]]),3))==[round(sum(f.values()),3) for f in r["flows_by_seat"]]
            arms[name]=summary_arm(r,ticket);traces[name]=r
        label=bind_terminal_returns(anchor,ticket,pr_sha,traces[parent],traces[alt]);y=round(arms["stay"]["margin"]-arms["leave"]["margin"],2)
        assert y==rr["delta_stay_minus_leave"]
        early={}
        for horizon in (1,3,5,10):
            if min(len(traces[k]["rows"]) for k in ("stay","leave"))>=horizon:
                differences=[]
                for k in ("stay","leave"):
                    wallet=traces[k]["rows"][horizon-1]["wallets"];differences.append(wallet[s]-wallet[1-s])
                early[str(horizon)]=round(differences[0]-differences[1],2)
        foe_divergences=[a["step"] for a,b in zip(traces["stay"]["rows"],traces["leave"]["rows"]) if a["actions"][1-s]!=b["actions"][1-s]]
        by_flow={}
        cats=set(c for arm in arms.values() for side in arm["score_by_category"] for c in side)
        for k in sorted(cats):
            effects=[]
            for mode in ("stay","leave"):
                q=arms[mode]["score_by_category"];effects.append(q[s].get(k,0)-q[1-s].get(k,0))
            by_flow[k]=round(effects[0]-effects[1],4)
        assert abs(sum(by_flow.values())-y)<.02
        incoming=[]
        for side,pl in enumerate(base["public_root_by_seat"][s]["players"]):
            for uid,worker in pl[2].items():
                cmd=traces["stay"]["rows"][0]["actions"][side].get(uid,"NONE")
                if destination(worker[0],cmd,cfg["size"])==ticket["cell"]:
                    incoming.append(dict(side=side,uid=uid,role=worker[2],src=worker[0],action=cmd))
        rows.append(dict(stay_arm_actual_arrivals=incoming,early_delta_from_same_root=early,opponent_different_complete_order_count=len(foe_divergences),
                         first_divergent_foe_step=foe_divergences[0] if foe_divergences else None,delta_official_cash_categories=by_flow,
                         seed=seed,terrain=ticket["terrain"],seat=s,step=t,uid=ticket["uid"],cell=ticket["cell"],old_net=ticket["estimated_net"],
                         old_payback_failure=ticket["estimated_nonpayback"],quote=ticket["quote"],parent_mode=parent,arms=arms,stay_minus_leave=y,
                         winner_switched=(arms["stay"]["margin"]>0)!=(arms["leave"]["margin"]>0),label=label))
    assert len(set(r["terrain"] for r in rows))==len(rows)
    x=np.array([r["old_net"] for r in rows]);y=np.array([r["stay_minus_leave"] for r in rows])
    for i,r in enumerate(rows):
        train=np.arange(len(rows))!=i;cx=x[train].mean();cy=y[train].mean();sd=x[train].std();sd=sd if sd>0 else 1.
        z=(x[train]-cx)/sd
        slope=float(z@(y[train]-cy)/(z@z+sum(train)));p=float(cy+(x[i]-cx)/sd*slope)
        r["cross_validation"]=dict(training_maps=[o["seed"] for j,o in enumerate(rows) if i!=j],fixed_lambda_per_map=1,
            feature_center=float(cx),feature_scale=float(sd),return_intercept=float(cy),learned_slope=slope,predicted_official_delta=p,
            training_mean_baseline_prediction=float(cy))
    patterns={}
    for name,p in (("unadjusted_first_tenure_net",x),("leave_one_map_out_ridge",np.array([r["cross_validation"]["predicted_official_delta"] for r in rows])),
                   ("leave_one_map_out_train_mean",np.array([r["cross_validation"]["training_mean_baseline_prediction"] for r in rows]))):
        choose=p>0
        patterns[name]=dict(mae=float(np.abs(p-y).mean()),correct_sign=int(((p>0)==(y>0)).sum()),
                          selected_stay=int(choose.sum()),observed_gain_vs_always_leave=float(np.mean(choose*y)))
    dest=HERE/"settlement.json";doc=dict(resource_note="One Linux CPU affinity 4; a co-scheduled local diagnostic also used this CPU for a portion of this run. Wall-clock search was live in every branch. No removals, restarts or claims of exclusive capacity or ladder timing certification.",
        protocol=pr_sha,source=sha(Path(__file__).read_bytes()),intervention_rows=rows,
        retrospective_four_map_predictions=patterns,arm_count=2*len(rows),total_joint_transitions=sum(u["terminal_step"]-r["step"] for r in rows for u in r["arms"].values()))
    dest.write_text(json.dumps(doc,indent=2,ensure_ascii=False,sort_keys=True,allow_nan=False)+"\n")
    print([(r["seed"],r["old_net"],r["stay_minus_leave"],r["winner_switched"]) for r in rows],patterns)
if __name__=="__main__":main()

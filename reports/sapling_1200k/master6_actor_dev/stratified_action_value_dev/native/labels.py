"""Independent offline settlement from exactly the signed raw history and both native response tapes."""
from collections import Counter, defaultdict


def measure(h, c, arm, d):
    p=d/f"{arm}.json.gz";z=h.read(p);binding=z["binding"];s=c["seat"];e=1-s
    h.check(binding["card_sha256"]==h.dig(c) and binding["arm"]==arm and binding["frozen_sha256"]==h.frozen_hash(),"arm source changed")
    env,own,_,obs,op,tape=h.root_state(c)
    cfg=tape["game_configuration"];sim=h.account.settled.FastCarbon(cfg).reset(seed=c["seed"])
    for j in tape["actions"][:c["t"]]:sim.step(*j)
    snap=sim.get_state()
    h.check(sim.obs(s)["players"]==obs["players"] and z["initial_public"]==[h.dig(obs),h.dig(op)],"root snapshots differ")
    root=z["joint0"]
    h.check(root[s]==c[arm] and root[e]==c["root_enemy"],"changed the opponent or an unregistered order")
    tail=[root];env.step(root)
    for item in z["history"]:
        u=h.live.public(env,s);v=h.live.public(env,e);j=item["joint"]
        h.check(not env.done and item["t"]==u["step"] and item["own_public_sha256"]==h.dig(u) and item["op_public_sha256"]==h.dig(v) and
                own.agent(u,cfg)==j[s],"physical response or continuation policy changed")
        env.step(j);tail.append(j)
    scores=[r.reward for r in env.states]
    h.check(env.done and [v.status for v in env.states]==z["statuses"]==["DONE","DONE"] and scores==z["official_scores"] and
            scores[s]-scores[e]==z["official_margin"] and len(tail)-1==z["completed_original_calls"],"unverified terminal or examiner call count")
    edits=c.get("edits") or [c["proposal"]]
    focal=[r["uid"] for r in edits if obs["players"][s][2].get(r["uid"],[0,0,""])[2]=="COLLECTOR"]
    h.check(focal,"this study requires a real focal collector for starting-principal labels")
    flows=h.account.account_arm(snap,cfg,tail,s,focal)
    h.check(flows["official_scores_physical"]==scores and flows["terminal_cash_physical"]==z["cash"] and
            flows["terminal_t"]==z["terminal_step"]==env.states[0].observation.step,"independent lot replay/engine differs")
    worker=[flows["per_worker"][u] for u in focal];team=flows["team"]
    treasuries=team["treasury_flows_physical"]
    metrics={"margin":z["official_margin"],"U":z["U"],"principal":sum(w["strict_principal_delivered"] for w in worker),
             "new_direct":sum(w["strict_fresh_delivered"] for w in worker),"mined":sum(w["strict_fresh_mined"] for w in worker),
             "own_delivery":team["own_delivery"],"focal_delivery":team["focal_delivery"],"others_delivery":team["other_delivery"],
             "team_fresh_direct":team["strict_new_delivered"],"enemy_delivery":treasuries[e].get("own_delivery",0.),
             "our_terminal_score":scores[s],"enemy_terminal_score":scores[e]}
    path=d/f"{arm}_account.json.gz";h.freeze(path,{"arm_sha256":h.sha(p),"account":flows,"measures":metrics})
    return metrics,h.sha(path)


def main(h,split):
    inp,cards=h.load_inputs();selected,selsha=h.choose(inp,cards,split);active={c["id"] for c in selected}
    receipt=h.read(h.HERE/f"{split}_complete.json")
    h.check(receipt["active"]==[c["id"] for c in selected] and receipt["held_selection_sha256"]==selsha and
            receipt["frozen_sha256"]==h.frozen_hash(),"wrong stage receipt")
    rows=[]
    states=defaultdict(list)
    for c in cards:
        if c["split"]!=split:continue
        state=c.get("state_id",f'{c["seed"]}-{c["seat"]}-{c["t"]}')
        states[(c["map"],c["seat"],state)].append(c["id"])
        if c["id"] not in active:continue
        row={k:c.get(k) for k in ("id","state_id","split","seed","seat","t","map","proposal","band","kind","stream","primary_id")}
        row["stream"]=c.get("stream",0);row["diagnostic"]=bool(c.get("diagnostic",False))
        d=h.HERE/c["id"];pair=h.read(d/"pair.json")
        h.check(pair["files"]=={a:h.sha(d/f"{a}.json.gz") for a in ("base","alt")} and pair["card_sha256"]==h.dig(c),"pair tampering")
        results={};hashes={}
        for a in ("base","alt"): results[a],hashes[a]=measure(h,c,a,d)
        row.update(status="paired_full_original",arms=results,account_sha256=hashes,
                   delta={k:results["alt"][k]-results["base"][k] for k in results["base"]})
        rows.append(row)
    state_rows=[{"map":key[0],"seat":key[1],"state_id":key[2],"eligible_cards":len(ids),"selected_ids":[i for i in ids if i in active]}
                for key,ids in states.items()]
    target=h.HERE/f"{split}_labels.json"
    h.freeze(target,{"split":split,"frozen_sha256":h.frozen_hash(),"runner_sha256":h.sha(h.HERE/"run.py"),
                     "source_protocol_sha256":h.sha(h.DEV/"native_input.json"),"verifier_sha256":h.sha(h.HERE/"labels.py"),
                     "counter_sha256":h.sha(h.OLD/"evidence/account.py"),"held_selection_sha256":selsha,
                     "cases":rows,"eligible_states":state_rows,"registered_map_seats":len({k[:2] for k in states})})
    print("verified",split,len(rows),"paired",len(states),"candidate-bearing states",flush=True)

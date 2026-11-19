"""Post-registration sensitivity: on each changed branch, let the exact production parent pick the last action."""
import copy
import importlib.util
import statistics
from collections import defaultdict

from study import HERE, REPLAY, ROOT, FastCarbon, protocol, read, require, sha, twin_step, write
from policy import moves, orders


def clean(commands):
    return {u:a for u,a in commands.items() if a and a != "NONE"}


def guaranteed_delivery(obs, commands, cfg):
    """Zero-fee amount furnished by this side's own cargo directed onto its own bank."""
    require(cfg["moveCost"] == cfg["smeltCost"] == 0, "principal identity needs zero fees")
    s = obs["player"]
    _, banks, workers, _ = obs["players"][s]
    payments = []
    for uid, (pos, load, kind) in workers.items():
        if kind != "COLLECTOR" or load <= 0:
            continue
        dest = moves(pos,cfg["size"])[commands.get(uid) or "NONE"]
        if dest in banks.values():
            payments.append({"uid": uid, "cell": dest, "cargo": load})
    return payments


def run():
    plan = protocol()
    pack = ROOT / "data/weights/sapling_1200k/terminal_return_fit_v1.py"
    require(sha(pack) == plan["source_parent"], "wrong native parent")
    spec = importlib.util.spec_from_file_location("verified_queue_parent", pack)
    agent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agent)
    caches=read(REPLAY)["games"]
    rows=read(HERE/"root_audit.json.gz")
    info = {(g["seed"],g["seat"]):g for b in plan["source_batches"] for g in read(ROOT / b["receipt"])["games"]}
    results = []
    for game in caches:
        seed,seat=game["seed"],game["seat"]
        k = (seed,seat)
        row = next(r for r in rows if (r["seed"],r["seat"],r["t"])==(*k,297))
        entry={"seed":seed,"seat":seat,"terrain":game["terrain"],"split":game["split"],"edits":row["changes"]}
        if not row["changed"]:
            entry["paired_margin_delta_by_enemy"]=[0,0,0]
            results.append(entry)
            continue
        path=ROOT/info[k]["raw_path"]
        require(sha(path)==game["raw_sha256"], "incorrect parent tape")
        raw=read(path)
        cfg=dict(game["config"],randomSeed=None)
        sim=FastCarbon(cfg).reset(full_carbon=raw["initial_full_carbon"])
        for t in range(297):
            public=sim.obs(seat)
            if t==0:
                require(clean(agent.agent(public,cfg)) == clean(raw["actions"][t][seat]), "opening parity")
            else:
                agent._FEAT.observe(public)
            sim.step(*raw["actions"][t])
        require(sim.get_state() == FastCarbon(cfg).set_state(game["states"]["297"]).get_state(), "historical state drift")
        own=clean(agent.agent(sim.obs(seat),cfg))
        require(own==clean(raw["actions"][297][seat]), "native t297 differs")
        before=sim.get_state()
        past_features=copy.deepcopy(agent._FEAT)
        arms=[]
        for name, proposed in (("v1",own),("candidate",row["candidate_own"])):
            sim.set_state(before)
            joint=list(raw["actions"][297])
            joint[seat]=proposed
            first=twin_step(sim,joint)
            require(first["status"]==["ACTIVE","ACTIVE"],"unexpected first-step extinction")
            agent._FEAT=copy.deepcopy(past_features)
            final_own=clean(agent.agent(sim.obs(seat),cfg))
            if name=="v1":
                require(final_own==clean(raw["actions"][298][seat]),"native t298 differs on factual arm")
            principal=guaranteed_delivery(sim.obs(seat),final_own,cfg)
            alt_state=sim.get_state()
            proxy_outcomes=[]
            for variant in range(3):
                sim.set_state(alt_state)
                opp=1-seat
                last_enemy=orders(sim.obs(opp),{297:game["history"]["297"][opp]},cfg,variant)
                joint=[None,None];joint[seat]=final_own;joint[opp]=last_enemy
                result=twin_step(sim,joint)
                require(result["status"]==["DONE","DONE"],"not actual terminal")
                proxy_outcomes.append({"last_enemy":last_enemy,"terminal":result["score"],"cash":result["cash"],
                                       "flow":result["flow"]})
            arms.append({"arm":name,"t298_cash_before_actions":first["cash"],"v1_last_commands":final_own,
                         "furnished_to_own_home":principal,"scenarios":proxy_outcomes})
        vs=[]
        for l,r in zip(arms[0]["scenarios"],arms[1]["scenarios"]):
            a,b=l["terminal"],r["terminal"]
            vs.append(round((b[seat]-a[seat])-(b[1-seat]-a[1-seat]),2))
        entry.update(arms=arms,paired_margin_delta_by_enemy=vs)
        initial_cash_diff=arms[1]["t298_cash_before_actions"][seat]-arms[0]["t298_cash_before_actions"][seat]
        entry["delta_own_cash_before_last_plus_own_committed_principal"]=round(initial_cash_diff+
                  sum(x["cargo"] for x in arms[1]["furnished_to_own_home"])-
                  sum(x["cargo"] for x in arms[0]["furnished_to_own_home"]),6)
        results.append(entry)
        print(seed,seat,vs,entry["delta_own_cash_before_last_plus_own_committed_principal"],flush=True)
    totals={}
    for split in ("diagnostic","evaluation"):
        selected=[r for r in results if r["split"]==split]
        maps=defaultdict(list)
        for r in selected:
            maps[r["terrain"]].append(r)
        require(all(len(one)==2 for one in maps.values()),"incomplete paired map")
        totals[split]={"games":len(selected),"maps":len(maps),"interventions":sum(bool(x["edits"]) for x in selected),
                       "equal_map_margin_changes":[statistics.mean(statistics.mean(r["paired_margin_delta_by_enemy"][k] for r in one) for one in maps.values()) for k in range(3)]}
    write(HERE/"v1_last_reply.json.gz",dict(original_pack_sha256=sha(pack),protocol_sha256=sha(HERE/"protocol.json"),
          offered_actions_sha256=sha(HERE/"root_audit.json.gz"),roots=results,summary=totals))
    print(totals)


if __name__=="__main__":
    run()

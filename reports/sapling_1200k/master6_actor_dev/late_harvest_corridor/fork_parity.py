"""Replay held-out dynamic interventions with the actual packaged policy on each new public frame."""
from collections import Counter
from copy import deepcopy
from pathlib import Path

import data
import evaluate as bench
import package_parity as parity
import policy

HERE=Path(__file__).resolve().parent
same=parity.same
clean=parity.clean


def one(pack, game, first, bindings):
    seed,s=game["seed"],game["seat"];opp=1-s;t=first["t"];key=(seed,s,t)
    file=HERE/"branches"/f"{seed}_{s}.json.gz";record=data.read(file)
    same(record["root"],first,"wrong first intervention")
    same(record["bindings"],bindings,"upstream input versions")
    case=record["scenarios"][0];arm=case["arms"]["dynamic"]
    data.must(case["variant"]==0 and arm["variant"]==0 and arm["role"]=="dynamic","required first public opponent")
    src=data.ROOT/game["path"]
    data.must(data.sha(src)==game["raw_sha256"],"source game")
    raw=data.read(src);truth=data.audit_root(key)
    cfg={k:v for k,v in truth["config"].items() if k!="randomSeed"}
    sim=data.reference().FastCarbon(cfg).reset(full_carbon=raw["initial_full_carbon"])
    hist=[{},{}]; opening_and_prelaunch=0
    for i in range(t):
        visible=[sim.obs(0),sim.obs(1)]
        if i==0 or i>=287:
            base=clean(pack.agent(visible[s],cfg));opening_and_prelaunch+=1
            same(base,clean(raw["actions"][i][s]),"candidate should not have launched yet")
            data.must(pack._HARVEST_PLAN is None and not pack._HARVEST_ISSUED,"wrong pre-launch mission")
        else:
            pack._FEAT.observe(visible[s]);pack._harvest_remember(visible[s]);pack._HARVEST_LAST=(s,i)
        if i>=t-12:
            for side in (0,1):hist[side][i]=visible[side]
        sim.step(*raw["actions"][i])
    registered=data.reference().FastCarbon(cfg).set_state(truth["state"])
    same(sim.get_state(),registered.get_state(),"root state")
    for side in (0,1): same(sim.obs(side),registered.obs(side),"numeric public state after restoration")
    data.must(bench.digest(sim.obs(s))==first["public_sha256"],"source public prefix drift")
    sim=registered
    stored_enemy=clean(truth["joint"][opp]); del truth
    target=first["meta"]["target_worker"];plan=None;issued=False;rows=[]
    counters=Counter();fates=[];last=None
    for tick in arm["steps"]:
        i=sim.step_count
        v=[sim.obs(0),sim.obs(1)]
        data.must(i==tick["t"],"wrong step")
        same([bench.digest(o) for o in v],tick["public_hash_by_seat"],"live seat observation hashes")
        own=clean(pack.agent(v[s],cfg)); stamped,seat,base,meta=pack._HARVEST_TRACE
        data.must(stamped==i and seat==s,"wrong package decision")
        same(clean(base),clean(tick["v1"]),"real v1 head vs reference v1")
        same(own,clean(tick["joint"][s]),"actual new-state friendly action")
        independent,plan_next,expected=policy.choose(v[s],hist[s],base,cfg,deepcopy(plan),allow_start=not issued)
        same(clean(independent),own,"independent policy emission");same(meta,expected,"independent metadata")
        same(plan_next,pack._HARVEST_PLAN,"independent new-state mission")
        if tick["meta"]["status"]=="v1":
            # The offline runner marks fallback itself rather than calling choose after completion.
            data.must(issued and plan is None and plan_next is None and meta=={
                "status":"fallback","stage":"out_of_window","target_worker":None}
                and tick["meta"]=={"status":"v1","stage":"no_active_mission","target_worker":None}
                and own==clean(base),"unexpected or semantic fallback mismatch")
            counters["inactive_telemetry_alias"]+=1
        else:
            same(meta,tick["meta"],"literal active branch telemetry")
            counters["active_meta_exact"]+=1
        same(plan_next,tick["plan_next"],"branch retained mission")
        issued|=meta["status"]=="start";plan=plan_next
        data.must(pack._HARVEST_ISSUED==issued,"wrong mission latch")
        enemy=stored_enemy if i==t else clean(bench.rival.orders(v[opp],hist[opp],cfg,0))
        same(enemy,clean(tick["joint"][opp]),"recomputed public opponent")
        joint=[None,None];joint[s]=own;joint[opp]=enemy
        pre=v[s]["players"][s][2].get(target)
        last=data.reference().twin_step(sim,joint)
        same(last,tick["cash"],"whole authoritative cash event")
        after=sim.obs(s)["players"][s][2].get(target)
        deposits=[(kind,round(float(value),5)) for kind,paid,value,*details in last["events"]
                  if kind=="own_delivery" and paid==s and details and details[0]==target]
        fates.extend((i,x[1]) for x in deposits)
        counters[meta["status"]]+=1
        rows.append(dict(t=i,uid=target,order=own.get(target,"NONE"),status=meta["status"],stage=meta["stage"],
                         reference_status=tick["meta"]["status"],before=pre,after=after,own_receipts=deposits,
                         opponent_source="simultaneous_archive_root" if i==t else "public_proxy_0",
                         focal_observation_sha256=bench.digest(v[s])))
        for side in (0,1):hist[side][i]=v[side]
    data.must(last["status"]==["DONE","DONE"],"incomplete branch")
    same(last["score"],arm["official_score"],"terminal scores")
    # Attributions below are the frozen paired-account reconstruction, not a new causal decomposition.
    return {"seed":seed,"seat":s,"root_t":t,"source_match":data.sha(src),"upstream_branch_sha256":data.sha(file),
            "prelaunch_neural_base_calls":opening_and_prelaunch,"active":dict(counters),"rows":rows,
            "actual_focal_deliveries":fates,"reconciled_terminal":last["score"],"tagged":arm["tagged_cargo"]}


def main():
    pack=parity.load_candidate();scan=data.read(HERE/"launch_scan.json.gz")
    bindings=bench.inputs()[0];same(bindings,scan["bindings"],"scan and runtime source binding")
    # Avoid dependence on results when selecting the eight unique boards/seats.
    games=[g for g in data.read(HERE/"source_registry.json")["games"] if g["split"]=="eval"]
    launch={(r["seed"],r["seat"]):r for r in scan["earliest"] if r["split"]=="eval"}
    data.must(len(games)==len(launch)==8 and len({g["terrain_sha256"] for g in games})==4,"held census mismatch")
    allrows=[];aggregate=Counter()
    for g in games:
        z=one(pack,g,launch[(g["seed"],g["seat"])],bindings);allrows.append(z);aggregate.update(z["active"])
        print(g["seed"],g["seat"],z["active"],flush=True)
    data.reference().write(HERE/"fork_parity.json.gz",{"scope":"8 held development branches with public opponent 0 after shared authentic simultaneous first command, actual live candidate on every own new observation; no original Master3s2 policy calls",
             "candidate_sha256":parity.PACK,"evaluation_sha256":data.sha(HERE/"evaluation.json"),
             "locked_scan_sha256":data.sha(HERE/"launch_scan.json.gz"),"source_sha256":data.sha(Path(__file__)),
             "aggregate":dict(aggregate),"real_branch_decisions":sum(len(r["rows"]) for r in allrows),
             "games":allrows,"input_bindings":bindings})


if __name__=="__main__":main()

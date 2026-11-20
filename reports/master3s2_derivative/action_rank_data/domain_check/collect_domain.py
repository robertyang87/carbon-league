"""Restore signed V2 deployment histories, capture two native H8 orders and seal their returns."""
import contextlib
import gzip
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import zipfile

HERE=Path(__file__).resolve().parent
D=HERE.parents[1]
sys.path.insert(0,str(HERE.parent))
import collect as paired
from collect import branches, freight, np, publish, save_json, sha, token
physical=freight.load("domain_cargo",D/"terminal_value_pairs/physical.py")
live=freight.load("domain_environment",freight.STUDY/"cargo_tree_action_learning_dev/ladder_live/live.py")
plan=json.loads((HERE/"protocol.json").read_bytes())
REG_SHA="e3fdfea9acaea17a25800202072e769760f07aa02fd3514d95e95567abc920a7"


def restore(ticket):
    s,t=ticket["seat"],ticket["step"]
    path=D/"cold_start/development/hosted"/ticket["path"]
    freight.check(sha(path.read_bytes())==ticket["archive"],"Signed archive digest")
    prefix=ticket["member"]+"/"
    with zipfile.ZipFile(path) as z:
        reg=z.read(prefix+"registration.json")
        turns=[json.loads(x) for x in z.read(prefix+"turns.jsonl").splitlines()[:t]]
        frames=[json.loads(x) for x in gzip.decompress(z.read(prefix+"observations.jsonl.gz")).splitlines()[:t+1]]
    with tempfile.TemporaryDirectory() as work:
        work=Path(work)
        p=work/"registration.json";p.write_bytes(reg)
        live.configure(p,work/"unused",work/"lock")
        env,cfg=live.official()
    sim=freight.econ.FastCarbon(cfg).reset(seed=ticket["seed"])
    sim.cash=freight.econ.CashEvents(sim.cash)
    freight.check(sha(bytes(env.states[0].observation.full_carbon))==ticket["terrain"]==sha(bytes(map(int,sim.carbon))),"Starting terrain")
    paths=[freight.ROOT/"data/master3s2_source.py"]*2
    paths[s]=D/"cold_start/candidate.py"
    freight.check(sha(paths[s].read_bytes())==plan["parent"] and sha(paths[1-s].read_bytes())==plan["opponent"],"Native identity")
    mods=[freight.load(f"domain_native_{s}_{k}",p) for k,p in enumerate(paths)]
    freight.check(all(m._TEMP==1 and m._SEED is None and m._S_ABORT==.58 and m._RX and m._SEARCH_OK for m in mods),"Native public path")
    orders=[[c["response"]["action"] for c in turn["calls"]] for turn in turns]
    freight.check(token([a[s] for a in orders])==ticket["pre_root_orders_sha256"],"Registered candidate precedents")
    bank=[5.,5.];public_hashes=[];costs=[]
    for i,frame in enumerate(frames):
        obs=[freight.public(env,k,bank) for k in (0,1)]
        hashes=[freight.digest(o) for o in obs]
        freight.check(hashes==frame["public_observation_sha256"] and obs==frame["observations"],f"Historic wire request {i}")
        freight.check(sim.cash==[p[0] for p in obs[0]["players"]],f"Cash at {i}")
        public_hashes.append(hashes)
        if i==t:break
        freight.check(hashes==turns[i]["public_observation_sha256"] and all(c["status"]=="ok" for c in turns[i]["calls"]),"Issued root binding")
        for k,m in enumerate(mods):
            players=obs[k]["players"];q=len(players[k][1])+len(players[k][2])
            if i<3:
                _,clock=freight.measured(m,obs[k],cfg);costs.append(dict(step=i,seat=k,clock=clock))
                if i==0:
                    m._ST["rng"]=np.random.RandomState((ticket["policy_rng"]+37*k)%2**32)
                    m._ST["rng"].random_sample(q)
            else:
                m._ST["tr"].step(players,i,obs[k]["carbon"])
                m._ST["rng"].random_sample(q)
            m._ST["tr"].set_actions(players,orders[i][k])
            m._ST["prev"]=players
        paired.advance(env,sim,orders[i])
        bank=turns[i]["overage_remaining_seconds"]
    freight.check(hashes==ticket["public_sha256"] and len(public_hashes)==t+1,"Root physical observation")
    provenance=dict(request_digests_by_seat=public_hashes,registered_sha256=REG_SHA,orders_prefix_sha256=token(orders),
                    historic_remaining_overage=bank,public_configuration=cfg,calibration=costs)
    return (mods,cfg,env,sim,obs,provenance)


def run(ticket):
    seat=ticket["seat"];prefix=f"seat{seat}";sealed=HERE/"sealed";sealed.mkdir(exist_ok=True)
    fd=HERE/(prefix+"_status.json")
    if fd.exists():
        prior=json.loads(fd.read_bytes())
        freight.check(prior["protocol_sha256"]==REG_SHA and prior["ticket"]==ticket,"Foreign previous result")
        return prior
    rp=HERE/(prefix+"_root.json.gz")
    freight.check(not rp.exists(),"Existing unfinished capture; preserve original trial")
    wall=time.perf_counter();cpu=time.process_time()
    restored=restore(ticket)
    mods,cfg,env,sim,obs,provenance=restored
    learner=mods[seat]
    stamp=freight.original_state.h([m._ST for m in mods]);rng=freight.original_state.h(learner._ST["rng"])
    cost=list(learner._S_COST)
    own,root,arrays=paired.rank_call(learner,obs[seat],cfg)
    root.update(ticket=ticket,diagnostic_protocol=REG_SHA,public_root=obs[seat],prefix_provenance=provenance,
                time_calibrated_cost_before_root=cost,native_pre_root_sha256=stamp,
                implementation_sha256=sha(Path(paired.__file__).read_bytes()),host_cpu=sorted(os.sched_getaffinity(0)))
    info=dict(protocol_sha256=REG_SHA,ticket=ticket,status=root["status"])
    if arrays:
        for arm in root["arms"]:
            # Match the production scorer's original return and matched roll set.
            for leaf in arm["leaves"]:
                leaf["physical_features"]=(np.zeros(2) if leaf["exact_terminal"] else physical.public_features(leaf["players"],leaf["leaf_step"],seat)).tolist()
        stream=io.BytesIO();np.savez_compressed(stream,**arrays)
        info["leaf_sha256"]=publish(HERE/(prefix+"_leaf.npz"),stream.getvalue())
    info["root_sha256"]=save_json(rp,root)
    if arrays:
        picks={a["name"]:a["order"] for a in root["arms"]}
        picks.update(menu_sha256=token(root["menu"]),cost=cost)
        # The fitter gets only signed public features and the hashes of unopened returns.
        with contextlib.redirect_stdout(io.StringIO()):
            outcome=branches.continue_pair(ticket,picks,ticket["arm_order"],restored,own,stamp,rng,lambda n,a:save_json(sealed/n,a))
        for a in outcome["arms"].values():
            a["terminal_value"]=learner._terminal(a["margin"])
        outcome["target_first_minus_second"]=outcome["arms"]["first"]["terminal_value"]-outcome["arms"]["second"]["terminal_value"]
        info["sealed_pair_sha256"]=save_json(sealed/(prefix+"_pair.json"),outcome)
        info["status"]="complete"
    else:
        info["reason"]=root["reason"]
    info.update(wall=time.perf_counter()-wall,cpu=time.process_time()-cpu)
    save_json(fd,info)
    print(prefix,info["status"],flush=True)
    return info


if __name__=="__main__":
    os.environ["CARBON_DEBUG"]="1"
    freight.check(os.sched_getaffinity(0)=={3},"Use the registered one CPU, number 3")
    freight.check(sha((HERE/"protocol.json").read_bytes())==REG_SHA and
                  sha((D/"terminal_value_pairs/physical.py").read_bytes())==plan["physical_features"] and
                  sha((HERE.parent/"protocol.json").read_bytes())==plan["paired_acquisition"],"Frozen method")
    for t in plan["tickets"]:run(t)

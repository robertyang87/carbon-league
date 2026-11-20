"""Replay the archived source prefix, reconstruct native state and run two registered TRAIN suffixes."""
import argparse
import copy
import gzip
import json
import os
from pathlib import Path
import subprocess
import tempfile
import traceback

import select_roots as selection

prior=selection.common
HERE=selection.HERE
PROTOCOL=HERE/"protocol.json"
ROOTS=HERE/"roots.json"


def load(path):
    raw=path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix==".gz" else raw)


def native_targets(obs, orders):
    return {u:selection.destination(w[0],orders.get(u,"NONE"))
            for u,w in obs["players"][obs["player"]][2].items()}


def prepare(source, ticket, cfg, actors,enemy):
    seat=ticket["seat"]
    env=prior.collect.begin(ticket["seed"],source["configuration"])
    prior.need(prior.sha(bytes(env.states[0].observation.full_carbon))==ticket["terrain"],"Incorrect terrain")
    bank=[5.,5.]
    warm=[]
    expected={}
    root=ticket["root_step"]
    for i in range(root+1):
        tape=source["rows"][i]
        mine,theirs=(prior.public(env,k,bank) for k in (seat,1-seat))
        prior.need(prior.digest(mine)==prior.digest(tape["public"]) and
                   prior.digest(theirs)==tape["enemy_public_sha256"] and prior.board_id(env)==tape["board_before"],
                   "Archived public input or physical board mismatch")
        native=enemy.mod
        if i:
            prior.need(native._ST["tr"].last_dest==expected and
                       native._S_COST==source["rows"][i-1]["clocks"][1-seat]["cost"],"Unrestored incoming intent or cost")
        guessed,shadow_clock=prior.call(enemy,theirs,cfg,examiner=True)
        issued=tape["orders"][1-seat]
        historical_clock=tape["clocks"][1-seat]
        prior.need(native._ST["tr"].last_dest==native_targets(theirs,guessed),"Native action hook incomplete")
        native._ST["tr"].set_actions(theirs["players"],issued)
        expected=native_targets(theirs,issued)
        native._S_COST[:]=historical_clock["cost"]
        prior.need(native._ST["tr"].last_dest==expected and native._ST["prev"]==theirs["players"],"History repair failed")
        warm.append({"step":i,"historical_order":issued,"shadow_order":guessed,
                     "historical":historical_clock,"shadow":shadow_clock,"corrected":guessed!=issued,
                     "restored_native_cost":list(native._S_COST),"native_public_sha256":prior.digest(theirs)})
        if i==root:
            prior.need(ticket["root_public_sha256"]==prior.digest(mine),"Root differs from prospective selection")
            break
        for player in actors.values():
            if i==0:
                prior.call(player,mine,cfg)
            else:
                prior.observe_only(player,mine)
        env.step(tape["orders"])
        bank=tape["bank_after"]
    students={n:copy.deepcopy(prior.actor_state(p)) for n,p in actors.items()}
    prior.need(len({prior.original_state.h(v[0]) for v in students.values()})==1,"Mismatched public feature history")
    return env,warm,{"ref":prior.official_state.take(env,bank),"native":prior.original_state.take(enemy.mod),
                     "actors":students,"their_reply":(issued,historical_clock),"board":prior.board_id(env),
                     "incoming_bank":bank}


def execute_branch(env,start,ticket,cfg,actors,enemy,snapshot,arm):
    bank=prior.official_state.restore(env,snapshot["ref"])
    prior.original_state.restore(enemy.mod,snapshot["native"])
    for k,p in actors.items():
        prior.restore_actor(p,snapshot["actors"][k])
        prior.need(prior.original_state.h(prior.actor_state(p))==prior.original_state.h(snapshot["actors"][k]),
                   "Student snapshot drift")
    prior.need(prior.board_id(env)==snapshot["board"] and bank==snapshot["incoming_bank"],"Paired root drift")
    rows=[]
    seat=ticket["seat"]
    while not env.done:
        ours,theirs=(prior.public(env,k,bank) for k in (seat,1-seat))
        t=ours["step"]
        rival=snapshot["their_reply"] if t==start else prior.call(enemy,theirs,cfg,examiner=True)
        if t<start+8:
            active=arm
            actions,state_clock=prior.call(actors[arm],ours,cfg)
            prior.observe_only(actors["bc"],ours)
        else:
            active="bc"
            actions,state_clock=prior.call(actors["bc"],ours,cfg)
        if t==start:
            prior.need(actions==ticket["orders"][arm] and prior.digest(ours)==ticket["root_public_sha256"],"Arm differs from registered root action")
        bank,row=prior.settle(env,bank,seat,ours,theirs,(actions,state_clock),rival)
        row["active_policy"]=active
        row["shared_historical_opponent_command"]=(t==start)
        rows.append(row)
    return {"rows":rows,"terminal":prior.terminal(env,seat),"host_end":prior.platform_state()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipts",required=True,type=Path)
    parser.add_argument("--seed",required=True,type=int)
    parser.add_argument("--out",required=True,type=Path)
    args=parser.parse_args()
    config=load(PROTOCOL)
    spec=load(ROOTS)
    prior.need(config["roots_file_sha256"]==prior.sha(ROOTS.read_bytes()) and
               config["runner_sha256"]==prior.sha(Path(__file__).read_bytes()) and
               config["selector_sha256"]==prior.sha(Path(selection.__file__).read_bytes()),"Frozen source drift")
    ticket=next(t for t in spec["tickets"] if t["seed"]==args.seed)
    prior.need(ticket["selected"],"No preselected root on this map")
    missing=set(os.sched_getaffinity(0))-set(config["allowed_local_cpu"])
    prior.need(not missing and os.sched_getaffinity(0),"Worker must run with registered CPU affinity")
    for k,v in selection.HASH.items():
        prior.need(v==config["candidate_sha256"][k]==prior.sha((HERE.parent/(k+".py")).read_bytes()),"Wrong candidate")
    sdir=args.receipts/"fit"/f'{args.seed}_{ticket["seat"]}'
    source_file=sdir/"source.json.gz"
    src_sha=prior.sha(source_file.read_bytes())
    prior.need(src_sha==ticket["source_sha256"]==load(sdir/"verified.json")["source_sha256"],"Wrong source")
    source=load(source_file)
    prior.need(source["actors"]["bc"]==prior.SIGNED["bc"] and source["actors"]["original"]==prior.SIGNED["original"]
               and source["terrain"]==ticket["terrain"],"Original source identity")
    metadata={"seed":args.seed,"seat":ticket["seat"],"terrain":ticket["terrain"],"root":ticket["root_step"],
              "source_sha256":src_sha,"root_public_sha256":ticket["root_public_sha256"],"protocol_sha256":prior.sha(PROTOCOL.read_bytes()),
              "runner_sha256":config["runner_sha256"],"host_start":prior.platform_state()}
    args.out.mkdir(parents=True,exist_ok=True)
    with (args.out/"claim.json").open("x") as f:
        json.dump(metadata,f)
    try:
        os.environ["CARBON_DEBUG"]="1"
        with tempfile.TemporaryDirectory() as temp:
            original=Path(temp)/"master3s2.py"
            original.write_bytes(subprocess.check_output(["git","-C",str(prior.REPO),"show","HEAD:sources/master3s2/master3s2.py"]))
            prior.need(prior.sha(original.read_bytes())==prior.SIGNED["original"] and prior.sha(prior.BC.read_bytes())==prior.SIGNED["bc"],"Frozen referees drifted")
            actors={k:prior.collect.Player(HERE.parent/(k+".py")) for k in selection.HASH}
            actors["bc"]=prior.collect.Player(prior.BC)
            enemy=prior.collect.Player(original)
            prior.need(enemy.mod._TEMP==1 and enemy.mod._SEED is None and enemy.mod._S_ABORT==.58 and enemy.mod._SEARCH_OK,"Original disabled")
            cfg={k:v for k,v in source["configuration"].items() if k!="randomSeed"}
            env,warm,snapshot=prepare(source,ticket,cfg,actors,enemy)
            metadata.update(root_native_counterfactual_sha256=snapshot["native"]["expected"],
                            root_native_cost=list(enemy.mod._S_COST),root_board_sha256=snapshot["board"],
                            root_actor_sha256={k:prior.original_state.h(v) for k,v in snapshot["actors"].items()})
            prior.save(args.out/"prefix.json.gz",{**metadata,"rows":warm})
            result={}
            for arm in config["branch_order"]:
                r=execute_branch(env,ticket["root_step"],ticket,cfg,actors,enemy,snapshot,arm)
                r["inprocess_time_limits"]=prior.local_time_limits(source["rows"][:ticket["root_step"]]+r["rows"])
                path=args.out/(arm+".json.gz")
                prior.save(path,{**metadata,"arm":arm,**r})
                result[arm]={"sha256":prior.sha(path.read_bytes()),"terminal":r["terminal"],
                             "steps":len(r["rows"]),"inprocess_time_limits":r["inprocess_time_limits"]}
                print(arm,result[arm],flush=True)
            w,u=(result[k]["terminal"] for k in ("weighted","uniform"))
            prior.save(args.out/"summary.json",{**metadata,"arms":result,"weighted_minus_uniform":w["margin"]-u["margin"],
                        "win_delta":w["win"]-u["win"],"source_terminal":source["terminal"],
                        "prefix_sha256":prior.sha((args.out/"prefix.json.gz").read_bytes())})
    except BaseException:
        prior.save(args.out/"failure.json",{"traceback":traceback.format_exc(),"seed":args.seed})
        raise


if __name__=="__main__":
    main()

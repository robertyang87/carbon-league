"""Eight real terminal callbacks of the unchanged original examiner, after the intake_08 court release."""
import argparse
import copy
import datetime
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from study import HERE, ROOT, REPLAY, FastCarbon, protocol, read, require, sha, twin_step, write
from v1_last_reply import clean
from policy import choose

FROZEN = "851a0903ab1aa0898ecaf6799528981872358fc14856905742c8a913deef3d04"
SUPPLEMENT = HERE.parent / "tree_response_training/supplement_01"
GRANT = SUPPLEMENT / "endgame_queue_grant.json"
COMPLETION = SUPPLEMENT / "completion.json"


def canonical(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def checked():
    path=HERE/"native_protocol.json"
    require(sha(path)==FROZEN, "changed response protocol")
    spec=read(path)
    for name, digest in (("root_audit.json.gz","raw_evidence_sha256"),("policy.py","policy_sha256"),
                         ("candidate.py","candidate_sha256"),("replay_roots.json.gz","sources_sha256"),
                         ("v1_last_reply.json.gz","parent_reply_sha256"),("native_streams.json.gz","shared_streams_sha256")):
        require(sha(HERE/name)==spec[digest], f"changed binding: {name}")
    cost=ROOT/spec["cost_history"]
    require(sha(cost)==spec["cost_history_sha256"] and read(cost)["warm_cost"]==spec["cost_before_each_arm"],
            "native timing calibration changed")
    master=ROOT/"data/master3s2_source.py"
    require(sha(master)==spec["source_examiner"],"wrong source on disk")
    published=subprocess.check_output(["git","show","HEAD:sources/master3s2/master3s2.py"],cwd=ROOT)
    require(hashlib.sha256(published).hexdigest()==spec["source_examiner"],"wrong canonical examiner")
    return spec,master


def shared_rng(row):
    rng=np.random.RandomState()
    rng.set_state((row["algorithm"],np.array(row["keys"],dtype=np.uint32),row["index"],row["gaussian"],row["cached"]))
    return rng


def plain(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    return value


def permission():
    if not GRANT.exists() or not COMPLETION.exists():
        return False
    data=read(GRANT)
    require(data["cpu"]==0 and data["permitted_native_calls"]==8 and data["completion_sha256"]==sha(COMPLETION),
            "unsigned or unrelated native court grant")
    return True


def other_workers():
    matches=[]
    for p in Path("/proc").glob("[0-9]*"):
        try:
            argv=(p/"cmdline").read_bytes().split(b"\0")
            state=(p/"stat").read_text().rsplit(")",1)[1].strip().split()[0]
        except (FileNotFoundError,ProcessLookupError,PermissionError):
            continue
        if state=="Z" or not argv or b"python" not in argv[0].lower():
            continue
        files=[os.fsdecode(a) for a in argv[1:] if a.endswith(b".py")]
        if any(Path(s).name in ("tree_pack_arena.py","pair_games.py") or
               ("live.py" in s and any(a in argv for a in (b"parent",b"alternative",b"pay",b"defer")))
               or ("native_response/run.py" in s and b"worker" in argv) for s in files):
            matches.append(int(p.name))
    return matches


def utility(rewards, p):
    gap=rewards[p]-rewards[1-p]
    wdl=1 if gap>0 else (0 if gap<0 else .5)
    return .5*wdl+.5/(1+math.exp(-gap/1000))


def score_cases(spec, path):
    os.environ["CARBON_DEBUG"]="1" # Turn silent exception fallbacks into test failures.
    module=importlib.util.spec_from_file_location("endgame_real_master3s2",path)
    master=importlib.util.module_from_spec(module)
    module.loader.exec_module(master)
    require(master._SEED is None and master._TEMP==1 and master._RX and master._SEARCH_OK and master._S_HOMEFIX,
            "full player disabled")
    master._net()
    costs=spec["cost_before_each_arm"]
    initial_stats=copy.deepcopy(master._S_STATS)
    wrapped=master._search
    calls=[]
    def count(*args,**kwargs):
        calls.append((args[0]["player"],args[0]["step"]))
        return wrapped(*args,**kwargs)
    master._search=count
    roots={(r["seed"],r["seat"]):r for r in read(HERE/"root_audit.json.gz") if r["t"]==297}
    qr={(r["seed"],r["seat"]):r for r in read(HERE/"v1_last_reply.json.gz")["roots"]}
    cache={(g["seed"],g["seat"]):g for g in read(REPLAY)["games"]}
    refs={(r["seed"],r["seat"]):r for b in protocol()["source_batches"] for r in read(ROOT/b["receipt"])["games"]}
    streams=read(HERE/"native_streams.json.gz")
    receipts=[]
    for i,item in enumerate(spec["cases"]):
        seed,seat=item["seed"],item["seat"]; enemy=1-seat; key=(seed,seat)
        g=cache[key]; record=roots[key]; cfg=dict(g["config"],randomSeed=None)
        stream=streams[i]
        require((stream["seed"],stream["seat"])==key and record["changes"]==item["edits"], "wrong registered root")
        target=HERE/f"native_{seed}_{seat}.json.gz"
        if target.exists():
            signed=read(target)
            require(signed["protocol_sha256"]==FROZEN and signed["runner_sha256"]==sha(Path(__file__)) and
                    len(signed["arms"])==2 and signed["source_sha256"]==item["source_sha256"],"stale pair")
            receipts.append(signed)
            continue
        tape=ROOT/refs[key]["raw_path"];require(sha(tape)==item["source_sha256"],"wrong transcript")
        raw=read(tape)
        sim=FastCarbon(cfg).reset(full_carbon=raw["initial_full_carbon"])
        tracker=master.BeliefTracker(master._Bel,enemy)
        digest=hashlib.sha256()
        for t, joint in enumerate(raw["actions"][:298]):
            vis=sim.obs(enemy)
            digest.update(canonical((vis,joint[enemy])).encode())
            tracker.step(vis["players"],t,vis["carbon"])
            tracker.set_actions(vis["players"],joint[enemy])
            if t<297:
                sim.step(*joint)
        state=sim.get_state()
        require(state==FastCarbon(cfg).set_state(g["states"]["297"]).get_state(), "wrong root truth")
        common=dict(tr=tracker,prev=vis["players"])
        outcomes={}
        for role in (("v1","candidate") if i%2==0 else ("candidate","v1")):
            partial = HERE / f"last_arm_{seed}_{seat}_{role}.json.gz"
            if partial.exists():
                sealed = read(partial)
                require(sealed["bindings"] == [FROZEN, item["source_sha256"], sha(Path(__file__)), role] and
                        sealed["outcome"]["search_call_count"] == 1 and sealed["outcome"]["end"]["status"] == ["DONE","DONE"],
                        "stale single-arm checkpoint")
                outcomes[role] = sealed["outcome"]
                continue
            sim.set_state(state)
            joint=list(raw["actions"][297])
            if role=="candidate":
                joint[seat]=record["candidate_own"]
            first=twin_step(sim,joint)
            require(first["status"]==["ACTIVE","ACTIVE"],"unexpected first-step termination")
            evaluation=next(a for a in qr[key]["arms"] if a["arm"]==role)
            own=evaluation["v1_last_commands"]
            if role=="candidate":
                # Prove that the final live wrapper chooses its own native-parent continuation on this branch.
                offer,_=choose(sim.obs(seat),{297:g["history"]["297"][seat]},own,cfg)
                require(clean(offer)==own,"candidate does not implement the frozen parent last action")
            master._ST.clear()
            master._ST.update(tr=copy.deepcopy(common["tr"]),prev=common["prev"],rng=shared_rng(stream))
            master._S_COST[:]=costs
            master._S_STATS.clear();master._S_STATS.update(copy.deepcopy(initial_stats))
            master._S_REC=[]
            actual_rng=master._ST["rng"].get_state()
            stream_readback={"seed":seed,"seat":seat,"algorithm":actual_rng[0],"keys":actual_rng[1].tolist(),
                             "index":actual_rng[2],"gaussian":actual_rng[3],"cached":actual_rng[4]}
            require(canonical(stream_readback)==canonical(stream) and list(master._S_COST)==costs,
                    "different RNG or measured starting cost across arms")
            cost_before=list(master._S_COST)
            own_last_observation_sha256=canonical(sim.obs(seat))
            before=len(calls);obs=sim.obs(enemy)
            t0=time.perf_counter()
            real=clean(master.agent(obs,cfg))
            wall=time.perf_counter()-t0
            require(calls[before:]==[(enemy,298)], "native last-turn search bypassed")
            groups=set(obs["players"][enemy][1])|set(obs["players"][enemy][2])
            require(set(real)<=groups,"foreign native command")
            # The focal actions in both seats were issued simultaneously from their own public streams.
            last=[None,None];last[enemy]=real;last[seat]=own
            end=twin_step(sim,last)
            require(end["status"]==["DONE","DONE"],"missing terminal")
            outcomes[role]={"first_joint":joint,"first_joint_sha256":canonical(joint),
                "last_joint":last,"last_joint_sha256":canonical(last),
                "own_root_visible_sha256":canonical(g["history"]["297"][seat]),
                "own_last_visible_sha256":own_last_observation_sha256,
                "enemy_last_visible_sha256":canonical(obs),"search_call_count":len(calls)-before,
                "search_callback_seat_step":calls[before:],"rng_before_sha256":canonical(stream_readback),
                "cost_before":cost_before,
                "native_search":copy.deepcopy(master._S_STATS),"root_candidates_reported":copy.deepcopy(master._S_REC),
                "rng_after_sha256":canonical([master._ST["rng"].get_state()[1].tolist(),master._ST["rng"].get_state()[2]]),
                "native_action_wall_s":wall,"cost_after":list(master._S_COST),"first":first,"end":end,
                "enemy_matches_original_recorded_last":real==clean(raw["actions"][298][enemy])}
            outcomes[role] = plain(outcomes[role])
            write(partial, {"bindings": [FROZEN, item["source_sha256"], sha(Path(__file__)), role],
                            "outcome": outcomes[role]})
        base=outcomes["v1"]["end"]; alt=outcomes["candidate"]["end"]
        delta=[round(a-b,2) for a,b in zip(alt["score"],base["score"])]
        original=g["original"]["score"]
        pair={"protocol_sha256":FROZEN,"runner_sha256":sha(Path(__file__)),"seed":seed,"seat":seat,
              "source_sha256":item["source_sha256"],"public_history_turns":298,"public_history_sha256":digest.hexdigest(),
              "paired_rng_state_sha256":canonical(stream),"original_scores_physical":original,"arms":outcomes,
              "native_enemy_orders_differ":outcomes["v1"]["last_joint"][enemy]!=outcomes["candidate"]["last_joint"][enemy],
              "delta_rewards_physical":delta,"delta_margin":round(delta[seat]-delta[enemy],2),
              "delta_u":utility(alt["score"],seat)-utility(base["score"],seat),
              "won_by_arm":{r:a["end"]["score"][seat]>a["end"]["score"][enemy] for r,a in outcomes.items()}}
        write(target,pair);receipts.append(pair);print(seed,seat,pair["delta_margin"],flush=True)
    return receipts


def main(wait):
    spec,path=checked()
    if not wait:
        require(permission(),"waiting for explicit court grant")
    while not permission() or other_workers():
        time.sleep(10)
    os.sched_setaffinity(0,{0})
    with (HERE/".native_slot.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        done=HERE/"native_result.json"
        if done.exists():
            require(read(done)["protocol_sha256"]==FROZEN and len(read(done)["pairs"])==4,
                    "invalid existing completion")
            return
        require(permission() and not other_workers(), "native court has active original games")
        started=time.monotonic()
        rows=score_cases(spec,path)
        report={"scope":"four previously registered development interventions, eight actual last-turn responses",
            "protocol_sha256":FROZEN,"runner_sha256":sha(Path(__file__)),"grant_sha256":sha(GRANT),
            "completion_sha256":sha(COMPLETION),"candidate_sha256":spec["candidate_sha256"],
            "examiner_sha256":spec["source_examiner"],"source_params":{"temperature":1,"seed":None,"search":True,
            "cost":spec["cost_before_each_arm"],"common_rng":True},"completed_at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "wall_seconds":time.monotonic()-started,"cpu_affinity":sorted(os.sched_getaffinity(0)),
            "native_calls":sum(a["search_call_count"] for r in rows for a in r["arms"].values()),
            "pairs":[{k:r[k] for k in ("seed","seat","delta_margin","delta_u","delta_rewards_physical","native_enemy_orders_differ","won_by_arm")} for r in rows],
            "mean_delta_all_16":sum(r["delta_margin"] for r in rows)/16,
            "mean_delta_u_all_16":sum(r["delta_u"] for r in rows)/16,
            "pair_sha256":{p.name:sha(p) for p in HERE.glob("native_6*.json.gz")}}
        require(report["native_calls"]==8 and len(report["pair_sha256"])==4,"incomplete native result")
        write(done,report)

if __name__=="__main__":
    cli=argparse.ArgumentParser()
    cli.add_argument("--wait",action="store_true",help="Wait for the signed court handoff")
    cli.add_argument("--check",action="store_true",help="Validate plan/inputs without loading the original agent")
    args=cli.parse_args()
    if args.check:
        p,_=checked();print("sealed",len(p["cases"]),FROZEN,"grant",permission())
    else:
        with (HERE / ".native_supervise.lock").open("a") as supervisor:
            try:
                fcntl.flock(supervisor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise SystemExit("native terminal study already supervised")
            main(args.wait)

"""Measure the frozen single file on an archived trajectory in two isolated platform sandboxes."""
import argparse
import asyncio
from contextlib import AsyncExitStack
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
HOST=HERE.parents[1]/"ladder_live"
sys.path.insert(0,str(HOST))
import live

PLAN=HERE/"protocol.json"
PLAN_SHA="0845be7bb6b202d17aee3db43c13f7cbf3d06df19f8e9041187f99e72e34d141"
KEYS={"player","step","players","trees","carbon","remainingOverageTime"}


def read_gzip(key):
    return json.loads(gzip.decompress((live.ROOT/live.PLAN["files"][key]["path"]).read_bytes()))


def check_inputs():
    if not __debug__ or live.sha(PLAN)!=PLAN_SHA:
        raise ValueError("registration or assert mode changed")
    live.configure(PLAN,HERE/".unused-claim",HERE/".unused-lock")
    # Both containers load the candidate single file in the regular candidate worker mode.
    live.ROLES=("candidate","candidate")
    for f in live.PLAN["files"].values():
        p=live.ROOT/f["path"]
        assert live.sha(p)==f["sha256"] and p.stat().st_size==f["bytes"]
    assert live.PLAN["files"]["candidate"]["bytes"]<=10_000_000
    raw=read_gzip("replay")
    ref=read_gzip("observed_actions")["student"]
    assert raw["seed"]==live.PLAN["seed"] and raw["initial_terrain_u8_sha256"]==live.PLAN["physical_map_sha256"]
    assert raw["game_configuration"]==read_gzip("official_config")["game_configuration"] and not raw["errors"]
    assert len(raw["actions"])==raw["steps"]==live.PLAN["expected_transitions"]==299
    assert len(ref)==2 and all(len(r["timing"]["rows"])==299 and r["timing"]["ready"]["pack_sha256"]==live.PLAN["files"]["candidate"]["sha256"] for r in ref)
    env,_=live.official()
    inputs=hashlib.sha256()
    for t,joint in enumerate(raw["actions"]):
        assert not env.done and env.states[0].observation.step==t
        for s in (0,1):
            obs=live.observe(env,s,5.)
            assert set(obs)==KEYS and obs["player"]==s and obs["step"]==t
            inputs.update(live.encoded(obs))
        env.step(joint)
    assert env.done and [state.reward for state in env.states]==raw["official_score_by_seat"]
    return raw,ref,inputs.hexdigest()


async def replay(args,result,rows,raw,ref):
    cpus=sorted(os.sched_getaffinity(0))[:2]
    if len(cpus)!=2:raise ValueError("two schedulable CPUs required")
    image_id=subprocess.check_output(["docker","image","inspect","--format","{{.Id}}",args.image],text=True).strip()
    assert image_id.startswith("sha256:")
    result.update(image_id=image_id,cold_loads_issued=0,command_match_counts=[0,0],reference_commands_by_seat=[299,299])
    async with AsyncExitStack() as stack:
        workers=[await stack.enter_async_context(live.sandbox(image_id,"candidate",",".join(map(str,cpus)))) for _ in (0,1)]
        procs,probes,settings=zip(*workers)
        issues=live.eligibility(probes,settings)
        if any(x["image"]!=image_id or not x["read_only_root"] or x["nano_cpus"]!=1_500_000_000 or x["memory"]!=536870912 for x in settings):
            issues.append("Docker inspection disagrees with frozen image or limits")
        result.update(probes=probes,containers=settings,blockers=issues)
        if issues:return "blocked_before_candidate_load"
        ready=[]
        for p in procs:
            result["cold_loads_issued"]+=1
            call=await live.exchange(p,{"load":True},20)
            ready.append(call)
        result["initialization"]=ready
        for call in ready:
            if call["status"]!="ok":return "cold_load_failure"
            if not call["response"].get("ready") or call["response"].get("sha256")!=live.PLAN["files"]["candidate"]["sha256"]:
                return "cold_identity_mismatch"
        env,cfg=live.official()
        bank=[5.,5.]
        slow=[0,0]
        with (args.out/"calls.jsonl").open("x") as log:
            for t,joint in enumerate(raw["actions"]):
                obs=[live.observe(env,s,bank[s]) for s in (0,1)]
                assert all(set(x)==KEYS and x["step"]==t and x["player"]==s for s,x in enumerate(obs))
                calls=[None,None]
                for seat in (t%2,1-t%2):
                    call=await live.exchange(procs[seat],{"observation":obs[seat],"configuration":cfg},live.allowance(slow[seat],bank[seat]))
                    calls[seat]=call
                    live.check_action(call,obs[seat])
                    if call["status"]=="ok":
                        result["command_match_counts"][seat]+=int(call["response"]["action"]==ref[seat]["timing"]["rows"][t]["action"])
                        if call["elapsed"]>1:
                            slow[seat]+=1
                            bank[seat]=max(0.,bank[seat]+1.-call["elapsed"])
                row=dict(step=t,calls=calls,public_observation_sha256=[hashlib.sha256(live.encoded(x)).hexdigest() for x in obs],
                         overage_remaining_seconds=bank.copy(),slow_steps=slow.copy())
                rows.append(row)
                log.write(live.encoded(row).decode())
                log.flush()
                if any(c["status"]!="ok" for c in calls):return "request_failure"
                env.step(joint)
        return "complete" if env.done and [s.reward for s in env.states]==raw["official_score_by_seat"] else "archive_endpoint_mismatch"


async def run(args,raw,ref,history):
    args.out.mkdir(exist_ok=True,parents=True)
    with (args.out/"attempt.json").open("x") as stream:
        stream.write(live.encoded(dict(protocol_sha256=PLAN_SHA,run_id=os.getenv("GITHUB_RUN_ID"))).decode())
    rows=[]
    result=dict(scope="limited-resource fixed replay of one historical map; no live opponent, game score or independent submission certificate",
                registration=PLAN_SHA,adapter_sha256=live.sha(Path(__file__)),pack=live.PLAN["files"]["candidate"],
                replay=live.PLAN["files"]["replay"],neutral_overage_public_history=history,run_id=os.getenv("GITHUB_RUN_ID"),
                git_commit=os.getenv("GITHUB_SHA"),seats=[0,1],expected_calls_per_seat=299)
    start=time.monotonic()
    try:
        result["status"]=await asyncio.wait_for(replay(args,result,rows,raw,ref),live.RULE["task_seconds"])
    except TimeoutError:
        result["status"]="whole_task_timeout"
    except Exception as exc:
        result.update(status="runner_error",error=repr(exc))
    result["whole_wall_seconds"]=time.monotonic()-start
    if result["whole_wall_seconds"]>live.RULE["task_seconds"]:result["status"]="whole_task_timeout"
    if (args.out/"calls.jsonl").exists():result["tape_sha256"]=live.sha(args.out/"calls.jsonl")
    result["measured_calls_per_seat"]=[len(rows),len(rows)]
    result["timing_by_seat"]=live.timing_by_seat(result,rows)
    result["fixed_replay_limits_pass"]=result["status"]=="complete" and len(rows)==299 and all(z["observed_waits_over_one_second"]<=3 and z["observed_excess_seconds"]<=5 and z["observed_wait_seconds_max"]<=3 and z["init_status"]=="ok" and z["init_seconds"]<=20 for z in result["timing_by_seat"])
    live.save(args.out/"receipt.json",result)
    return int(not result["fixed_replay_limits_pass"])


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--check",action="store_true");parser.add_argument("--out",type=Path);parser.add_argument("--image")
    args=parser.parse_args()
    raw,ref,history=check_inputs()
    if args.check:print("Public replay verified",history,flush=True)
    elif args.out and args.image:sys.exit(asyncio.run(run(args,raw,ref,history)))
    else:parser.error("use --check or --out DIR --image TAG")

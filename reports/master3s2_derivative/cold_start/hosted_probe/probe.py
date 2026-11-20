"""Measure concurrent cold loading and complete step-zero JSON exchanges on eight fresh pairs."""
import argparse
import asyncio
from contextlib import AsyncExitStack
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
DERIV = HERE.parent.parent
CARGO = ROOT / "reports/sapling_1200k/master6_actor_dev/cargo_tree_action_learning_dev"
ARENA = CARGO / "ladder_live"
sys.path.insert(0, str(ARENA))
import live

PLAN = HERE / "protocol.json"
WORKFLOW = ROOT / ".github/workflows/master3s2-cold-start-probe.yml"
PACKS = {"v1": DERIV / "opponent_policy/candidate.py", "v2": HERE.parent / "candidate.py",
         "opponent": ROOT / "sources/master3s2/master3s2.py"}
IDENTITIES = {"v1":"245b5814d1855c5449435ee6bd4a53560017a8cc7a46d08729fa9dfdc13e9654",
              "v2":"7cf5c6c91fc59b46b2a38924c4c5ab56143fb1242d082f9426303af7336ac3ed",
              "opponent":"65f268b2919ce30fe19355115a928eada032a6beaa018d801662d434c083f2b5"}


def check(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_bytes())


def entry(path):
    return dict(path=path.relative_to(ROOT).as_posix(), sha256=live.sha(path), bytes=path.stat().st_size)


def contract():
    original = read(ARENA / "protocol.json")
    inputs = {"runner":Path(__file__), "workflow":WORKFLOW, "live_runner":ARENA/"live.py",
              "worker":ARENA/"worker.py", "reference":ARENA/"protocol.json", "container":ARENA/"Dockerfile",
              "source_maps":DERIV/"opponent_policy/protocol.json", "release":HERE.parent/"release.json",
              "official_engine":ROOT/"code/env_src/carbon_challenge-master/zerosum_env/envs/carbon/carbon.py",
              "opening_config":CARGO/"direct_original/games/candidate/raw_621758346_1.json.gz",
              "official_helpers":ROOT/original["files"]["official_helpers"]["path"]}
    files = {k:entry(p) for k,p in {**inputs, **PACKS}.items()}
    check(all(files[k]["sha256"] == h for k,h in IDENTITIES.items()), "Changed examiner or derivative")
    cards = [c for c in read(inputs["source_maps"])["games"] if c["split"] == "train"]
    maps = sorted({c["terrain"] for c in cards})[:4]
    tickets = []
    for i,h in enumerate(maps):
        seat = i%2
        card, = [c for c in cards if c["terrain"] == h and c["expert_seat"] == seat]
        # The physical map identity is verified by live.official at initialization.
        live.PLAN = dict(seed=card["seed"], physical_map_sha256=h)
        env, cfg = live.official()
        obs = [live.observe(env,p,5.) for p in (0,1)]
        tickets.append(dict(seed=card["seed"], terrain=h, seat=seat, source_episode_sha256=card["sha256"],
                            cfg_sha256=hashlib.sha256(live.encoded(cfg)).hexdigest(),
                            public_sha256=[hashlib.sha256(live.encoded(o)).hexdigest() for o in obs],
                            order=["v1","v2"] if i in (0,3) else ["v2","v1"]))
    check(len(tickets) == 4 and len({t["seed"] for t in tickets}) == 4, "Expected four source training maps")
    return dict(schema="derivative-cold-start-probe/1", files=files, tickets=tickets, platform=original["platform"],
                base_image="python@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b",
                workload="One hosted machine. Fresh independent cgroup/process per role and trial, same two CPUs "
                         "for both, simultaneous two-role load like the full runner. Once both are ready, seat 0 and "
                         "then seat 1 receive the initial official public request. Stop before any transition. "
                         "Preserve initializations, individual and paired failures in all eight tickets; no retries.",
                scope="Short timing diagnostic only. First-use shape coverage and full-match ladder qualification are separate.")


def resource_paths(containers):
    paths = []
    for c in containers:
        pid = subprocess.check_output(["docker","inspect","--format","{{.State.Pid}}",c["id"]],text=True).strip()
        groups = Path("/proc",pid,"cgroup").read_text().splitlines()
        rel, = [line.removeprefix("0::") for line in groups if line.startswith("0::")]
        path = Path("/sys/fs/cgroup") / rel.lstrip("/")
        paths.append(path)
    return paths


def snapshot(paths):
    rows = []
    for path in paths:
        cpu = {k:int(v) for k,v in (line.split() for line in (path/"cpu.stat").read_text().splitlines())}
        rows.append(dict(memory_peak_bytes=int((path/"memory.peak").read_text()), cpu=cpu,
                         captured_ns=time.monotonic_ns()))
    return rows


async def pair(ticket, arm, plan, image, cpus, out):
    seat = ticket["seat"]
    live.PLAN = dict(seed=ticket["seed"], physical_map_sha256=ticket["terrain"], platform=plan["platform"],
                     files={**plan["files"], "candidate":plan["files"][arm]})
    live.RULE = plan["platform"]
    live.ROLES = ("candidate","opponent") if seat == 0 else ("opponent","candidate")
    env, cfg = live.official()
    views = [live.observe(env,s,5.) for s in (0,1)]
    check([hashlib.sha256(live.encoded(o)).hexdigest() for o in views] == ticket["public_sha256"] and
          hashlib.sha256(live.encoded(cfg)).hexdigest() == ticket["cfg_sha256"], "Wrong public request")
    rec = dict(seed=ticket["seed"], seat=seat, arm=arm, sources_by_seat=[live.PLAN["files"][r]["sha256"] for r in live.ROLES],
               public_sha256=ticket["public_sha256"], cfg_sha256=ticket["cfg_sha256"], start_ns=time.monotonic_ns(),
               initializations=0, action_calls=[0,0], official_transitions=0, status="preflight")
    live.save(out,rec)
    try:
        async with AsyncExitStack() as stack:
            boxes = [await stack.enter_async_context(live.sandbox(image,role,cpus)) for role in live.ROLES]
            process, probes, containers = zip(*boxes)
            issues = live.eligibility(probes,containers)
            for c in containers:
                if not (c["image"] == image and c["nano_cpus"] == 1500000000 and
                        c["memory"] == c["memory_swap"] == 536870912 and c["cpuset"] == cpus and c["read_only_root"]):
                    issues.append("Docker inspect settings differ from frozen limits")
            rec.update(containers=containers, probes=probes, resource_blockers=issues)
            if issues:
                rec["status"]="resource_preflight_failed"
                return rec
            paths = resource_paths(containers)
            check(all(p.stat().st_ino == expected["cgroup_inode"] for p,expected in zip(paths,probes)),
                  "Host/guest cgroup ownership differs")
            rec["before_load"] = snapshot(paths)
            rec.update(initializations=2, status="load_requested")
            live.save(out,rec)
            loads = await asyncio.gather(*(live.exchange(p,{"load":True},20.) for p in process))
            for call in loads:
                call["limit"] = 20.
            rec["initialization"] = loads
            rec["after_load"] = snapshot(paths)
            for side,call in enumerate(loads):
                if call["status"] != "ok":
                    continue
                r = call["response"]
                check(r["ready"] and r["sha256"] == rec["sources_by_seat"][side], "Cold load identity differs")
                if side != seat:
                    check(r["original"] == {"_TEMP":1,"_SEED":None,"_S_ABORT":.58,"_SEARCH_OK":True},
                          "Examiner configuration differs")
            if live.failures(loads):
                rec["status"]="cold_load_failed"
                return rec
            rec["step_zero"] = []
            for side,proc in enumerate(process):
                rec["action_calls"][side] += 1
                call = await live.exchange(proc,{"observation":views[side],"configuration":cfg},3.)
                call["limit"] = 3.
                live.check_action(call,views[side])
                call["exceeds_one_second"] = call["elapsed"]>1.
                rec["step_zero"].append(call)
            rec["after_step_zero"] = snapshot(paths)
            rec["status"] = "step_zero_failed" if live.failures(rec["step_zero"]) else "complete"
    except Exception as exc:
        rec.update(status="infrastructure_error", error=type(exc).__name__+": "+str(exc))
    finally:
        rec["end_ns"]=time.monotonic_ns()
        live.save(out,rec)
    return rec


async def run(args, plan):
    out=args.out
    out.mkdir(parents=True,exist_ok=True)
    image=subprocess.check_output(["docker","image","inspect",args.image,"--format","{{.Id}}"],text=True).strip()
    cpus=sorted(os.sched_getaffinity(0))[:2]
    check(len(cpus)==2,"Insufficient CPUs")
    header=dict(run_id=os.environ["GITHUB_RUN_ID"],attempt=os.environ["GITHUB_RUN_ATTEMPT"],commit=os.environ["GITHUB_SHA"],
                protocol_sha256=live.sha(PLAN), image=image, cpus=cpus, host=os.uname().nodename,
                start_ns=time.monotonic_ns(), cpu=os.uname().machine)
    check(header["attempt"]=="1","Workflow reattempt")
    with (out/"claim.json").open("x") as stream:
        json.dump(header,stream)
    for ticket in plan["tickets"]:
        for arm in ticket["order"]:
            dst=out/f'{ticket["seed"]}-{arm}.json'
            check(not dst.exists(),"Previous exposure")
            receipt=await pair(ticket,arm,plan,image,",".join(map(str,cpus)),dst)
            print(receipt["seed"], arm, receipt["status"],flush=True)
    live.save(out/"summary.json", summarize(out, plan))


def summarize(out, plan):
    rows=[]
    all_ids=set()
    previous=0
    header=read(out/"claim.json")
    check(header["protocol_sha256"]==live.sha(PLAN) and header["attempt"]=="1", "Wrong run registration")
    for ticket in plan["tickets"]:
        for arm in ticket["order"]:
            r=read(out/f'{ticket["seed"]}-{arm}.json')
            s=ticket["seat"]
            check((r["seed"],r["seat"],r["arm"],r["public_sha256"])==
                  (ticket["seed"],s,arm,ticket["public_sha256"]) and r["start_ns"]>=previous and
                  r["official_transitions"]==0, "Unexpected probe order or request")
            check(r["sources_by_seat"][s]==IDENTITIES[arm] and r["sources_by_seat"][1-s]==IDENTITIES["opponent"],
                  "Source identities changed")
            ids=[c["id"] for c in r.get("containers",[])]
            check(not all_ids.intersection(ids) and len(set(ids))==len(ids),"Reused container")
            all_ids.update(ids)
            previous=r["end_ns"]
            init=r.get("initialization",[{},{}])
            steps=[*r.get("step_zero",[]),{},{}]
            peaks=r.get("after_step_zero",r.get("after_load",[{},{}]))
            if r["status"] in ("complete","step_zero_failed","cold_load_failed"):
                live.PLAN={"files":plan["files"]}
                live.RULE=plan["platform"]
                live.ROLES=("candidate","opponent") if s==0 else ("opponent","candidate")
                check(not live.eligibility(r["probes"],r["containers"]) and all(
                    c["image"]==header["image"] for c in r["containers"]), "Unqualified trial presented as measured")
                check(r["initializations"]==2 and len(init)==2, "Missing a cold process")
                for i,call in enumerate(init):
                    check(call["limit"]==20. and (call["status"]!="ok" or
                          (call["elapsed"]<=20. and call["response"]["sha256"]==r["sources_by_seat"][i])),
                          "Bad load bound or identity")
                if r["status"]=="cold_load_failed":
                    check(live.failures(init) and r["action_calls"]==[0,0], "Invalid cold failure")
                else:
                    check(not live.failures(init) and r["action_calls"]==[1,1] and len(r["step_zero"])==2,
                          "Incomplete zero-step exchange")
                    for call in r["step_zero"]:
                        check(call["limit"]==3. and call["exceeds_one_second"]==(call["elapsed"]>1.) and
                              (call["status"]!="ok" or (call["elapsed"]<=3. and call["response"]["step"]==0)),
                              "Invalid first step or timing")
                    check((not live.failures(r["step_zero"]))==(r["status"]=="complete"), "Incorrect first step outcome")
            rows.append(dict(seed=r["seed"],seat=s,arm=arm,status=r["status"],init=init[s].get("status"),
                             load_seconds=init[s].get("elapsed"),original_init=init[1-s].get("status"),
                             original_load_seconds=init[1-s].get("elapsed"),
                             step_zero=steps[s].get("status"),step_seconds=steps[s].get("elapsed"),
                             slow=steps[s].get("exceeds_one_second"),peak_bytes=peaks[s].get("memory_peak_bytes")))
    per_arm={}
    for arm in ("v1","v2"):
        group=[r for r in rows if r["arm"]==arm]
        successful=[r["load_seconds"] for r in group if r["init"]=="ok"]
        per_arm[arm]=dict(registered=len(group), candidate_cold_ready=len(successful),
                          original_cold_ready=sum(r["original_init"]=="ok" for r in group),
                          bilateral_cold_failure=sum(r["init"] in ("timeout","error") and
                                                     r["original_init"] in ("timeout","error") for r in group),
                          complete_step_zero=sum(r["status"]=="complete" for r in group),
                          candidate_slow_step_zero=sum(r["slow"] is True for r in group),
                          max_successful_candidate_load_seconds=max(successful,default=None))
    return dict(**header, registered_pairs=8, full_games=0, rows=rows, by_version=per_arm)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("freeze","check","run","verify"))
    parser.add_argument("--digest")
    parser.add_argument("--image")
    parser.add_argument("--out",type=Path)
    args=parser.parse_args()
    plan=contract()
    if args.mode=="freeze":
        if PLAN.exists():check(read(PLAN)==plan,"Changed existing protocol")
        else:live.save(PLAN,plan)
        print(live.sha(PLAN))
    else:
        check(live.sha(PLAN)==args.digest and read(PLAN)==plan,"Source or protocol digest differs")
        if args.mode=="run":asyncio.run(run(args,plan))
        if args.mode=="verify":
            check(summarize(args.out,plan)==read(args.out/"summary.json"),"Summary differs")

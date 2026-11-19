"""Prepare a frozen root schedule from public decisions; run or verify its two live arms."""
import argparse
import fcntl
import gzip
import json
from pathlib import Path
import tempfile
import os
import subprocess
import sys

from branches import ROOT,PACK_HASH,MOVES,begin,candidate_roots,clean,destination,digest,public
from live import prepare as prepare_stream,run as run_live,save
from live_verify import main as verify_live


def load_case(source,seed,seat):
    cf=json.loads((source/"official_replay.json").read_bytes())["runs"][source.name]
    c=next(r for r in cf["games"] if (r["seed"],r["seat"])==(seed,seat))
    source_path=source/f"raw_{seed}_{seat}.json.gz";b=source_path.read_bytes()
    if digest(b)!=c["transcript_sha256"]:raise ValueError("recorded map identity")
    raw=json.loads(gzip.decompress(b))
    if raw["pack_sha256"]!=PACK_HASH or raw["examiner_md5"]!="57f9a594cce4f2ec9e3573d8d9d26ca1" or cf["pack_sha256"]!=PACK_HASH or raw["errors"]:
        raise ValueError("wrong policy identity or source")
    raw["initial_terrain_u8_sha256"]=c["map_sha256"]
    return raw,dict(source=str(source_path.relative_to(ROOT)),source_sha256=digest(b),map_sha256=c["map_sha256"],seed=seed,seat=seat)


def options(obs,raw,action,request):
    """Select among registered tree opportunities, or validate an explicit board-legal worker action."""
    if request["family"]!="explicit_worker":
        return candidate_roots(obs,raw,action)
    if not all(k in request for k in ("root_t","worker","alternative")):
        raise ValueError("explicit_worker requires root_t, worker and alternative")
    seat=raw["candidate_seat"];p=obs["players"][seat];uid=request["worker"]
    new=request["alternative"];old=action.get(uid,"NONE")
    if uid not in p[2] or new not in MOVES or new==old:return []
    cell,amount,role=p[2][uid];size=raw["game_configuration"]["size"]
    target=destination(cell,new,size)
    # Same convention as the registered diagnostic: do not force a friendly
    # collision or enter the opposing recruitment centre.
    occupied={destination(w[0],action.get(k,"NONE"),size) for k,w in p[2].items() if k!=uid}
    occupied.update(pos for base,pos in p[1].items() if action.get(base) in ("RECCOLLECTOR","RECPLANTER"))
    if target in occupied or target in obs["players"][1-seat][1].values():return []
    tree=request.get("tree","")
    return [dict(family="explicit_worker",t=obs["step"],worker=uid,origin=cell,cargo=amount,role=role,
                 old=old,alternative=new,tree=tree,cell=target,cash=p[0])]


def select(base):
    plan_bytes=(base/"plan.json").read_bytes();plan=json.loads(plan_bytes)
    case_names=[c["name"] for c in plan["cases"]]
    if len(set(case_names))!=len(case_names) or any(Path(k).name!=k for k in case_names):raise ValueError("invalid case names")
    selected=[]
    allmaps=set()
    for c in plan["cases"]:
        seed=c["seed"]
        if seed in allmaps:raise ValueError("map reused across cases")
        allmaps.add(seed)
        seats=[c["preferred_seat"]] if "root_t" in c else [c["preferred_seat"],1-c["preferred_seat"]]
        choice=None;coverage={}
        for seat in seats:
            source=ROOT/c["source"];raw,g=load_case(source,seed,seat);env=begin(raw)
            pool=[]
            for t,joint in enumerate(raw["actions"]):
                if t>plan["decision_window"][1]:break
                if t>=plan["decision_window"][0] and ("root_t" not in c or t==c["root_t"]):
                    obs=public(env,seat)
                    for r in options(obs,raw,clean(joint[seat]),c):
                        if r["family"]!=c["family"] or any(c[key]!=r[name] for key,name in (("worker","worker"),("alternative","alternative"),("tree","tree")) if key in c):
                            continue
                        key=f"{plan['salt']}|{seed}|{seat}|{t}|{r['worker']}|{r['tree']}|{r['alternative']}"
                        pool.append((digest(key.encode()),r))
                env.step(joint)
            coverage[str(seat)]=len(pool)
            if pool:
                signature,root=min(pool,key=lambda x:x[0])
                choice=dict(seed=seed,seat=seat,root_t=root["t"],family=c["family"],root=root,record=g,
                            source_dir=c["source"],selection_hash=signature,examiner_after_root="full original responsive")
                break
        if choice is None:raise ValueError(f"no eligible predeclared root for {c['name']}: {coverage}")
        choice["candidate_counts_by_seat_scanned"]=coverage
        choice["plan_sha256"]=digest(plan_bytes)
        wd=base/c["name"];wd.mkdir(exist_ok=True)
        path=wd/"live_protocol.json"
        if path.exists() and json.loads(path.read_bytes())!=choice:raise ValueError("root schedule changed")
        if not path.exists():save(path,choice)
        prepare_stream(wd)
        selected.append(dict(name=c["name"],seed=choice["seed"],seat=choice["seat"],root=choice["root"],protocol_sha256=digest(path.read_bytes())))
    schedule=base/"schedule.json"
    record=dict(plan_sha256=digest(plan_bytes),cases=selected)
    if schedule.exists() and json.loads(schedule.read_bytes())!=record:raise ValueError("frozen schedule changed")
    if not schedule.exists():save(schedule,record)
    return record


def entry(base,action,case,arm,stop):
    if action=="select":
        with (base/"plan.json").open("rb") as handle:
            fcntl.flock(handle,fcntl.LOCK_EX)
            print(select(base),flush=True)
        return
    schedule=json.loads((base/"schedule.json").read_bytes())
    if digest((base/"plan.json").read_bytes())!=schedule["plan_sha256"]:raise ValueError("plan changed")
    task=next(c for c in schedule["cases"] if c["name"]==case)
    wd=base/case;cp=wd/"live_protocol.json"
    if digest(cp.read_bytes())!=task["protocol_sha256"]:raise ValueError("root changed")
    if action=="verify":verify_live(wd);return
    lock=Path(tempfile.gettempdir())/("sapling-root-"+digest(str((wd/arm).resolve()).encode())+".lock")
    with lock.open("a") as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run_live(arm,wd,json.loads(cp.read_bytes()),resume=True,stop_after=stop)


def run_all(base, cpus):
    if not 1<=len(cpus)<=2 or len(set(cpus))!=len(cpus) or not set(cpus)<=os.sched_getaffinity(0):
        raise ValueError("choose one or two distinct allowed CPU cores")
    with (base/"plan.json").open("rb") as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        schedule=select(base)
    env=dict(os.environ,OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",CARBON_DEBUG="1")
    for case in schedule["cases"]:
        processes=[]
        try:
            for i,arm in enumerate(("parent","alternative")):
                cmd=["taskset","-c",str(cpus[i%len(cpus)]),sys.executable,str(Path(__file__).resolve()),
                     str(base),"run","--case",case["name"],"--arm",arm]
                if len(cpus)==1:subprocess.run(cmd,env=env,check=True)
                else:processes.append(subprocess.Popen(cmd,env=env))
            for proc in processes:
                if proc.wait()!=0:raise RuntimeError("arm failed; its completed checkpoints are preserved")
        finally:
            for proc in processes:
                if proc.poll() is None:proc.terminate();proc.wait()
        verify_live(base/case["name"])


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory",type=Path);p.add_argument("action",choices=["select","run","verify","run-all"])
    p.add_argument("--case");p.add_argument("--arm",choices=["parent","alternative"]);p.add_argument("--stop-after",type=int)
    p.add_argument("--cpus",type=int,nargs="+")
    a=p.parse_args()
    if a.action in ("run","verify") and not a.case or a.action=="run" and a.arm is None:p.error("missing case/arm")
    if a.action=="run-all":
        if not a.cpus:p.error("run-all needs --cpus")
        run_all(a.directory.resolve(),a.cpus)
    else:entry(a.directory.resolve(),a.action,a.case,a.arm,a.stop_after)

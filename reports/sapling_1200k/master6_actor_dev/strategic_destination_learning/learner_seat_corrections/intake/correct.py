"""Collect original expert labels on an already-executed Sapling seat; never execute a teacher label."""
import argparse
from collections import Counter
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import tempfile
import traceback

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p/".git").exists())
PLAN = HERE/"protocol.json"
PROTOCOL = "83b7a9257615d1b34b18802bb158aeeafe5838c899805fba774d035fe20a50e4"
DIRECTIONS = {"NONE":(0,0), "UP":(-1,0), "RIGHT":(0,1), "DOWN":(1,0), "LEFT":(0,-1)}
sha = lambda data:hashlib.sha256(data).hexdigest()


def require(flag, why):
    if not flag:raise ValueError(why)


def encode(obj, sort=False):
    return json.dumps(obj, sort_keys=sort, separators=(",",":"), allow_nan=False).encode()


def digest(obj):
    return sha(encode(obj, True))


def save(path, obj):
    raw=encode(obj)
    if path.suffix==".gz":raw=gzip.compress(raw, compresslevel=6, mtime=0)
    temp=path.with_suffix(path.suffix+".partial")
    temp.write_bytes(raw)
    os.replace(temp,path)


def destinations(players, seat, order):
    result={}
    for uid, unit in players[seat][2].items():
        row,col=divmod(unit[0],15)
        dr,dc=DIRECTIONS[order.get(uid) or "NONE"]
        result[uid]=(row+dr)%15*15+(col+dc)%15
    return result


def run_one(card,p):
    os.environ["CARBON_DEBUG"]="1"
    import numpy as np
    import random
    sys.path.insert(0,str(ROOT/"code/env_src/carbon_challenge-master"))
    from zerosum_env import make
    tape_path=ROOT/card["source_path"]
    b=tape_path.read_bytes() if tape_path.is_file() else subprocess.check_output(["git","-C",str(ROOT),"show","origin/main:"+card["source_path"]])
    require(sha(b)==card["source_sha256"],"Wrong historical trajectory")
    tape=json.loads(gzip.decompress(b))
    require(tape["pack_sha256"]==card["learner_pack_sha256"] and tape["candidate_seat"]==card["seat"]
        and digest(tape["actions"])==card["body_actions_sha256"] and tape["game_configuration"]==card["game_configuration"],"Wrong acting policy, orders or rules")
    local=HERE/"traces";local.mkdir(exist_ok=True)
    stem=f'{card["seed"]}_{card["seat"]}'; dest=local/(stem+".json.gz")
    claim=local/(stem+".claim.json")
    require(not dest.exists() and not claim.exists(),"Ticket already queried or claimed")
    src=ROOT/p["teacher_source_path"]
    native_bytes=src.read_bytes() if src.is_file() else subprocess.check_output(["git","-C",str(ROOT),"show","origin/main:"+p["teacher_source_path"]])
    require(sha(native_bytes)==p["teacher_sha256"],"Wrong original teacher source")
    stable=tempfile.TemporaryDirectory(prefix="sapling-shadow-3s2-")
    src=Path(stable.name)/"master3s2.py";src.write_bytes(native_bytes)
    spec=importlib.util.spec_from_file_location("retrospective_native_teacher",src)
    teacher=importlib.util.module_from_spec(spec)
    start=time.perf_counter();spec.loader.exec_module(teacher);cold=time.perf_counter()-start
    require(teacher._SEARCH_OK and teacher._S_START==0 and teacher._S_ABORT==.58 and teacher._TEMP==1. and teacher._SEED is None,"Different native teacher behavior")
    seed,seat,cfg=card["seed"],card["seat"],tape["game_configuration"]
    random.seed(seed);np.random.seed(seed)
    env=make("carbon",configuration={**cfg,"randomSeed":seed})
    require(sha(bytes(env.states[0].observation.full_carbon))==card["terrain"],"Different physical opening")
    claim.open("xb").write(encode({"protocol":PROTOCOL, "seed":seed, "seat":seat,"pid":os.getpid()}))
    rawlog=local/(stem+".journal");rows=[];previous=None;prev_players=None
    with rawlog.open("xb") as journal:
        for t in p["root_steps"]:
            shared=env.states[0].observation;own=env.states[seat].observation
            observation=json.loads(encode(dict(step=shared.step,player=seat,carbon=own.carbon,
                players=shared.players,trees=shared.trees,remainingOverageTime=own.remainingOverageTime)))
            require(not env.done and observation["step"]==t and "full_carbon" not in observation,"Unexpected game end or nonpublic observation")
            if t: require(teacher._ST["tr"].last_dest==previous and teacher._ST["prev"]==prev_players,
                          "Teacher history lost recorded own move or observed players")
            before=dict((k,teacher._S_STATS[k]) for k in ("searched","single","changed","aborted"))
            clock0,cpu0=time.perf_counter(),time.process_time()
            order=teacher.agent(observation,cfg)
            cpu=time.process_time()-cpu0;wall=time.perf_counter()-clock0
            calls={k:int(teacher._S_STATS[k]-v) for k,v in before.items()}
            if calls["single"]==1 and calls["searched"]==calls["aborted"]==0:
                n,depth,rollouts,native=1,0,0,None
            else:
                require(calls["single"]==0 and (calls["searched"] or calls["aborted"]),"Public original did not report search")
                n,depth,rollouts,native,*_=teacher._S_STATS["last"]
                require(n>=2 and bool(depth)==bool(calls["searched"]),"Inconsistent native depth/candidate counts")
            _,bases,workers,_=observation["players"][seat]
            require(isinstance(order,dict) and all(
                a in (("RECCOLLECTOR","RECPLANTER","NONE") if uid in bases else DIRECTIONS if uid in workers else ())
                for uid,a in order.items()),"Teacher returned an invalid physical order")
            actual=tape["actions"][t][seat];players=observation["players"]
            tr=teacher._ST["tr"];emitted=destinations(players,seat,order);expected=destinations(players,seat,actual)
            require(tr.last_dest==emitted and teacher._ST["prev"]==players,"Source left an unaccounted private state")
            tr.set_actions(players,actual)
            require(tr.last_dest==expected,"Original bookkeeping did not match independent physical actions")
            rng=teacher._ST["rng"].get_state()
            row=dict(t=t,observation=observation,public_sha256=digest(observation),teacher_order=order,
                learner_order=actual,opponent_order=tape["actions"][t][1-seat],
                teacher_clock=dict(wall=wall,cpu=cpu,K=int(n),H=int(depth),rolls=int(rollouts),native_seconds=native,**calls),
                teacher_state=dict(before_intended_destinations_sha256=digest(emitted),
                    executed_destinations_sha256=digest(expected),past_executed_destinations_observed=True,
                    repaired_workers=sum(emitted[k]!=v for k,v in expected.items()),
                    rng_stream_sha256=sha(rng[1].tobytes()),rng_index=int(rng[2])))
            rows.append(row);journal.write(encode(row)+b"\n");journal.flush()
            previous=expected;prev_players=players
            env.step(tape["actions"][t],[])
    trace=dict(schema="sapling-learner-seat-shadow-trace/1",protocol_sha256=PROTOCOL,collector_sha256=sha(Path(__file__).read_bytes()),
               split=card["split"],seed=seed,seat=seat,terrain=card["terrain"],source_path=card["source_path"],source_sha256=card["source_sha256"],
               behavior_package_sha256=card["learner_pack_sha256"],game_configuration=cfg,teacher_sha256=p["teacher_sha256"],
               cold_init_wall=cold,pid=os.getpid(),affinity=sorted(os.sched_getaffinity(0)),torch_threads=teacher._torch.get_num_threads(),rows=rows)
    save(dest,trace);rawlog.unlink()
    return dest


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--ticket",type=int);ap.add_argument("--all",action="store_true");a=ap.parse_args()
    p=json.loads(PLAN.read_bytes());require(sha(PLAN.read_bytes())==PROTOCOL,"Unregistered source")
    if a.ticket is not None:
        run_one(p["tickets"][a.ticket],p);return
    require(a.all and os.sched_getaffinity(0)=={0,1},"Require registered two-CPU serial intake")
    rows=[];issues=[];streams=[]
    for i,c in enumerate(p["tickets"]):
        f=HERE/"traces"/f'{c["seed"]}_{c["seat"]}.json.gz'
        try:
            if not f.is_file():
                subprocess.run([sys.executable,__file__,"--ticket",str(i)],check=True)
            b=f.read_bytes();d=json.loads(gzip.decompress(b));ls=d["rows"]
            require(d["protocol_sha256"]==PROTOCOL and d["source_sha256"]==c["source_sha256"] and
               (d["seed"],d["seat"],d["split"])==(c["seed"],c["seat"],c["split"]) and
               [r["t"] for r in ls]==p["root_steps"] and d["collector_sha256"]==sha(Path(__file__).read_bytes()),"Invalid trace")
            multi=sum(r["teacher_clock"]["K"]>=2 for r in ls);pos=sum(r["teacher_clock"]["H"]>0 for r in ls)
            h=ls[0]["teacher_state"]["rng_stream_sha256"];streams.append(h)
            eligible=multi>0 and pos/multi>=.80
            row=dict(seed=c["seed"],seat=c["seat"],split=c["split"],terrain=c["terrain"],sha256=sha(b),
                     trace_path=f.relative_to(ROOT).as_posix(),n_rows=len(ls),eligible=eligible,multi=multi,positive=pos,
                     different_orders=sum(r["teacher_order"]!=r["learner_order"] for r in ls),
                     repaired_rows=sum(r["teacher_state"]["repaired_workers"]>0 for r in ls),
                     teacher_wall=sum(r["teacher_clock"]["wall"] for r in ls),
                     maximum_public_wall=max(r["teacher_clock"]["wall"] for r in ls))
            rows.append(row);print(i,c["seed"],c["seat"],eligible,multi,pos,row["different_orders"],flush=True)
            if not eligible:issues.append(dict(ticket=i,reason="insufficient positive-depth support"))
        except Exception as e:
            issues.append(dict(ticket=i,seed=c["seed"],seat=c["seat"],reason=repr(e)));traceback.print_exc()
    if len(set(streams))!=len(streams):issues.append(dict(reason="Repeated observed native RNG stream"))
    save(HERE/"collection.json",dict(schema="sapling-learner-seat-correction-collection/1",protocol_sha256=PROTOCOL,
        collector_sha256=sha(Path(__file__).read_bytes()),training_eligible=len(rows)==12 and not issues,
        tickets=rows,issues=issues,roots=sum(r["n_rows"] for r in rows),new_games=0))
    if issues:sys.exit(1)


if __name__=="__main__":
    main()

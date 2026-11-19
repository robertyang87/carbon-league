"""Register public late-game decisions, run responsive full-search pairs, and audit cash labels."""
import argparse
from collections import Counter
import fcntl
import gzip
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
ROOT=BASE.parents[1]
sys.path.insert(0,str(BASE/"root_recovery_branches"))
from branches import MOVES,begin,public,destination,distances,clean,digest
from batch import load_case
import live

PLAN=HERE/"protocol.json"
FAMILIES=("own_return","deny_enemy_return")


def read(path):return json.loads(path.read_bytes())
def packed(path):return json.loads(gzip.decompress(path.read_bytes()))
def persist(path,data):live.save(path,data)


def offers(obs,raw,action,code):
    s=obs["player"];t=obs["step"];cfg=raw["game_configuration"];n=cfg["size"];D=distances(n)
    rem=cfg["episodeSteps"]-1-t
    own,opp=obs["players"][s],obs["players"][1-s];ours,theirs=own[2],opp[2]
    homes=tuple(own[1].values());enemy_homes=tuple(opp[1].values())
    actual={u:destination(w[0],action.get(u,"NONE"),n) for u,w in ours.items()}
    hired={cell for bid,cell in own[1].items() if action.get(bid) in ("RECCOLLECTOR","RECPLANTER")}
    h=lambda pos,bs:min(D[pos][b] for b in bs)
    for uid,(pos,load,role) in ours.items():
        if role!="COLLECTOR":continue
        old=action.get(uid,"NONE")
        banned=hired|set(enemy_homes)|{v for u,v in actual.items() if u!=uid}
        for alt in MOVES:
            z=destination(pos,alt,n)
            if alt==old or z in banned:continue
            danger={u for u,w in theirs.items() if w[2]=="COLLECTOR" and w[1]<=load and D[z][w[0]]<=1}
            common=dict(t=t,seat=s,worker=uid,origin=pos,cargo=load,old=old,alternative=alt,target_cell=z,
                        own_home_distance=h(pos,homes),original_home_distance=h(actual[uid],homes),alternative_home_distance=h(z,homes),
                        remaining=rem,own_cash=own[0],enemy_cash=opp[0],tree="",recorded_own_base_cells=homes,enemy_base_cells=enemy_homes)
            def offer(family,prize,**extra):
                tie=f"late-cash-01|{raw['seed']}|{s}|{t}|{uid}|{alt}|{extra.get('enemy_worker','')}"
                return dict(**common,family=family,priority_cargo=prize,selection_hash=digest(tie.encode()),**extra)
            if not danger and load>=20 and 0<h(pos,homes)<=rem and h(actual[uid],homes)>=h(pos,homes)>h(z,homes):
                yield offer(FAMILIES[0],load)
            for eid,(ep,ec,er) in theirs.items():
                if er!="COLLECTOR" or ec<40 or load>=ec or danger-{eid}:continue
                eh=h(ep,enemy_homes)
                if 2<=eh<=min(5,rem) and D[z][ep]==1 and h(z,enemy_homes)==eh-1:
                    shortest=[destination(ep,m,n) for m in MOVES if m!="NONE" and h(destination(ep,m,n),enemy_homes)==eh-1]
                    yield offer(FAMILIES[1],ec,enemy_worker=eid,enemy_origin=ep,enemy_cargo=ec,enemy_home_distance=eh,enemy_shortest_cells=shortest)


def select():
    plan=read(PLAN);sha=digest(PLAN.read_bytes());saved=HERE/"roots.json"
    if saved.exists():
        doc=read(saved)
        if doc["protocol_sha256"]!=sha:raise ValueError("frozen plan changed")
        return doc
    games=[];all_seeds=set()
    for c in plan["maps"]:
        seed=c["seed"]
        if seed in all_seeds or seed==621456920:raise ValueError("prohibited or duplicate map seed")
        all_seeds.add(seed);pool={k:[] for k in FAMILIES};source={}
        for seat in (0,1):
            raw,g=load_case(ROOT/c["source"],seed,seat);source[str(seat)]=g
            env=begin(raw);T=raw["game_configuration"]["episodeSteps"]-1
            for t,joint in enumerate(raw["actions"]):
                rem=T-t
                if rem<min(plan["remaining"]):break
                if rem<=max(plan["remaining"]):
                    obs=public(env,seat)
                    for e in offers(obs,raw,clean(joint[seat]),c):
                        pool[e["family"]].append(e)
                env.step(joint)
        selected={};taken=set()
        for family in FAMILIES:
            for r in sorted(pool[family],key=lambda e:(-e["priority_cargo"],e["selection_hash"])):
                key=(r["seat"],r["t"],r["worker"])
                if key in taken:continue
                selected[family]=r;taken.add(key);break
        games.append(dict(**c,by_seat=source,eligible_actions={k:len(v) for k,v in pool.items()},
                          eligible_map_seat_turns={k:len({(v["seat"],v["t"]) for v in vs}) for k,vs in pool.items()},
                          selected=selected))
    result=dict(protocol_sha256=sha,games=games);persist(saved,result);return result


def folder(g,r,i):return HERE/"arms"/str(g["seed"])/r["family"]/str(i)


def jobs(reg):
    for g in reg["games"]:
        for r in g["selected"].values():
            for i in range(read(PLAN)["streams"]):yield g,r,i


def init(g,r,i):
    wd=folder(g,r,i);wd.mkdir(parents=True,exist_ok=True)
    case=dict(seed=g["seed"],seat=r["seat"],family=r["family"],root_t=r["t"],root=r,source_dir=g["source"],record=g["by_seat"][str(r["seat"])],
              roots_sha256=digest((HERE/"roots.json").read_bytes()),split=g["split"],stream=i,live_code_sha256=digest(Path(live.__file__).read_bytes()))
    path=wd/"live_protocol.json"
    if path.exists() and read(path)!=case:raise ValueError("changed case")
    if not path.exists():persist(path,case)
    live.prepare(wd)
    return wd,case


def native_run(task,arm):
    wd=HERE/task;cfg=read(wd/"live_protocol.json")
    if digest(Path(live.__file__).read_bytes())!=cfg["live_code_sha256"]:raise ValueError("live runner differs from frozen code")
    lock=Path(tempfile.gettempdir())/("sapling-late-arm-"+digest(str((wd/arm).resolve()).encode())+".lock")
    with lock.open("a") as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        live.run(arm,wd,cfg,resume=True)


def accounting():
    # Economic verifier reused from the first (different-map) experiment.
    directory=BASE/"planter_routes_dev/endgame_responses"
    sys.path.insert(0,str(directory))
    spec=importlib.util.spec_from_file_location("existing_endgame_ledger",directory/"verify.py")
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m.ledger


def verify_pair(g,r,i,ledger):
    wd=folder(g,r,i);cfg=read(wd/"live_protocol.json")
    raw,src=load_case(ROOT/g["source"],g["seed"],r["seat"])
    if cfg["record"]!=src or cfg["roots_sha256"]!=digest((HERE/"roots.json").read_bytes()):raise ValueError("identity mismatch")
    s=r["seat"];o=1-s; arms={}
    for arm in ("parent","alternative"):
        info=read(wd/f"live_{arm}.json");log=wd/info["action_tape"]
        if digest(log.read_bytes())!=info["action_sha256"] or info["source_sha256"]!=live.MASTER_SHA or info["pack_sha256"]!=read(PLAN)["actor"]:raise ValueError("actor or command tape identity")
        if info["protocol_sha256"]!=digest((wd/"live_protocol.json").read_bytes()) or info["shared_init_sha256"]!=digest((wd/"live_shared_init.json").read_bytes()):raise ValueError("stream or registry mismatch")
        acts=packed(log);parent=raw["actions"][r["t"]]
        if clean(acts[0][o])!=clean(parent[o]):raise ValueError("changed simultaneous enemy move")
        orig=clean(parent[s]);new=clean(acts[0][s])
        if {uid for uid in set(orig)|set(new) if orig.get(uid)!=new.get(uid)}!=({r["worker"]} if arm=="alternative" else set()):raise ValueError("joint action diff")
        if arm=="alternative" and new.get(r["worker"],"NONE")!=r["alternative"]:raise ValueError("alternative identity")
        v=ledger(raw,r,acts)
        if v["score_by_physical_seat"]!=info["score"] or info["search_calls"]!=len(acts)-1:raise ValueError("official result or actual searches")
        v.update(terminal_statuses=info["statuses"],search_calls=info["search_calls"],search_completed=info["search_completed"],
                 action_sha256=info["action_sha256"],action_path=str(log.relative_to(HERE)))
        arms[arm]=v
    a,b=arms["parent"],arms["alternative"];keys=a["gross"][0].keys()
    effect={k:round(b["gross"][s][k]-b["gross"][o][k]-a["gross"][s][k]+a["gross"][o][k],4) for k in keys}
    if abs(sum(effect.values()) - (b["margin"]-a["margin"]))>.03:raise ValueError("official score delta vs cash accounting")
    out=dict(seed=g["seed"],split=g["split"],seat=s,family=r["family"],root=r,stream=i,
             stream_sha256=digest(json.dumps(read(wd/"live_shared_init.json")["mt_keys"]).encode()),
             delta_margin=round(b["margin"]-a["margin"],4),delta_u=b["u"]-a["u"],
             focal_own_deposit_delta=round(b["focal_own_deposits"]-a["focal_own_deposits"],3),
             focal_enemy_deposit_delta=round(b["focal_enemy_deposits"]-a["focal_enemy_deposits"],3) if r["family"]==FAMILIES[1] else None,
             delta_by_source=effect,arms=arms)
    persist(wd/"verified.json.gz",out)
    return out


def assemble(reg):
    rows=[]
    for g,r,i in jobs(reg):
        p=folder(g,r,i)/"verified.json.gz"
        if p.exists():rows.append(packed(p))
    if len({v["stream_sha256"] for v in rows})!=len(rows):raise ValueError("reused random stream")
    summary=[]
    for g in reg["games"]:
        for family in FAMILIES:
            part=[v for v in rows if v["seed"]==g["seed"] and v["family"]==family]
            if not part:continue
            r=part[0]["root"]
            summary.append(dict(seed=g["seed"],seat=r["seat"],split=g["split"],family=family,t=r["t"],worker=r["worker"],
                                old=r["old"],alternative=r["alternative"],remaining=r["remaining"],root_features=r,
                                streams=len(part),delta_margin=[v["delta_margin"] for v in part],delta_u=[v["delta_u"] for v in part],
                                tagged_own_delta=[v["focal_own_deposit_delta"] for v in part],tagged_enemy_delta=[v["focal_enemy_deposit_delta"] for v in part],
                                outcomes=[[v["arms"][k]["wdl"] for k in ("parent","alternative")] for v in part],
                                mean_delta_u=statistics.mean(v["delta_u"] for v in part),mean_margin=statistics.mean(v["delta_margin"] for v in part),
                                evidence=[str((folder(g,r,v["stream"])/"verified.json.gz").relative_to(HERE)) for v in part]))
    for split in ("train","holdout"):
        persist(HERE/f"{split}_labels.json",[v for v in summary if v["split"]==split])
    complete=len(rows)==sum(1 for _ in jobs(reg))
    persist(HERE/"progress.json",dict(complete=complete,maps=len(reg["games"]),selected_roots=sum(len(g["selected"]) for g in reg["games"]),
                                        expected_pairs=sum(1 for _ in jobs(reg)),verified_pairs=len(rows),verified_arms=2*len(rows)))
    return summary


def main(cpus):
    if not 1<=len(cpus)<=2 or len(set(cpus))!=len(cpus) or not set(cpus)<=os.sched_getaffinity(0):raise ValueError("CPU assignment")
    key=digest(str(HERE).encode());lock=Path(tempfile.gettempdir())/("sapling-late-cash-"+key+".lock")
    with lock.open("a") as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        reg=select();ledger=accounting();assemble(reg)
        print("registered",[(g["seed"],g["eligible_actions"],[(r["family"],r["seat"],r["t"]) for r in g["selected"].values()]) for g in reg["games"]],flush=True)
        for g,r,i in jobs(reg):
            wd,case=init(g,r,i);rel=str(wd.relative_to(HERE));procs=[]
            try:
                for j,arm in enumerate(("parent","alternative")):
                    cmd=["taskset","-c",str(cpus[(i+j)%len(cpus)]),sys.executable,str(Path(__file__)),"--arm",arm,"--task",rel]
                    if len(cpus)==1:subprocess.run(cmd,check=True)
                    else:procs.append(subprocess.Popen(cmd))
                if any(p.wait()!=0 for p in procs):raise RuntimeError("failed arm")
            finally:
                for p in procs:
                    if p.poll() is None:p.terminate();p.wait()
            pair=verify_pair(g,r,i,ledger);assemble(reg)
            print("audited",g["seed"],r["family"],i,pair["delta_margin"],pair["delta_u"],flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--cpus",nargs="+",type=int);parser.add_argument("--task");parser.add_argument("--arm",choices=("parent","alternative"));args=parser.parse_args()
    if args.arm:native_run(args.task,args.arm)
    elif args.cpus:main(args.cpus)
    else:parser.error("select one or two available cores with --cpus")

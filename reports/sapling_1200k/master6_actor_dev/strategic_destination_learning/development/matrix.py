"""Freeze and execute the strategic-destination development tournament on the existing dual-cgroup referee."""
import argparse
import asyncio
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
TRAIN=HERE.parent
CARGO=TRAIN.parent/"cargo_tree_action_learning_dev"
NATURAL=CARGO/"native_actor_corpus/league_policy/natural_original"
LIVE=CARGO/"ladder_live"
MATCH=CARGO/"territory_sequence/fresh_original"
ROOT=next(p for p in HERE.parents if (p/".git").exists())
PLAN=HERE/"protocol.json"
ARMS=("parent","bc","joint")
SEEDS=(621758500,621758501)
WORKFLOW="sapling-strategic-destination-development.yml"
DESIGN="18e949898c1898dd207439be6fdb0ff11ea3a42548f1d79ac1e418d037e75b48"
LEARNING="97f5cad8d56c07f21020560df5ad0d5e180d4176179461890071b7763bda21a7"
sys.path.insert(0,str(MATCH))
import play
from verify_pair import verify_game
spec=importlib.util.spec_from_file_location("native_timing_audit",NATURAL/"settle.py")
timing=importlib.util.module_from_spec(spec)
spec.loader.exec_module(timing)
read=play.load
sha=play.live.sha


def require(ok, why):
    if not ok: raise ValueError(why)


def record(path):
    relative=path.relative_to(ROOT).as_posix()
    raw=path.read_bytes() if path.is_file() else subprocess.check_output(["git","-C",str(ROOT),"show","HEAD:"+relative])
    return dict(path=relative,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def contract():
    require(sha(HERE/"design.json")==DESIGN and sha(TRAIN/"learning.json")==LEARNING,"Design or training protocol changed")
    design=read(HERE/"design.json")
    clearing=read(HERE/"clearance.json")
    require(clearing["by_seed"]==design["physical_boards"] and not clearing["seed_intersections"]
            and not clearing["physical_intersections"],"Development map not cleared")
    more=read(HERE/"clearance_increment.json")
    require(more["source_clearance_sha256"]==sha(HERE/"clearance.json") and
            more["physical_boards"]==design["physical_boards"] and not more["seed_and_opening_intersections"]
            and not more["current_unresolved_incremental_replay_faults"], "Fresh replay increment unavailable or overlapping")
    sources=read(TRAIN/"sources.json")
    selected=sources["maps"]
    require(not {row["seed"] for row in selected}.intersection(design["seeds"]) and
            not {row["terrain"] for row in selected}.intersection(design["physical_boards"].values()),
            "Fresh development map overlaps this learner's train, validation or previously exposed development maps")
    report=read(TRAIN/"fit.json")
    require(report["parent_sha256"]==design["parent_sha256"] and
            report["parent_weights_sha256"]==design["parent_npz_sha256"] and report["learning_sha256"]==LEARNING and
            report["fit_code_sha256"]==sha(TRAIN/"train.py") and report["data_sha256"]==sha(TRAIN/"data.json"),
            "Training evidence does not bind these data, sources and parent")
    release=read(HERE/"release.json")
    base=read(LIVE/"protocol.json")
    items={"parent":CARGO/"candidate.py","parent_weights":CARGO/"candidate.npz",
           "opponent":ROOT/"sources/master3s2/master3s2.py","ladder_runner":LIVE/"live.py",
           "worker":LIVE/"worker.py","base_registration":LIVE/"protocol.json","container":LIVE/"Dockerfile",
           "original_game_configuration":CARGO/"direct_original/games/candidate/raw_621758346_1.json.gz",
           "current_guide":NATURAL/"platform_guide.md","match_runner":MATCH/"play.py","replayer":MATCH/"verify_pair.py",
           "native_timing_audit":NATURAL/"settle.py","previous_exposure_guard":NATURAL/"fresh_repeat/workflow_cli.py","design":HERE/"design.json","clearance":HERE/"clearance.json",
           "clearance_increment":HERE/"clearance_increment.json",
           "release":HERE/"release.json","matrix":Path(__file__).resolve(),"cli":HERE/"workflow_cli.py",
           "workflow":ROOT/".github/workflows"/WORKFLOW,"training":TRAIN/"learning.json","fit":TRAIN/"fit.json",
           "training_map_inventory":TRAIN/"sources.json","trainer":TRAIN/"train.py","data_receipt":TRAIN/"data.json"}
    for arm in ARMS[1:]:
        items[arm]=TRAIN/(arm+".py")
        items[arm+"_weights"]=TRAIN/(arm+".npz")
    files={k:record(path) for k,path in items.items()}
    for key in ("rules","environment","official_helpers"):
        entry=base["files"][key]
        require(entry==record(ROOT/entry["path"]),"Rules, interpreter or helpers drift")
        files[key]=entry
    require(files["parent"]["sha256"]==design["parent_sha256"] and files["parent_weights"]["sha256"]==design["parent_npz_sha256"]
            and files["current_guide"]["sha256"]==design["current_platform_guide_sha256"]
            and files["opponent"]["sha256"]==design["original_sha256"]==base["files"]["opponent"]["sha256"],"Changed reference identity")
    require(release["fit_sha256"]==files["fit"]["sha256"] and release["learning_sha256"]==LEARNING,"Training release differs")
    for arm in ARMS[1:]:
        stamp=report["arms"][arm]
        require(all(stamp["quantized_changed_by_tensor"][k]>0 for k in ("stem.weight","gru.weight_ih","whead.weight")),
                "Quantized student update failed to change the encoder, decoder or action head")
        require(stamp["package_sha256"]==release["arms"][arm]["package_sha256"]==files[arm]["sha256"] and
                stamp["weights_sha256"]==release["arms"][arm]["weights_sha256"]==files[arm+"_weights"]["sha256"] and
                stamp["package_bytes"]==files[arm]["bytes"]<=10_000_000,"Wrong or oversized "+arm)
    fields=(b"_W64 = ",b"_WMETA = ",b"_QMETA = ")
    def serving_bytes(path):
        lines=path.read_bytes().splitlines(keepends=True)
        require(all(sum(line.startswith(f) for line in lines)==1 for f in fields),"Unrecognized student payload")
        return b"".join(line for line in lines if not line.startswith(fields))
    reference=serving_bytes(items["parent"])
    require(all(serving_bytes(items[arm])==reference for arm in ARMS[1:]), "Deployed code changed outside Sapling payload")
    require(len({files[a]["sha256"] for a in ARMS})==3,"Identical arm packages")
    return {**design,"schema":"sapling-strategic-development/frozen/1","platform":base["platform"],"files":files}


def bind(commit):
    """Bind only main-published candidate, weight and fit bytes. Run before dispatch."""
    rev=subprocess.check_output(["git","-C",str(ROOT),"rev-parse","--verify",commit+"^{commit}"],text=True).strip()
    require(rev==commit,"A full immutable revision is required")
    subprocess.run(["git","-C",str(ROOT),"merge-base","--is-ancestor",rev,"origin/main"],check=True)
    files=[TRAIN/n for n in ("bc.py","joint.py","bc.npz","joint.npz","fit.json","learning.json","sources.json","train.py","data.json")]
    for path in files:
        blob=subprocess.check_output(["git","-C",str(ROOT),"show",rev+":"+path.relative_to(ROOT).as_posix()])
        require(blob==path.read_bytes(),"Training working file differs from published release: "+path.name)
    src={a:dict(package_sha256=sha(TRAIN/(a+".py")),weights_sha256=sha(TRAIN/(a+".npz"))) for a in ARMS[1:]}
    release=dict(training_commit=rev,arms=src,fit_sha256=sha(TRAIN/"fit.json"),learning_sha256=sha(TRAIN/"learning.json"))
    for path,value in ((HERE/"release.json",release),(PLAN,None)):
        obj=contract() if value is None else value
        if path.exists(): require(read(path)==obj,"Refusing to replace a different frozen "+path.name)
        else: play.live.save(path,obj)
    print(sha(PLAN))


def check(digest):
    require(PLAN.is_file() and sha(PLAN)==digest and read(PLAN)==contract(),"Different registration or source bytes")
    return read(PLAN)


def verify_block(directory,seed,seat):
    p=Path(directory)
    matrix=read(PLAN);games={};errors={};sets=[];timeline=[]
    for arm in ARMS:
        try:
            dest=p/arm
            game=verify_game(dest,seed,seat,arm,PLAN)
            receipt=read(dest/"receipt.json")
            game["timing_audit"]=timing.check_timing(dest,game)
            claim=dest/"attempt.json"
            if claim.exists():
                require(read(claim)==dict(seed=seed,runtime_image=receipt["image_id"],protocol_sha256=sha(dest/"registration.json")),
                        "Unbound attempt")
            else:
                require(receipt.get("strategy_initializations",0)==0,"Missing on-field attempt")
            if game["timing_audit"]["executor_verified_for_scored_attempt"]:
                sets.append(tuple(sorted({c["cpuset"] for c in receipt["containers"]})))
            games[arm]=game
        except (ValueError,OSError,TypeError,KeyError,IndexError) as exc:
            errors[arm]=str(exc)
    try:
        timeline=[json.loads(row) for row in (p/"execution.jsonl").read_bytes().splitlines()]
        row=next(x for x in matrix["schedule"] if (x["seed"],x["seat"])==(seed,seat))
        require([x["arm"] for x in timeline]==row["order"] and len({(x["hostname"],x["pid"]) for x in timeline})==1
                and all(x["seed"]==seed and x["seat"]==seat and x["matrix_sha256"]==sha(PLAN) and
                        x["receipt_sha256"]==sha(p/x["arm"]/"receipt.json") and x["start_ns"]<=x["end_ns"] for x in timeline)
                and all(a["end_ns"]<=b["start_ns"] for a,b in zip(timeline,timeline[1:])),
                "Different host, order, ticket, or overlapping physical intervals")
        require(len({x["image_id"] for x in games.values()})<=1 and
                len({str(x["workflow_run"]) for x in [*games.values(),*timeline]})<=1 and
                all(len(c)==1 and c[0] for c in sets) and len(set(sets))<=1,
                "Container image, hosted run or CPU set changed within ticket")
    except (OSError,ValueError,KeyError,StopIteration) as exc:
        errors["sequence"]=str(exc)
    return dict(seed=seed,seat=seat,games=games,host_sequence=timeline,verification_errors=errors)


def run(digest,seed,seat,out,image):
    check(digest)
    require(seed in SEEDS and seat in (0,1),"Unregistered ticket")
    out.mkdir(parents=True,exist_ok=True)
    timeline=out/"execution.jsonl"
    if timeline.exists(): raise ValueError("Map-seat timeline already populated")
    play.HERE=HERE
    def checked(matrix):
        require(matrix==read(PLAN),"Modified plan")
        return read(LIVE/"protocol.json")
    play.check=checked
    original=play.live.run
    async def measured(args):
        start=time.monotonic_ns()
        try:
            return await original(args)
        finally:
            receipt=args.out/"receipt.json"
            event=dict(arm=args.out.name,seed=seed,seat=seat,hostname=socket.gethostname(),pid=os.getpid(),
                       workflow_run=os.getenv("GITHUB_RUN_ID"),matrix_sha256=digest,start_ns=start,
                       end_ns=time.monotonic_ns(),receipt_sha256=sha(receipt) if receipt.exists() else None)
            with timeline.open("a") as f: print(json.dumps(event),file=f,flush=True)
    play.live.run=measured
    sys.argv=["play","--seed",str(seed),"--seat",str(seat),"--out",str(out),"--image",image]
    return play.main()


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("mode",choices=("bind","check","list","play","verify"))
    p.add_argument("--training-commit");p.add_argument("--matrix-sha");p.add_argument("--seed",type=int);p.add_argument("--seat",type=int,choices=(0,1))
    p.add_argument("--out",type=Path);p.add_argument("--image");a=p.parse_args()
    if a.mode=="bind": bind(a.training_commit)
    elif a.mode=="play": sys.exit(run(a.matrix_sha,a.seed,a.seat,a.out,a.image))
    else:
        m=check(a.matrix_sha)
        if a.mode=="list":print(json.dumps({"include":[{"seed":r["seed"],"seat":r["seat"]} for r in m["schedule"]]}))
        if a.mode=="verify":
            result=verify_block(a.out,a.seed,a.seat)
            print(json.dumps(result,allow_nan=False));sys.exit(int(bool(result["verification_errors"])))

"""Freeze and run the two naturally sampled Sapling packages on two previously unused development maps."""
import argparse
import hashlib
import importlib.util
import json
import os
import socket
import subprocess
import time
from pathlib import Path
import sys
import zipfile

HERE=Path(__file__).resolve().parent
PREVIOUS=HERE.parent
spec=importlib.util.spec_from_file_location("exposed_original_matrix", PREVIOUS/"matrix.py")
old=importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
ROOT=old.ROOT
PLAN=HERE/"protocol.json"
RELEASE=PREVIOUS/"hosted/settlement.json"
SOURCE_RUN=44622271891
SOURCE_COMMIT="7df5b37ca961419aee96a835db3d37b3b93b88b4"
SOURCE_SETTLEMENT="4041b1385445ef7ad20a2ab9134ca9ed8a520aa321c13ea1cf3d2826f0e86b7f"
SOURCE_PLAN="059d05d7594018e89c05194b2aaced81a3d1ccf1da1335c70bbf958cf1a87cb2"
SEEDS=(621758450,621758451)
BOARDS={
    "621758450":"ea58a1d18d9e92d6d9da7e5738fbec7769e7b6735a35b17d4e3d86de01246878",
    "621758451":"e4b8147d1c170997262db3c88772b66185a42acc9eb0a246d487df89d49c1184",
}
A,B="sampled_parent","sampled_child"
ARMS=(A,B)
ORDERS=((A,B),(B,A),(B,A),(A,B))
sha,read,file,require=old.sha,old.read,old.file,old.require


def evidence_file(path):
    if path.is_file():
        return file(path)
    relative=path.relative_to(ROOT).as_posix()
    if relative!="sources/master3s2/master3s2.py":
        raise FileNotFoundError(path)
    content=subprocess.check_output(["git","-C",str(ROOT),"show","HEAD:"+relative])
    return dict(path=relative,sha256=hashlib.sha256(content).hexdigest(),bytes=len(content))


def schedule():
    return [dict(seed=s,seat=p,order=list(ORDERS[2*i+p])) for i,s in enumerate(SEEDS) for p in (0,1)]


def prior_receipt():
    require(sha(PREVIOUS/"protocol.json")==SOURCE_PLAN and sha(RELEASE)==SOURCE_SETTLEMENT,"Source protocol or independent settlement digest differs")
    oldplan=read(PREVIOUS/"protocol.json"); doc=read(RELEASE)
    require(doc["run_id"]==SOURCE_RUN and doc["commit"]==SOURCE_COMMIT and doc["plan_sha256"]==SOURCE_PLAN
            and doc["records_reconciled"] is True and doc["verification_errors"]==[]
            and doc["missing_triples"]==[] and doc["remote_artifacts_missing_locally"]==[],
            "Twelve-game independent verification incomplete")
    prior_seats={(s,p) for s in oldplan["seeds"] for p in (0,1)}
    archives={}
    require(len(doc["archives"])==4 and {(r["seed"],r["seat"]) for r in doc["archives"]}==prior_seats,
            "Source artifact census differs")
    for r in doc["archives"]:
        p=RELEASE.parent/(str(r["id"])+".zip")
        require(p.stat().st_size==r["bytes"] and sha(p)==r["sha256"], "Source artifact bytes or SHA differ")
        archives[r["seed"],r["seat"]]=p
    arms=oldplan["arms"]
    seen=set();wins={arm:0 for arm in arms};margins={arm:[] for arm in arms}
    for g in doc["games"]:
        seed,seat,arm=g["seed"],g["seat"],g["arm"]; ident=(seed,seat,arm)
        proof=g["action_support"];timing=g["timing_by_seat"]
        require(ident not in seen and (seed,seat) in prior_seats and arm in arms,"Source arm or seat identity differs")
        seen.add(ident)
        require(str(g["workflow_run"])==str(SOURCE_RUN) and g["status"]=="complete" and g["counted"] and
                g["terrain"]==oldplan["physical_boards"][str(seed)] and g["timing_audit"]["executor_verified_for_scored_attempt"] and
                proof["match_status"]=="complete" and proof["matched_all_steps"] and proof["deployment_sha256"]==oldplan["files"][arm]["sha256"]
                and proof["official_scores_by_seat"]==g["official_scores_by_seat"] and
                proof["n_steps"]==proof["official_transitions"]==g["official_transitions"] and
                0<g["official_transitions"]<=299 and g["replayed_public_observations"]==2*g["official_transitions"]
                and proof["receipt_sha256"]==g["receipt_sha256"], "Incomplete shipped policy, reward, clock or post-game action proof")
        require(len(timing)==2 and g["whole_wall_seconds"]<=1500 and
                all(t["init_seconds"]<=20 and t["observed_wait_seconds_max"]<=3 and
                    t["observed_waits_over_one_second"]<=3 and t["observed_excess_seconds"]<=5.00000001 for t in timing),
                "Source game violates the registered public time budgets")
        with zipfile.ZipFile(archives[seed,seat]) as z:
            for name,key in (("registration.json","registration_sha256"),("receipt.json","receipt_sha256"),("turns.jsonl","tape_sha256")):
                require(hashlib.sha256(z.read(arm+"/"+name)).hexdigest()==proof[key],"Source audit is not linked to its signed archive")
        scores=g["official_scores_by_seat"]
        wins[arm]+=int(scores[seat]>scores[1-seat]);margins[arm].append(scores[seat]-scores[1-seat])
    require(len(seen)==12 and wins[B]==1 and wins[A]==0 and
            abs((sum(margins[B])-sum(margins[A]))/4-599.21)<.011,"Source aggregate differs from the registered development observation")
    return [archives[k] for k in sorted(archives)]


def frozen():
    prior=prior_receipt()
    require(sha(HERE/"clearance.json")=="0a6a7593e934a7d294b3203383dc4e801a0c84ecef1192f6b6b3013c0914b080","Map-clearance receipt changed")
    clearance=read(HERE/"clearance.json")
    require(clearance["seeds"]==list(SEEDS) and clearance["by_seed"]==BOARDS and
            clearance["seed_intersections"]==[] and clearance["physical_intersections"]==[],"Fresh development map clearance differs")
    oldplan=read(PREVIOUS/"protocol.json")
    current=read(old.LIVE/"protocol.json")
    require(current["platform"]==oldplan["platform"],"Platform rules changed")
    # Use the exact installed and tested actor, weights, interpreter, public runner, guide and timing verifier.
    files={k:oldplan["files"][k] for k in (*ARMS,"opponent",A+"_weights",B+"_weights","environment","official_helpers","rules",
                                         "live_registration","live","worker","dockerfile","runner","verifier","guide")}
    for key,path in {
        "clearance":HERE/"clearance.json","clearance_code":HERE/"clearance.py","previous_plan":PREVIOUS/"protocol.json",
        "previous_result":RELEASE,"exposed_matrix":PREVIOUS/"matrix.py","original_settle":PREVIOUS/"settle.py",
        "support_check":PREVIOUS/"audit_menu.py","exporter":PREVIOUS/"export_behavior.py",
        "collector":PREVIOUS.parent/"collect.py","fit":PREVIOUS.parent/"fit.json",
        "fresh_matrix":Path(__file__).resolve(),"fresh_merge":HERE/"merge.py","workflow_helper":HERE/"workflow_cli.py",
        "workflow":ROOT/".github/workflows/sapling-natural-fresh-repeat.yml",
        "initial_configuration":old.CARGO/"direct_original/games/candidate/raw_621758346_1.json.gz",
        "reference_source":ROOT/"sources/master3s2/master3s2.py",
    }.items():
        files[key]=evidence_file(path)
    for i,path in enumerate(prior):files[f"previous_archive_{i}"]=file(path)
    # Each keyed role is verified from the actual repository bytes; the original has a production mirror.
    for k,r in files.items():
        require(r==evidence_file(ROOT/r["path"]),"Frozen dependency mismatch: "+k)
    require(files["opponent"]["sha256"]==files["reference_source"]["sha256"] and
            all(files[k]["sha256"]==old.PARTICIPANTS[k][1] for k in (*ARMS,"opponent")),
            "Production package or original entry changed")
    return dict(schema="sapling-natural-fresh-repeat/1",parent_experiment=dict(run=SOURCE_RUN,commit=SOURCE_COMMIT,
                protocol_sha256=SOURCE_PLAN,settlement_sha256=files["previous_result"]["sha256"]),
        scope="Two fresh development maps; both frozen sampled actors face the intact original Master3s2 on both seats. The independent 100-map exam stays unopened.",
        hypotheses={"primary":"Does the updated Sapling policy win an official whole game on either unseen map? Report the four-attempt child W/D/L and all four scores.",
                    "paired":"Compare all four child and parent actual outcomes; report the differences in official margins by map and seat and by map averaged over its two seats. Opponent and policy draw naturally.",
                    "interpretation":"Two maps and a single draw per arm per seat support descriptive development evidence; they do not estimate a 90% population win rate."},
        endpoint="Retain all eight attempts, including all losses, draws, timeouts and missing infrastructure receipts. Numerical margins only when official final scores exist. No early stopping, side selection or replacement after exposure.",
        seeds=list(SEEDS),expected_games=8,arms=list(ARMS),physical_boards=BOARDS,
        schedule_rule="A=sampled parent 93d70be4, B=sampled child 851c901a; successive tickets AB, BA, BA, AB. Both arms on each map/seat use one host, matching private containers and the same physical CPU set, and run serially.",
        schedule=schedule(),platform=current["platform"],files=files)


def check(protocol_sha):
    require(len(protocol_sha or "")==64 and sha(PLAN)==protocol_sha and
            (HERE/"matrix.sha256").read_text()==protocol_sha+"  protocol.json\n","Unreleased matrix identity")
    require(frozen()==read(PLAN),"Current preconditions or dependencies differ from the frozen protocol")


def verify_block(directory, seed, seat):
    # Reuse the original game's independent official replay, per-round timing and resource verification.
    sys.path.insert(0,str(PREVIOUS))
    import settle
    games={};errors={};directory=Path(directory)
    for arm in ARMS:
        arm_dir=directory/arm
        try:
            item=settle.verify_game(arm_dir,seed,seat,arm,PLAN)
            attempt=arm_dir/"attempt.json"
            receipt=read(arm_dir/"receipt.json")
            if attempt.exists():
                require(read(attempt)==dict(seed=seed,runtime_image=receipt["image_id"],
                                           protocol_sha256=sha(arm_dir/"registration.json")),"Different first-attempt registration")
            else:
                require(receipt.get("strategy_initializations",0)==0,"Strategy exposed with no saved attempt")
            item["timing_audit"]=settle.check_timing(arm_dir,item)
            games[arm]=item
        except (OSError,ValueError,TypeError,KeyError,IndexError,AssertionError) as exc:
            errors[arm]=f"{type(exc).__name__}: {exc}"
    for key in ("image_id","workflow_run"):
        if len({g[key] for g in games.values()})>1: errors[key]="Same-host pair runtime identity mismatch"
    if len(games)==2:
        r=[read(directory/arm/"receipt.json") for arm in ARMS]
        cpu=[{c["cpuset"] for c in t.get("containers",[]) if "cpuset" in c} for t in r]
        if cpu[0]!=cpu[1] or (all(g["status"] in ("complete","forfeit","bilateral_failure") for g in games.values())
                              and (len(cpu[0])!=1 or not next(iter(cpu[0]),None))):
            errors["cpu_set"]="Different or missing per-ticket public CPU set"
    seq=directory/"execution.jsonl"
    timeline=[]
    try:
        timeline=[json.loads(line) for line in seq.read_bytes().splitlines()]
        expected=next(r["order"] for r in schedule() if (r["seed"],r["seat"])==(seed,seat))
        require([r["arm"] for r in timeline]==expected and len({(r["hostname"],r["pid"]) for r in timeline})==1,
                "Different host/process or missing registered arm order")
        require(all(q["start_ns"]<=q["end_ns"] and q["seed"]==seed and q["seat"]==seat and
                    q["matrix_sha256"]==sha(PLAN) and q["receipt_sha256"]==sha(directory/q["arm"]/"receipt.json")
                    for q in timeline) and timeline[0]["end_ns"]<=timeline[1]["start_ns"], "Overlapping or unbound hosted pair")
        if games:require(all(str(v["workflow_run"])==str(timeline[0]["workflow_run"]) for v in games.values()),"Wrong host execution record")
    except (OSError,ValueError,KeyError,StopIteration) as exc:
        errors["sequence"]=str(exc)
    return dict(seed=seed,seat=seat,games=games,host_sequence=timeline,verification_errors=errors)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode",choices=("freeze","list","check","play","verify"))
    ap.add_argument("--matrix-sha");ap.add_argument("--seed",type=int);ap.add_argument("--seat",type=int,choices=(0,1))
    ap.add_argument("--out",type=Path);ap.add_argument("--image")
    args=ap.parse_args()
    if args.mode=="freeze":
        data=old.canonical(frozen())
        if PLAN.exists():require(PLAN.read_text()==data,"Refusing to replace a different map registration")
        else:
            with PLAN.open("x") as f:f.write(data)
        digest=sha(PLAN)+"  protocol.json\n";dest=HERE/"matrix.sha256"
        if dest.exists():require(dest.read_text()==digest,"Different prior digest")
        else:
            with dest.open("x") as f:f.write(digest)
        print(sha(PLAN));return 0
    if args.mode=="list":
        require(sha(PLAN)==args.matrix_sha and read(PLAN)["schedule"]==schedule(),"Unregistered matrix")
        print(json.dumps({"include":[{"seed":r["seed"],"seat":r["seat"]} for r in schedule()]}));return 0
    check(args.matrix_sha)
    if args.mode=="check":return 0
    require(args.seed in SEEDS and args.seat in (0,1) and args.out is not None,"Unregistered ticket")
    if args.mode=="verify":
        result=verify_block(args.out,args.seed,args.seat)
        print(json.dumps(result,allow_nan=False));return int(bool(result["verification_errors"]))
    require(args.image,"Missing frozen image")
    old.play.HERE=HERE
    def checked(matrix):
        require(matrix==read(PLAN),"Matrix changed after check")
        return read(old.LIVE/"protocol.json")
    old.play.check=checked
    previous_run=old.play.live.run
    timeline=args.out/"execution.jsonl"
    require(not timeline.exists(),"Existing pair execution")
    async def measured(params):
        arm=params.out.name
        before=time.monotonic_ns()
        try:
            return await previous_run(params)
        finally:
            receipt=params.out/"receipt.json"
            row=dict(arm=arm,seed=args.seed,seat=args.seat,hostname=socket.gethostname(),
                     pid=os.getpid(),workflow_run=os.getenv("GITHUB_RUN_ID"),matrix_sha256=args.matrix_sha,
                     start_ns=before,end_ns=time.monotonic_ns(),receipt_sha256=sha(receipt) if receipt.exists() else None)
            with timeline.open("a") as f:
                print(json.dumps(row,sort_keys=True,allow_nan=False),file=f,flush=True)
    old.play.live.run=measured
    sys.argv=["play","--seed",str(args.seed),"--seat",str(args.seat),"--out",str(args.out),"--image",args.image]
    return old.play.main()


if __name__=="__main__":
    sys.exit(main())

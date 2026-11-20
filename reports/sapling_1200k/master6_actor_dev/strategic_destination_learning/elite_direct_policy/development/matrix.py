"""Register two immutable neural packages for an original-opponent, four-map development comparison."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ELITE = HERE.parent
STRATEGY = ELITE.parent
BASE = STRATEGY/"development"
REPL = STRATEGY/"replication"
spec = importlib.util.spec_from_file_location("native_strategic_runner",BASE/"matrix.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
import gate
ROOT, play, timing = runner.ROOT, runner.play, runner.timing
read, sha, record, require = runner.read, runner.sha, runner.record, runner.require
ARMS = ("control","elite")
SEEDS = (621758520,621758521,621758522,621758523)
PLAN = HERE/"protocol.json"
RELEASE = HERE/"release.json"
WORKFLOW = "sapling-elite-bc-comparison.yml"
DESIGN = "3f8bde681620bfb246886f99fe8ee6da1fa4b6d7b8b12bd177225e3425bd6fed"
PRIOR = "f09bbf8e8db3504ad5e4abff47adcc11330224d49b9645b5c464265b6861d762"


def contract():
    require(sha(HERE/"design.json")==DESIGN and sha(REPL/"protocol.json")==PRIOR,"Different development registration")
    design = read(HERE/"design.json")
    clear, delta = (read(HERE/n) for n in ("clearance.json","score_delta.json"))
    expected = design["physical_boards"]
    require(clear["by_seed"]==expected==delta["physical_boards"] and clear["source_snapshot"]==delta["through"] and
            not any(x[k] for k in ("seed_intersections","physical_intersections") for x in (clear,delta)) and
            not delta["unresolved_incremental_replays"],"Incomplete fresh-map clearance")
    old = read(REPL/"protocol.json")
    require(not set(SEEDS).intersection(old["seeds"]) and not set(expected.values()).intersection(old["physical_boards"].values()),"Old evaluation overlap")
    learn=read(ELITE/"protocol.json")
    registered=[x for rows in learn["entries"].values() for x in rows]
    require(not {x["seed"] for x in registered}.intersection(SEEDS) and not {x["terrain"] for x in registered}.intersection(expected.values()),
            "Elite learner source terrain overlap")
    require([(r["seed"],r["seat"],r["order"]) for r in design["schedule"]]==
            [(s,p,list(ARMS if (i+p)%2==0 else ARMS[::-1])) for i,s in enumerate(SEEDS) for p in (0,1)],
            "Changed map, seat or order")
    released = read(RELEASE)
    offline = gate.validate(None)
    require(offline["status"]=="ELIGIBLE_LOCALLY_AWAIT_PUBLISHED_MAIN" and
            released == {**offline,"status":"ELIGIBLE","release":released.get("release")} and
            len(released["release"])==40,"Gate or release artifacts changed")
    files={}
    shared=("opponent","official_helpers","rules","environment","ladder_runner","worker","container","base_registration",
            "original_game_configuration","current_guide","match_runner","replayer","native_timing_audit","previous_exposure_guard")
    for k in shared:
        src=old["files"][k]
        require(record(ROOT/src["path"])==src,"Platform/original dependency drift: "+k)
        files[k]=src
    paths={"control":STRATEGY/"bc.py","control_weights":STRATEGY/"bc.npz","elite":ELITE/"trained.py","elite_weights":ELITE/"weights.npz",
           "learning_plan":ELITE/"protocol.json","training_receipt":ELITE/"fit.json","trainer":ELITE/"fit.py",
           "assessment":ELITE/"assessment.json","assessed_actions":ELITE/"assessment_actions.json.gz","gate":HERE/"gate.py", "economic_protocol":HERE/"economy_question.md",
           "design":HERE/"design.json","clearance":HERE/"clearance.json","score_delta":HERE/"score_delta.json","census":HERE/"clearance.py",
           "released_identity":RELEASE,"workflow":ROOT/".github/workflows"/WORKFLOW,"matrix":HERE/"matrix.py","cli":HERE/"workflow_cli.py",
           "settle":HERE/"settle.py","registered_runner":BASE/"matrix.py","registered_sparse":BASE/"workflow_cli.py",
           "replay_auditor":BASE/"settle.py","four_map_auditor":REPL/"settle.py","previous_registration":REPL/"protocol.json"}
    files.update({k:record(v) for k,v in paths.items()})
    require(all(files[k]["sha256"]==design[k+"_sha256"] for k in ("control","elite","control_weights","elite_weights")) and
            files["opponent"]["sha256"]==design["original_sha256"] and files["current_guide"]["sha256"]==design["current_platform_guide_sha256"],
            "Different policy package, weights, guide or original examiner")
    frozen=[paths[n].read_bytes().splitlines(keepends=True) for n in ARMS]
    require(len(frozen[0])==len(frozen[1]) and [(x.startswith(b"_W64 = "),y.startswith(b"_W64 = "))
            for x,y in zip(*frozen) if x!=y]==[(True,True)],"Policies differ in runtime code outside trained tensor payload")
    return {**design,"schema":"sapling-elite-policy-versus-frozen-bc/frozen/1","platform":old["platform"],"files":files}


def check(digest):
    require(sha(PLAN)==digest and read(PLAN)==contract(),"Unregistered data or executable bytes")
    return read(PLAN)


def bind(rev):
    require(len(rev)==40 and subprocess.check_output(["git","-C",str(ROOT),"rev-parse","--verify",rev+"^{commit}"],text=True).strip()==rev,
            "Exact main revision required")
    subprocess.run(["git","-C",str(ROOT),"merge-base","--is-ancestor",rev,"origin/main"],check=True)
    release=gate.validate(rev)
    require(release["status"]=="ELIGIBLE","Offline evidence does not qualify")
    if RELEASE.exists():require(read(RELEASE)==release,"Different previously released candidate")
    else:play.live.save(RELEASE,release)
    plan=contract()
    if PLAN.exists():require(read(PLAN)==plan,"Different previously registered evaluation")
    else:play.live.save(PLAN,plan)
    print(sha(PLAN))


runner.HERE,runner.PLAN,runner.SEEDS,runner.ARMS,runner.check=HERE,PLAN,SEEDS,ARMS,check
verify_block=runner.verify_block

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("op",choices=("bind","check","list","play","verify"))
    p.add_argument("--release");p.add_argument("--matrix-sha");p.add_argument("--seed",type=int);p.add_argument("--seat",type=int,choices=(0,1))
    p.add_argument("--out",type=Path);p.add_argument("--image");a=p.parse_args()
    if a.op=="bind":bind(a.release)
    elif a.op=="play":sys.exit(runner.run(a.matrix_sha,a.seed,a.seat,a.out,a.image))
    else:
        plan=check(a.matrix_sha)
        if a.op=="list":
            print(json.dumps({"include":[{"seed":r["seed"],"seat":r["seat"]} for r in plan["schedule"]]}))
        if a.op=="verify":
            info=verify_block(a.out,a.seed,a.seat);print(json.dumps(info,allow_nan=False));sys.exit(int(bool(info["verification_errors"])))

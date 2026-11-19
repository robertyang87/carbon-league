"""Bind the three exact packages to a repeated, two-map, both-seat original-opponent diagnosis."""
import argparse
import gzip
import itertools
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
LEAGUE = HERE.parent
CARGO = HERE.parents[2]
ROOT = CARGO.parents[3]
ORIGINAL_RUNNER = CARGO / "territory_sequence/fresh_original"
LIVE = CARGO / "ladder_live"
sys.path.insert(0, str(ORIGINAL_RUNNER))
import play

ARMS = ("greedy_parent","sampled_parent","sampled_child")
PARENT, PARENT_DRAW, CHILD_DRAW = ARMS
SEEDS = (621758345, 621758346)
ORDER = ((0,1,2),(1,2,0),(2,0,1),(2,1,0))
PARTICIPANTS = {
    PARENT: (CARGO/"candidate.py","8da4b465f7fa6e601e51bd7c2e81f25825cb59b9b5ab285f7f56d0f6bbfe1cac"),
    PARENT_DRAW: (HERE/"parent_sampled.py","93d70be43a8cc358c05e4ac3d0870bded1e720aa539208ed0550523fc718e789"),
    CHILD_DRAW: (HERE/"candidate_sampled.py","851c901aa561ce80bf04d03e7c2a8e7876bfa2f24dc6c57ffe22b501979320ee"),
    "opponent": (ROOT/"data/master3s2_source.py","65f268b2919ce30fe19355115a928eada032a6beaa018d801662d434c083f2b5")
}
sha = play.live.sha

def require(ok, msg):
    if not ok: raise ValueError(msg)

def read(p):
    return json.loads(p.read_bytes())

def file(p):
    return {"path":p.relative_to(ROOT).as_posix(),"sha256":sha(p),"bytes":p.stat().st_size}

def canonical(x):
    return json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+"\n"

def frozen():
    base=read(LIVE/"protocol.json")
    files={k:file(p) for k,(p,h) in PARTICIPANTS.items()}
    for k,(_,expected) in PARTICIPANTS.items():
        require(files[k]["sha256"]==expected and files[k]["bytes"]<=10_000_000,"Wrong package or size: "+k)
    official="sources/master3s2/master3s2.py"
    mirror=ROOT/official
    original=mirror.read_bytes() if mirror.is_file() else subprocess.check_output(["git","-C",str(ROOT),"show","HEAD:"+official])
    require(original==(ROOT/files["opponent"]["path"]).read_bytes(),"Original differs from registered public source")
    files.update({k:base["files"][k] for k in ("environment","official_helpers","rules")})
    static={"parent_weights":CARGO/"candidate.npz","student_weights":LEAGUE/"candidate.npz",
            "fit":LEAGUE/"fit.json","held":LEAGUE/"evidence/results.json","training":LEAGUE/"protocol.json",
            "historical_exam":CARGO/"direct_original/protocol.json","live_registration":LIVE/"protocol.json",
            "live":LIVE/"live.py","worker":LIVE/"worker.py","dockerfile":LIVE/"Dockerfile",
            "runner":ORIGINAL_RUNNER/"play.py","verifier":ORIGINAL_RUNNER/"verify_pair.py",
            "adapter":HERE/"matrix.py","guide":HERE/"platform_guide.md"}
    files.update({k:file(p) for k,p in static.items()})
    require(all(file(ROOT/f["path"])==f for f in files.values()),"Dependency hash mismatch")
    fit=read(static["fit"])
    require(fit["parent_sha256"]==files[PARENT]["sha256"] and fit["parent_weights_sha256"]==files["parent_weights"]["sha256"]
            and fit["export"]["weights_sha256"]==files["student_weights"]["sha256"]
            and fit["package_sha256"]=="510c527a6f3e88f718e71e4b456e6e065abaeb1474587dc76fd11a5744d7c3d4","Unexpected source checkpoint")
    require(read(static["held"])["pass_held_league"] is False,"Unexpected historical greedy result")
    require(files["guide"]["sha256"]==read(static["training"])["guide_sha256"],"Current guide snapshot changed")
    files["sampler"]=file(CARGO/"onpolicy_native/training_policy.py")
    files["training_registration"]=file(LEAGUE/"collection_registration.json")
    files["trained_greedy_source"]=file(LEAGUE/"candidate.py")
    require(files["trained_greedy_source"]["sha256"]==fit["package_sha256"],"Underlying trained source differs")
    for arm in ARMS: files[arm+"_weights"]=files["student_weights" if arm==CHILD_DRAW else "parent_weights"]
    for arm,tag,underlying in ((PARENT_DRAW,"parent",files[PARENT]),(CHILD_DRAW,"candidate",files["trained_greedy_source"])):
        audit=HERE/(tag+"_audit.json"); draws=HERE/(tag+"_natural.json")
        files[tag+"_audit"]=file(audit); files[tag+"_natural"]=file(draws)
        result=read(audit); stamp=result["deployment"]; natural=read(draws)
        require(stamp["sha256"]==files[arm]["sha256"]==natural["package_sha256"]
                and stamp["source_sha256"]==underlying["sha256"] and stamp["training_weights_sha256"]==files[arm+"_weights"]["sha256"]
                and stamp["training_sampler_sha256"]==files["sampler"]["sha256"] and stamp["registration_sha256"]==files["training_registration"]["sha256"]
                and stamp["schedule"]==list(range(12,277,24)), "Sampling adapter or release provenance differs")
        require(result["totals"]["alternate"]>0 and {r["seat"] for r in natural["training_roots"]}=={0,1}
                and all(len(row["options"])==2 and all(n>0 for n in row["counts"]) for row in natural["training_roots"]), "Both physical choices need support")
    boards={}
    for s in SEEDS:
        history=CARGO/f"direct_original/games/candidate/raw_{s}_1.json.gz"
        raw=json.loads(gzip.decompress(history.read_bytes()))
        require(raw["seed"]==s and len(raw["actions"])==raw["steps"],"Wrong historical receipt")
        files["prior_"+str(s)]=file(history)
        boards[str(s)]=raw["initial_terrain_u8_sha256"]
    require(len(set(boards.values()))==2 and set(SEEDS)==set(read(static["historical_exam"])["seeds"]),"Incorrect historical maps")
    schedule=[dict(seed=s,seat=seat,order=[ARMS[i] for i in ORDER[2*idx+seat]])
              for idx,s in enumerate(SEEDS) for seat in (0,1)]
    return dict(schema="sapling-natural-original/1",scope="Exposed-map development comparison. Two known boards, both physical seats, each of three models on each seat. One naturally sampled trajectory per arm and seat; no inference about population win rate. Training, independent development and the formal examination are distinct.",
                hypotheses={"sampling":"sampled_parent minus greedy_parent, same weights","weight_update":"sampled_child minus sampled_parent, same sampler","combined":"sampled_child minus greedy_parent"},
                endpoint="Record all 12 attempts, including infrastructure failures, timeouts, forfeits, draws, terminal official scores where defined, and wall clocks. No replacement or early stopping. Do not assign a made-up official numerical margin to a forfeit with no official score.",
                sample_contract="Natural game entropy in the packaged policy and unchanged examiner. No global RNG seed is injected into either policy. Replaying a command tape recovers official state and score; an independently drawn command need not match. The offline action-menu check uses the actual public history.",
                protocol_prior="The separate held greedy test for 510c failed and remains closed. These new stochastic policies test different deployed behaviors on two previously exposed maps; 444/445 and all final maps are excluded.",
                schedule_rule="Three cyclic Latin-square rows followed by reversal CBA on the fourth seed/seat. Four map-seat blocks cannot exercise all six permutations. Each arm appears in all three positions; within each position arm counts differ by at most one.",
                possible_orders=[list(q) for q in itertools.permutations(ARMS)],seeds=list(SEEDS),physical_boards=boards,expected_games=12,arms=list(ARMS),schedule=schedule,platform=base["platform"],files=files)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("freeze","check","play","verify"))
    parser.add_argument("--matrix-sha");parser.add_argument("--seed",type=int);parser.add_argument("--seat",type=int,choices=(0,1))
    parser.add_argument("--out",type=Path);parser.add_argument("--image")
    a=parser.parse_args()
    matrix=frozen(); path=HERE/"protocol.json"
    if a.mode=="freeze":
        expected=canonical(matrix)
        if path.exists(): require(path.read_text()==expected,"Different registration exists")
        else:
            with path.open("x") as f:f.write(expected)
        expected=sha(path)+"  protocol.json\n"; checksum=HERE/"matrix.sha256"
        if checksum.exists():require(checksum.read_text()==expected,"Digest already differs")
        else:
            with checksum.open("x") as f:f.write(expected)
        return 0
    require(a.matrix_sha==sha(path) and matrix==read(path),"Registration or SHA changed")
    if a.mode=="check":return 0
    require(a.seed in SEEDS and a.seat in (0,1) and a.out is not None,"Unregistered map or seat")
    if a.mode=="verify":
        from settle import verify_triple
        result=verify_triple(a.out,a.seed,a.seat,path)
        print(json.dumps(result,allow_nan=False))
        return int(bool(result["verification_errors"]))
    require(a.image,"Registered container image required")
    play.HERE=HERE
    def check(plan):
        require(plan==matrix,"Registered session changed")
        return read(LIVE/"protocol.json")
    play.check=check
    sys.argv=["play","--seed",str(a.seed),"--seat",str(a.seat),"--out",str(a.out),"--image",a.image]
    return play.main()

if __name__=="__main__":sys.exit(main())

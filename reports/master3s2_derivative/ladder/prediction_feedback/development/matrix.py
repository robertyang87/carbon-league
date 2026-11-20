"""Counterbalance two frozen packages on two new maps with the external ladder referee."""
import argparse
import gzip
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
FEEDBACK = HERE.parent
FAMILY = HERE.parents[2]
ROOT = next(p for p in HERE.parents if (p / ".git").exists())
STRATEGY = ROOT / "reports/sapling_1200k/master6_actor_dev/strategic_destination_learning"
BASE = STRATEGY / "development"
SPEC = importlib.util.spec_from_file_location("registered_ladder_runner", BASE / "matrix.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)
play, timing, read, sha, record, require = runner.play, runner.timing, runner.read, runner.sha, runner.record, runner.require
PLAN = HERE / "protocol.json"
SEEDS = (621758592, 621758593)
ARMS = ("parent", "calibrated")
WORKFLOW = "master3s2-feedback-development.yml"
V3 = "8b4c4bafba2b3b2887139163bd6397b3f7d1acf178a43ea1756d8f80306efe43"
V2 = "7cf5c6c91fc59b46b2a38924c4c5ab56143fb1242d082f9426303af7336ac3ed"
ORIGINAL = "65f268b2919ce30fe19355115a928eada032a6beaa018d801662d434c083f2b5"
PREVIOUS_PROTOCOL = "a9990ea7f5066ea7351f2f882011275978800d25e7130fd5730779ef8d3df164"
GUIDE = "97adbc81e39ef2b1834af96e261cd10e76424352abde5e04868b3467c2a5cec8"



def contract():
    previous_path = FAMILY/"cold_start/development/protocol.json"
    require(sha(previous_path) == PREVIOUS_PROTOCOL, "Different original/referee reference")
    previous = read(previous_path)
    keys = ("opponent", "official_helpers", "rules", "environment", "ladder_runner", "worker", "container",
            "base_registration", "original_game_configuration", "current_guide", "match_runner", "replayer",
            "native_timing_audit", "previous_exposure_guard", "registered_runner", "registered_settler",
            "registered_sparse", "observation_replay")
    files = {k:previous["files"][k] for k in keys}
    require(all(record(ROOT/r["path"])==r for r in files.values()), "Referee or original source changed")
    release = read(FEEDBACK/"package/manifest.json")
    other = {"parent":FAMILY/"cold_start/candidate.py","parent_weights":FAMILY/"opponent_policy/weights.npz",
             "calibrated":FEEDBACK/"package/candidate.py","calibrated_weights":FEEDBACK/"weights.json",
             "release":FEEDBACK/"package/manifest.json","builder":FEEDBACK/"package/build.py",
             "feedback_runtime":FEEDBACK/"runtime.py","fit_protocol":FEEDBACK/"protocol.json",
             "readout_train":FEEDBACK/"train_result.json","readout_held":FEEDBACK/"held_result.json",
             "v2_registration":previous_path,"conditional_registration":FEEDBACK/"package/conditional_protocol.json",
             "conditional_work":FEEDBACK/"package/conditional_roots.json.gz", "conditional_runner":FEEDBACK/"package/conditional.py",
             "clearance":HERE/"clearance.json","clearance_calculator":HERE/"clearance.py",
             "additional_checks":HERE/"inventory.json","additional_checker":HERE/"inventory.py",
             "workflow":ROOT/".github/workflows"/WORKFLOW,"matrix":HERE/"matrix.py",
             "cli":HERE/"workflow_cli.py","settler":HERE/"settle.py"}
    files.update({key:record(path) for key,path in other.items()})
    require(files["opponent"]["sha256"]==ORIGINAL and files["current_guide"]["sha256"]==GUIDE, "Wrong reference")
    require(release["parent_sha256"]==files["parent"]["sha256"]==V2 and
            release["candidate_sha256"]==files["calibrated"]["sha256"]==V3 and release["bytes"]==files["calibrated"]["bytes"]<=10_000_000,
            "Wrong deployment bytes")
    for field,key in (("weights_sha256","calibrated_weights"),("protocol_sha256","fit_protocol"),("held_result_sha256","readout_held"),
                      ("runtime_sha256","feedback_runtime"),("builder_sha256","builder")):
        require(release[field]==files[key]["sha256"], "Release evidence differs: "+key)
    require(read(other["readout_held"])["verdict"]=="PASS","Independent action prediction gate failed")
    proof=json.loads(gzip.decompress(other["conditional_work"].read_bytes()))
    require(proof["candidate_sha256"]==V3 and proof["parent_sha256"]==V2 and
            proof["protocol_sha256"]==files["conditional_registration"]["sha256"] and
            proof["probe_sha256"]==files["conditional_runner"]["sha256"] and
            proof["counts"]["attributable_changed"]==2 and len(proof["roots"])==8,
            "Registered deployment mechanism evidence differs")
    census=read(other["clearance"]); boards=census["by_seed"]; extra=read(other["additional_checks"])
    require(set(map(int,boards))==set(SEEDS) and len(set(boards.values()))==len(SEEDS)
            and not census["seed_intersections"] and not census["physical_intersections"]
            and len(extra["ladders"])==10 and extra["boards"]==boards and extra["clearance_sha256"]==files["clearance"]["sha256"]
            and extra["verifier_sha256"]==files["additional_checker"]["sha256"] and extra["passed"],
            "Historical board or seed is unresolved")
    schedule=[]
    for i,seed in enumerate(SEEDS):
        for seat in (0,1):
            schedule.append(dict(seed=seed,seat=seat,order=list(ARMS if (i+seat)%2==0 else reversed(ARMS))))
    return dict(schema="master3s2-paired-ladder-feedback-development/1",seeds=list(SEEDS),arms=list(ARMS),
                schedule=schedule,physical_boards=boards,files=files,platform=previous["platform"], registration_class="development",
                examiner=previous["examiner"],
                scope="Two new maps, both seats, both frozen versions face the original in alternating order on the same isolated host. Eight games, four per arm. Each game uses fresh, natural agent RNG, with the same map but distinct trajectories. No outcome-triggered extension or replacement.",
                outcomes="All eight tickets, bilateral faults and own/opponent terminals. Disclose numeric margin only when official scores exist. Report per-arm wins over four, per-seat and two physical-map estimates of paired official margin and win fraction. Descriptive development only; final exam remains unused.")


def check(digest):
    require(sha(PLAN)==digest and read(PLAN)==contract(), "Protocol or registered bytes changed")
    return read(PLAN)


runner.HERE,runner.PLAN,runner.SEEDS,runner.ARMS,runner.check = HERE,PLAN,SEEDS,ARMS,check
verify_block=runner.verify_block


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("freeze","list","check","play","verify","observations"))
    p.add_argument("--matrix-sha");p.add_argument("--seed",type=int,choices=SEEDS);p.add_argument("--seat",type=int,choices=(0,1))
    p.add_argument("--out",type=Path);p.add_argument("--image")
    a=p.parse_args()
    if a.mode=="freeze":
        actual=contract()
        if PLAN.exists():require(read(PLAN)==actual,"Different existing freeze")
        else:play.live.save(PLAN,actual)
        print(sha(PLAN))
    else:
        plan=check(a.matrix_sha)
        if a.mode=="list":print(json.dumps({"include":[{"seed":v["seed"],"seat":v["seat"]} for v in plan["schedule"]]}))
        if a.mode in ("play","verify","observations"):
            require(any((r["seed"],r["seat"])==(a.seed,a.seat) for r in plan["schedule"]),"Wrong ticket")
            if a.mode=="play":sys.exit(runner.run(a.matrix_sha,a.seed,a.seat,a.out,a.image))
            if a.mode=="verify":
                result=verify_block(a.out,a.seed,a.seat);print(json.dumps(result,allow_nan=False))
                sys.exit(int(bool(result["verification_errors"])))
            sys.path.append(str(STRATEGY/"native_action_onpolicy_intake"))
            from observations import recover
            for arm in ARMS:
                folder=a.out/arm
                if (folder/"turns.jsonl").is_file():
                    (folder/"observations.jsonl.gz").write_bytes(recover(folder))

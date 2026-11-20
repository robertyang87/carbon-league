"""Freeze and run four natural original-vs-cold-repair games on two untouched development boards."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
COLD = HERE.parent
FAMILY = COLD.parent
ROOT = next(p for p in HERE.parents if (p / ".git").exists())
STRATEGY = ROOT / "reports/sapling_1200k/master6_actor_dev/strategic_destination_learning"
BASE = STRATEGY / "development"
SPEC = importlib.util.spec_from_file_location("registered_ladder_runner", BASE / "matrix.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)
play, timing, read, sha, record, require = runner.play, runner.timing, runner.read, runner.sha, runner.record, runner.require
PLAN = HERE / "protocol.json"
SEEDS = (621758590, 621758591)
ARM = "cold_repair"
ARMS = (ARM,)
WORKFLOW = "master3s2-cold-repair-development.yml"
V1 = "245b5814d1855c5449435ee6bd4a53560017a8cc7a46d08729fa9dfdc13e9654"
V2 = "7cf5c6c91fc59b46b2a38924c4c5ab56143fb1242d082f9426303af7336ac3ed"
ORIGINAL = "65f268b2919ce30fe19355115a928eada032a6beaa018d801662d434c083f2b5"
PREVIOUS_PROTOCOL = "4468d809a4c308f5ccaf056963f6ad7f240a323c8cc70e312ff1cd43eedc1dde"
GUIDE = "97adbc81e39ef2b1834af96e261cd10e76424352abde5e04868b3467c2a5cec8"


def contract():
    prior_path = FAMILY / "dev1/protocol.json"
    require(sha(prior_path) == PREVIOUS_PROTOCOL, "Different original/referee reference")
    prior = read(prior_path)
    keys = ("opponent", "official_helpers", "rules", "environment", "ladder_runner", "worker", "container",
            "base_registration", "original_game_configuration", "current_guide", "match_runner", "replayer",
            "native_timing_audit", "previous_exposure_guard", "registered_runner", "registered_settler",
            "registered_sparse", "observation_replay")
    files = {k: prior["files"][k] for k in keys}
    require(all(record(ROOT / r["path"]) == r for r in files.values()), "Original or referee changed")
    require(files["opponent"]["sha256"] == ORIGINAL and files["current_guide"]["sha256"] == GUIDE,
            "Wrong original or platform requirements")
    additions = {ARM: COLD / "candidate.py", ARM + "_weights": FAMILY / "opponent_policy/weights.npz",
                 "v1": FAMILY / "opponent_policy/candidate.py", "v1_registration": prior_path,
                 "v2_builder": COLD / "build.py", "v2_release": COLD / "release.json",
                 "cold_probe": COLD / "hosted_probe/results.json", "clearance": HERE / "clearance.json",
                 "clearance_increment": HERE / "clearance_at_freeze.json", "clearance_extension": HERE / "extend_clearance.py",
                 "clearance_verifier": HERE / "clearance.py", "matrix": HERE / "matrix.py",
                 "cli": HERE / "workflow_cli.py", "settler": HERE / "settle.py",
                 "workflow": ROOT / ".github/workflows" / WORKFLOW}
    files.update({k:record(p) for k,p in additions.items()})
    release = read(additions["v2_release"])
    require((files["v1"]["sha256"], files[ARM]["sha256"]) == (V1,V2)
            and files[ARM]["bytes"] == release["candidate_bytes"] <= 10_000_000
            and release["parent"] == V1 and release["candidate"] == V2
            and files[ARM+"_weights"] == prior["files"]["opponent_policy_weights"],"Different release or learned weights")
    p1, p2 = additions["v1"].read_bytes(), additions[ARM].read_bytes()
    old, new = release["old_line"].encode(), release["new_line"].encode()
    require(p2.count(new) == 1 and p1.count(old) == 1 and p2.replace(new,old) == p1, "Candidate changed outside the frozen warmup")
    probe = read(additions["cold_probe"])
    probe_rows = [r for r in probe["summary"]["rows"] if r["arm"] == "v2"]
    require(probe["run"] == 44671315099 and len(probe_rows) == 4 and
            all(r["status"] == "complete" and r["init"] == "ok" and r["step_zero"] == "ok" for r in probe_rows),
            "Hosted cold-start probe does not bind four completed V2 loads")
    c = read(additions["clearance"]); boards=c["by_seed"]
    require(set(boards) == {str(s) for s in SEEDS} and len(set(boards.values())) == 2
            and not c["seed_intersections"] and not c["physical_intersections"] and
            not set(boards.values()) & set(prior["physical_boards"].values()), "Development board overlap")
    increment = read(additions["clearance_increment"])
    require(increment["source_census_sha256"] == files["clearance"]["sha256"] and increment["by_seed"] == boards
            and increment["verifier_sha256"] == files["clearance_extension"]["sha256"]
            and not increment["seed_intersections"] and not increment["physical_intersections"], "Incremental map evidence is unresolved")
    return dict(schema="master3s2-cold-repair-development/1", seeds=list(SEEDS), arms=list(ARMS),
                registration_class="development", physical_boards=boards, files=files, platform=prior["platform"],
                schedule=[dict(seed=s,seat=p,order=[ARM]) for s in SEEDS for p in (0,1)],
                scope="Four new development tickets. No replacement, outcome selection, extension or extrapolation to the 200-game final. Keep V1 results separate.",
                examiner="Original public entry, T=1, natural draws, native 0.58s search. Separate 1.5-CPU/512-MiB offline cgroups; pair of 20s cold starts, full external request timers including step zero.",
                outcomes="All registered tickets and unresolved bilateral faults; numeric official margin only where the official referee produced a score. Report seats and both physical-map clusters.")


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
            folder=a.out/ARM;(folder/"observations.jsonl.gz").write_bytes(recover(folder))

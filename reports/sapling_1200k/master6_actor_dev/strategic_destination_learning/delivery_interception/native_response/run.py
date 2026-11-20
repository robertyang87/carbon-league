"""Run two prespecified retired-map prefix/suffix comparisons against the original examiner."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
M6 = HERE.parents[2]
INTAKE = M6 / "cargo_tree_action_learning_dev/native_actor_corpus/league_policy/native_return_intake"
TERMINAL = INTAKE / "economy_proposals/terminal_intake"
sys.path[:0] = [str(TERMINAL), str(HERE.parent)]
import menu


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


legacy = load("signed_semantic_runner", TERMINAL / "run.py")
play = legacy.play
runner = legacy.runner
need = legacy.require
student_state = runner.student_state
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()


def propose(student, obs, cfg, category):
    if "full_carbon" in obs or category != "incoming_delivery":
        raise ValueError("Only an acting seat's public observation is accepted")
    tick, cpu = time.perf_counter(), time.process_time()
    own, event = runner.action(student, obs, cfg)
    selected = menu.select(obs, own["order"])
    if selected is not None:
        actions = selected["options"]
        need(actions[0] == own["order"] and actions[0] != actions[1], "Wrong native A or unchanged B")
        prob = event["p"][event["pick"]] if event else 1.
        selected.update(native_A=own["order"], native_choice=event, native_A_probability=float(prob),
                        parent_after_sha256=student_state.fingerprint(student),
                        parent_rng_after_sha256=play.student_history.behavior_rng_snapshot(student),
                        public_observation_sha256=play.sha(play.encode(obs)))
    measured = {**own, "native_wall":own["wall"], "native_cpu":own["cpu"],
                "wall":time.perf_counter()-tick, "cpu":time.process_time()-cpu}
    return measured, selected


def issue_student(student, obs, cfg, index, category, expected):
    measurement, selected = propose(student, obs, cfg, category)
    need(index in (0, 1) and selected == expected, "Restored student produced a different offer")
    return {**measurement, "order":dict(selected["options"][index])}, {**selected, "pick":index}


class InterceptionSelector:
    probe = staticmethod(propose)
    issue_student = staticmethod(issue_student)


def check():
    plan = play.read(HERE/"protocol.json")
    trace = {name:sha(play.ROOT/name) for name in plan["sources"]}
    need(trace == plan["sources"], "Frozen code, board registry or deployed package changed")
    intake = play.read(INTAKE/"protocol.json")
    need([[r["seed"],r["seat"],r["arm_order"]] for r in plan["fixtures"]] ==
         [[621758460,0,["A","B"]],[621758461,1,["B","A"]]], "Wrong retired seats or arm order")
    need(all(r["board_sha256"]==intake["physical_boards"][str(r["seed"])] for r in plan["fixtures"]),
         "Training terrains disagree")
    need(plan["probe_turns"] == list(range(40,241)), "Changed first-eligible window")
    return plan, trace


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check",action="store_true")
    p.add_argument("--seed",choices=(621758460,621758461),type=int)
    args=p.parse_args()
    plan,trace=check()
    if args.check:
        print("bound",len(trace),"files and both training tickets")
        return
    need(all(os.environ.get(name)=="1" for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS")),"Thread limits")
    legacy.select_root = InterceptionSelector
    runner.set_affinity()
    for f in plan["fixtures"]:
        if args.seed is not None and args.seed != f["seed"]:
            continue
        folder=HERE/"outputs"/f"{f['seed']}_{f['seat']}"
        folder.mkdir(parents=True,exist_ok=True)
        release=plan["frozen_source_commit"]
        with (folder/"claim.json").open("x") as stream:
            json.dump(dict(seed=f["seed"],seat=f["seat"],terrain=f["board_sha256"],
                      category=f["category"],arm_order=f["arm_order"],release=release,
                      pid=os.getpid(),time=runner.utc(),protocol_sha256=sha(HERE/"protocol.json"),source_sha256=trace),stream)
        try:
            legacy.play_fixture(f,plan,folder,release,trace)
        except BaseException as error:
            play.save(folder/"failure.json",dict(time=runner.utc(),exception=repr(error),traceback=traceback.format_exc()))
            raise


if __name__=="__main__":
    main()

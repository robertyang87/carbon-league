"""Reuse eight signed receipts and run exactly four full local-head games on the exposed boards."""
import argparse
from collections import Counter
import fcntl
import importlib.util
import os
from pathlib import Path
from statistics import mean, stdev
import sys
import time

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "balanced_return_dev"
OUT = HERE / "development"
SEEDS = (621684210, 621684211)
SOURCE = "30edfdf836bf5309b3f21b3db4a98bff8af1d6b918f103f5a69ecbd68a998043"
ENGINE = "ba0dcd85adeeacf1f85b85c03c5886b618b9fd47f1c2d169bd6019cabe86227e"
PARENT = "c5ad3f43959a4812e84e89574ee0c6e69765ebb06a8b3ec93f839b0dde128dd3"
UNGATED = "7c7878ede1ef27a9ab2fe79ddef082ad40fd8d6139faa467e3bd819ab57f8cce"
OPPONENT = "9c54df17eba7cfecf6ac2698eccbbadc2607665e16ab613925188131ce5514e0"

sys.path.insert(0, str(OLD))
import common
import worlds
import return_accounts

read, write, sha = common.read, common.write, common.sha


def check(ok, message):
    if not ok:
        raise ValueError(message)


def load_local(name, file):
    spec = importlib.util.spec_from_file_location(name, file)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def signed_engine():
    check(sha(OLD/"development.py") == ENGINE, "prior recovery and validation source changed")
    eng = load_local("balanced_native_development_reused", OLD/"development.py")
    eng.OUT = OUT
    old_commands = eng.commands

    def actions(packs, obs, seat, t):
        candidate = packs["candidate"].module
        if t == 0:
            true_slot = candidate._local_slot
            candidate._observed_slots = []
            def observe_return(x, aux, enc, step):
                actual = true_slot(x, aux, enc, step)
                uid = candidate._FEAT.own_uid[actual] if actual >= 0 else None
                candidate._observed_slots.append((int(step), int(actual), uid))
                return actual
            candidate._local_slot = observe_return
        n = len(candidate._observed_slots)
        row, enemy = old_commands(packs, obs, seat, t)
        if t in (205,271):
            check(len(candidate._observed_slots) == n+1, "expected exactly one decision-time focal-slot result")
            k,slot,uid=candidate._observed_slots[-1]
            check(k == t and (uid is not None) == (slot >= 0),"candidate focus result inconsistent with current public worker")
            row["local_endpoint"]=dict(step=t,slot=slot,uid=uid)
        else:
            check(len(candidate._observed_slots) == n, "out-of-window local head evaluated")
        if t not in (205,271):
            check(row["candidate"] == row["parent"],"different physical actions outside preregistered local-head support")
        row["ungated"] = packs["ungated"].observe(obs[seat])
        return row, enemy

    eng.commands = actions
    return eng


def load_tickets():
    protocol = HERE/"protocol.json"
    reg=read(protocol)
    check(reg["maps"] == list(SEEDS) and reg["physical_seats"] == [0,1] and reg["windows"] == [205,271]
          and reg["diagnosis_only"] and reg["endpoint_only"] and reg["references"]["historical_comparison"] == SOURCE
          and reg["references"]["parent"] == PARENT and reg["references"]["ungated"] == UNGATED
          and reg["references"]["maps"] == sha(OLD/"maps.json") and reg["references"]["screen"] == sha(OLD/"screen.json"),
          "unregistered diagnostic or comparison")
    release = read(HERE/"release.json"); mech = read(HERE/"mechanism.json")
    candidate = HERE/"candidate.py"
    check(release["protocol_sha256"] == mech["protocol_sha256"] == sha(protocol), "different protocol")
    check(release["candidate_sha256"] == mech["candidate_sha256"] == sha(candidate), "different candidate")
    check(sha(common.PARENT) == PARENT == release["parent_sha256"] and
          release["ungated_candidate_sha256"] == UNGATED and
          release["trained_checkpoint_sha256"] == sha(OLD/"candidate.npz") == reg["references"]["weights"],
          "parent, ungated or trained checkpoint mismatch")
    check(sha(common.IT50) == OPPONENT and sha(OLD/"candidate.py") == UNGATED, "ungated actor or proxy changed")
    check(type(mech["passed"]) is bool, "ambiguous mechanism verdict")
    frame = read(OLD/"development/comparison.json")
    check(sha(OLD/"development/comparison.json") == SOURCE and frame["status"] == "complete"
          and frame["binding"]["opponent_sha256"] == OPPONENT and frame["binding"]["parent_sha256"] == PARENT
          and frame["binding"]["candidate_sha256"] == UNGATED and tuple(worlds.registry()["development"]) == SEEDS,
          "wrong previous characterization or boards")
    binding = dict(protocol_sha256=sha(protocol), release_sha256=sha(HERE/"release.json"),
                   mechanism_sha256=sha(HERE/"mechanism.json"), candidate_sha256=sha(candidate),
                   parent_sha256=PARENT, ungated_sha256=UNGATED, opponent_sha256=OPPONENT,
                   source_comparison_sha256=SOURCE, prior_runner_sha256=ENGINE,
                   runner_sha256=sha(Path(__file__)), economy_runner_sha256=sha(HERE/"economy.py"),
                   terrain_sha256={str(s):worlds.locked_terrain(s,"development") for s in SEEDS})
    return mech["passed"], frame, binding


def physical_receipt(case, file, seat):
    off = case["official"]
    return dict(path=str(file.relative_to(HERE.parent)),tape_sha256=sha(file),
                actions_sha256=case["actions_sha256"],scores_by_physical_seat=off["scores"],
                outcome=off["result_by_seat"][seat], steps=off["steps"], termination=off["termination"],
                margin=off["margin_by_seat"][seat], u=off["u_by_seat"][seat],attribution=case["attribution"])


def historical(seed,seat,label,prior,eng):
    old_arm = "candidate" if label == "ungated" else "parent"
    item = prior["arms"][old_arm]
    path = OLD/"development"/f"{old_arm}_{seed}_{seat}.json.gz"
    check(sha(path) == item["tape_sha256"], "signed historical tape changed")
    case = read(path)
    t = case["ticket"]
    check((t["arm"],t["seed"],t["seat"]) == (old_arm,seed,seat) and
          t["binding"]["opponent_sha256"] == OPPONENT and t["binding"]["candidate_sha256"] == UNGATED
          and t["binding"]["parent_sha256"] == PARENT, "historical subject changed")
    eng.validate_case(case,t,return_accounts)
    return case,path


def aggregate(rows,arm):
    c = Counter(r["arms"][arm]["outcome"] for r in rows)
    return dict(games=len(rows),wins=c["win"],draws=c["draw"],losses=c["loss"],
                mean_margin=mean(r["arms"][arm]["margin"] for r in rows),
                mean_u=mean(r["arms"][arm]["u"] for r in rows))


def paired(rows,base):
    items = []
    for seed in SEEDS:
        matching = [r for r in rows if r["seed"] == seed]
        check(len(matching)==2 and set(r["seat"] for r in matching)=={0,1}, "incomplete map cluster")
        items.append(dict(seed=seed,**{f"delta_{k}":mean(r["arms"]["local"][k]-r["arms"][base][k] for r in matching)
                                     for k in ("u","margin")}))
    return dict(map_means=items,**{f"mean_delta_{k}":mean(r[f"delta_{k}"] for r in items) for k in ("u","margin")},
                two_map_cluster_se={k:stdev(r[f"delta_{k}"] for r in items)/(2**.5) for k in ("u","margin")})


def run():
    check(os.sched_getaffinity(0)=={2} and os.getpriority(os.PRIO_PROCESS,0)==19 and
          all(os.environ.get(k)=="1" for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS")),
          "registered one-thread CPU2 nice19 allocation is required")
    passed,source,binding=load_tickets()
    OUT.mkdir(exist_ok=True)
    if not passed:
        write(OUT/"skip.json",dict(status="mechanism_failed",new_games=0,binding=binding))
        return
    eng=signed_engine()
    economy=load_local("balanced_local_economy",HERE/"economy.py")
    write(OUT/"binding.json",binding)
    indexed={(r["seed"],r["seat"]):r for r in source["pairs"]}
    check(len(source["pairs"])==len(indexed)==4,"unregistered number of signed physical pairs")
    paths=dict(parent=common.PARENT,candidate=HERE/"candidate.py",opponent=common.IT50,ungated=OLD/"candidate.py")
    rows=[]; expenses=[]
    for seed in SEEDS:
        for seat in (0,1):
            cases={}; arms={}
            for a in ("parent","ungated"):
                case,path=historical(seed,seat,a,indexed[seed,seat],eng);cases[a]=case
                arms[a]=physical_receipt(case,path,seat)
            local=eng.play(seed,seat,"candidate",paths,binding,return_accounts)
            eng.validate_case(local,local["ticket"],return_accounts)
            cases["local"]=local
            arms["local"]=physical_receipt(local,OUT/f"candidate_{seed}_{seat}.json.gz",seat)
            check(all(isinstance(r["ungated"],dict) for r in local["shadow"]),"missing same-root ungated decisions")
            check(all(r["candidate"] == r["parent"] for r in local["shadow"] if r["step"] not in (205,271)),
                  "archived command differs outside the two registered steps")
            endpoint = [r for r in local["shadow"] if r["step"] in (205,271)]
            check(all(("local_endpoint" in r) == (r["step"] in (205,271)) for r in local["shadow"])
                  and all(r["local_endpoint"]["slot"]>=0 or r["candidate"]==r["parent"] for r in endpoint),
                  "local head intervention violates released endpoint or physical eligibility")
            attr={**local["attribution"],"endpoint_roots":len(endpoint),
                  "endpoint_selected":sum(r["local_endpoint"]["slot"]>=0 for r in endpoint),"local_vs_ungated_commands":sum(r["candidate"]!=r["ungated"] for r in local["shadow"])}
            pair=dict(seed=seed,seat=seat,arms=arms,local_action_attribution=attr,
                      endpoints=[{**r["local_endpoint"],"changed":r["candidate"]!=r["parent"]} for r in endpoint],
                      pairwise={a:{k:arms["local"][k]-arms[a][k] for k in ("margin","u")} for a in ("parent","ungated")})
            rows.append(pair)
            expenses.append(economy.case_pair(seed,seat,cases,arms))
            print("signed development case",seed,seat,flush=True)
    coverage={a:aggregate(rows,a) for a in ("parent","ungated","local")}
    diffs={a:paired(rows,a) for a in ("parent","ungated")}
    attribution={k:sum(r["local_action_attribution"][k] for r in rows) for k in
                 ("all_roots","parent_candidate_different","student_ablation_roots","candidate_differs_from_no_student","local_vs_ungated_commands","endpoint_roots","endpoint_selected")}
    checks=dict(both_map_u_positive=all(r["delta_u"]>0 for r in diffs["parent"]["map_means"]),
                both_map_margin_nonnegative=all(r["delta_margin"]>=0 for r in diffs["parent"]["map_means"]),
                no_fewer_observed_wins=coverage["local"]["wins"]>=coverage["parent"]["wins"],
                actual_changed_joint=attribution["parent_candidate_different"]>0,
                actual_student_contribution=attribution["candidate_differs_from_no_student"]>0,
                complete_block=len(rows)==4)
    write(OUT/"economy.json",dict(scope="Four new own-proxy games and eight frozen games repriced under official terminal accounting.",
                                 binding=binding,pairs=expenses))
    worker_wall=sum(read(OUT/f"candidate_{s}_{i}.json.gz")["execution"]["wall_seconds"] for s in SEEDS for i in (0,1))
    worker_cpu=sum(read(OUT/f"candidate_{s}_{i}.json.gz")["execution"]["process_cpu_seconds"] for s in SEEDS for i in (0,1))
    write(OUT/"comparison.json",dict(status="complete",new_execution=dict(joint_steps=attribution["all_roots"],
          game_worker_wall_seconds=worker_wall,game_process_cpu_seconds=worker_cpu,logical_cpu=2,
          includes="Per-game public inference, same-root shadows and official/cash verification; excludes postgame cross-tape analysis."),scope="repeated, previously exposed development maps, four new own-proxy games",
          original_master3s2_games=0,formal_games=0,new_full_games=4,reused_full_games=8,
          binding=binding,preregistered_checks=checks,retain=all(checks.values()),pairs=rows,by_arm=coverage,
          by_seat={str(s):{a:aggregate([r for r in rows if r["seat"]==s],a) for a in coverage} for s in (0,1)},
          contrasts=diffs,local_action_attribution=attribution,economy_sha256=sha(OUT/"economy.json")))
    return read(OUT/"comparison.json")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait",action="store_true")
    args=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    # Do not make an active lease or candidates into repository artifacts.
    lockpath=Path("/tmp/sapling-balanced-local-development.lock")
    with lockpath.open("a") as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
        if args.wait:
            while not all((HERE/f).is_file() for f in ("protocol.json","release.json","mechanism.json")):
                time.sleep(5)
        run()

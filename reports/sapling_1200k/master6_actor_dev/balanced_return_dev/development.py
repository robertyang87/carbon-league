"""Paired full games on sealed proxy boards, starting from the public packaged agents' first move."""
from collections import Counter
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import time

import common
import worlds

HERE = Path(__file__).resolve().parent
OUT = HERE / "development"
ARMS = ("parent", "candidate")
PUBLIC_STRIDE = 10
read, sha, write = common.read, common.sha, common.write


def require(condition, message):
    if not condition:
        raise ValueError(message)


def command_hash(value):
    return hashlib.sha256(common.encode(value)).hexdigest()


def load_source(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PublicPack:
    def __init__(self, path, student=True):
        self.module = load_source(path, "_balanced_frozen_public_actor")
        mix = self.module._HYBRID
        require((mix["expert_weight"], mix["student_weight"], mix["style"]) == (.75, .25, 2),
                "unexpected deployed-policy mixture")
        if not student:
            # Remove only the student's logits. The expert coefficient and the return residual stay identical.
            mix["student_weight"] = 0.
        self.step = 0

    def observe(self, obs, issue=True):
        require(obs["step"] == self.step, "public features must see the complete physical history")
        if issue or self.module._FEAT is None:
            command = self.module.agent(obs, worlds.CFG)
        else:
            # Only for the shadow student's unsampled actions. Feat.observe updates its history;
            # the per-call recurrent decoder is rebuilt from context and has no cross-turn state.
            self.module._FEAT.observe(obs)
            command = None
        self.step += 1
        return command


def configuration(binding):
    maps = worlds.registry()
    screen = read(HERE / "screen.json")
    choice = screen["selected"]
    require(choice in ("it20_self","frozen_it50") and screen["summary"][choice]["eligible"], "no eligible proxy")
    package = read(HERE / "package.json")
    paths = {"parent": common.PARENT, "candidate": HERE / "candidate.py",
             "opponent": common.PARENT if choice == "it20_self" else common.IT50}
    opponent_sha = read(HERE / "screen_protocol.json")["policies"][choice]
    require(sha(paths["parent"]) == package["parent_sha256"] and sha(paths["candidate"]) == package["candidate_sha256"]
            and sha(HERE / "candidate.npz") == package["weights_sha256"] and sha(paths["opponent"]) == opponent_sha,
            "frozen deployed actor, student checkpoint or screened opponent differs")
    require(len(maps["development"]) == 2 and maps["physical_seats"] == [0,1], "incomplete preregistered development block")
    identity = dict(package_sha256=sha(HERE / "package.json"), package=package,
                    opponent=choice, opponent_sha256=opponent_sha, screen_sha256=sha(HERE / "screen.json"),
                    maps_sha256=sha(HERE / "maps.json"), clearance_sha256=sha(HERE / "map_clearance.json"),
                    runner_sha256=sha(Path(__file__)), world_sha256=sha(HERE/"worlds.py"),
                    caller_binding=binding, parent_sha256=sha(paths["parent"]), candidate_sha256=sha(paths["candidate"]))
    return maps, paths, identity


def checkpoint(path, data):
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_bytes(gzip.compress(common.encode(data), compresslevel=1, mtime=0))
    os.replace(temp, path)


def commands(packs, obs, seat, t):
    parent = packs["parent"].observe(obs[seat])
    candidate = packs["candidate"].observe(obs[seat])
    foe = packs["opponent"].observe(obs[1 - seat])
    off = packs["without_student"].observe(obs[seat], issue=t % PUBLIC_STRIDE == 0)
    item = dict(step=t, physical_seat=seat, public_observation_sha256=command_hash(obs[seat]),
                parent=parent, candidate=candidate)
    if t % PUBLIC_STRIDE == 0:
        item["candidate_without_sapling"] = off
    return item, foe


def attribution(rows, tape, seat, arm):
    require(len(rows) == len(tape) and len(rows) > 0,"missing full public action attribution")
    changed = samples = student_changed = 0
    for t,(r,joint) in enumerate(zip(rows,tape)):
        require((r["step"],r["physical_seat"]) == (t,seat) and r[arm] == joint[seat]
                and isinstance(r["parent"],dict) and isinstance(r["candidate"],dict),
                "logged actual policy disagrees with the native joint action")
        changed += r["candidate"] != r["parent"]
        due = t % PUBLIC_STRIDE == 0
        require(("candidate_without_sapling" in r) == due, "partial or unregistered student ablation")
        if due:
            samples += 1
            student_changed += r["candidate_without_sapling"] != r["candidate"]
    return dict(all_roots=len(rows), parent_candidate_different=changed,
                student_ablation_roots=samples, candidate_differs_from_no_student=student_changed,
                ablation_stride=PUBLIC_STRIDE, ablation_weights=[.75,0.])


def validate_case(case, target, accounting):
    require(case["ticket"] == target and command_hash(case["actions"]) == case["actions_sha256"],
            "development receipt is bound to different actions or arms")
    seed,seat,arm = (target[k] for k in ("seed","seat","arm"))
    official = worlds.replay_official(seed, case["actions"], "development")
    require(case["official"] == official and official["terrain_sha256"] == target["terrain_sha256"],
            "terminal rewards, terrain or physical seat changed")
    require(case["attribution"] == attribution(case["shadow"],case["actions"],seat,arm),
            "development shadow action count differs")
    require(case["economy"] == accounting.account(worlds.World(seed,"development"),case["actions"],official,seat),
            "cash events differ from the recorded joint commands")


def play(seed, seat, arm, paths, identity, accounting):
    started, cpu_started = time.perf_counter(), time.process_time()
    world = worlds.World(seed, "development")
    target = dict(seed=seed,seat=seat,arm=arm,terrain_sha256=world.terrain_sha256,binding=identity)
    path = OUT / f"{arm}_{seed}_{seat}.json.gz"
    ck = OUT / f"progress_{arm}_{seed}_{seat}.json.gz"
    if path.exists():
        case = read(path)
        validate_case(case,target,accounting)
        if ck.exists():
            stale = read(ck)
            require(stale["ticket"] == target and stale["actions"] == case["actions"][:len(stale["actions"])]
                    and stale["shadow"] == case["shadow"][:len(stale["shadow"])], "contradictory leftover progress")
            ck.unlink()
        return case
    packs = {name: PublicPack(path) for name,path in paths.items()}
    packs["without_student"] = PublicPack(paths["candidate"], student=False)
    if ck.exists():
        progress = read(ck)
        require(progress["ticket"] == target and len(progress["actions"]) == len(progress["shadow"]),
                "cannot resume another policy, board or partial command pair")
        tape, rows = progress["actions"],progress["shadow"]
        prior_clock = progress["execution"]
    else:
        tape, rows = [],[]
        prior_clock = dict(wall_seconds=0.,process_cpu_seconds=0.,attempts=0)
    def elapsed():
        return dict(wall_seconds=prior_clock["wall_seconds"]+time.perf_counter()-started,
                    process_cpu_seconds=prior_clock["process_cpu_seconds"]+time.process_time()-cpu_started,
                    attempts=prior_clock["attempts"]+1,logical_cpus=sorted(os.sched_getaffinity(0)))
    # Rebuild each deployed module's history from its own public observations. Previously committed
    # turns are checked, never overwritten or counted as more games.
    while world.terminal is None:
        t = world.step
        obs = [world.obs(0),world.obs(1)]
        item,foe = commands(packs,obs,seat,t)
        joint = [item[arm],foe] if seat == 0 else [foe,item[arm]]
        world.advance(joint)
        if t < len(tape):
            require(tape[t] == joint and rows[t] == item, "committed deterministic public decisions differ")
        else:
            tape.append(joint); rows.append(item)
            checkpoint(ck,dict(ticket=target,actions=tape,shadow=rows,execution=elapsed()))
    require(len(tape) == world.step, "trailing stored commands after the official terminal")
    official = worlds.replay_official(seed,tape,"development")
    require(official["terrain_sha256"] == target["terrain_sha256"] and
            all(world.terminal[k] == official[k] for k in ("steps","termination","extinct_by_seat","result_by_seat"))
            and max(abs(a-b) for a,b in zip(world.terminal["scores"],official["scores"])) <= .005001,
            "independent original interpreter and forward simulator differ")
    case = dict(ticket=target,actions=tape,shadow=rows,actions_sha256=command_hash(tape),
                official=official,attribution=attribution(rows,tape,seat,arm),
                economy=accounting.account(worlds.World(seed,"development"),tape,official,seat))
    case["execution"] = elapsed()  # includes package load, checkpointing, shadow inference and both terminal audits
    write(path,case)
    ck.unlink(missing_ok=True)
    return case


def summarize(cases, seeds, identity, accounting):
    expected = {(seed,seat,arm) for seed in seeds for seat in (0,1) for arm in ARMS}
    require(len(cases) == 8 and set(cases) == expected, "incomplete or duplicate paired development block")
    pairs = []
    for seed in seeds:
        for seat in (0,1):
            arms={}
            for a in ARMS:
                c=cases[seed,seat,a];o=c["official"]
                arms[a]=dict(score_by_seat=o["scores"],margin=o["margin_by_seat"][seat],u=o["u_by_seat"][seat],
                             result=o["result_by_seat"][seat],termination=o["termination"],steps=o["steps"],
                             actions_sha256=c["actions_sha256"],attribution=c["attribution"],
                             tape_sha256=sha(OUT/f"{a}_{seed}_{seat}.json.gz"),execution=c["execution"])
            delta=accounting.contrast(cases[seed,seat,"parent"]["economy"],cases[seed,seat,"candidate"]["economy"])
            pairs.append(dict(seed=seed,seat=seat,arms=arms,delta_margin=arms["candidate"]["margin"]-arms["parent"]["margin"],
                              delta_u=arms["candidate"]["u"]-arms["parent"]["u"],income_delta=delta))
    def group(items, a):
        v=[r["arms"][a] for r in items]; results=Counter(r["result"] for r in v)
        return dict(games=len(v),wins=results["win"],draws=results["draw"],losses=results["loss"],
                    mean_margin=statistics.mean(r["margin"] for r in v),mean_u=statistics.mean(r["u"] for r in v))
    per_map={str(s):{k:statistics.mean(r[k] for r in pairs if r["seed"]==s) for k in ("delta_margin","delta_u")} for s in seeds}
    attr={a:{k:sum(r["arms"][a]["attribution"][k] for r in pairs) for k in
                ("all_roots","parent_candidate_different","student_ablation_roots","candidate_differs_from_no_student")} for a in ARMS}
    by_arm = {a:group(pairs,a) for a in ARMS}
    seconds = sum(r["arms"][a]["execution"]["wall_seconds"] for r in pairs for a in ARMS)
    steps = sum(r["arms"][a]["steps"] for r in pairs for a in ARMS)
    timing = dict(registered_games=8,joint_transitions=steps,accounted_worker_wall_seconds=seconds,
                  accounted_process_cpu_seconds=sum(r["arms"][a]["execution"]["process_cpu_seconds"] for r in pairs for a in ARMS),
                  joint_transitions_per_worker_second=steps/seconds if seconds else None,
                  scope="Single-worker active game, public counterfactual inference, checkpointing, official replay and cash accounting. Excludes queue time and lost work after the last stored checkpoint.")
    checks = dict(mean_u_positive_on_both_maps=all(v["delta_u"]>0 for v in per_map.values()),
                  mean_margin_nonnegative_on_both_maps=all(v["delta_margin"]>=0 for v in per_map.values()),
                  no_fewer_observed_wins=by_arm["candidate"]["wins"]>=by_arm["parent"]["wins"],
                  changed_actual_joint_on_candidate_trajectory=attr["candidate"]["parent_candidate_different"]>0,
                  candidate_sapling_changes_execution=attr["candidate"]["candidate_differs_from_no_student"]>0,
                  official_all_eight=len(cases)==8)
    return dict(status="complete",scope="eight greedy complete games against the frozen screened in-house proxy",
                preregistered_checks=checks,retain_for_next_study=all(checks.values()),
                formal_games=0,original_master3s2_games=0,binding=identity,pairs=pairs,
                by_arm=by_arm,
                by_seat={str(seat):{a:group([r for r in pairs if r["seat"]==seat],a) for a in ARMS} for seat in (0,1)},
                by_map=per_map,mean_delta_margin=statistics.mean(v["delta_margin"] for v in per_map.values()),
                mean_delta_u=statistics.mean(v["delta_u"] for v in per_map.values()),
                two_map_cluster_se={k:statistics.stdev(v[k] for v in per_map.values())/(2**.5) for k in ("delta_margin","delta_u")},
                attribution_by_arm=attr,execution=timing)


def run(binding=None):
    """Use the caller's existing single-core supervisor. Completed games are rechecked from their action tapes."""
    import return_accounts as accounting
    common.assert_cpu()
    maps,paths,identity=configuration(binding)
    OUT.mkdir(parents=True,exist_ok=True)
    writes = OUT/"binding.json"
    write(writes,identity)
    cases={}
    for seed in maps["development"]:
        for seat in (0,1):
            for arm in ARMS:
                cases[seed,seat,arm]=play(seed,seat,arm,paths,identity,accounting)
    report=summarize(cases,maps["development"],identity,accounting)
    write(OUT/"comparison.json",report)
    return report

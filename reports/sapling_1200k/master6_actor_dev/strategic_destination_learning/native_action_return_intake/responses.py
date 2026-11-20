"""Execute two registered one-step interventions, then the same incumbent against the original."""
import argparse
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import traceback

import select_orders as sel

HERE = sel.HERE
common = sel.common
need = common.need


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


previous = load_module("signed_pair_reconstruction", sel.PREVIOUS / "run.py")
auditor = load_module("signed_native_replay", common.HERE / "verify.py")


def restore(env, snap, actors, enemy):
    bank = common.official_state.restore(env, snap["ref"])
    common.original_state.restore(enemy.mod, snap["native"])
    for key, instance in actors.items():
        common.restore_actor(instance, snap["actors"][key])
        need(common.original_state.h(common.actor_state(instance)) ==
             common.original_state.h(snap["actors"][key]), "Wrong actor snapshot")
    actors["sampler"].mod._NATIVE_RNG.bit_generator.state = copy.deepcopy(snap["sampler_rng"])
    need(common.board_id(env) == snap["board"] and bank == snap["incoming_bank"],
         "Wrong shared board or clock")
    return bank


def branch(env, snap, actors, enemy, cfg, ticket, arm):
    seat, start = ticket["seat"], ticket["root_step"]
    bank = restore(env, snap, actors, enemy)
    rows = []
    while not env.done:
        own, opposing = (common.public(env, s, bank) for s in (seat, 1-seat))
        at_root = own["step"] == start
        their_reply = snap["their_reply"] if at_root else common.call(enemy, opposing, cfg, examiner=True)
        student_reply = common.call(actors["bc"], own, cfg)
        if at_root:
            need(common.digest(own) == ticket["root_public_sha256"] and
                 student_reply[0] == ticket["orders"]["parent"], "Different registered parent")
            if arm == "alternative":
                sampled, time_used = common.call(actors["sampler"], own, cfg)
                info = actors["sampler"].mod._LAST_POLICY
                need(info["issued"] == sampled and info["orders"] == ticket["menu"], "Wrong root sampler")
                student_reply = ticket["orders"][arm], time_used
        bank, row = common.settle(env, bank, seat, own, opposing, student_reply, their_reply)
        row.update(shared_root_opponent=at_root, acting_policy=arm if at_root else "bc")
        rows.append(row)
    return rows, common.terminal(env, seat)


def scores(s):
    margin = s["margin"]
    wdl = 1. if margin > 0. else 0. if margin < 0. else .5
    sigmoid = 1. / (1. + math.exp(-min(700., max(-700., margin / 1000.))))
    return {"win": int(margin > 0.), "margin": margin, "U": .5 * (wdl + sigmoid)}


def check_plan():
    protocol = sel.read_json(HERE / "protocol.json")
    for name, expected in protocol["bound_sources"].items():
        need(sel.sha(common.REPO / name) == expected, "Frozen source changed: " + name)
    need(sel.sha(sel.PACKAGE) == protocol["proposer"] == sel.PROPOSER, "Proposer changed")
    need(sel.sha(common.BC) == protocol["continuation"] == common.SIGNED["bc"], "Continuation changed")
    return protocol


def measure(seed, directory):
    plan = check_plan()
    affinity = sorted(os.sched_getaffinity(0))
    need(len(affinity) == 1, "Conditional training uses one logical CPU at a time")
    ticket = next(t for t in sel.read_json(HERE / "roots.json")["tickets"] if t["seed"] == seed)
    need(ticket["selected"], "No registered choice on this map")
    src, fields = sel.source_for(seed)
    need(all(ticket[k] == v for k, v in fields.items()), "Unregistered trajectory")
    cfg = {k: v for k, v in src["configuration"].items() if k != "randomSeed"}
    metadata = {**fields, "root": ticket["root_step"], "root_public": ticket["root_public_sha256"],
                "protocol": sel.sha(HERE / "protocol.json"), "affinity": affinity,
                "host": common.platform_state()}
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "claim.json").open("x") as f:
        json.dump(metadata, f)
    try:
        os.environ["CARBON_DEBUG"] = "1"
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "master3s2.py"
            path.write_bytes(subprocess.check_output(["git", "-C", str(common.REPO), "show",
                             "HEAD:sources/master3s2/master3s2.py"]))
            need(sel.sha(path) == plan["examiner"] == common.SIGNED["original"], "Wrong examiner")
            enemy = common.collect.Player(path)
            need(enemy.mod._TEMP == 1 and enemy.mod._SEED is None and enemy.mod._S_ABORT == .58
                 and enemy.mod._SEARCH_OK, "Wrong examiner settings")
            actors = {"bc": common.collect.Player(common.BC), "sampler": common.collect.Player(sel.PACKAGE)}
            env, warm, snap = previous.prepare(src, ticket, cfg, actors, enemy)
            snap["sampler_rng"] = copy.deepcopy(actors["sampler"].mod._NATIVE_RNG.bit_generator.state)
            root = ticket["root_step"]
            need(len(warm) == root+1 and all(
                w["historical_order"] == src["rows"][t]["orders"][1-ticket["seat"]] and
                w["restored_native_cost"] == src["rows"][t]["clocks"][1-ticket["seat"]]["cost"]
                for t,w in enumerate(warm)), "Restored prefix differs")
            metadata.update(snapshot=snap["native"]["expected"], board=snap["board"],
                            calibration=snap["their_reply"][1]["cost"],
                            sampler_rng_sha256=common.digest(snap["sampler_rng"]))
            common.save(directory / "prefix.json.gz", {**metadata, "rows": warm})
            arms = {}
            for name in plan["arm_order"][str(seed)]:
                rows, terminal = branch(env, snap, actors, enemy, cfg, ticket, name)
                first = rows[0]
                need(first["orders"][ticket["seat"]] == ticket["orders"][name] and
                     first["orders"][1-ticket["seat"]] == src["rows"][root]["orders"][1-ticket["seat"]] and
                     first["clocks"][1-ticket["seat"]] == src["rows"][root]["clocks"][1-ticket["seat"]] and
                     all(r["acting_policy"] == "bc" and not r["shared_root_opponent"] for r in rows[1:]),
                     "Unshared opponent or changed continuation")
                trace = src["rows"][:root] + rows
                auditor.replay(src, trace, terminal)
                file = directory / (name + ".json.gz")
                common.save(file, {**metadata, "arm": name, "rows": rows, "terminal": terminal})
                clocks = [r["clocks"][1-ticket["seat"]] for r in rows[1:]]
                arms[name] = {"sha256": sel.sha(file), "steps": len(rows), "terminal": terminal,
                              "value": scores(terminal), "inprocess_timing": common.local_time_limits(trace),
                              "native_fresh_calls": len(clocks), "K": [c["K"] for c in clocks],
                              "H": [c["H"] for c in clocks], "aborts": sum(c["aborted"] for c in clocks),
                              "root_order": first["orders"][ticket["seat"]]}
                print(seed, name, terminal["scores"], flush=True)
            common.save(directory / "summary.json", {**metadata, "arms": arms,
                        "alternative_minus_parent": {k: arms["alternative"]["value"][k] - arms["parent"]["value"][k]
                                                     for k in ("win", "margin", "U")},
                        "prefix_sha": sel.sha(directory / "prefix.json.gz"),
                        "corrected_prefix_orders": sum(w["corrected"] for w in warm),
                        "original_opening_games_added": 0, "root_and_suffix_replayed": True})
    except BaseException:
        common.save(directory / "failure.json", {"seed": seed, "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    measure(args.seed, HERE / "tickets" / str(args.seed))

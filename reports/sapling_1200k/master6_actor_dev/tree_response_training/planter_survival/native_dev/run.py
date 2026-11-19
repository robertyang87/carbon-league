"""Resume the two-map development comparison after transfer receipts and the training-owner grant."""

import argparse
from collections import Counter
import fcntl
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
from statistics import mean, stdev
import subprocess
import sys
import tempfile
import time
import types

HERE = Path(__file__).resolve().parent
TRIAL = HERE.parents[1]
for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[name] = "1"
# The upstream driver's module name 'build' is already used by trial.py.
spec = importlib.util.spec_from_file_location("survival_release_build", HERE / "build.py")
build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build)
sys.path.insert(0, str(TRIAL))
import resource_slot as slot
import source_games as sg
from source_games import ENV, gate, sha, trial

CANDIDATE = "2383979ecebb012c980f12ce9bfbd7c162c57746ab7e7a8f7b3859b04061834e"
BATCH_MAPS = {"03": (621473120, 621473121), "04": (621474120, 621474121)}
ARMS = ("parent", "candidate")


def read(p):
    return json.loads(p.read_bytes())


def check(ok, msg):
    if not ok:
        raise ValueError(msg)


def prepare():
    manifest = build.build()
    plan = read(HERE / "protocol.json")
    maps = plan["maps"]
    check(
        manifest["pack_sha256"] == CANDIDATE
        and sha(build.ROOT / "data/master3s2_source.py") == plan["examiner_sha256"]
        and sha(build.ROOT / "data/weights/sapling_1200k/master6_actor_v1.npz") == plan["student_npz_sha256"],
        "source, model or pack changed",
    )
    info = {}
    accepted = True
    evidence = {
        p: sha(HERE / p)
        for p in (
            "protocol.json",
            "release_manifest.json",
            "candidate.py",
            "run.py",
            "parity.py",
            "clearance.json",
        )
    }
    clearance = read(HERE / "clearance.json")
    check(
        clearance["declaration_sha256"] == build.PLAN
        and clearance["verdict"] == "eligible_in_checked_scope"
        and set(map(int, clearance["boards"])) == set(maps)
        and len(set(clearance["boards"].values())) == 2,
        "invalid map clearance",
    )
    for batch, expected in BATCH_MAPS.items():
        path = HERE / f"transfer{batch}.json"
        if not path.is_file():
            return None
        r = read(path)
        check(
            r["batch"] == batch
            and r["protocol_sha256"] == build.PLAN
            and r["candidate_sha256"] == CANDIDATE
            and r["parity_passed"]
            and r["all_actions_matched"]
            and r["all_opportunity_predictions_matched"]
            and r["all_original_v1_scores_matched"]
            and set(map(int, r["maps"])) == set(expected)
            and r["identity"]["reference_chooser_sha256"] == evidence["parity.py"],
            "unbound transfer",
        )
        required = {f"transfer{batch}/{s}_{p}.json.gz" for s in expected for p in (0, 1)}
        check(set(r["per_seat_files"]) == required, "incomplete seat witnesses")
        for name, h in r["per_seat_files"].items():
            check(sha(HERE / name) == h, "transfer tape changed")
            evidence[name] = h
        passed = all(
            m["net_saved"] == m["saved"] - m["harmed"] and m["net_saved"] > 0 and m["harmed"] == 0
            for m in r["maps"].values()
        )
        check(
            r["passed"] == passed and r["decision"] == ("pass" if passed else "fail"),
            "decision differs from the preregistered rule",
        )
        accepted &= passed
        info[batch] = {"maps": r["maps"], "pass": passed}
        evidence[path.name] = sha(path)
    decision = {
        "decision": "run_native" if accepted else "do_not_run_native",
        "started_native_games": 0,
        "transfer_by_batch": info,
        "evidence_files_sha256": evidence,
    }
    slot.seal(HERE / "gate.json", decision)
    if not accepted:
        slot.cancel(HERE / "gate.json", "public transfer did not pass the registered four-map survival rule")
        return decision
    request = {
        "seeds": maps,
        "boards": clearance["boards"],
        "examiner_md5": plan["examiner_md5"],
        "packs": {
            "candidate": {"pack_sha256": CANDIDATE, "weight_sha256": sha(HERE / "release_manifest.json")},
            "parent": {
                "pack_sha256": plan["parent_pack_sha256"],
                "weight_sha256": plan["parent_weight_manifest_sha256"],
            },
        },
        "evidence_files_sha256": {**evidence, "gate.json": sha(HERE / "gate.json")},
    }
    slot.seal(HERE / "slot_request.json", request)
    return decision


def module(path, name):
    m = types.ModuleType(name)
    exec(compile(path.read_bytes(), str(path), "exec"), m.__dict__)
    return m


def edits(tape):
    from fastsim import FastCarbon

    # Independent v1 at every step distinguishes a meaningful new order from a self-reported intervention.
    candidate = module(HERE / "candidate.py", "survival_native_attribution")
    parent = module(build.PARENT, "frozen_parent_attribution")
    sim = FastCarbon(tape["game_configuration"]).reset(seed=tape["seed"])
    seat = tape["candidate_seat"]
    chosen = []
    visits = 0
    for t, joint in enumerate(tape["actions"]):
        obs = sim.obs(seat)
        base = parent.agent(obs, sim.config)
        actual = candidate.agent(obs, sim.config)
        trace = candidate._SURV_TRACE
        check(
            base == trace["v1_commands"]
            and actual == joint[seat]
            and trace["complete"]
            and trace["step"] == t,
            "different live actions on exact public history",
        )
        eligible = trace["opportunities"]
        visits += len(eligible)
        triggers = [r for r in eligible if r["trigger"]]
        selected = triggers[0] if triggers else None
        check(
            eligible == sorted(eligible, key=lambda r: r["uid"])
            and all(
                r["trigger"] == (r["p_stay"] >= 0.2 and r["p_stay"] >= r["p_move"] + 0.05) for r in eligible
            ),
            "unbound decision",
        )
        if selected:
            uid = selected["uid"]
            check(
                trace["selected"] == uid and not base.get(uid) and actual == {**base, uid: selected["move"]},
                "more than the allowed isolated replacement",
            )
            chosen.append({"step": t, **selected})
        else:
            check(actual == base and trace["selected"] is None, "unregistered change")
        sim.step(*joint)
    return {"eligible": visits, "actual_edits": len(chosen), "decisions": chosen}


def run_arm(name, pack, weights, req, plan):
    d = HERE / name
    pkg = req["packs"][name]
    check(
        sha(pack) == pkg["pack_sha256"] and sha(weights) == pkg["weight_sha256"], "wrong live evidence input"
    )
    subprocess.run(
        [
            sys.executable,
            "-u",
            str(sg.BASE / "tree_pack_arena.py"),
            "--pack",
            str(pack),
            "--weights",
            str(weights),
            "--pack-sha256",
            pkg["pack_sha256"],
            "--maps",
            *map(str, plan["maps"]),
            "--cpus",
            *map(str, plan["cpu_affinity"]),
            "--run",
            str(d),
        ],
        check=True,
        env=ENV,
    )
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "official.json"
        subprocess.run(
            [sys.executable, str(sg.BASE / "official_game_replay.py"), str(d), "--out", str(out)],
            env=ENV,
            check=True,
        )
        trial.frozen_file(d / "official_replay.json", read(out))
    official = read(d / "official_replay.json")
    check(
        official["engine_source_sha256"]
        == read(TRIAL / "source/official_replay.json")["engine_source_sha256"],
        "changed official engine",
    )
    native = read(d / "results.json")
    header = native["header"]
    check(
        native["status"] == "complete"
        and header["seeds"] == req["seeds"]
        and header["examiner_md5"] == plan["examiner_md5"]
        and header["cpus"] == plan["cpu_affinity"]
        and header["plan_sha256"] is None
        and header["source_sha256"] == gate.protocol_hashes()
        and all(header[k] == v for k, v in pkg.items()),
        "native arena settings differ",
    )
    expected = {(s, p) for s in plan["maps"] for p in (0, 1)}
    off = official["runs"][name]
    check(
        off["pack_sha256"] == pkg["pack_sha256"] and len(native["games"]) == len(off["games"]) == 4,
        "incomplete arm",
    )
    officiated = {(r["seed"], r["seat"]): r for r in off["games"]}
    check(
        set(officiated) == {(r["seed"], r["candidate_seat"]) for r in native["games"]} == expected,
        "missing or unequal map/seats",
    )
    games = {}
    for r in native["games"]:
        s, p = r["seed"], r["candidate_seat"]
        o = officiated[s, p]
        raw = json.loads(gzip.decompress((d / r["raw_file"]).read_bytes()))
        gate.validate_examiner_configuration(raw["configuration"])
        gate.validate_game(r)
        cash = gate.verify_raw(
            d,
            r,
            pkg["pack_sha256"],
            pkg["weight_sha256"],
            income=True,
            expected_terrain=req["boards"][str(s)],
        )
        check(
            r["cpu_affinity"] == [plan["cpu_affinity"][p]]
            and raw["audit_messages"] == []
            and o["map_sha256"] == req["boards"][str(s)]
            and o["transcript_sha256"] == r["raw_sha256"]
            and o["official_reward_by_seat"] == r["official_score_by_seat"]
            and o["result"] == r["result"]
            and o["steps"] == r["actual_transitions"]
            and o["statuses"] == ["DONE", "DONE"],
            "official or opponent discrepancy",
        )
        games[s, p] = dict(
            o,
            margin=r["margin"],
            U=trial.terminal(r["margin"])[0],
            cash=cash,
            search=r["search_return_types"],
            steps_with_search_invocation=r["search_invocations"],
            termination=r["termination"],
            wall_seconds=r["wall_seconds"],
            call_seconds=r["call_seconds_by_seat"],
        )
        if name == "candidate":
            games[s, p]["attribution"] = edits(raw)
    return games


def compare(arms, req):
    pairs = []
    by_map = {}
    for s in req["seeds"]:
        for p in (0, 1):
            a, b = arms["parent"][s, p], arms["candidate"][s, p]
            pairs.append(
                dict(
                    seed=s,
                    seat=p,
                    parent=a,
                    candidate=b,
                    delta_margin=b["margin"] - a["margin"],
                    delta_U=b["U"] - a["U"],
                )
            )
        sub = [x for x in pairs if x["seed"] == s]
        by_map[str(s)] = dict(
            delta_margin=mean(x["delta_margin"] for x in sub), delta_U=mean(x["delta_U"] for x in sub)
        )
    wins = {a: sum(x[a]["result"] == "win" for x in pairs) for a in ARMS}
    gains = [m["delta_margin"] for m in by_map.values()]
    center = mean(gains)
    width = 12.706204736 * stdev(gains) / math.sqrt(2)
    conditions = [
        all(v > 0 for v in gains),
        sum(x["delta_U"] > 0 for x in pairs) >= 3,
        wins["candidate"] >= wins["parent"],
    ]
    return {
        "games": 8,
        "games_per_arm": 4,
        "release_manifest_sha256": sha(HERE / "release_manifest.json"),
        "gate_sha256": sha(HERE / "gate.json"),
        "grant_sha256": sha(slot.SLOT / "granted.json"),
        "pairs": pairs,
        "maps": by_map,
        "wins": wins,
        "equal_map_delta_margin": center,
        "illustrative_margin_delta_map_t95": [center - width, center + width],
        "equal_map_delta_U": mean(x["delta_U"] for x in pairs),
        "positive_U_pairs": sum(x["delta_U"] > 0 for x in pairs),
        "candidate_actual_edits": sum(x["candidate"]["attribution"]["actual_edits"] for x in pairs),
        "directional_followup_signal": all(conditions),
        "promotion": False,
        "formal": False,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=("supervise", "status"))
    cmd = ap.parse_args().command
    if cmd == "status":
        files = {
            k: (HERE / k).exists()
            for k in (
                "transfer03.json",
                "transfer04.json",
                "gate.json",
                "slot_request.json",
                "comparison.json",
            )
        }
        print(
            json.dumps(
                {
                    "files": files,
                    "games": {
                        a: (
                            len(read(HERE / a / "results.json")["games"])
                            if (HERE / a / "results.json").exists()
                            else 0
                        )
                        for a in ARMS
                    },
                }
            )
        )
        return
    with (HERE / ".supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prerequisites = ["clearance.json", "transfer03.json", "transfer04.json"]
        while not all((HERE / f).is_file() for f in prerequisites):
            time.sleep(5)
        decision = prepare()
        if decision["decision"] == "do_not_run_native":
            print("native cancelled, training lane available")
            return
        if not (slot.SLOT / "release.json").exists():
            with slot.native_session(), tempfile.TemporaryDirectory() as tmp:
                req = read(HERE / "slot_request.json")
                plan = read(HERE / "protocol.json")
                parent, weights, _ = trial.recipe("learned", Path(tmp))
                arms = {}
                for a, pack, w in [
                    ("parent", parent, weights),
                    ("candidate", HERE / "candidate.py", HERE / "release_manifest.json"),
                ]:
                    arms[a] = run_arm(a, pack, w, req, plan)
                trial.frozen_file(HERE / "comparison.json", compare(arms, req))
            slot.release()
        print(
            json.dumps({k: v for k, v in read(HERE / "comparison.json").items() if k != "pairs"}), flush=True
        )


if __name__ == "__main__":
    main()

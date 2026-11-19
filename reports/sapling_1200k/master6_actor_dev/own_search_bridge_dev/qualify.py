"""Run the deployable Sapling and two locally instantiated controls on untouched recorded public histories."""

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import time
from statistics import mean
import numpy as np
import build
import bridge


def package(path, label):
    spec = importlib.util.spec_from_file_location("own_bridge_" + label, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def norm(cmd):
    return {uid: a for uid, a in cmd.items() if a and a != "NONE"}


def run():
    root = bridge.HERE
    proto = bridge.load(root / "entry_protocol.json")
    p = Path("/workspace/sapling-opening-trained-agent.py")
    bridge.require(
        bridge.sha(p) == proto["parent"], "materialize opening parent with its registered build.py"
    )
    release = build.make(build.DEFAULT)
    bridge.save(root / "release.json", release)
    trained = {r["terrain"] for r in bridge.load(root / "fit.json")["trajectories"]}
    registered = {r["physical_terrain"] for r in bridge.load(root / "root_scores.json.gz")}
    native_progress = bridge.load(bridge.ACTOR / "first_win_dev/clearance.json")
    eight = set(native_progress["physical_boards"].values())
    bridge.require(
        len(eight) == 8 and not eight & (trained | registered), "training/label maps intersect live arena"
    )
    cases = []
    maps = set()
    n_calls = 0
    changed = 0
    contributed = 0
    all_times = []
    parent = package(p, "parent")
    cand = package(build.DEFAULT, "serving")
    ablate = package(build.DEFAULT, "ablated")
    ablate._HYBRID["student_weight"] = 0.0
    for tag in proto["tapes"]:
        for path in sorted((bridge.ACTOR / tag).glob("raw_*.json.gz")):
            raw = bridge.load(path)
            s = raw["candidate_seat"]
            terrain = raw["initial_terrain_u8_sha256"]
            maps.add(terrain)
            bridge.require(terrain not in trained | registered | eight, "entry maps overlap fit/validation")
            sim = bridge.FastCarbon(raw["game_configuration"]).reset(seed=raw["seed"])
            group = []
            ca = 0
            wit = None
            for t, j in enumerate(raw["actions"]):
                obs = sim.obs(s)
                base = parent.agent(obs, raw["game_configuration"])
                tick = time.perf_counter()
                cmd = cand.agent(obs, raw["game_configuration"])
                elapsed = time.perf_counter() - tick
                disabled = ablate.agent(obs, raw["game_configuration"])
                last = cand._BR_LAST
                bridge.require(norm(base) == norm(last["base"]), "unsearched joint changed")
                local = norm(cmd) != norm(base)
                witness = norm(cmd) != norm(disabled)
                if local:
                    bridge.require(
                        last["status"] == "completed" and last["selected"] == 1,
                        "changed without complete candidate",
                    )
                if witness and wit is None:
                    wit = {
                        "t": t,
                        "focal": s,
                        "without_student": disabled,
                        "with_student": cmd,
                        "search_eligibility": last["status"],
                    }
                if last["status"] != "outside_or_ineligible":
                    group.append(
                        {
                            "t": t,
                            "status": last["status"],
                            "selected": last.get("selected"),
                            "parent": base,
                            "actual": cmd,
                            "proposed": last.get("alternate"),
                            "wall": elapsed,
                            "details": last.get("comparison"),
                        }
                    )
                all_times.append(elapsed)
                n_calls += 1
                changed += local
                contributed += witness
                ca += witness
                sim.step(*j)
            cases.append(
                dict(
                    source=str(path.relative_to(bridge.ROOT)),
                    sha256=bridge.sha(path),
                    seed=raw["seed"],
                    seat=s,
                    terrain=terrain,
                    turns=len(raw["actions"]),
                    opportunities=group,
                    student_changes=ca,
                    first_student_witness=wit,
                )
            )
    bridge.require(len(cases) == 6 and len(maps) == 3, "missing an entry tape")
    report = {
        "release": release,
        "source": bridge.sha(Path(__file__)),
        "protocol_sha256": bridge.sha(root / "entry_protocol.json"),
        "original_games_opened": 0,
        "eight_map_exclusion_sha256": bridge.sha(bridge.ACTOR / "first_win_dev/clearance.json"),
        "map_count": len(maps),
        "public_calls": n_calls,
        "changed": changed,
        "student_ablation_changed": contributed,
        "wall_all": {
            "mean": float(np.mean(all_times)),
            "p95": float(np.percentile(all_times, 95)),
            "max": max(all_times),
        },
        "cases": cases,
    }
    bridge.save(root / "qualification.json.gz", report)
    print({k: v for k, v in report.items() if k not in ("release", "cases")}, flush=True)


if __name__ == "__main__":
    run()

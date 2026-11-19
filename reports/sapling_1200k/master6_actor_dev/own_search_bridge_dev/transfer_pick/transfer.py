"""Read-only transfer audit on six complete stored nonwinning public histories."""

from collections import Counter
from pathlib import Path
import gzip
import hashlib
import importlib.util
import json
import math
import sys
import time

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent
ACTOR = RESEARCH.parent
ROOT = ACTOR.parents[2]
sys.path.insert(0, str(ROOT / "code"))
from fastsim import FastCarbon

CURRENT = ACTOR / "actor_partner_dev/candidate.py"
PICK = ACTOR / "blended_actor_dev/serving/candidate.py"
CURRENT_SHA = "c5ad3f43959a4812e84e89574ee0c6e69765ebb06a8b3ec93f839b0dde128dd3"
PICK_SHA = "85c47d5d1f93d39dd66d1b959c0c2b3bdc6ee7187553b3ad992b4915056c6fe4"
SOURCES = (
    "tree_response_training/planter_survival/native_dev/parent",
    "freight_deployment/first_response/transfer_parent",
)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    return json.loads(gzip.decompress(p.read_bytes()) if p.suffix == ".gz" else p.read_bytes())


def mod(p, label):
    spec = importlib.util.spec_from_file_location("transfer_" + label, p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def save(p, obj):
    p.write_bytes(
        gzip.compress(
            json.dumps(
                obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode(),
            mtime=0,
        )
    )


def same(ok, why):
    if not ok:
        raise ValueError(why)


def clean(a):
    return {k: v for k, v in a.items() if v and v != "NONE"}


def run():
    same(sha(CURRENT) == CURRENT_SHA and sha(PICK) == PICK_SHA, "packed policy drift")
    first = ACTOR / "first_win_dev"
    live = set(read(first / "clearance.json")["physical_boards"].values())
    live_seeds = set(read(first / "protocol.json")["seeds"])
    prior_wins = []
    # Only archived incumbent games that predate the new eight. Neither selection nor evaluation opens its current arena.
    old_dirs = (
        "actor_partner_dev/native/candidate",
        "actor_partner_dev/replication/games/candidate",
        "blended_actor_arena/games/parent",
    )
    for d in old_dirs:
        for f in sorted((ACTOR / d).glob("raw_*.json.gz")):
            a = read(f)
            same(a["pack_sha256"] == CURRENT_SHA, "wrong historical incumbent")
            if a["margin"] > 0:
                prior_wins.append((a["seed"], a["initial_terrain_u8_sha256"]))
    same(len(prior_wins) == 1, "incumbent prior record changed")
    training = read(ACTOR / "data_split.json")["games"]
    excluded = live | {p[1] for p in prior_wins} | {r["terrain_sha256"] for r in training}
    excluded_seeds = live_seeds | {p[0] for p in prior_wins} | {r["seed"] for r in training}
    runners = [mod(CURRENT, "incumbent"), mod(PICK, "candidate")]
    cases = []
    for d in SOURCES:
        for f in sorted((ACTOR / d).glob("raw_*.json.gz")):
            a = read(f)
            side = a["candidate_seat"]
            seed = a["seed"]
            board = a["initial_terrain_u8_sha256"]
            same(
                board not in excluded and seed not in excluded_seeds,
                "overlap with win/live/candidate-training map",
            )
            same(a["margin"] < 0, "source was not predeclared as a negative historical tape")
            sim = FastCarbon(a["game_configuration"]).reset(seed=seed)
            turns = Counter()
            slots = Counter()
            opp = Counter()
            rows = []
            for t, pair in enumerate(a["actions"]):
                obs = sim.obs(side)
                outputs = [clean(m.agent(obs, a["game_configuration"])) for m in runners]
                own = obs["players"][side]
                enemy = obs["players"][1 - side]
                hist = "opening" if t < 50 else "middle" if t < 200 else "late"
                turns[hist + "/all"] += 1
                turns["all"] += 1
                es = [loc for loc, mass, role in enemy[2].values() if role == "COLLECTOR"]

                def dist(i, j):
                    x = abs(i // 15 - j // 15)
                    y = abs(i % 15 - j % 15)
                    return min(x, 15 - x) + min(y, 15 - y)

                changed = []
                for uid in (*own[1], *own[2]):
                    role = "base" if uid in own[1] else own[2][uid][2].lower()
                    c0, c1 = [cm.get(uid, "NONE") for cm in outputs]
                    slots[role + "/all"] += 1
                    if c0 != c1:
                        slots[role + "/changed"] += 1
                        changed.append([uid, c0, c1])
                    if role == "collector":
                        cell, mass, _ = own[2][uid]
                        flags = []
                        if mass > 0 and min(dist(cell, home) for home in own[1].values()) <= 1:
                            flags.append("loaded_with_bank_in_one")
                        if mass >= 40 and any(
                            v <= mass and dist(cell, pos) <= 2
                            for pos, v, r in enemy[2].values()
                            if r == "COLLECTOR"
                        ):
                            flags.append("loaded_enemy_contact")
                        if any(
                            v > mass and dist(cell, pos) <= 2
                            for pos, v, r in enemy[2].values()
                            if r == "COLLECTOR"
                        ):
                            flags.append("can_contact_heavier_enemy")
                        for fl in flags:
                            opp[fl + "/all"] += 1
                            opp[fl + "/changed"] += int(c0 != c1)
                same(bool(changed) == (outputs[0] != outputs[1]), "unmatched physical UID")
                turns[hist + "/changed"] += bool(changed)
                turns["changed"] += bool(changed)
                if changed:
                    rows.append([t, changed])
                sim.step(*pair)
            cases.append(
                dict(
                    seed=seed,
                    side=side,
                    terrain=board,
                    source=str(f.relative_to(ROOT)),
                    source_sha256=sha(f),
                    turns=dict(turns),
                    slots=dict(slots),
                    cargo_context=dict(opp),
                    changes=rows,
                )
            )
    same(
        len(cases) == 6 and len({a["terrain"] for a in cases}) == 3,
        "expected three physical maps and both seats",
    )
    rec = dict(
        candidate=PICK_SHA,
        incumbent=CURRENT_SHA,
        prior_wins_excluded=len(prior_wins),
        live_maps_excluded=len(live),
        input_manifest_sha256=sha(ACTOR / "data_split.json"),
        active_map_exclusion_sha256=sha(first / "clearance.json"),
        source_sha256=sha(Path(__file__)),
        cohorts=SOURCES,
        candidate_real_games=0,
        formal=0,
        scope="All submitted calls here are hypothetical on the exact same archived public trajectory; underlying real terminal outcomes do not label them.",
        cases=cases,
    )
    save(HERE / "same_history.json.gz", rec)
    for k in ("turns", "slots", "cargo_context"):
        cc = Counter()
        for a in cases:
            cc.update(a[k])
        print(k, dict(cc), flush=True)


if __name__ == "__main__":
    run()

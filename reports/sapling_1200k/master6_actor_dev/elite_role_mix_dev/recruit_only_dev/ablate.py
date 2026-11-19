"""Freeze then measure the conditional base-only ablation under normal serving."""
import argparse
import ast
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
MIX = HERE.parent
ACTOR = MIX.parent
sys.path.insert(0, str(ACTOR / "blended_actor_dev/serving"))
import probe

PARENT = ACTOR / "actor_partner_dev/candidate.py"
FULL = MIX / "conditional/candidate.py"
PANELS = (
    "blended_actor_dev/panel.json", "data_split.json", "data_loss_split.json",
    "elite_role_mix_dev/protocol.json", "elite_role_mix_dev/conditional/protocol.json",
    "late_advantage_actor_dev/panel.json", "planter_threat_transfer_dev/panel.json",
    "planter_threat_transfer_dev/parent_panel/panel.json", "planter_option/value_split.json",
    "denial_solvency_dev/recovery_head/split.json",
)
ROLES = ("collector", "planter", "base")
COHORT = "claude-AlphaCarbon-Master6"


def digest(b):
    return hashlib.sha256(b).hexdigest()


def dump(path, data):
    with path.open("x") as f:
        json.dump(data, f, ensure_ascii=False, allow_nan=False, indent=2)
        f.write("\n")


def identity_rows(obj):
    if isinstance(obj, dict):
        yield obj
        for value in obj.values():
            yield from identity_rows(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from identity_rows(value)


def choose(exclusion_files):
    seen = {k: set() for k in ("id", "seed", "terrain_sha256")}
    for path in exclusion_files:
        for row in identity_rows(json.loads(path.read_bytes())):
            for key in seen:
                value = row.get(key)
                if isinstance(value, (str, int)):
                    seen[key].add(value)
    seen["seed"].update(json.loads((ACTOR / "data_prior_local_seeds.json").read_bytes())["seeds"])
    universe = {}
    for manifest in sorted((ACTOR.parent / "all_top5").glob("appendix_*/manifest.json")):
        for card in json.loads(manifest.read_bytes())["games"]:
            if card["seat_names"].count(COHORT) == 1 and card["clean_terminal"] and not card["self_play"]:
                universe[card["id"]] = card
    priority = lambda c: digest(("base-only-signed-v1|" + c["id"]).encode())
    selected = []
    for seat in (0, 1):
        n = 0
        for g in sorted(universe.values(), key=priority):
            if g["seat_names"][seat] != COHORT or g["id"] in seen["id"] or g["seed"] in seen["seed"]:
                continue
            path = probe.CACHE / (g["id"] + ".json.gz")
            if not path.exists():
                continue
            raw = path.read_bytes()
            if digest(raw) != g["sha256"]:
                raise ValueError("source hash mismatch")
            env = json.loads(gzip.decompress(raw))
            terrain = digest(bytes(env["steps"][0][0]["observation"]["full_carbon"]))
            if terrain in seen["terrain_sha256"]:
                continue
            m = g["score_by_seat"][seat] - g["score_by_seat"][1-seat]
            selected.append(dict(id=g["id"], seed=g["seed"], terrain_sha256=terrain, seat=seat, sha256=g["sha256"],
                                 turns=g["transitions"], split="external", wdl="W" if m>0 else "L" if m<0 else "D",
                                 opponent=g["seat_names"][1-seat]))
            for k in seen:
                seen[k].add(selected[-1][k])
            n += 1
            if n == 12: break
        if n != 12:
            raise ValueError(f"only {n} fresh maps available for seat {seat}")
    return selected


def freeze():
    original, full = PARENT.read_bytes(), FULL.read_bytes()
    fit = json.loads((MIX / "conditional/fit.json").read_bytes())
    if digest(original) != fit["parent_sha256"] or digest(full) != fit["candidate_sha256"]:
        raise ValueError("unverified parent or source")
    marks = list(re.finditer(rb"(?m)^_ROLE_THETA = np.asarray\((\[.*\]), dtype=np.float32\)$", full))
    if len(marks) != 1: raise ValueError("ambiguous matrix declaration")
    m = marks[0]
    theta = np.array(ast.literal_eval(m.group(1).decode()), dtype=np.float32)
    if theta.shape != (3, 59) or not np.isfinite(theta).all():
        raise ValueError("unexpected role geometry")
    isolated = np.zeros_like(theta)
    isolated[2] = theta[2]
    package = full[:m.start(1)] + repr(isolated.tolist()).encode() + full[m.end(1):]
    if not np.array_equal(theta[2], isolated[2]) or np.any(isolated[:2]):
        raise ValueError("base ablation failed")
    names = [ACTOR / x for x in PANELS]
    selected = choose(names)
    with (HERE / "candidate.py").open("xb") as f:
        f.write(package)
    dump(HERE / "protocol.json", dict(parent_sha256=digest(original), source_sha256=digest(full),
         candidate_sha256=digest(package), coefficients_shape=list(theta.shape),
         full_matrix_sha256=digest(theta.tobytes()), ablated_matrix_sha256=digest(isolated.tobytes()),
         base_row_sha256=digest(theta[2].tobytes()), nonzero_per_role=np.count_nonzero(isolated,axis=1).tolist(),
         excluded_panels_sha256={str(p.relative_to(ACTOR)):digest(p.read_bytes()) for p in names},
         prior_seed_ledger_sha256=digest((ACTOR/"data_prior_local_seeds.json").read_bytes()),
         panel_selection="Hash order of base-only-signed-v1|match_id, first 12 eligible per seat. Disjoint preceding model panels and within-arm IDs, seeds, terrains.",
         games=selected, steps=probe.STEPS, probe_sha256=digest(Path(probe.__file__).read_bytes()),
         score="Signed candidate minus original accuracy against the logged expert using actual deployed commands and each arm's own prefix. Mean across eligible maps, no game-strength inference."))


def measure():
    plan = json.loads((HERE/"protocol.json").read_bytes())
    if any(digest(p.read_bytes()) != plan[key] for p,key in ((PARENT,"parent_sha256"), (FULL,"source_sha256"),
        (HERE/"candidate.py","candidate_sha256"),(Path(probe.__file__),"probe_sha256"))):
        raise ValueError("frozen input identity mismatch")
    models = {"parent":probe.package(PARENT,"recruitonlyparent"),
              "candidate":probe.package(HERE/"candidate.py","recruitonlycandidate")}
    evidence = [probe.run_game(models, game, plan["steps"]) for game in plan["games"]]
    summaries = []
    base_transitions = Counter()
    for game in evidence:
        counts = Counter()
        for root in game["roots"]:
            counts["roots"] += 1
            changed = root["arms"]["parent"]["actual_command"] != root["arms"]["candidate"]["actual_command"]
            counts["changed_joint"] += changed
            if root["t"] == 0 and changed: raise ValueError("opening action changed")
            for arm in models:
                counts[arm+"_whole_joint_agreement"] += root["arms"][arm]["actual_command"] == root["demonstration_command"]
            for row in root["slots"].values():
                role, out, expert = row["role"], row["outputs"], row["demo_canonical"]
                a,b = out["parent"]["actual"], out["candidate"]["actual"]
                counts["occupied_"+role] += 1
                counts["changed_"+role] += a != b
                if role == "base":
                    base_transitions[(a,b)] += 1
                if any(len(x["legal"]) <= 1 for x in out.values()): continue
                for group in (role, "all"):
                    counts["free_"+group] += 1
                    counts["parent_"+group] += a == expert
                    counts["candidate_"+group] += b == expert
                    counts["gained_"+group] += b == expert and a != expert
                    counts["lost_"+group] += a == expert and b != expert
        summaries.append({"id":game["id"], "counts":dict(counts)})
    totals = sum((Counter(g["counts"]) for g in summaries), Counter())
    deltas = {}
    for role in (*ROLES, "all"):
        vals = [((s["counts"].get("candidate_"+role,0)-s["counts"].get("parent_"+role,0)) / s["counts"]["free_"+role])
                for s in summaries if s["counts"].get("free_"+role, 0)]
        boot = np.random.default_rng(621758336).choice(vals,(5000,len(vals)),replace=True).mean(1)
        deltas[role] = dict(eligible_maps=len(vals), map_mean=float(np.mean(vals)), ci95=np.quantile(boot,[.025,.975]).tolist(),
                            positive_maps=sum(v>0 for v in vals),negative_maps=sum(v<0 for v in vals))
    raw = gzip.compress(json.dumps(evidence, separators=(",",":"),allow_nan=False).encode(),mtime=0)
    with (HERE/"serving_roots.json.gz").open("xb") as f: f.write(raw)
    dump(HERE/"result.json",dict(protocol_sha256=digest((HERE/"protocol.json").read_bytes()),
         candidate_sha256=plan["candidate_sha256"], parent_sha256=plan["parent_sha256"],
         roots_sha256=digest(raw), counts=dict(totals), by_map=summaries, signed_expert_delta=deltas,
         changed_joint_frequency=totals["changed_joint"]/totals["roots"], changed_maps=sum(x["counts"]["changed_joint"]>0 for x in summaries),
         base_actual_parent_to_candidate={str(a)+"->"+str(b):n for (a,b),n in sorted(base_transitions.items())},
         base_canonical="0 NONE; 1 recruit collector; 2 recruit planter. Transitions include forced decisions."))
    print(deltas, flush=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("mode", choices=("freeze","measure")); args=p.parse_args()
    freeze() if args.mode=="freeze" else measure()

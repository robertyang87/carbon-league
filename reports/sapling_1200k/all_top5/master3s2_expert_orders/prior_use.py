"""Historical cross-project split evidence for the direct ladder corpus, fixed to one Git tree."""
import gzip
import json
import subprocess
from pathlib import Path

REV = "7574f70c0ae34f9314e9aa5da0e967af3b437c7d"
ROOT = Path(__file__).resolve().parents[4]
BASE = "reports/sapling_1200k/"
ACTOR = "master6_actor_dev/"


def load(path):
    data = subprocess.check_output(["git", "show", f"{REV}:{BASE}{path}"], cwd=ROOT)
    return json.loads(gzip.decompress(data) if path.endswith(".gz") else data)


def canonical_split(raw):
    if raw.startswith(("test_", "val_")):
        return raw.split("_")[0]
    if raw in {"held_map", "held_rival", "known_new_map", "new_opponent", "appendix"}:
        return "holdout"
    if raw.startswith("excluded"):
        return "diagnostic"
    return raw


def prior_uses(cards):
    by_id = {r["id"]: r for r in cards}
    by_terrain = {r["terrain"]: r for r in cards}
    by_seed = {r["seed"]: r for r in cards}
    if len(cards) != len(by_id) or len(cards) != len(by_terrain) or len(cards) != len(by_seed):
        raise ValueError("expected one registered replay per physical map")
    out = {k: [] for k in by_id}

    def register(row, path, raw, role, **detail):
        terrain = row.get("terrain", row.get("terrain_sha256"))
        seed = row.get("seed", row.get("random_seed", row.get("map")))
        key = row.get("id")
        card = by_terrain.get(terrain) or by_id.get(key) or by_seed.get(seed)
        if card is None:
            return
        if terrain is not None and terrain != card["terrain"] or seed is not None and seed != card["seed"]:
            raise ValueError(f"{path}: inconsistent terrain/seed for {card['id']}")
        out[card["id"]].append(dict(project=BASE+path, split=canonical_split(raw), raw_split=raw,
                                     role=role, terrain=card["terrain"], **detail))

    registries = [
        ("tree_tenure_dev/maps.json.gz", "maps", "tree-tenure map assignment"),
        ("planter_intent_labels/maps.json.gz", "maps", "planter-intent map assignment"),
        ("planter_threat_transfer_dev/splits.json.gz", "maps", "threat-transfer eligible split population"),
        ("planter_threat_transfer_dev/panel.json", "maps", "threat-transfer actual panel"),
        ("two_arm_plant_economy_dev/plan.json", "maps", "two-arm planting-economy actual panel"),
        ("cargo_tree_action_learning_dev/territory_sequence/sources.json", None, "territory sequence"),
        ("blended_actor_dev/panel.json", "games", "blended actor"),
        ("blended_actor_dev/opening_training/panel.json.gz", "games", "opening-order model"),
        ("freight_horizon/dataset.json", "rows", "enemy freight horizon"),
    ]
    for tail, field, role in registries:
        path = ACTOR + tail
        data = load(path)
        for row in data[field] if field else data:
            register(row, path, row["split"], role)
            for seat, held in enumerate(row.get("held_rival", [])):
                if held:
                    register(row, path, "held_rival", role, seat=seat)

    path = "all_top5_pressure_dev/split.json"
    data = load(path)
    for row in data["maps"]:
        for seat, info in sorted(row["victim_seats"].items()):
            register(row, path, info["set"], "pressure-response victim split", seat=int(seat), victim=info["victim"])
    for seed in data["excluded_prior_26"]:
        register({"seed": seed}, path, "excluded_prior_26", "pressure-response source exclusion")

    path = "planter_routes_dev/integrity.json"
    data = load(path)
    for split, maps in data["maps_sha256_by_split"].items():
        for k, terrain in maps.items():
            register(dict(id=k, terrain=terrain), path, split, "planter route")
    for k, terrain in data["excluded_old26"].items():
        register(dict(id=k, terrain=terrain), path, "excluded_old26", "planter route source exclusion")

    path = ACTOR + "elite_role_mix_dev/protocol.json"
    for k in load(path)["train_ids"]:
        register({"id": k}, path, "train", "elite role mix")
    for path in ["elite_3s2_agreement.json", "tree_v4_master3s2_agreement.json"]:
        for row in load(path)["games"]:
            register(row, path, "holdout", "disjoint original-opponent action-agreement evaluation", seat=row["seat"])
    return out

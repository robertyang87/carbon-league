"""Rebuild the map-exposure receipt for the six scored games at 628e28d0."""
import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SAPLING = HERE.parents[1]
ROOT = SAPLING.parents[1]
REVISION = "628e28d06b0610f6ceebdac3cc632057766bddd2"
PREVIOUS = HERE.parent / "appendix_56e1e613/exposure.json"
SEQUENCE = SAPLING / "master6_actor_dev/cargo_tree_action_learning_dev/territory_sequence"
CHECKER = SEQUENCE / "fresh_original/check_maps.py"
RECENT = SEQUENCE / "windowed/fresh_original/clearance.json"

sys.path.insert(0, str(CHECKER.parent))
import check_maps as checks
from tree_win_gate import RESERVED_DEVELOPMENT_PLANS


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def frozen(path):
    return checks.git("show", f"{REVISION}:{path.relative_to(ROOT)}")


def validate_runtime_inputs():
    paths = {
        Path(module.__file__).resolve()
        for module in list(sys.modules.values())
        if getattr(module, "__file__", None) and Path(module.__file__).resolve().is_relative_to(ROOT)
    }
    paths.discard(Path(__file__).resolve())
    paths.add(SAPLING / checks.map_identity.LOCAL_EXPOSURE)
    paths.update(SAPLING / item[0] for item in RESERVED_DEVELOPMENT_PLANS)
    paths.update(SAPLING / item[0] for item in checks.map_identity.SOURCES)
    subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet", REVISION, "--",
                    *[str(path.relative_to(ROOT)) for path in paths]], check=True)


def rows(raw):
    return [json.loads(line) for line in gzip.decompress(raw).splitlines()]


def calculate():
    validate_runtime_inputs()
    history = json.loads(frozen(PREVIOUS))
    hashes, old_games, blocks = {}, {}, {}
    fault_ids, nominal_seeds, declared_terrains = set(), set(), set()
    source_counts = {"metadata": 0, "ladder": 0}
    sapling = SAPLING.relative_to(ROOT)
    paths = checks.git("ls-tree", "-r", "--name-only", REVISION, str(sapling), "data/k0_calib")
    for path in map(Path, paths.decode().splitlines()):
        in_sapling = path.is_relative_to(sapling)
        if in_sapling:
            nominal_seeds.update(map(int, checks.NUMBER.findall(path.name.encode())))
        metadata = in_sapling and path.suffix == ".json" and any(s in path.name for s in checks.REGISTERS)
        ladder = path.is_relative_to(sapling / "all_top5") and path.name == "validation_maps.jsonl.gz"
        pool = path.is_relative_to("data/k0_calib") and "seed_blocks_" in path.name and path.suffix == ".json"
        if not (metadata or ladder or pool):
            continue
        data = frozen(ROOT / path)
        hashes[str(path)] = sha(data)
        if ladder:
            source_counts["ladder"] += 1
            for game in rows(data):
                if game["id"] in old_games:
                    raise ValueError(f"Duplicate ladder game in {path}")
                old_games[game["id"]] = game
                if "unscored" in path.parts:
                    fault_ids.add(game["id"])
        elif pool:
            blocks[str(path)] = [tuple(map(int, b["seeds"].split("-"))) for b in json.loads(data)["blocks"]]
        else:
            source_counts["metadata"] += 1
            nominal_seeds.update(map(int, checks.NUMBER.findall(data)))
            declared_terrains.update(value.decode() for value in checks.DIGEST.findall(data))

    old_seeds = {game["seed"] for game in old_games.values()}
    old_terrains = {game["initial_terrain_u8_sha256"] for game in old_games.values()}
    counts = [len(old_games), len(old_seeds), len(old_terrains)]
    historical_counts = [history[f"distinct_archived_{field}_including_fault"] for field in ("ids", "seeds", "terrains")]
    if counts != historical_counts or sorted(fault_ids) != history["fault_ids"]:
        raise ValueError("Historical identity ledgers differ from the preceding receipt")

    _, known_seeds, known_terrains = checks.exam_exposures(sorted(s for s in nominal_seeds if s < 2**32))
    recent_data = frozen(RECENT)
    recent = json.loads(recent_data)
    recent_boards = recent["explicit_earlier_development_reservations"] | recent["physical_boards"]
    known_seeds.update(map(int, recent_boards))
    known_terrains.update(recent_boards.values())

    registry = (HERE / "validation_maps.jsonl.gz").read_bytes()
    validation = json.loads((HERE / "validation.json").read_bytes())
    manifest = json.loads((HERE / "manifest.json").read_bytes())
    games = rows(registry)
    if (manifest["revision"] != REVISION or validation["revision"] != REVISION
            or sha(registry) != validation["map_registry_sha256"]
            or {g["id"] for g in games} != {g["id"] for g in manifest["games"]}
            or not len(games) == validation["cards"] == 6):
        raise ValueError("New map registry differs from the frozen games")
    result_rows = []
    for game in games:
        gid, seed, terrain = game["id"], game["seed"], game["initial_terrain_u8_sha256"]
        if gid in old_games or seed in old_seeds or terrain in old_terrains:
            raise ValueError(f"Duplicate ladder ID, seed or physical terrain for {gid}")
        old_games[gid] = game
        old_seeds.add(seed)
        old_terrains.add(terrain)
        result_rows.append({
            **game,
            "known_development_or_training_seed_match": seed in known_seeds,
            "known_or_declared_physical_identity_match": terrain in known_terrains or terrain in declared_terrains,
            "reserved_final_seed_interval_match": checks.RESERVED_RANGE[0] <= seed < checks.RESERVED_RANGE[1],
            "k0_registered_seed_blocks_matching": sorted(p for p, ranges in blocks.items()
                                                         if any(lo <= seed <= hi for lo, hi in ranges)),
            "nominal_fast_sim_map_equals_observed": checks.map_identity.local_board(seed) == terrain,
        })
    numeric = len(old_games) - len(fault_ids)
    if not numeric == manifest["accumulated_games"] == validation["union_distinct"]["seed"]:
        raise ValueError("Numerically scored games differ across the validation receipts")
    return {
        "ladder_revision": REVISION,
        "registered_local_metadata_revision": REVISION,
        "complete_scored_games": numeric,
        "official_fault_prefixes": len(fault_ids),
        "fault_ids": sorted(fault_ids),
        "distinct_archived_ids_including_fault": len(old_games),
        "distinct_archived_seeds_including_fault": len(old_seeds),
        "distinct_archived_terrains_including_fault": len(old_terrains),
        "new_source_compressed_sha256": sha(registry),
        "previous_increment_exposure_sha256": sha(frozen(PREVIOUS)),
        "local_input_manifest_sha256": sha(json.dumps(hashes, sort_keys=True).encode()),
        "source_metadata_registers": source_counts["metadata"],
        "baseline_ladder_terrain_files": source_counts["ladder"],
        "k0_seed_registry_files": len(blocks),
        "last_six_registered_dev_boards": recent_boards,
        "recent_clearance_sha256": sha(recent_data),
        "known_local_or_conservatively_excluded_seed_ids": len(known_seeds),
        "known_local_or_conservatively_excluded_physical_boards": len(known_terrains),
        "frozen_clearance_checker_sha256": sha(frozen(CHECKER)),
        "reserved_formal_half_open": list(checks.RESERVED_RANGE),
        "rows": result_rows,
        "new_prior_ladder_duplicate_ids_seeds_or_boards": 0,
        "new_development_or_training_seed_matches": sum(r["known_development_or_training_seed_match"] for r in result_rows),
        "new_development_or_training_terrain_matches": sum(r["known_or_declared_physical_identity_match"] for r in result_rows),
        "new_reserved_final_seed_interval_matches": sum(r["reserved_final_seed_interval_match"] for r in result_rows),
        "new_k0_seed_block_matches": sum(bool(r["k0_registered_seed_blocks_matching"]) for r in result_rows),
        "scope": "Historical opening terrain comes from the 225 cells recorded in the official replay. Development comparisons cover the Sapling registry snapshot, local exposure roster and reserved seed blocks. The final draw excludes every published seed and observed terrain.",
    }


if __name__ == "__main__":
    receipt = calculate()
    (HERE / "exposure.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")

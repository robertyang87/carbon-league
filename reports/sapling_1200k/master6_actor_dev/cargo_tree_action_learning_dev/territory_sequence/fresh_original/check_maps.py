"""Recompute the two development terrain clearances. Run from the registered checkout."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent
SAPLING = HERE.parents[3]
ROOT = SAPLING.parents[1]
sys.path[:0] = [str(SAPLING), str(ROOT / "code"), str(ROOT / "code/env_src/carbon_challenge-master")]
import numpy as np
import map_identity
from tree_audited_game import exam
from tree_win_gate import RESERVED_RANGE, exam_exposures
from zerosum_env import make

REVISION = "12affc533122f5331f6d298bf639a701ff4803b7"
SEEDS = (621758352, 621758353)
REGISTERS = ("protocol", "clearance", "registr", "seed", "declar", "plan", "panel", "sources", "split", "map")
DIGEST = re.compile(rb"(?<![a-zA-Z0-9])[0-9a-f]{64}(?![a-zA-Z0-9])")
# Broad exclusion includes numeric IDs, even when their original schema had no named seed field.
NUMBER = re.compile(rb"(?<![.a-zA-Z0-9])\d{8,10}(?![.a-zA-Z0-9])")


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, env={**os.environ, "GIT_NO_LAZY_FETCH": "1"})


def inspect():
    base = str(SAPLING.relative_to(ROOT))
    paths = git("ls-tree", "-r", "--name-only", REVISION, base, "data/k0_calib").decode().splitlines()
    ledgers = {}
    ids, from_files, declared_digests, ladder_terrains, ladder_seeds = set(), set(), set(), set(), set()
    metadata_count = ladder_count = k0_count = 0
    # Development-run receipts have seeds encoded in their filenames as well as in registries.
    for rel in paths:
        p = Path(rel)
        if rel.startswith(base + "/"):
            from_files.update(map(int, NUMBER.findall(p.name.encode())))
        previous_map_register = rel.startswith(base + "/") and p.suffix == ".json" and any(k in p.name for k in REGISTERS)
        ladder_register = p.name == "validation_maps.jsonl.gz" and rel.startswith(base + "/all_top5/")
        alternative_pool = rel.startswith("data/k0_calib/") and "seed_blocks_" in p.name and p.suffix == ".json"
        if not (previous_map_register or ladder_register or alternative_pool):
            continue
        data = git("show", f"{REVISION}:{rel}")
        ledgers[rel] = hashlib.sha256(data).hexdigest()
        if ladder_register:
            for line in gzip.decompress(data).splitlines():
                r = json.loads(line)
                if r["id"] in ids:
                    raise ValueError(f'duplicate replay {r["id"]}')
                ids.add(r["id"])
                ladder_seeds.add(r["seed"])
                ladder_terrains.add(r["initial_terrain_u8_sha256"])
            ladder_count += 1
        elif alternative_pool:
            for block in json.loads(data)["blocks"]:
                low, high = map(int, block["seeds"].split("-"))
                if any(low <= seed <= high for seed in SEEDS):
                    raise ValueError(f"prior seed block overlaps {rel}")
            k0_count += 1
        else:
            from_files.update(map(int, NUMBER.findall(data)))
            declared_digests.update(d.decode() for d in DIGEST.findall(data))
            metadata_count += 1

    map_seeds = [s for s in from_files if 0 <= s < 2**32]
    _, known_seeds, known_terrains = exam_exposures(map_seeds)
    known_seeds.update(ladder_seeds)
    known_terrains.update(ladder_terrains)
    if any(RESERVED_RANGE[0] <= s < RESERVED_RANGE[1] for s in SEEDS):
        raise ValueError("formal examination pool overlap")
    rows = map_identity.checked_maps(SEEDS, known_seeds, known_terrains | declared_digests)
    boards = {}
    for r in rows:
        seed = r["seed"]
        np.random.seed(seed)
        env = make("carbon", configuration={**exam.CFG, "randomSeed": seed})
        official = map_identity.terrain_sha256(env.states[0].observation.full_carbon)
        if official != r["initial_terrain_u8_sha256"]:
            raise ValueError("official opening differs from the fast simulator")
        boards[str(seed)] = official
    return dict(snapshot=REVISION, purpose="Fresh whole-game development only. Both terrains are permanently exposed by this registration.",
                physical_boards=boards, seed_collisions=0, terrain_collisions=0,
                independent_terrain_sources=["FastCarbon.reset", "zerosum_env.make"],
                reserved_formal_half_open=list(RESERVED_RANGE),
                legacy_and_current_seed_blocks_checked=k0_count,
                top5_map_files=ladder_count, unique_top5_entries_including_unscored=len(ids),
                metadata_files_examined=metadata_count,
                conservative_excluded_seed_ids=len(known_seeds),
                conservative_excluded_physical_hashes=len(known_terrains),
                input_manifest_sha256=hashlib.sha256(json.dumps(ledgers, sort_keys=True).encode()).hexdigest(),
                scope="Pinned Sapling historical register names, receipt filenames, archive/training splits, all published Top5 terrain indexes, and the six K0 seed-block files. The standard map-identity checker adds the frozen local exposure roster and previously reserved development cohorts.")


if __name__ == "__main__":
    print(json.dumps(inspect(), indent=2, sort_keys=True))

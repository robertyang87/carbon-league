"""Keep all observed ladder terrains out of Sapling's one-shot test draw."""
import gzip
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCES = (
    ("all_top5/validation_maps.jsonl.gz", 1324,
     "15558ec38db5cb45c8dd5666b0829a2902e6248857780615955dde2bc277036d",
     "1922c0d3677bd21c7453c755ac8b3d05ffd8b5a01b43430a0e84f689b195fa18"),
    ("all_top5/appendix_499886f1/validation_maps.jsonl.gz", 8,
     "26bdf38714dbc37a06a3a83a2d0b43d352b1db0bf0e2b06bb0d9eea707b44434",
     "91bc4810a0c02f18c461e3743ce8a76e1ccaee73e746c5d4ab77778a015bf4a4"),
)
IDENTITY = "sha256(row-major 15x15 full initial carbon, one unsigned integer byte per cell)"
SCHEMA = "sapling-90-and-income/4"

def terrain_sha256(full_carbon):
    if len(full_carbon) != 225 or any(
        isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
        or v != int(v) or not 0 <= v <= 255 for v in full_carbon
    ):
        raise ValueError("expected the 225 integral initial full-carbon cells in row-major order")
    return hashlib.sha256(bytes(map(int, full_carbon))).hexdigest()


def local_board(seed):
    from tree_audited_game import exam
    return terrain_sha256(exam.FastCarbon(exam.CFG).reset(seed=seed).carbon)


def ladder_boards():
    all_ids, seeds, boards = set(), set(), set()
    artifacts = []
    for filename, expected_count, evidence_hash, id_hash in SOURCES:
        packed = (HERE / filename).read_bytes()
        plain = gzip.decompress(packed)
        if hashlib.sha256(plain).hexdigest() != evidence_hash:
            raise ValueError("frozen ladder terrain evidence was altered")
        rows = [json.loads(line) for line in plain.splitlines()]
        ids = [r["id"] for r in rows]
        if (len(rows) != expected_count or len(set(ids)) != expected_count or all_ids.intersection(ids)
                or hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest() != id_hash):
            raise ValueError("incomplete or overlapping frozen Top 5 exposure ledger")
        all_ids.update(ids)
        artifacts.append(hashlib.sha256(packed).digest())
        for row in rows:
            seed, terrain = row["seed"], row["initial_terrain_u8_sha256"]
            if type(seed) is not int or not isinstance(terrain,str) or len(terrain) != 64 or len(bytes.fromhex(terrain)) != 32:
                raise ValueError("malformed frozen Top 5 terrain evidence")
            seeds.add(seed)
            boards.add(terrain)
    return hashlib.sha256(b"".join(artifacts)).hexdigest(), seeds, boards


class OverlappingDraw(ValueError):
    """Retire this entire nonce before any game is played."""


def checked_maps(seeds, excluded_seeds, excluded_boards):
    seen = set()
    rows = []
    for seed in seeds:
        if seed in excluded_seeds:
            raise OverlappingDraw("draw includes an exposed seed")
        terrain = local_board(seed)
        if terrain in excluded_boards or terrain in seen:
            raise OverlappingDraw("draw includes an exposed or repeated terrain")
        seen.add(terrain)
        rows.append({"seed":seed, "initial_terrain_u8_sha256":terrain})
    return rows


def verify_observed_map(expected, row, tape):
    if (row.get("initial_terrain_u8_sha256") != expected or tape.get("initial_terrain_u8_sha256") != expected
            or terrain_sha256(tape.get("initial_full_carbon", [])) != expected):
        raise ValueError("actual initial terrain on the physical seat differs from the frozen map")

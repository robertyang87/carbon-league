"""Keep known ladder and local development terrains out of the one-shot test draw."""
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
    ("all_top5/appendix_4bfdabf5/validation_maps.jsonl.gz", 15,
     "f0bed3cb04f2b6653b8f2f84fc2301c721672f671878eea66e8fd3fcf786bba7",
     "672479cfbd86d0876f1497b08de69b63e382345e5e3d2f637f144f64d6b5c411"),
    ("all_top5/appendix_555c0c56/validation_maps.jsonl.gz", 10,
     "4d8220e64ad2cf3c6bfd4bb64004cca06ee4d8d1dce4f7b6b9f1aa1e04615dfb",
     "16daea51472bd2d6c977877b6b052c89d21c6a783e41dfefac2aaae146a786eb"),
    ("all_top5/appendix_bf049043/validation_maps.jsonl.gz", 4,
     "f5389ae5b1a4e2990ac03d84e84811cc098ce92fef1e6d9237a05a8f34cfa5a0",
     "6bebdc439366ea8fb1cf8def9da38591ecbcc671d0db5b308c53831888b524a5"),
    ("all_top5/appendix_bd3c0021/validation_maps.jsonl.gz", 12,
     "148f8d87ed41e377e77aa37ef4dbe9f24ff441e3957cae4939813f347ad9daf2",
     "39fabae6a683c5f3119865669711cd9134246b6e1ec21556ed8e9f4d36d5fd85"),
    ("all_top5/appendix_5484590b/validation_maps.jsonl.gz", 5,
     "eced1e30fe392db5f484b7b2a69b57b08c04a31aab13b391cde0f9b6c83ce99b",
     "334fdeb43c100fd3fed44ce043bcf41742187cb92ac81aa088e65e3666684565"),
    ("all_top5/appendix_99919cbf/validation_maps.jsonl.gz", 4,
     "b107dff27997b4f75ed8efd76570e8e7131fd047da63fe697a912cc7d9fdec9e",
     "26c95e2a358b06dcb7d1ebda0606456f533f8bb24cad69805deb35bff2d49b93"),
    ("all_top5/appendix_0137949d/validation_maps.jsonl.gz", 4,
     "7ffca4cca4767ca1d3c01e31f89492a0724edb62127dca2b4533edfab8c00e40",
     "d3c084606893d7d591fe0b5e431fbbe7a46ca4033bcfe0c9ed440d061a72b60b"),
    ("all_top5/appendix_41e4177e/validation_maps.jsonl.gz", 16,
     "bde062a866a068896d1556f62af5b9ca876218c6485a6d81e6038d14ed7cce7f",
     "c4466d9ad2da8db715bbfa150199aae1f95582ac74ce63dbc0c69d3c12376e82"),
    ("all_top5/appendix_f7109df6/validation_maps.jsonl.gz", 4,
     "87f686afb1310e24dc0cf7ceb2c5411ef9d1557fff042c9c1729be86d7e9f1ad",
     "5894089617371157b76eed1b94d92747378ff55ea11900c80985afde95ef7447"),
    ("all_top5/appendix_1fdacb1b/validation_maps.jsonl.gz", 1,
     "d4d57014b3fe536010119e282f02f0d2c0e694a26d697b115db11cdc2cd9a4fc",
     "3d4127e2746741d4a5ce0c90401356a95ac8c76c006776d05e00244f8c5968e8"),
    ("all_top5/appendix_32b555ee/validation_maps.jsonl.gz", 14,
     "25f474cb90dde1f0c9403db709fa0f12e8b19efd547706dcfaf3dd5044f13c58",
     "7c9c0555234870e29c5885f72a37ee27744a1f379a4fa3c005e6d68aaa2e8b4d"),
    ("all_top5/appendix_c5c8cf8d/validation_maps.jsonl.gz", 11,
     "fcb635eabc5851f829f522651f8e68943b2a7873bafe85f981821be042afaa34",
     "25efbc7d357dbcbec36cb55a8335ee98dfe0ef6c8c8e248514968edeef0cdaa0"),
    ("all_top5/appendix_39a745e0/validation_maps.jsonl.gz", 8,
     "4faa0557e9b95d75ab258f93ebcf3a830460dbceaf12b23ea6a6476b27d08e45",
     "766a9a12a9036ac1b3c9696c82362ca153f7abc793a6c20769ae91c44386cdb7"),
    ("all_top5/appendix_cbd6d545/validation_maps.jsonl.gz", 25,
     "18a1a3385eb99095ca6ab54149f9358bde2fe79ac3e1766215571ce640a7b54e",
     "1dee70025d245beddbdc18878daf612d6a8b1ac6e3c1ee6d7f852fc666eece8e"),
    ("all_top5/appendix_336e8b8c/validation_maps.jsonl.gz", 10,
     "045714019f1b94f6440175939ff7e58511ee24b54718e647bfe97868baf1f137",
     "66eafb838520bc68c32cfa2e8fea1a46318892eba1a0908ace2cf08f2280cefc"),
    ("all_top5/appendix_97de1ebc/validation_maps.jsonl.gz", 12,
     "a3e113a2d0b74b6e926c630fb6124a297bf22ef92e7f13e4710e688d82f17487",
     "910def6fcf7537bf3d5b84efe55c2e37fc48ca5fe63ad3109834b47a1234c467"),
    ("all_top5/appendix_97de1ebc/unscored/validation_maps.jsonl.gz", 1,
     "e1fcfdf27564b0b6326e38d3fce4618c02bb4da0e25407d9c65403141f1b2ffe",
     "345f4168d8988c77d4c34fcfb741add8fe38bd3cc94af3d013f3b719dedfbef1"),
    ("all_top5/appendix_2cc7c358/validation_maps.jsonl.gz", 15,
     "9ce4bdcc32a4aa16beb08c5767d84a1a3574641e2ee0d2510dc4d7151320a1ea",
     "ee2eaf3c15964d4c0ebc85496f8acfeb73b68419ba96894b9afc3ba057033104"),
    ("all_top5/appendix_b5e6a7f8/validation_maps.jsonl.gz", 19,
     "0425fd448a167872110e8c856f888d3181b17ee123794161123693d8dc496460",
     "6d60b966dddcdd5ffb3bf49b1a2f5d957c721b577c2a2b5b460164eb936f6c05"),
    ("all_top5/appendix_aca9b8a3/validation_maps.jsonl.gz", 7,
     "310da4887caeb813358d703d5beb2f0c6979f7e89ad2abea6190ffc61837a42c",
     "f267bc5b5474476612e8960198a4fba2eff758c4e685c4ee980af8c04445d295"),
    ("all_top5/appendix_4a4677c9/validation_maps.jsonl.gz", 14,
     "2fc0d08d5afdb2af1df51244f1942178d878069efa425bb0edd929dca1b63196",
     "2957661bf1ff35a187cc6d920c18d28ab4dfc607dd833fd6f03c58b7be2f9d98"),
)
IDENTITY = "sha256(row-major 15x15 full initial carbon, one unsigned integer byte per cell)"
SCHEMA = "sapling-90-and-income/5"
LOCAL_EXPOSURE = "known_local_maps/exposure.json"

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


def exposed_boards(additional_seeds, observed_maps=()):
    """Combine actual ladder terrains, local anchors and operator-supplied local seeds."""
    if not isinstance(additional_seeds, list) or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in additional_seeds):
        raise ValueError("additional local exposures must be integer game seeds")
    ladder_hash, occupied_seeds, occupied_terrains = ladder_boards()
    known = json.loads((HERE / LOCAL_EXPOSURE).read_bytes())["maps"]
    local_seeds = set()
    for row in known:
        local_seeds.add(row["seed"])
        occupied_terrains.add(row["initial_terrain_u8_sha256"])
    for seed in set(additional_seeds) - local_seeds:
        occupied_terrains.add(local_board(seed))
    occupied_seeds.update(local_seeds)
    occupied_seeds.update(additional_seeds)
    for observation in observed_maps:
        if not isinstance(observation, dict) or type(observation.get("seed")) is not int:
            raise ValueError("an observed local map needs its seed and the initial 225 cells")
        occupied_terrains.add(terrain_sha256(observation.get("initial_full_carbon", ())))
        occupied_seeds.add(observation["seed"])
    return ladder_hash, occupied_seeds, occupied_terrains


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

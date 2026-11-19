"""Verify the registered original-opponent study, play one same-host pair, or replay its receipts."""

import argparse
import gzip
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
C = HERE.parents[1]
ROOT = C.parents[3]
OLD = C / "territory_sequence/fresh_original"
LANE = C / "native_actor_corpus"
A = LANE / "original_response"
T = LANE / "ladder_timing"
sys.path.insert(0, str(OLD))
import play
import verify_pair

FULL = "6c54899562382552b95bc1299d83119088ef1d1206b93135e33769815ed69c9f"
PARENT = "8da4b465f7fa6e601e51bd7c2e81f25825cb59b9b5ab285f7f56d0f6bbfe1cac"
FROZEN = {
    "student": (LANE / "full.py", FULL),
    "student_weights": (
        LANE / "full.npz",
        "f52573c9a91b8638bd33ec062000f38eeb44c658d45da4e67add038759564f9e",
    ),
    "parent": (C / "candidate.py", PARENT),
    "parent_weights": (
        C / "candidate.npz",
        "7a677b76f30769eb097162a42d355b13b21acf3bab9ae7e22bfe2e4148ff9923",
    ),
    "opponent": (
        ROOT / "data/master3s2_source.py",
        "65f268b2919ce30fe19355115a928eada032a6beaa018d801662d434c083f2b5",
    ),
    "runner": (OLD / "play.py", "17695a8b5f055ef9ae9cc14674b20efe0b776bce0b72365a4336dea393d525d2"),
    "verifier": (OLD / "verify_pair.py", "a8f42ded66c694c2f9668cfe7c6785b08ae02c231b83d3f4440bf5296cc82051"),
}
sha = play.live.sha


def check(x, description):
    if not x:
        raise ValueError(description)


def read(p):
    return json.loads(gzip.decompress(p.read_bytes()) if p.suffix == ".gz" else p.read_bytes())


def identities():
    base = read(C / "ladder_live/protocol.json")
    for role, (p, h) in FROZEN.items():
        check(
            sha(p) == h
            and (
                role not in ("student", "parent", "opponent")
                or p.stat().st_size <= base["platform"]["source_limit_bytes"]
            ),
            role,
        )
    mirror = ROOT / "sources/master3s2/master3s2.py"
    mirror_raw = (
        mirror.read_bytes()
        if mirror.is_file()
        else subprocess.check_output(
            ["git", "-C", str(ROOT), "cat-file", "blob", "HEAD:sources/master3s2/master3s2.py"]
        )
    )
    check(mirror_raw == FROZEN["opponent"][0].read_bytes(), "original mirrored file")
    for role in ("environment", "official_helpers", "rules"):
        f = base["files"][role]
        check(sha(ROOT / f["path"]) == f["sha256"], role)
    return base


def activation():
    paths = [A / "settlement.json", T / "hosted/verification.json", T / "hosted/artifact.zip"]
    if any(not p.is_file() for p in paths):
        return {
            "eligible": False,
            "status": "pending",
            "missing": [str(p.relative_to(ROOT)) for p in paths if not p.is_file()],
        }
    rel = read(A / "release.json")
    result = read(paths[0])
    timing = read(paths[1])
    check(
        sha(A / "release.json")
        == result["release_file_sha"]
        == "6742a7d20a76254ac2237f23d50b19db0dd9add629f10f803bd9b993a32a7232",
        "old window release",
    )
    check(
        sha(T / "protocol.json")
        == timing["registration_sha256"]
        == "c997aabef84fd387e6c9dd81a877b9b11a34e5d6351c1e5fcde9a2b5751006c4",
        "timer contract",
    )
    check(
        result["packages"] == rel["packages"]
        and rel["packages"]["full"]["sha256"] == FULL
        and rel["packages"]["parent"]["sha256"] == PARENT,
        "conditional model identity",
    )
    check(
        timing["source_sha256"] == FULL
        and timing["verifier_sha256"] == sha(T / "hosted/verify.py")
        and timing["artifact_sha256"] == sha(paths[2]),
        "verified hosted clock identity",
    )
    m = read(T / "hosted/archive.json")
    check(
        m["run_id"] == timing["run_id"]
        and m["head_sha"] == timing["frozen_main"]
        and m["github_digest"] == "sha256:" + timing["artifact_sha256"],
        "hosted archive provenance",
    )
    timer = (
        timing["fixed_replay_limits_pass"]
        and timing["replayed_public_requests"] == 598
        and timing["command_matches_against_local"] == [299, 299]
    )
    check(len(result["cases"]) == len(rel["cases"]) == 2, "requires both signed responses")
    evidence = []
    for case, r in zip(rel["cases"], result["cases"], strict=True):
        seed, seat = case["seed"], case["seat"]
        check(
            (seed, seat, case["t0"], case["terrain"]) == (r["seed"], r["seat"], r["t0"], r["terrain"]),
            "old case identity",
        )
        scored = {}
        changes = 0
        for arm in ("parent", "student"):
            entry = r["arms"][arm]
            path = A / "native" / f"{seed}_{seat}" / (arm + ".json.gz")
            check(sha(path) == entry["receipt_file_sha"], "conditional receipt hash")
            raw = read(path)
            tape = raw["tape"]
            check(
                raw["official_scores"] == entry["official_scores"] and entry["status"] == ["DONE", "DONE"],
                "official terminal",
            )
            scored[arm] = entry["official_scores"][seat] - entry["official_scores"][1 - seat]
            if arm == "student":
                check(len(raw["comparisons"]) == min(32, len(tape)), "window length differs")
                for i, row in enumerate(raw["comparisons"]):
                    check(
                        row["t"] == case["t0"] + i
                        and i < 32
                        and row["executed"] == row["student"] == tape[i][seat],
                        "actually sent student command",
                    )
                    changes += sum(
                        "worker" in u and row["student"].get(u, "NONE") != row["parent"].get(u, "NONE")
                        for u in set(row["student"]) | set(row["parent"])
                    )
        delta = scored["student"] - scored["parent"]
        check(abs(delta - r["margin_delta"]) < 1e-6, "signed official margin")
        evidence.append(dict(seed=seed, seat=seat, changed_worker_commands=changes, margin_delta=delta))
    return dict(
        status="settled",
        timing_pass=timer,
        windows=evidence,
        eligible=bool(
            timer and all(r["changed_worker_commands"] > 0 and r["margin_delta"] > 0 for r in evidence)
        ),
    )


def matrix_check(matrix):
    base = identities()
    g = activation()
    check(g["eligible"], f"activation {g}")
    check(
        matrix["platform"] == base["platform"] and matrix["expected_games"] == 8,
        "platform/attempt denominator changed",
    )
    seeds = matrix["seeds"]
    check(len(set(seeds)) == len(seeds) == 2, "requires two distinct future terrains")
    check(
        set(matrix["physical_boards"]) == set(map(str, seeds))
        and len(set(matrix["physical_boards"].values())) == 2,
        "matrix terrains",
    )
    files = matrix["files"]
    for label, (p, h) in FROZEN.items():
        f = files[label]
        check(ROOT / f["path"] == p and f["sha256"] == h, label)
    for label, path in (("conditional", A / "settlement.json"), ("timing", T / "hosted/verification.json")):
        check(ROOT / files[label]["path"] == path, label)
    for label, item in files.items():
        p = ROOT / item["path"]
        check(sha(p) == item["sha256"] and p.stat().st_size == item["bytes"], label)
    clearance = read(ROOT / files["clearance"]["path"])
    check(
        clearance["physical_boards"] == matrix["physical_boards"]
        and clearance["seed_collisions"] == clearance["terrain_collisions"] == 0
        and clearance["score_complete_ladder_games"] == 1630
        and clearance["distinct_ladder_terrain_hashes"] == 1631,
        "future terrain clearance",
    )
    schedule = [
        dict(
            seed=s, seat=seat, order=(["parent", "student"] if (i + seat) % 2 == 0 else ["student", "parent"])
        )
        for i, s in enumerate(seeds)
        for seat in (0, 1)
    ]
    check(matrix["schedule"] == schedule, "all 8 arms in alternating same-host pairs")
    check(matrix["eligibility"] == g, "frozen gate evidence changed")
    check(matrix["guide_rechecked"]["sha256"] == sha(HERE / "platform_guide.md"), "live guide snapshot")
    return base


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("check", "pair", "verify"))
    p.add_argument("--matrix", type=Path)
    p.add_argument("--matrix-sha")
    p.add_argument("--seed", type=int)
    p.add_argument("--seat", type=int, choices=(0, 1))
    p.add_argument("--out", type=Path)
    p.add_argument("--image")
    a = p.parse_args()
    check(
        a.matrix
        and a.matrix.resolve() == HERE / "protocol.json"
        and a.matrix_sha
        and sha(a.matrix) == a.matrix_sha,
        "matrix path or pinned digest differs",
    )
    matrix = read(a.matrix)
    matrix_check(matrix)
    if a.mode == "check":
        print(
            json.dumps(
                {"eligibility": matrix["eligibility"], "schedule": matrix["schedule"]}, ensure_ascii=False
            )
        )
        return 0
    check(a.seed is not None and a.seat is not None and a.out is not None, "registered seed/seat/output")
    check(
        any((s["seed"], s["seat"]) == (a.seed, a.seat) for s in matrix["schedule"]),
        "unregistered verification pair",
    )
    if a.mode == "verify":
        verify_pair.HERE = a.matrix.parent
        v = verify_pair.verify_pair(a.out, a.seed, a.seat)
        print(json.dumps(v))
        return int(bool(v["verification_errors"]))
    check(a.image, "pinned runner image")
    play.HERE = a.matrix.parent
    play.check = matrix_check
    sys.argv = [
        "paired",
        "--seed",
        str(a.seed),
        "--seat",
        str(a.seat),
        "--out",
        str(a.out),
        "--image",
        a.image,
    ]
    return play.main()


if __name__ == "__main__":
    sys.exit(main())

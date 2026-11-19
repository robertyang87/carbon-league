"""Cross-check archived physical boards and cash against the registered development maps."""
import gzip
import hashlib
import json
import tarfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
ROOT = BASE.parents[1]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    data = path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix == ".gz" else data)


def require(test, message):
    if not test:
        raise ValueError(message)


def run():
    development = BASE / "master6_actor_dev/first_win_dev"
    protocol = read(development / "protocol.json")
    clearance = read(development / "clearance.json")
    bind = digest((development / "protocol.json").read_bytes())
    boards = {int(k): v for k, v in clearance["physical_boards"].items()}
    require(clearance["registered_sha256"] == bind, "registration changed")
    require(len(boards) == len(set(boards.values())) == 8 and set(boards) == set(protocol["seeds"]),
            "incomplete physical opening registration")
    validation = read(HERE / "validation.json")
    manifest = read(HERE / "manifest.json")
    exposure = read(HERE / "exposure.json")
    paths = [HERE.parent / p["relative_dir"] / "validation_maps.jsonl.gz" for p in validation["prior_game_lists"]]
    previous = list(paths)
    paths.extend(path.parent / "unscored/validation_maps.jsonl.gz" for path in previous
                 if (path.parent / "unscored/manifest.json").exists())
    paths.append(HERE / "validation_maps.jsonl.gz")
    seeds, terrains, games = set(), set(), set()
    source_hashes = []
    for path in paths:
        raw = path.read_bytes()
        rows = [json.loads(line) for line in gzip.decompress(raw).splitlines()]
        source_hashes.append([str(path.relative_to(BASE)), len(rows), digest(raw)])
        for row in rows:
            gid, seed, terrain = row["id"], row["seed"], row["initial_terrain_u8_sha256"]
            require(gid not in games and seed not in seeds and terrain not in terrains, "repeated ladder identity")
            seeds.add(seed)
            terrains.add(terrain)
            games.add(gid)
    require(len(games) == exposure["distinct_ids_seeds_and_initial_terrains"] == manifest["accumulated_games"] + 1,
            "unbound history or fault-prefix map")
    require(not (seeds & boards.keys() or terrains & set(boards.values())), "development overlap")
    cash = {g["id"]: g for g in read(HERE / "turn_cash.json.gz")["games"]}
    turns = {g["id"]: g for g in read(HERE / "turn_actions.json.gz")["games"]}
    ledger = {r["id"]: r for r in rows}
    checked = []
    for entry in manifest["new_archives"]:
        raw_path = ROOT / entry["archive"]
        require(digest(raw_path.read_bytes()) == entry["sha256"], "archive changed")
        with tarfile.open(raw_path) as archive:
            for game in manifest["games"]:
                gid = game["id"]
                raw = archive.extractfile(game["path"]).read()
                require(digest(raw) == game["sha256"], "replay changed")
                replay = json.loads(gzip.decompress(raw))
                frames = replay["steps"]
                opening = frames[0][0]["observation"]["full_carbon"]
                require(len(opening) == 225 and all(type(c) is int and 0 <= c <= 255 for c in opening),
                        "bad raw opening")
                require(digest(bytes(opening)) == ledger[gid]["initial_terrain_u8_sha256"]
                        and replay["configuration"]["randomSeed"] == ledger[gid]["seed"], "replay-map mismatch")
                timeline = cash[gid]["turns"]
                require(len(timeline) == len(frames) - 1 == game["transitions"], "incomplete cash ledger")
                error = 0.0
                for t, joint in enumerate(timeline):
                    before, after = [f[0]["observation"]["players"] for f in frames[t:t+2]]
                    require(joint[0] == t + 1 and joint[1] == [p[0] for p in after], "cash-state mismatch")
                    for seat in (0, 1):
                        error = max(error, abs(after[seat][0] - before[seat][0] - sum(joint[2][seat])))
                require(error < .005, "cash components fail to reconcile at displayed precision")
                require([s["reward"] for s in frames[-1]] == game["score_by_seat"], "seat rewards differ")
                rows_by_t = {r["t_before"]: r for r in turns[gid]["transitions"]}
                for _, event, _ in turns[gid]["windows"].values():
                    require(all(max(0, event + off) in rows_by_t for off in (-10, -5, -3)), "missing lead-in")
                for t, row in rows_by_t.items():
                    require(row["commands_as_recorded"] == [s["action"] or {} for s in frames[t+1]],
                            "command time/seat mismatch")
                    require(row["public_players_before"] == frames[t][0]["observation"]["players"],
                            "post-action state used as context")
                checked.append({"id": gid, "seats": game["seat_names"], "official_by_seat": game["score_by_seat"],
                                "joint_cash_transitions": len(timeline), "unique_joint_action_leadin_rows": len(rows_by_t),
                                "windows": turns[gid]["windows"], "max_abs_per_seat_cash_error": round(error, 8)})
    require(len(checked) == len(ledger) == 2, "incomplete new batch")
    return {"score_revision": manifest["revision"], "registered_protocol_sha256": bind,
            "physical_boards_sha256": digest(json.dumps(clearance["physical_boards"], sort_keys=True,
                                                        separators=(",", ":")).encode()),
            "physical_board_provenance": "first_win_dev/clearance.json physical_boards verified by official and local initializations",
            "physical_development_boards": len(boards), "new_scored_boards": len(checked),
            "unique_ladder_boards_including_fault": len(games), "fault_boards": 1,
            "seed_intersection_count": len(seeds & boards.keys()),
            "actual_terrain_intersection_count": len(terrains & set(boards.values())), "games": checked,
            "ladder_registry_files": source_hashes}


if __name__ == "__main__":
    result = run()
    (HERE / "first_win_exclusion.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")

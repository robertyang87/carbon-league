"""Reproduce board clearance from the full scanner receipts and verify both engines on old signed tapes."""
import argparse
import copy
import gzip
import hashlib
import json
from pathlib import Path

import world
from map_identity import ladder_boards


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load(path):
    data = path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix == ".gz" else data)


def publish(path, data):
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"frozen geography evidence differs at {path}")
    else:
        path.write_bytes(data)


def publish_json(path, value):
    publish(path, (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())


def clearance(scans):
    here = world.HERE
    declaration = here / "map_declaration.json"
    plan = load(declaration)
    game_ids = {}
    ladder_files = {}
    for path in sorted((world.BASE / "all_top5").rglob("validation_maps.jsonl.gz")):
        b = path.read_bytes()
        ladder_files[str(path.relative_to(world.ROOT))] = sha(b)
        for line in gzip.decompress(b).splitlines():
            r = json.loads(line)
            game_ids[r["id"]] = (r["seed"], r["initial_terrain_u8_sha256"])
    old = {}
    for name in ["data_split.json", "data_loss_split.json"]:
        for g in load(here.parent / name)["games"]:
            old[g["id"]] = (g["seed"], g["terrain_sha256"])
    dep = {}
    ladder_hash, registered_seeds, registered_terrains = ladder_boards()
    for seed in plan["new_proxy_development"]:
        data = (scans / f"{seed}.json").read_bytes() if scans else gzip.decompress((here / f"map_audit_{seed}.json.gz").read_bytes())
        r = json.loads(data)
        terrain = r["official_opening_sha256"]
        if not (r["candidate_seed"] == seed and terrain == r["candidate_initial_terrain_u8_sha256"]
                and r["frozen_git_revision"] == plan["source_revision"] and r["verdict"] == "eligible_in_checked_scope"
                and not r["reserved"] and not r["conflicts"] and not r["same_terrain_matches"]
                and not r["ladder_seed_or_terrain_match"] and not r["current_declaration"]["frozen_exact_references"]
                and r["current_declaration"]["sha256"] == sha(declaration.read_bytes())):
            raise ValueError("public or local registration/physical board collision")
        if (seed in registered_seeds or terrain in registered_terrains
                or any(seed == s or terrain == h for s,h in list(game_ids.values()) + list(old.values()))):
            raise ValueError("current actor or ladder map collision")
        audit_path = here / f"map_audit_{seed}.json.gz"
        publish(audit_path, gzip.compress(data, compresslevel=6, mtime=0))
        dep[str(seed)] = dict(terrain=terrain, audit_sha256=sha(audit_path.read_bytes()),
                              audited_paths=r["coverage"]["file_or_member_paths"], known_conflicts=0,
                              historical_catalog_sha256=r["historical_catalog_loaded_sha256"])
    if len({v["terrain"] for v in dep.values()}) != len(dep):
        raise ValueError("duplicate development terrain")
    return dict(scope="New proxy development only; source freeze and scanner limitations retained in individual receipts.",
                eligible=True, source_revision=plan["source_revision"], declaration_sha256=sha(declaration.read_bytes()),
                development=dep, registered_ladder_fingerprint=ladder_hash, unique_ladder_game_ids=len(game_ids),
                actor_game_ids_checked=len(old), ladder_registries=ladder_files,
                official_games_played=0)


def histories():
    receipts = [world.HERE.parent / "tree_response_training" / p for p in
                ("source_receipt.json", "batches/intake_02/source_receipt.json")]
    registration = load(world.HERE / "map_declaration.json")
    by_game = []
    proxy_checks = []
    for receipt in receipts:
        for g in load(receipt)["games"]:
            seed, seat = g["seed"], g["seat"]
            b = (world.ROOT / g["raw_path"]).read_bytes()
            if sha(b) != g["raw_sha256"]:
                raise ValueError("changed historical tape")
            raw = json.loads(gzip.decompress(b))
            w = world.World(seed)
            if (raw["seed"], raw["candidate_seat"], raw["initial_terrain_u8_sha256"]) != (seed, seat, w.terrain_sha256):
                raise ValueError("source map or seat differs")
            probes = {name: world.make_opponent(name) for name in world.OPPONENTS} if seed == registration["training"][0] else {}
            for t, acts in enumerate(raw["actions"]):
                if t <= 16:
                    for name, bot in probes.items():
                        public = w.obs(1 - seat)
                        if t == 16:
                            other = copy.deepcopy(bot)
                            alternate = other(public, world.CFG)
                            actual = bot(public, world.CFG)
                            if actual != alternate or other.policy is bot.policy or other.policy.belief_state is bot.policy.belief_state:
                                raise ValueError("shared proxy history or non-deterministic fork")
                            branch = copy.deepcopy(w)
                            branch.advance(acts)
                            if w.step != t or branch.step != t + 1:
                                raise ValueError("shared world between forks")
                            proxy_checks.append(dict(seed=seed, seat=1-seat, proxy=name, step=t,
                                joint_action_sha256=sha(json.dumps(actual,sort_keys=True).encode())))
                        else:
                            bot(public, world.CFG)
                w.advance(acts)
            terminal = world.replay_official(seed, raw["actions"])
            if terminal != w.terminal or terminal["scores"] != g["scores"] or terminal["scores"] != raw["official_score_by_seat"]:
                raise ValueError("official recorded, FastCarbon, or replayed outcome differs")
            by_game.append(dict(seed=seed, seat=seat, raw_sha256=sha(b), **terminal))
    expected = {(s,i) for s in registration["training"] for i in (0,1)}
    if {(g["seed"],g["seat"]) for g in by_game} != expected or len(by_game) != len(expected):
        raise ValueError("training source tapes are missing, duplicate, or not predeclared")
    return dict(scope="Historical command replay, no new live games. Proxy clone parity on old public prefixes only.",
                policy_sha256={k:v[1] for k,v in world.OPPONENTS.items()}, tapes=by_game,
                jointly_replayed_steps=sum(g["steps"] for g in by_game), proxy_clones=proxy_checks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scans", type=Path, help="directory containing the two original full scanner receipts")
    args = parser.parse_args()
    publish_json(world.HERE / "map_clearance.json", clearance(args.scans))
    publish_json(world.HERE / "world_verification.json", histories())

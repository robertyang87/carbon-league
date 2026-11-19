"""Validate registered boards, official scoring and both seats using existing tapes, without a live opponent."""
import argparse
import copy
import gzip
import hashlib
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import worlds
from map_identity import ladder_boards


def load(path):
    data = path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix == ".gz" else data)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def publish(path, data):
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"immutable evidence differs at {path}")
    else:
        path.write_bytes(data)


def emit(path, value):
    publish(path, (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())


def seal(scans=None):
    plan = worlds.registry()
    root, base, here = worlds._base.ROOT, worlds._base.BASE, worlds.HERE
    previous = load(root / plan["prior_proxy_development_exclusion"])
    ladder = {}
    for p in sorted((base / "all_top5").rglob("validation_maps.jsonl.gz")):
        for line in gzip.decompress(p.read_bytes()).splitlines():
            r = json.loads(line)
            key, identity = r["id"], (r["seed"], r["initial_terrain_u8_sha256"])
            if key in ladder and ladder[key] != identity:
                raise ValueError("conflicting ladder identity")
            ladder[key] = identity
    actor = {}
    for p in ("data_split.json", "data_loss_split.json"):
        for r in load(here.parent / p)["games"]:
            actor[r["id"]] = (r["seed"], r["terrain_sha256"])
    occupied = {int(s): v["terrain"] for s,v in previous["development"].items()}
    occupied.update({int(s): v["terrain_sha256"] for s,v in plan["historical_maps"].items()})
    expected_sha = sha(worlds.DECL.read_bytes())
    registry_hash, seed_index, terrain_index = ladder_boards()
    dev = {}
    for seed in plan["development"]:
        archive = here / f"map_audit_{seed}.json.gz"
        raw = (scans / f"{seed}.json").read_bytes() if scans else gzip.decompress(archive.read_bytes())
        d = json.loads(raw)
        terrain = d["official_opening_sha256"]
        if (d["verdict"] != "eligible_in_checked_scope" or d["candidate_seed"] != seed
                or d["frozen_git_revision"] != plan["source_revision"] or d["current_declaration"]["sha256"] != expected_sha
                or d["current_declaration"]["frozen_exact_references"] or d["reserved"] or d["ladder_seed_or_terrain_match"]
                or d["candidate_initial_terrain_u8_sha256"] != terrain or d["conflicts"] or d["same_terrain_matches"]
                or seed in seed_index or terrain in terrain_index or seed in occupied or terrain in occupied.values()
                or any(seed == s or terrain == h for s,h in [*ladder.values(),*actor.values()])):
            raise ValueError(f"prior exposure, incompatible opening, or unregistered map {seed}")
        occupied[seed] = terrain
        publish(archive, gzip.compress(raw, compresslevel=6, mtime=0))
        dev[str(seed)] = dict(terrain_sha256=terrain, prior_matches=0, audit_file=archive.name,
                              audit_sha256=sha(archive.read_bytes()), scanned_paths=d["coverage"]["file_or_member_paths"])
    return dict(scope="Two proxy development boards only, prior workspace/archives and known training/ladder maps; exact scope in frozen receipts.",
                source_revision=plan["source_revision"], declaration_sha256=expected_sha, eligible=True,
                development=dev, previous_proxy_development_exclusion_sha256=sha((root / plan["prior_proxy_development_exclusion"]).read_bytes()),
                ladder_games=len(ladder), ladder_registry_sha256=registry_hash, actor_game_ids=len(actor))


def historic():
    plan = worlds.registry()
    verified = []
    for split in ("screen","training"):
        for seed in plan[split]:
            identity = plan["historical_maps"][str(seed)]
            p = worlds._base.ROOT / identity["source_receipt"]
            if sha(p.read_bytes()) != identity["source_receipt_sha256"]:
                raise ValueError("history source receipt changed")
            for g in (r for r in load(p)["games"] if r["seed"] == seed):
                seat = g["seat"]
                path = worlds._base.ROOT / g["raw_path"]
                if sha(path.read_bytes()) != g["raw_sha256"] or g["raw_sha256"] != identity["seat_raw_sha256"][str(seat)]:
                    raise ValueError("historical joint action tape changed")
                raw = load(path)
                w = worlds.World(seed,split)
                if (raw["seed"],raw["candidate_seat"],raw["initial_terrain_u8_sha256"]) != (seed,seat,w.terrain_sha256):
                    raise ValueError("raw seed/seat/actual terrain differs")
                for joint in raw["actions"]:
                    w.advance(joint)
                official = worlds.replay_official(seed, raw["actions"], split)
                if official != w.terminal or official["scores"] != raw["official_score_by_seat"] or official["scores"] != g["scores"]:
                    raise ValueError("official terminal scores or physical terrain differ")
                verified.append(dict(seed=seed,seat=seat,split=split,raw_sha256=g["raw_sha256"],**official))
    if len(verified) != 20 or len({(r["seed"],r["seat"]) for r in verified}) != 20:
        raise ValueError("incomplete 10 old maps in both seats")
    return dict(scope="Archived trajectories only; no new games or policy calls",tapes=verified,
                joint_steps=sum(r["steps"] for r in verified))


class WorldsTests(unittest.TestCase):
    def test_roles_and_physical_terrains(self):
        plan = worlds.registry()
        seen = set()
        for role in worlds.SPLITS:
            for seed in plan[role]:
                w = worlds.World(seed,role)
                self.assertNotIn(w.terrain_sha256, seen)
                seen.add(w.terrain_sha256)
                self.assertEqual([w.obs(0)["player"], w.obs(1)["player"]], [0,1])
                self.assertNotIn("full_carbon", w.obs(0))
                for other in worlds.SPLITS:
                    if other != role:
                        with self.assertRaises(ValueError):
                            worlds.World(seed,other)
        self.assertEqual(len(seen),12)
        prior = load(worlds._base.ROOT / plan["prior_proxy_development_exclusion"])
        self.assertFalse({r["terrain"] for r in prior["development"].values()} & seen)
        for exposed in prior["development"]:
            with self.assertRaises(ValueError):
                worlds.World(int(exposed))
        with self.assertRaises(ValueError):
            worlds.World(-1)

    def test_physical_board_mismatch_and_experiment_namespace(self):
        seed = worlds.registry()["training"][0]
        with patch.object(worlds,"locked_terrain",return_value="0"*64), self.assertRaises(ValueError):
            worlds.World(seed)
        old_path = worlds.PRIOR
        import importlib.util
        spec = importlib.util.spec_from_file_location("unchanged_former_experiment_world",old_path)
        former = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(former)
        with self.assertRaisesRegex(ValueError, "outside"):
            former.World(worlds.registry()["screen"][0])

    def test_three_step_extinction_and_fork(self):
        s = worlds.registry()["screen"][0]
        original = worlds.World(s,"screen")
        b = [next(iter(original.obs(p)["players"][p][1])) for p in (0,1)]
        joint = [[{b[0]:"RECCOLLECTOR"},{}],
                 [{b[0]:"RECCOLLECTOR","player-0-worker-0":"UP"},{}],
                 [{"player-0-worker-0":"DOWN"},{b[1]:"RECCOLLECTOR"}]]
        branch = copy.deepcopy(original)
        for commands in joint:
            branch.advance(commands)
        self.assertEqual(original.step,0)
        self.assertEqual(branch.terminal,worlds.replay_official(s,joint,"screen"))
        self.assertEqual(branch.terminal["scores"],[-298,80])
        self.assertEqual(branch.terminal["extinct_by_seat"],[True,False])
        with self.assertRaises(ValueError):
            branch.advance([{},{}])
        with self.assertRaises(ValueError):
            worlds.replay_official(s,joint[:-1],"screen")

    def test_signed_new_board_receipts_and_old_dual_seat_scores(self):
        frozen = load(worlds.HERE/"map_clearance.json")
        for row in frozen["development"].values():
            path = worlds.HERE / row["audit_file"]
            self.assertEqual(sha(path.read_bytes()),row["audit_sha256"])
            x = load(path)
            self.assertEqual(x["official_opening_sha256"],row["terrain_sha256"])
        self.assertEqual(historic(),load(worlds.HERE/"world_verification.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,add_help=False)
    parser.add_argument("--seal", action="store_true")
    parser.add_argument("--scans", type=Path)
    args, rest = parser.parse_known_args()
    if args.seal:
        emit(worlds.HERE/"map_clearance.json",seal(args.scans))
        emit(worlds.HERE/"world_verification.json",historic())
    else:
        unittest.main(argv=[sys.argv[0],*rest])

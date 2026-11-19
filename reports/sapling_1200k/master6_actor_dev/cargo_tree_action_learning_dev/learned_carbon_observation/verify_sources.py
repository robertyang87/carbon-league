"""Verify physical terrain separation and the archive-only cohort without rerunning policies."""
import sys
from common import FastCarbon, HERE, PROTOCOL, ROOT, read, save, sha
sys.path.insert(0, str(HERE.parents[2]))
import map_identity

original = read(HERE.parent / "panel.json")
cards = PROTOCOL["cards"]
rank = sorted((c for c in original["cards"] if c["seat"] == 0 and original["split"][c["terrain"]] == "train" and 621470000 <= c["seed"] < 621500000), key=lambda c:sha(("public-hidden-carbon-v1/"+c["terrain"]).encode()))
assert [(c["terrain"],c["sha256"]) for c in rank] == [(c["terrain"],c["sha256"]) for c in cards]
assert all(c["split"] == ("train" if i < len(cards)-3 else "val") for i,c in enumerate(cards))
boards = {c["terrain"] for c in cards}
seeds = {c["seed"] for c in cards}
assert len(cards) == len(boards) == len(seeds)
prior = {h for h, split in original["split"].items() if split != "train"}
ladder_sig, ladder_seeds, ladder_terrains = map_identity.ladder_boards()
reserved = {621461321,621461322,621565120,621565121,621675585,621675586}
for run in ("first_win_dev","first_win_blend_dev"):
    reserve_record = read(HERE.parents[1] / run / "protocol.json")
    reserved.update(reserve_record["seeds"])
    reserved.update(reserve_record.get("historical_control_maps", []))
reserved.update(range(621758343,621758356))
cfg = read(ROOT / cards[0]["path"])["game_configuration"]
physical = {seed:sha(bytes(map(int,FastCarbon(cfg).reset(seed=seed).carbon))) for seed in reserved}
for c in cards:
    tape = read(ROOT/c["path"])
    assert c["terrain"] == tape["initial_terrain_u8_sha256"] == sha(bytes(map(int,tape["initial_full_carbon"])))
    assert all(a==b for a,b in zip(tape["initial_full_carbon"],FastCarbon(cfg).reset(seed=c["seed"]).carbon))
assert seeds.isdisjoint(ladder_seeds | reserved)
assert boards.isdisjoint(ladder_terrains | prior | set(physical.values()))
assert not any(1887000000 <= s < 1888000000 for s in seeds)
save(HERE/"source_clearance.json",dict(cards=len(cards),distinct_physical_hashes=len(boards),prior_parent_validation_and_test_terrains=len(prior),
    matched_public_ladder_hashes=0,matched_reserved_hashes=0,matched_parent_holdout_hashes=0,formal_seed_collisions=0,
    official_ladder_roster_sha256=ladder_sig,ladder_terrain_count=len(ladder_terrains),excluded_first_win_controls_and_343_355=sorted(reserved),
    seed_reconstruction_matches=len(cards),protocol_sha256=sha((HERE/"protocol.json").read_bytes())))

"""Verify fresh board identities across the source-tracked corpus, placements and final-exam reserve."""
import gzip
import importlib.util
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
FAMILY = HERE.parents[2]
SOURCE = "05b1f28644ef015cdf9cd67c02174dc326da05a0"
SEEDS = [621758592, 621758593]
TOKEN = re.compile(rb"(?<![.\da-zA-Z])62175859[23](?![.\da-zA-Z])")
spec = importlib.util.spec_from_file_location("v2_clearance", FAMILY/"cold_start/development/clearance.py")
v2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v2)

def run():
    v2.BASE, v2.SEEDS, v2.TOKEN = SOURCE, SEEDS, TOKEN
    census = v2.generate()
    boards = census["by_seed"]
    old = v2.prior.old
    prev = json.loads(v2.prior.frozen(FAMILY/"cold_start/development/protocol.json",SOURCE))
    assert not set(boards.values()) & set(prev["physical_boards"].values()) and not set(SEEDS) & set(prev["seeds"])
    placement = json.loads(v2.prior.frozen(FAMILY/"ladder/ds-pro-v1-placement.json",SOURCE))
    ladder = []
    for row in placement["games"]:
        match, game = row["match_id"], row["game"]
        blob = v2.prior.frozen(FAMILY/f"ladder/replays/{match}-{game}.json.gz",SOURCE)
        env = json.loads(gzip.decompress(blob))
        obs = env["steps"][0][0]["observation"]
        assert obs["step"]==0
        terrain = old.engine.map_identity.terrain_sha256(obs["full_carbon"])
        seed = env["configuration"]["randomSeed"]
        assert seed not in SEEDS and terrain not in boards.values()
        ladder.append(dict(match=match,game=game,seed=seed,terrain=terrain,replay_sha256=v2.prior.sha(blob)))
    assert len(ladder)==len({r["match"] for r in ladder})==placement["settled_games"]
    census.update(schema="master3s2-ladder-feedback/development-clearance/1",
                  purpose="Two new development maps with both candidates in both seats against original Master3s2. Exclude from formal exam and training.",
                  verifier_sha256=v2.prior.sha(Path(__file__).read_bytes()),
                  historical_verifier_sha256=v2.prior.sha((FAMILY/"cold_start/development/clearance.py").read_bytes()),
                  groups={"paired_development_only":SEEDS}, ladder_placement_openings=ladder)
    return census

if __name__=="__main__":
    result=run()
    print(v2.shared.save(HERE/"clearance.json",result),result["by_seed"])

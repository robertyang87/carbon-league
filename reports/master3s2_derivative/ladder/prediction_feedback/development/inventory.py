"""Extend the frozen all-history board census and compare both official and replay seed namespaces."""
import argparse
import gzip
import json
from pathlib import Path

from clearance import v2, HERE, FAMILY, SEEDS, SOURCE, TOKEN

def collect(endpoint):
    prior=v2.prior
    census=json.loads((HERE/"clearance.json").read_bytes())
    boards=census["by_seed"]
    prior.SEEDS=prior.old.SEEDS=SEEDS
    prior.IDS=TOKEN
    require_snapshot=prior.git("rev-parse",endpoint+"^{commit}").decode().strip()
    if endpoint != require_snapshot:
        raise ValueError("Expected a full frozen revision")
    delta=dict(by_seed=boards,remote_extension=[])
    prior.extend(delta,SOURCE,endpoint)
    added=prior.verify_added_boards(SOURCE,endpoint,boards)
    placement=json.loads(prior.frozen(FAMILY/"ladder/ds-pro-v1-placement.json",endpoint))
    layouts={}
    for row in placement["games"]:
        match=row["match_id"]; game=row["game"]
        raw=prior.frozen(FAMILY/f"ladder/replays/{match}-{game}.json.gz",endpoint)
        env=json.loads(gzip.decompress(raw))
        initial=env["steps"][0][0]["observation"]
        observed=prior.old.engine.map_identity.terrain_sha256(initial["full_carbon"])
        card_seed=row["map_seed"]; physical_seed=env["configuration"]["randomSeed"]
        if initial["step"]!=0 or {card_seed,physical_seed}&set(SEEDS) or {observed,row["map_sha256"]}&set(boards.values()):
            raise ValueError("Placement map overlap: "+match)
        layouts[match]=dict(game=game,platform_map_seed=card_seed,replay_random_seed=physical_seed,
                            replay_opening_u8_sha256=observed,platform_map_sha256=row["map_sha256"],
                            replay_sha256=prior.sha(raw))
    if len(layouts)!=placement["settled_games"]:raise ValueError("Incomplete ladder placement ledger")
    groups={}
    for split in ("train","held"):
        path=HERE.parent/(split+"_result.json");data=path.read_bytes()
        members=json.loads(data)["games"]
        if len(members)!=2:raise ValueError("Wrong fit split")
        for row in members:
            if row["terrain"]!=layouts[row["match"]]["replay_opening_u8_sha256"] or row["terrain"] in boards.values():
                raise ValueError("Wrong physical fit source")
        groups[split]=dict(sha256=prior.sha(data),maps=[r["terrain"] for r in members])
    return dict(through=endpoint,full_census_revision=SOURCE,boards=boards,
                clearance_sha256=prior.sha((HERE/"clearance.json").read_bytes()),
                verifier_sha256=prior.sha(Path(__file__).read_bytes()),new_replay_layouts=added,
                extension=delta["remote_extension"],ladders=layouts,learner=groups,passed=True)

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--through",required=True)
    args=p.parse_args()
    value=collect(args.through)
    print(v2.shared.save(HERE/"inventory.json",value))

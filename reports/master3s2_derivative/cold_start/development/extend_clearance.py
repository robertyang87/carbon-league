"""Check changes after the full inventory, including new archived initial boards and remote ladder scores."""
import argparse
import json
from pathlib import Path

from clearance import BASE, HERE, SEEDS, TOKEN, prior, shared


def run(through):
    frozen = (HERE/"clearance.json").read_bytes()
    old = json.loads(frozen)
    if old["source_snapshot"]!=BASE or old["verifier_sha256"]!=prior.sha((HERE/"clearance.py").read_bytes()):
        raise ValueError("Full census or implementation differs")
    prior.SEEDS = prior.old.SEEDS = SEEDS
    prior.IDS = TOKEN
    roots=old["by_seed"]
    state=dict(by_seed=roots,remote_extension=[])
    prior.extend(state,BASE,through)
    layouts=prior.verify_added_boards(BASE,through,roots)
    state.update(after=BASE,through=through,through_tree=prior.git("rev-parse",through+"^{tree}").decode().strip(),
                 source_census_sha256=prior.sha(frozen),verifier_sha256=prior.sha(Path(__file__).read_bytes()),
                 added_replay_openings=layouts,seed_intersections=[],physical_intersections=[])
    return state


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--through",required=True)
    args=parser.parse_args()
    print(shared.save(HERE/"clearance_at_freeze.json",run(args.through)),flush=True)

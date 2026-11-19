"""Clear the windowed candidate's predeclared development maps without loading match results."""
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parents[1] / "fresh_original"
REFERENCE = "b4e0b78c2fdcb43ff6e81dc6fd1c5ddbce07a7d6"
CHECKER = "b9f04b7900253baf649aa3a25b397c7d2eb51a64cc72fa71974d4096bc9d78a7"
PRIOR_CLEARANCE = "132763b246551cbc5443600f8fe8d7c7ee99bf4a06dab55747053f72eb4474db"
SEEDS = (621758354, 621758355)
RESERVED = (621758350, 621758351, 621758352, 621758353)


def inspect():
    filename = PRIOR / "check_maps.py"
    if hashlib.sha256(filename.read_bytes()).hexdigest() != CHECKER:
        raise ValueError("earlier clearance checker differs from registered code")
    spec = importlib.util.spec_from_file_location("prior_sapling_map_clearance", filename)
    prior = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prior)
    prior.REVISION = REFERENCE
    prior.SEEDS = SEEDS
    receipt = prior.inspect()

    existing = (PRIOR / "clearance.json").read_bytes()
    if hashlib.sha256(existing).hexdigest() != PRIOR_CLEARANCE:
        raise ValueError("earlier development clearance changed")
    excluded = {str(seed): prior.map_identity.local_board(seed) for seed in RESERVED}
    earlier = json.loads(existing)["physical_boards"]
    if any(excluded[str(seed)] != earlier[str(seed)] for seed in RESERVED[2:]):
        raise ValueError("previously reserved official terrain changed")
    registered = receipt["physical_boards"]
    if set(registered) & set(excluded) or set(registered.values()) & set(excluded.values()):
        raise ValueError("windowed development cohort overlaps prior reservation")
    receipt["explicit_earlier_development_reservations"] = excluded
    receipt["earlier_clearance_sha256"] = PRIOR_CLEARANCE
    receipt["frozen_checker_sha256"] = CHECKER
    return receipt


if __name__ == "__main__":
    print(json.dumps(inspect(), indent=2, sort_keys=True))

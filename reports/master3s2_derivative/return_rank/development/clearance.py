"""Reuse the earlier whole-repo census and live ladder identity checks for two new development maps."""
import importlib.util
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
FAMILY = HERE.parents[1]
BASE = '341d222f35866da9f8fad6ee51a6819b7094fbfd'
SEEDS = [621758598, 621758599]
matcher = re.compile(rb'(?<![.\da-zA-Z])62175859[89](?![.\da-zA-Z])')
archive_dir = FAMILY / 'ladder/cargo_defense/development'
spec = importlib.util.spec_from_file_location('cargo_map_census', archive_dir/'clearance.py')
cargo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cargo)
cargo.SOURCE, cargo.SEEDS, cargo.TOKEN = '91d1e13719060a50efd36cfc443510f0456c2b0c', SEEDS, matcher


def run(census=None):
    # The shared scanner checks numerical seeds as well as official constructed zero-turn maps.
    # This immutable revision predates this registration. Later experiment schedules are checked again at freeze.
    result = cargo.complete(census if census is not None else cargo.full_census(), BASE)
    boards = result['by_seed']
    for lane in ('ladder/cargo_defense/development','ladder/prediction_feedback/development','cold_start/development','dev1'):
        path = FAMILY/lane/'protocol.json'
        other = json.loads(cargo.prior.frozen(path,BASE))
        assert not set(SEEDS).intersection(other['seeds']) and not set(boards.values()).intersection(other['physical_boards'].values())
    result.update(schema='master3s2-return-rank-development-clearance/1',groups={'return_rank_full_development_only':SEEDS},
                  verifier_sha256=cargo.prior.sha(Path(__file__).read_bytes()),source_census_runner_sha256=cargo.prior.sha((archive_dir/'clearance.py').read_bytes()))
    return result


if __name__ == '__main__':
    result=run()
    print(cargo.old.shared.save(HERE/'clearance.json',result),result['by_seed'])

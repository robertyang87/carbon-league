"""Measure whether the rent-guided offer actually survives native search."""
import argparse
import importlib.util
import json

from build import FAMILY, HERE, PARENT, PARENT_SHA, build, sha

path = FAMILY/'native_critic_update/public_search/check.py'
spec = importlib.util.spec_from_file_location('counterfactual_rent_site_probe', path)
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)
STEPS = (30, 65)


def register():
    path = FAMILY/'live_tree_income/search_protocol.json'
    source = path.read_bytes()
    source_rows = json.loads(source)['tickets']
    # Leave the four maps used to test the earlier leaf re-ranking untouched.
    earlier = {r['seed'] for r in source_rows}
    positions = {461, 463, 471, 473}
    cases = []
    for panel in ('native_speed_dev', 'switch_margin_dev'):
        import zipfile
        for file in sorted((FAMILY/panel/'hosted').glob('*.zip')):
            with zipfile.ZipFile(file) as z:
                reg = z.read('parent/registration.json')
            r = json.loads(reg)
            seed, seat = r['seed'], r['candidate_seat']
            if seed % 1000 not in positions:
                continue
            assert seed not in earlier and r['files']['candidate']['sha256'] == PARENT_SHA
            assert r['files']['opponent']['sha256'] == shared.ORIGINAL
            for step in STEPS:
                ticket = dict(archive=str(file.relative_to(FAMILY)), archive_sha256=sha(file.read_bytes()),
                              arm='parent', registration_sha256=sha(reg), seed=seed, seat=seat,
                              terrain=r['physical_map_sha256'], step=step,
                              rng=int(sha(f'rent-guided-menu/1/{seed}/{seat}/{step}'.encode())[:8],16))
                cases.append(ticket)
    assert len(cases) == 16 and len({(r['seed'],r['seat'],r['step']) for r in cases}) == 16
    return dict(schema='rent-guided-candidate-menu/1', packages=shared.DIGESTS, compare_to=sha(source),
                intervention='In t20–92 reserve the final existing candidate slot for one unblocked empty tree location. '
                             'Frozen 16-turn rent estimate ranks locations; native leaf and original clock rank complete commands.',
                tickets=cases)


if __name__ == '__main__':
    cli = argparse.ArgumentParser()
    cli.add_argument('--register', action='store_true')
    cli.add_argument('--cpu', type=int)
    args = cli.parse_args()
    shared.HERE, shared.PARENT, shared.CANDIDATE = HERE, PARENT, HERE/'candidate.py'
    shared.DIGESTS, shared.STEPS, shared.registration = (PARENT_SHA, build()), STEPS, register
    if args.register:
        raw = shared.encode(register())
        dst = HERE/'protocol.json'
        assert not dst.exists() or dst.read_bytes() == raw
        dst.write_bytes(raw)
        print(sha(raw), flush=True)
    else:
        shared.execute(args.cpu)

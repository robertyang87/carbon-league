"""Reuse the official native response harness to measure the changed first actions."""
import argparse
import gzip
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PUBLIC_SEARCH = HERE.parent
FAMILY = HERE.parents[2]
COMPARISON = FAMILY / 'root_rotation_policy/response'


def load_runner():
    # The existing runner binds its own register module at import time.
    sys.path.insert(0, str(COMPARISON))
    spec = importlib.util.spec_from_file_location('official_public_response', COMPARISON / 'run.py')
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(COMPARISON))
    module.HERE = HERE
    return module


def registration(sha):
    source = (PUBLIC_SEARCH / 'results.json.gz').read_bytes()
    study = json.loads(gzip.decompress(source))
    proto = (PUBLIC_SEARCH / 'protocol.json').read_bytes()
    assert sha(proto) == study['protocol_sha256']
    parent, candidate = json.loads(proto)['packages']
    by_board = {}
    for row in study['rows']:
        if row['reliable_change'] and row['ticket']['step'] == 247:
            ticket = row['ticket']
            by_board.setdefault(ticket['seed'], []).append(row)
    tickets = []
    for seed in sorted(by_board)[:2]:
        row = min(by_board[seed], key=lambda r: r['ticket']['seat'])
        ticket = row['ticket']
        orders = {package: row['live'][package][0]['issued'] for package in ('v4', 'critic')}
        assert orders['v4'] != orders['critic']
        order = ['v4', 'critic'] if not tickets else ['critic', 'v4']
        tickets.append(ticket | dict(history_source_package=parent, orders=orders, own_public_sha256=row['public_sha256'],
                                     initial_cost=row['cost'], rival_rng=ticket['rng'] ^ 0x7D4A1F6B,
                                     branch_order=order))
    return dict(schema='native_critic_actual_root_response/1', source_probe_sha256=sha(source),
                continuation_parent=parent, source_candidate=candidate,
                examiner='65f268b2919ce30fe19355115a928eada032a6beaa018d801662d434c083f2b5',
                selection='At t247, first two maps with reliable changed commands in ascending seed order; lower seat breaks ties.',
                method='One changed root action followed by native V4 against the complete original with identical pre-branch state and random streams, including the same original root order.',
                tickets=tickets)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--cpu', type=int)
    ap.add_argument('--register', action='store_true')
    args = ap.parse_args()
    runner = load_runner()
    plan = registration(runner.sha)
    filename = HERE / 'protocol.json'
    raw = (json.dumps(plan, indent=2, allow_nan=False) + '\n').encode()
    if args.register:
        assert not filename.exists() or filename.read_bytes() == raw
        filename.write_bytes(raw)
        print(runner.sha(raw), [(t['seed'], t['seat']) for t in plan['tickets']])
    else:
        assert filename.read_bytes() == raw and plan['tickets']
        runner.PLAN, runner.LOCK = plan, runner.sha(raw)
        runner.go(args.cpu)

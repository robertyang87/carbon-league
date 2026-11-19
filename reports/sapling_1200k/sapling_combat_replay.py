"""Replay archived joint actions to audit collision transfers and changes in tree ownership."""
import argparse
import collections
import gzip
import json
from pathlib import Path
import statistics

import tree_audited_game as arena
from tree_win_gate import validate_game, verify_raw


def replay(run, row, pack_sha):
    verify_raw(run, row, pack_sha)
    validate_game(row)
    with gzip.open(run / row['raw_file'], 'rt') as source:
        record = json.load(source)
    side = row['candidate_seat']
    sim = arena.exam.FastCarbon(record['game_configuration']).reset(seed=row['seed'])
    periods = {phase: collections.Counter() for phase in ('t1_100', 't101_200', 't201_299')}
    for t, actions in enumerate(record['actions'], 1):
        counts = periods['t1_100' if t <= 100 else 't101_200' if t <= 200 else 't201_299']
        ownership = {tid: tree.player for tid, tree in sim.trees.items()}
        sim.step(*actions)
        for _, owner, cargo, recipient, _ in sim.last_collisions:
            role = 'candidate' if owner == side else 'opponent'
            counts[f'{role}_workers_lost_in_collisions'] += 1
            counts[f'{role}_cargo_on_workers_lost_in_collisions'] += cargo
            if recipient == 1-owner:
                counts[f'{role}_cargo_transferred_to_enemy_in_collisions'] += cargo
        for tid, tree in sim.trees.items():
            if tid in ownership and tree.player != ownership[tid]:
                loser = 'candidate' if ownership[tid] == side else 'opponent'
                counts[f'{loser}_trees_taken_by_enemy'] += 1
    if arena.prior.hashed(sim.get_state()) != row['final_state_sha256']:
        raise ValueError(f'Terminal state mismatch for {row["seed"]}/{side}')
    total = sum(periods.values(), collections.Counter())
    return dict(seed=row['seed'], candidate_seat=side, margin=row['margin'], raw_sha256=row['raw_sha256'],
                final_state_sha256=row['final_state_sha256'], phases=periods, total=total)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    arms = {}
    for run in args.runs:
        report = json.loads((run / 'results.json').read_text())
        if report['status'] != 'complete' or run.name in arms:
            raise ValueError('A completed run with a unique name is required')
        games = [replay(run, row, report['header']['pack_sha256']) for row in report['games']]
        by_phase = {}
        for phase in ('t1_100', 't101_200', 't201_299', 'total'):
            ledgers = [g['total'] if phase == 'total' else g['phases'][phase] for g in games]
            keys = set().union(*(ledger.keys() for ledger in ledgers))
            by_phase[phase] = {k: statistics.mean(ledger[k] for ledger in ledgers) for k in sorted(keys)}
        arms[run.name] = dict(report_sha256=arena.prior.digest(run/'results.json'),
                              mean_per_scheduled_game=by_phase, games=games)
    arena.prior.shared.atomic_json(args.out, dict(
        definition='Loss counts one deleted worker, including empty workers and friendly collisions. Transfers require the opposite seat to receive the lost cargo via worker, tree or base. Cargo carried by a winning collector is only realized as cash if later deposited. Tree losses count live ID ownership changes after a full transition, excluding expiry. Phase values use all scheduled games as denominator.',
        source_sha256=arena.prior.digest(__file__), fastsim_sha256=arena.prior.digest(arena.exam.ROOT/'code/fastsim.py'), arms=arms))


if __name__ == '__main__':
    main()

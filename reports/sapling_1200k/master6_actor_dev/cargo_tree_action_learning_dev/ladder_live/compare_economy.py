"""Settle the same exposed seat's old win and independently timed retest from both tapes."""

from collections import Counter
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import math
import sys

import live

sys.path.insert(0, str(live.ROOT / 'reports/sapling_1200k'))
from turning_point_ledger import CashEvents, FastCarbon


def settle(raw, rows, official_scores, recompute_public):
    seat = live.PLAN['candidate_seat']
    env, _ = live.official()
    sim = FastCarbon(raw['game_configuration']).reset(seed=raw['seed'])
    if list(sim.carbon) != list(env.states[0].observation.full_carbon):
        raise ValueError('Physical openings disagree')
    sim.cash = CashEvents(sim.cash)
    initial = list(sim.cash)
    bank = [5., 5.]
    periods = {}
    turns = []
    for row in rows:
        t = sim.step_count
        if t != row['step'] or env.done:
            raise ValueError('Nonconsecutive or post-terminal move')
        observations = [live.observe(env, s, bank[s]) for s in (0, 1)]
        if recompute_public:
            hashes = [hashlib.sha256(live.encoded(o)).hexdigest() for o in observations]
            if hashes != row['public_observation_sha256']:
                raise ValueError(f'Wrong public history at step {t}')
            if any(c['status'] != 'ok' for c in row['calls']):
                break
            actions = [c['response']['action'] for c in row['calls']]
            bank = row['overage_remaining_seconds']
        else:
            actions = row['actions']
        before = list(sim.cash)
        before_players = json.loads(json.dumps(observations[seat]['players']))
        sim.cash.events.clear()
        sim.step(*actions)
        env.step(actions)
        official_cash = [p[0] for p in env.states[0].observation.players]
        if any(abs(a-b) > 1e-6 for a, b in zip(sim.cash, official_cash)):
            raise ValueError(f'Physical cash differs from official interpreter at {t}')
        totals = periods.setdefault(str(t//100), [Counter(), Counter()])
        detail = [Counter(), Counter()]
        for kind, side, amount, *fields in sim.cash.events:
            if kind == 'tree_absorption':
                detail[side]['ground_rent'] += amount - fields[2]
                detail[side]['cargo_tax'] += fields[2]
            else:
                detail[side][kind] += amount
        for s in (0, 1):
            if not math.isclose(sum(detail[s].values()), sim.cash[s]-before[s], abs_tol=1e-6):
                raise ValueError(f'Ledger failed at step {t}, seat {s}')
            totals[s].update(detail[s])
        delta = [sim.cash[s]-before[s] for s in (0, 1)]
        transactions = [e for e in sim.cash.events if e[0] not in ('tree_absorption', 'rounding') and e[2]]
        turns.append(dict(t=t, players_before=before_players, actions=actions, physical_cash_after=list(sim.cash),
                          candidate_margin_change=round(delta[seat]-delta[1-seat], 2), postings=detail,
                          attributed_transactions=transactions))
    scores = [s.reward for s in env.states] if env.done else None
    if official_scores is not None and scores != official_scores:
        raise ValueError('Official terminal scores differ from the artifact')
    pivots = {}
    if turns:
        for name, select in (('largest_one_step_loss', min), ('largest_one_step_gain', max)):
            selected = select(turns, key=lambda r: r['candidate_margin_change'])
            t = selected['t']
            pivots[name] = dict(t=t, change=selected['candidate_margin_change'],
                                from_t_minus_10=turns[max(0,t-10):t+1])
    return dict(official_scores=scores, settled_transitions=len(turns), opening_cash=initial, final_cash=list(sim.cash),
                periods_by_physical_seat=periods, pivots=pivots)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('game', type=Path, help='Directory with hosted receipt.json and turns.jsonl')
    cli.add_argument('out', type=Path)
    args = cli.parse_args()
    card = live.HERE.parent / 'direct_original/games/candidate/raw_621758346_1.json.gz'
    raw = json.loads(gzip.decompress(card.read_bytes()))
    receipt = json.loads((args.game/'receipt.json').read_text())
    if receipt['protocol_sha256'] != live.sha(live.HERE/'protocol.json') or receipt['seed'] != raw['seed']:
        raise ValueError('Wrong registration')
    ledger = args.game/'turns.jsonl'
    if ledger.is_file() and live.sha(ledger) != receipt['turns_sha256']:
        raise ValueError('Incorrect action tape')
    history = [dict(step=i, actions=a) for i,a in enumerate(raw['actions'])]
    archive = settle(raw, history, raw['official_score_by_seat'], False)
    rows = [json.loads(line) for line in ledger.read_text().splitlines()] if ledger.is_file() else []
    hosted = settle(raw, rows, receipt.get('official_scores_by_seat'), True)
    if hosted['settled_transitions'] != receipt['official_transitions']:
        raise ValueError('Settled transition count differs')
    same = 0
    for row, old in zip(rows, history):
        if any(call['status'] != 'ok' for call in row['calls']):
            break
        if [c['response']['action'] for c in row['calls']] != old['actions']:
            break
        same += 1
    result = dict(archive_sha256=live.sha(card), hosted_receipt_sha256=live.sha(args.game/'receipt.json'),
                  game_status=receipt['status'], same_joint_orders_from_start=same,
                  candidate_seat=live.PLAN['candidate_seat'], historical=archive, hosted=hosted)
    args.out.write_bytes(gzip.compress(live.encoded(result), mtime=0))


if __name__ == '__main__':
    main()

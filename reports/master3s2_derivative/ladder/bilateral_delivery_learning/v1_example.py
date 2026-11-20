"""Audit three destination-wallet receipts on previously observed ladder examples."""
from functools import lru_cache
import hashlib
import importlib.util
import json
from math import fsum
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


old = load_module(HERE / 'ryan_example.py', 'exposed_ryan_receipts')
KINDS = ('own_delivery', 'opponent_worker_delivery_to_this_base', 'base_collision_cash')


def wallet_components(events, seat):
    """Each column is oriented to (our wallet, opposing wallet), regardless of carrier."""
    return {k: [fsum(x[2] for x in events if x[0] == k and x[1] == s)
                for s in (seat, 1-seat)] for k in KINDS}


def combined(components):
    return [fsum(components[k][i] for k in KINDS) for i in (0, 1)]


@lru_cache(maxsize=2)
def original(mid):
    _, folder, key, _, ledger_sha = next(x for x in old.CASE_FILES if x[0] == mid)
    cards = json.loads((folder / 'summary.json').read_text())[key]
    card, = [x for x in cards if x['match'] == mid]
    digest = card.get('archive_sha256') or card['replay_sha256']
    rep = old.read_gzip(folder / card['replay'], digest)
    ledger = old.read_gzip(folder / card['ledger'], ledger_sha)
    if ledger['replay_sha256'] != digest or ledger['official'] != dict(matched=299, first_divergence=None):
        raise ValueError('Uncertified source trajectory')
    frames = rep['steps']
    cfg = rep['configuration']
    econ = load_module(HERE.parents[2] / 'sapling_1200k/all_top5/economy_extract.py', 'v1_cash_events')
    sim = econ.FastCarbon(cfg).load_observation(frames[0][0]['observation'])
    sim.cash = econ.CashEvents(sim.cash)
    events = []
    for step, frame in enumerate(frames[1:], 1):
        sim.cash.events.clear()
        sim.step(*(p['action'] or {} for p in frame))
        expected = frame[0]['observation']
        if (sim.obs(0)['players'] != expected['players'] or list(sim.carbon) != expected['full_carbon']
                or sim.step_count != step):
            raise ValueError(f'Cash reconstruction diverged at {mid}, {step}')
        events.append([x for x in sim.cash.events if x[0] in KINDS])
    return card, rep, events


def corrected_example(r):
    """Extend one old real_examples() row. Keep the pre-amendment narrow amounts."""
    card, _, events = original(r['match'])
    t, seat = r['step'], r['seat']
    if card['seat'] != seat or len(events[t:t+10]) != 10:
        raise ValueError('Missing full ten-transition window or wrong perspective')
    components = wallet_components([e for row in events[t:t+10] for e in row], seat)
    if max(abs(x-y) for x, y in zip(r['actual'], components['own_delivery'])) > 1e-6:
        raise ValueError('Narrow receipt totals disagree')
    amended = {**r, 'actual_own_delivery': r['actual'], 'actual_components': components,
               'actual': combined(components), 'actual_horizon': 10, 'alternatives': []}
    if not r.get('alternatives'):
        return amended
    source = old.read_gzip(old.ARCHIVE / 'ryan_intercepts.json.gz',
                          '1cabc133d6442cc3c7943e367cd1006d5995922f0fbe51ff33838031aa5ca7f2')
    for a in r['alternatives']:
        scenario, = [s for s in source if s['t'] == t and s['our'] == a['worker']
                     and s['orders'] == a['worker_first_commands']]
        rows = scenario[a['continuation']]['rows'][:10]
        if [row['t'] for row in rows] != list(range(t, t+10)):
            raise ValueError('Incomplete historical branch')
        term = wallet_components([e for row in rows for e in row['events']], seat)
        if max(abs(x-y) for x, y in zip(a['conditional10'], term['own_delivery'])) > 1e-6:
            raise ValueError('Narrow branch amounts disagree')
        amended['alternatives'].append({**a, 'conditional10_own_delivery': a['conditional10'],
                                        'conditional_components': term, 'conditional10': combined(term)})
    return amended


def alphacarbon_279():
    """Enemy worker 56 returns 189.106446, or our 68 pays their bank 219.241446."""
    card, replay, events = original(old.ALPHA)
    source = old.read_gzip(HERE.parent / 'v4_alphacarbon_decisions/branches.json.gz',
                          '7f6b9ed0ee78043e7d8871ab8ee26ea70cf5084ecb433b677be3415625cb7011')
    if source['entire_replay_sha'] != card['replay_sha256'] or source['verified_official_steps'] != 299:
        raise ValueError('Mismatched source game')
    branches = [b for b in source['branches'] if b['t'] == 279]
    if len(branches) != 2 or {b['replacement'] for b in branches} != {True, False}:
        raise ValueError('Expected two archived root commands')
    seat = card['seat']
    t = 279
    frames = replay['steps']
    actual_order = frames[t+1][seat]['action']
    components = wallet_components([e for row in events[t:t+10] for e in row], seat)
    res = dict(seed=replay['configuration']['randomSeed'], terrain=card['terrain'], seat=seat,
               step=t, players=frames[t][0]['observation']['players'], order=actual_order,
               actual=combined(components), actual_components=components, actual_horizon=10,
               actual_own_delivery=components['own_delivery'], match=old.ALPHA, opponent=card['opponent'],
               branches=[], notes='Original ten-transition label is realized. Each archived fork below ends at t280.')
    for b in sorted(branches, key=lambda x: x['replacement']):
        if b['until'] != t+1 or len(b['trace']) != 1 or b['enemy_at_269'] != 'recorded':
            raise ValueError('Different archived continuation contract')
        row, = b['trace']
        order, other = row['actions'][seat], row['actions'][1-seat]
        if row['step'] != t or other != frames[t+1][1-seat]['action']:
            raise ValueError('Unexpected opponent response')
        if not b['replacement'] and order != actual_order:
            raise ValueError('Original command differs')
        term = {k: [row['flows'][s].get(k, 0.0) for s in (seat, 1-seat)] for k in KINDS}
        if not b['replacement'] and term != wallet_components(events[t], seat):
            raise ValueError('One-step original records differ')
        changes = {uid for uid in set(order) | set(actual_order)
                   if order.get(uid, 'NONE') != actual_order.get(uid, 'NONE')}
        if changes != ({'player-0-worker-68'} if b['replacement'] else set()):
            raise ValueError('Unexpected changed unit')
        res['branches'].append(dict(replacement=b['replacement'], order=order,
                                    worker='player-0-worker-68', command=order['player-0-worker-68'],
                                    horizon=1, conditional1=combined(term), conditional_components=term,
                                    conditional10=None if b['replacement'] else res['actual'],
                                    historical_enemy_orders_fixed=True,
                                    enemy_recorded_command_worker_56=other['player-1-worker-56'],
                                    wallets_after=row['cash'], official_margin_after=row['margin']))
    return res

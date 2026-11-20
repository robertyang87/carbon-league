"""Read already exposed ladder positions and audited ten-action delivery labels.

Only players, step, seat and a proposed own order belong in the predictor.
All amounts/scores in the other fields describe the subsequent replay or a
specified historical fork, and may be opened only after the model is frozen.
"""
import gzip
import hashlib
import importlib.util
import json
from math import fsum
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARCHIVE = HERE.parent / 'v4_observations'
RYAN = '1c80b916b203238a43267baeabc88fbe'
ALPHA = '6e994d80674dc2430a4925b71bb6a947'
CASE_FILES = (
    (RYAN, ARCHIVE, 'games', (148, 149, 150),
     '35f24f7ccccbf8e16b969541cf1510d6dd935fd9b34349d80374766b3089efab'),
    (ALPHA, ARCHIVE / 'increment_0904', 'added', (277,),
     '9e7a22fce3e5ec25d2938780c2955955dfc6abf6c8a5d3288479a6eb54c88933'),
)


def read_gzip(file, digest):
    data = file.read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f'Unexpected bytes in {file}')
    return json.loads(gzip.decompress(data))


def horizon_total(events, seat):
    return [fsum(amount for side, _, amount in events if side == s) for s in (seat, 1-seat)]


def counterfactuals(scenarios, step, order, seat, actual):
    alternatives = []
    for scenario in scenarios:
        if scenario['t'] != step:
            continue
        worker = scenario['our']
        commands = scenario['orders']
        changed = dict(order)
        changed[worker] = commands[0]
        for mode in ('actual', 'fixed', 'escort'):
            rows = scenario[mode]['rows'][:10]
            if [r['t'] for r in rows] != list(range(step, step+10)):
                raise ValueError('Incomplete archived response window')
            events = [(e[1], e[3], e[2]) for row in rows for e in row['events']
                      if e[0] == 'own_delivery']
            amounts = horizon_total(events, seat)
            if mode == 'actual':
                if max(abs(a-b) for a, b in zip(amounts, actual)) > 1e-6:
                    raise ValueError('Original-path deposits differ from the verified replay')
                continue
            alternatives.append(dict(
                order=changed.copy(), worker=worker, enemy_target=scenario['enemy'],
                worker_first_commands=commands.copy(), continuation=mode,
                conditional10=amounts, historical_enemy_orders_fixed=True,
                notes=('After the script, the changed worker follows archived orders.'
                       if mode == 'fixed' else
                       'After the script, the changed worker uses the archived public-state return policy until its first deposit.'),
            ))
    return alternatives


def real_examples():
    """Return three overlapping Ryan roots and one registered AlphaCarbon root."""
    # Reuse the existing validated cash ledger extractor. It verifies every
    # observed roster and cash balance along the stored physical trajectory.
    source = HERE.parents[2] / 'sapling_1200k/all_top5/economy_extract.py'
    spec = importlib.util.spec_from_file_location('ladder_example_economy_extract', source)
    economy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(economy)
    scenarios = read_gzip(ARCHIVE / 'ryan_intercepts.json.gz',
                          '1cabc133d6442cc3c7943e367cd1006d5995922f0fbe51ff33838031aa5ca7f2')
    examples = []
    for mid, directory, card_key, turns, ledger_hash in CASE_FILES:
        cards = json.loads((directory / 'summary.json').read_text())[card_key]
        card, = [c for c in cards if c['match'] == mid]
        replay_hash = card.get('archive_sha256') or card['replay_sha256']
        data = read_gzip(directory / card['replay'], replay_hash)
        ledger = read_gzip(directory / card['ledger'], ledger_hash)
        tape = data['steps']
        seat = card['seat']
        terrain = hashlib.sha256(bytes(tape[0][0]['observation']['full_carbon'])).hexdigest()
        if (terrain != card['terrain'] or ledger['replay_sha256'] != replay_hash
                or ledger['seat'] != seat or ledger['official'] != dict(matched=299, first_divergence=None)
                or len(tape) != 300):
            raise ValueError(f'Unreconciled replay {mid}')
        _, _, banks, _ = economy.reconstruct(tape, data['configuration'])
        for t in turns:
            audit = ledger['receipts_by_transition'][t:t+10]
            for archived, precise in zip(audit, banks[t:t+10]):
                archived_events = [(e['seat'], e['worker'], e['amount']) for e in archived]
                if sorted(archived_events) != sorted((s, uid, round(q, 2)) for s, uid, q in precise):
                    raise ValueError(f'Delivery ledger disagrees at {mid}, root {t}')
            actual = horizon_total([e for row in banks[t:t+10] for e in row], seat)
            order = tape[t+1][seat]['action'] or {}
            players = tape[t][0]['observation']['players']
            at_end = tape[t+10][0]['observation']['players']
            changes = [round(at_end[s][0] - players[s][0], 2) for s in (seat, 1-seat)]
            notes = ('Overlapping windows of one exposed game. At frame 153 Ryan delivered '
                     'with two separate carriers. Conditional forks only exist at t149 and t150; '
                     'the opponent can deviate from its recorded route.'
                     if mid == RYAN else 'Exposed ladder game against the registered AlphaCarbon '
                     'account. Only the actual joint trajectory is labelled here.')
            examples.append(dict(
                seed=data['configuration']['randomSeed'], terrain=terrain, seat=seat, step=t,
                players=players, order=order, actual=actual, match=mid, opponent=card['opponent'],
                actual_wallet_change=changes, notes=notes, replay_sha256=replay_hash,
                alternatives=counterfactuals(scenarios if mid == RYAN else [], t, order, seat, actual),
            ))
    return examples

"""Read independent development games as ten-transition own-base cash-delivery examples."""
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import zipfile


FAMILY = Path(__file__).resolve().parents[2]
ROOT = FAMILY.parents[1]
HOSTED = FAMILY / 'v4_replication_dev/hosted'
CASH_SOURCE = (ROOT / 'reports/sapling_1200k/master6_actor_dev/cargo_tree_action_learning_dev'
               / 'native_actor_corpus/league_policy/natural_original/economy.py')
CASH_SHA = 'e7d46fcbe7731c2ed7af27715be00de4b7b34f5eea7560d7bde98f9fbaec95c1'
SETTLEMENT_SHA = 'f3c8212869dae39b1578728038e151bb1285c423c3dc5eaadf1e81bd4e723bf8'
ECONOMY_SHA = '6a831ef53eb2786d8e9756fc078f135294b02fbf3494144113906ec13b32b242'
RUN = 44706116852
PROTOCOL = 'c8d5ac6b75f61a0eed2971c3891c94070b9816f7b8bd7bd95991e654f7ae5a3b'
SEEDS = (621758610, 621758611, 621758612, 621758613)
STEPS = range(6, 289, 6)
WINDOW = 10


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def require(ok, why):
    if not ok:
        raise ValueError(why)


def read_bytes(path, expected):
    raw = path.read_bytes()
    require(sha(raw) == expected, 'Different SHA-256 for ' + str(path))
    return raw


def cash_engine():
    require(sha(CASH_SOURCE.read_bytes()) == CASH_SHA, 'Different official-cash replay implementation')
    spec = importlib.util.spec_from_file_location('verified_bilateral_delivery_cash', CASH_SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def game_rows(engine, config, game, reference, archive, run):
    seat, arm, seed = game['seat'], game['arm'], game['seed']
    ledger = engine.replay(game, archive, config)
    tape = ledger['transitions']
    require(game['status'] == 'complete' and len(tape) == game['official_transitions'],
            'A scored game with its full actual action tape is required')
    for side in (0, 1):
        account = reference['by_seat'][side]
        ledger_total = ledger['signed_account_by_seat'][side]
        engine.check_equal(sum(ledger_total.values()), game['official_scores_by_seat'][side], 'signed official payoff')
        engine.check_equal(ledger_total['own_delivery'], account['own_delivery'], 'archived own-base deliveries')
    require(ledger['action_tape_sha256'] == reference['signed_actions_sha256']
            and ledger['official_scores_by_seat'] == reference['official_scores_by_seat'],
            'Different archive accounted for in cash record')
    raw_views = gzip.decompress(archive.read(arm + '/observations.jsonl.gz')).splitlines()
    require(len(raw_views) == len(tape), 'Incomplete pre-action observations')
    result = []
    for step in STEPS:
        if step + WINDOW > len(tape):
            continue
        side_view = json.loads(raw_views[step])
        obs = side_view['observations'][seat]
        encoded = json.dumps(obs, separators=(',', ':'), allow_nan=False).encode() + b'\n'
        start = tape[step]
        require(sha(encoded) == side_view['public_observation_sha256'][seat] == start['public_observation_sha256'][seat]
                and obs['player'] == seat and side_view['step'] == obs['step'] == start['t'] == step,
                'Different perspective or move number for public pre-action input')
        # Only the current own order and the preceding public board reach the predictor.
        # Opponent commands and later positions enter the cash accountant only.
        window = tape[step:step + WINDOW]
        target = [math.fsum(move['signed_cash_by_seat'][s].get('own_delivery', 0.) for move in window)
                  for s in (seat, 1-seat)]
        result.append(dict(seed=seed, terrain=game['terrain'], seat=seat, source_arm=arm, step=step,
                           players=obs['players'], order=start['commands_by_seat'][seat], target=target,
                           match=f'{run}:{seed}:{seat}:{arm}'))
    return result


def rows():
    """Return one selected side per game and each complete registered [step, step+10) window."""
    settlement = json.loads(read_bytes(HOSTED / 'settlement.json', SETTLEMENT_SHA))
    financial = json.loads(gzip.decompress(read_bytes(HOSTED / 'economy.json.gz', ECONOMY_SHA)))
    require(settlement['records_reconciled'] and settlement['workflow_run'] == RUN
            and settlement['protocol_sha256'] == PROTOCOL and financial['settlement_sha256'] == SETTLEMENT_SHA,
            'Different development registration or financial receipt')
    expected = {(seed, seat, arm) for seed in SEEDS for seat in (0, 1) for arm in ('challenger', 'parent')}
    source_games = settlement['games']
    financial_games = financial['games']
    key = lambda g: (g['seed'], g['seat'], g['arm'])
    require(len(source_games) == len(financial_games) == len(expected) == 16
            and set(map(key, source_games)) == set(map(key, financial_games)) == expected, 'Incomplete game census')
    accounts = {key(g): g for g in financial_games}
    tickets = {(a['seed'], a['seat']): a for a in settlement['archives']}
    require(len(tickets) == len(settlement['archives']) == 8, 'Expected eight distinct switched-seat ZIPs')
    engine = cash_engine()
    cfg = engine.config()
    result = []
    for game in source_games:
        ticket = tickets[game['seed'], game['seat']]
        path = HOSTED / (str(ticket['id']) + '.zip')
        account = accounts[key(game)]
        require(sha(path.read_bytes()) == account['archive_sha256'] == ticket['sha256']
                and path.stat().st_size == ticket['bytes'], 'Different signed hosted ZIP')
        with zipfile.ZipFile(path) as archive:
            result.extend(game_rows(engine, cfg, game, account, archive, RUN))
    require(len(result) == 768 and len({(r['match'], r['step']) for r in result}) == 768,
            'Expected 48 complete registered windows for each of 16 full-length games')
    return result

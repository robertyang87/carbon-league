"""Reconcile the two sides' realized cash, deliveries and heavy-cargo collisions in the eight signed games."""
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / '.git').exists())
ECONOMY = ROOT / 'reports/sapling_1200k/master6_actor_dev/cargo_tree_action_learning_dev/native_actor_corpus/league_policy/natural_original/economy.py'
TREE_COMPONENTS = ('ground_rent', 'cargo_tax', 'plant', 'seize')



def exposed_collectors(public_turn, seat, distances):
    players = public_turn['observations'][seat]['players']
    own = players[seat][2]
    rival = [w for w in players[1-seat][2].values() if w[2] == 'COLLECTOR']
    return {uid for uid, (pos, load, kind) in own.items() if kind == 'COLLECTOR' and load >= 200. and
            any(other[1] <= load and distances[pos][other[0]] <= 2 for other in rival)}


def main():
    spec = importlib.util.spec_from_file_location('game_cash_accounting', ECONOMY)
    ledger = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ledger)
    settlement_bytes = (HERE / 'settlement.json').read_bytes()
    settlement = json.loads(settlement_bytes)
    registration = (HERE.parent / 'protocol.json').read_bytes()
    ledger.require(settlement['records_reconciled'] and hashlib.sha256(registration).hexdigest() == settlement['protocol_sha256'],
                   'Expected reconciled signed games for this registration')
    plan = json.loads(registration)
    keys = {(seed, seat, arm) for seed in plan['seeds'] for seat in (0, 1) for arm in plan['arms']}
    games = settlement['games']
    ledger.require(len(games) == len(keys) == 8 and {(g['seed'], g['seat'], g['arm']) for g in games} == keys,
                   'Wrong or incomplete eight-game denominator')
    archives = {(a['seed'], a['seat']): a for a in settlement['archives']}
    cfg = ledger.config()
    distances = ledger.distances(cfg['size'])
    out = []
    for game in games:
        archive = archives[game['seed'], game['seat']]
        src = HERE / (str(archive['id']) + '.zip')
        ledger.require(ledger.sha(src.read_bytes()) == archive['sha256'], 'Archive differs from the signed digest')
        with zipfile.ZipFile(src) as z:
            book = ledger.replay(game, z, cfg)
            member = game['arm'] + '/observations.jsonl.gz'
            public = ([json.loads(line) for line in gzip.decompress(z.read(member)).splitlines()]
                      if member in z.namelist() else [])
            ledger.require(len(public) >= book['executed_transitions'], 'Missing actual public observations')
        accounts = book['signed_account_by_seat']
        rows = book['transitions']
        sides = []
        for seat, account in enumerate(accounts):
            source_cash = {k: account.get(k, 0.) for k in ledger.KIND}
            if book['official_scores_by_seat'] is not None:
                ledger.check_equal(sum(source_cash.values()), book['official_scores_by_seat'][seat], 'official terminal payoff')
            heavy = [dict(step=row['t'], worker=uid, cargo=load, recipient=recipient, surviving_collector_seat=survivor)
                     for row in rows for uid, lost, load, recipient, survivor in row['collisions']
                     if lost == seat and load >= 200.]
            stolen = [c for c in heavy if c['recipient'] == 1-seat]
            exposed = [exposed_collectors(turn, seat, distances) for turn in public[:len(rows)]]
            stolen_exposed = [c for c in stolen if c['worker'] in exposed[c['step']]]
            sides.append(dict(cash_components=source_cash, own_delivery=source_cash['own_delivery'],
                              net_tree_cash=sum(source_cash[k] for k in TREE_COMPONENTS),
                              heavy_cargo_to_enemy=sum(c['cargo'] for c in stolen), heavy_cargo_collisions=heavy,
                              heavy_threatened_worker_turns=sum(map(len, exposed)),
                              threatened_worker_losses_to_enemy=len(stolen_exposed),
                              threatened_cargo_to_enemy=sum(c['cargo'] for c in stolen_exposed)))
        out.append(dict(seed=game['seed'], seat=game['seat'], arm=game['arm'], archive_sha256=archive['sha256'],
                        terrain=game['terrain'], status=book['status'], official_scores_by_seat=book['official_scores_by_seat'],
                        official_margin=book['official_margin'], transitions=book['executed_transitions'],
                        signed_actions_sha256=book['action_tape_sha256'], by_seat=sides))
    output = dict(schema='cargo-defense-actual-game-economy/1', source_ledger=ECONOMY.relative_to(ROOT).as_posix(),
                  source_ledger_sha256=ledger.sha(ECONOMY.read_bytes()), tree_net_components=TREE_COMPONENTS,
                  settlement_sha256=ledger.sha(settlement_bytes), games=out)
    path = HERE / 'economy.json.gz'
    raw = json.dumps(output, sort_keys=True, allow_nan=False, separators=(',', ':')).encode()
    temp = path.with_suffix('.tmp')
    temp.write_bytes(gzip.compress(raw, mtime=0))
    temp.replace(path)


if __name__ == '__main__':
    main()

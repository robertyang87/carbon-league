"""Replay all archived V2-v-original development games selected by package SHA."""
from collections import Counter, defaultdict
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import zipfile

HERE = Path(__file__).resolve().parent
FAMILY = HERE.parent
ROOT = FAMILY.parents[1]
LEDGER = ROOT/'reports/sapling_1200k/master6_actor_dev/cargo_tree_action_learning_dev/native_actor_corpus/league_policy/natural_original/economy.py'
PARENT = '8b4c4bafba2b3b2887139163bd6397b3f7d1acf178a43ea1756d8f80306efe43'
EXAMINER = '65f268b2919ce30fe19355115a928eada032a6beaa018d801662d434c083f2b5'
PANELS = ('ladder/prediction_feedback/development', 'v3_release/development', 'nn_policy_development',
          'v4_replication_dev', 'two_response_dev', 'native_speed_dev', 'switch_margin_dev')
GROUPS = {
    'delivery': ('own_delivery', 'opponent_worker_delivery_to_this_base'),
    'tree_net': ('ground_rent', 'cargo_tax', 'tree_collision_cash', 'plant', 'seize'),
    'recruit': ('hire_collector', 'hire_planter'),
    'other': ('base_collision_cash', 'rounding', 'terminal_scoring_adjustment', 'opening_cash'),
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def save(name, obj):
    raw = json.dumps(obj, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()+b'\n'
    if name.endswith('.gz'):
        raw = gzip.compress(raw, mtime=0)
    path = HERE/name
    staged = path.with_suffix('.partial')
    staged.write_bytes(raw)
    staged.replace(path)


def select():
    tickets = []
    for name in PANELS:
        hosted = FAMILY/name/'hosted'
        raw = (hosted/'settlement.json').read_bytes()
        receipt = json.loads(raw)
        assert receipt['records_reconciled'] and not receipt['formal_exam_games'] and not receipt['verification_errors']
        archives = {(a['seed'], a['seat']): a for a in receipt['archives']}
        picked = []
        for game in receipt['games']:
            archive = archives[game['seed'], game['seat']]
            path = hosted/(str(archive['id'])+'.zip')
            with zipfile.ZipFile(path) as z:
                reg_bytes = z.read(game['arm']+'/registration.json')
                reg = json.loads(reg_bytes)
            identity = tuple(reg['files'][k]['sha256'] for k in ('candidate', 'opponent'))
            if identity != (PARENT, EXAMINER):
                continue
            assert sha(path.read_bytes()) == archive['sha256']
            assert game['counted'] and reg['physical_map_sha256'] == game['terrain']
            picked.append(dict(panel=name, archive=str(path.relative_to(ROOT)), archive_sha256=archive['sha256'],
                               settlement_sha256=sha(raw), registration_sha256=sha(reg_bytes), game=game))
        assert picked, name
        tickets.extend(picked)
    assert len(tickets) == len({(r['game']['terrain'], r['game']['seat']) for r in tickets})
    counts = Counter(r['game']['terrain'] for r in tickets)
    assert all(count == 2 for count in counts.values())
    return tickets


def grouped(ledger, account, seat):
    values = ledger.minus(*[account[i] for i in (seat, 1-seat)])
    return {tag: sum(values[k] for k in keys) for tag, keys in GROUPS.items()}


def window(ledger, rows, seat, start, end):
    prof = ledger.profile(seat, rows, start, end)
    return dict(start=start, end=end, by_seat=prof['by_physical_seat'],
                margin=prof['learner_margin_change'], components=grouped(ledger, prof['by_physical_seat'], seat))


def analyze(ledger, t, cfg):
    game = t['game']
    path = ROOT/t['archive']
    assert sha(path.read_bytes()) == t['archive_sha256']
    with zipfile.ZipFile(path) as z:
        prefix = game['arm']+'/'
        assert sha(z.read(prefix+'registration.json')) == t['registration_sha256']
        rec = json.loads(z.read(prefix+'receipt.json'))
        seat = game['seat']
        actual = rec['initialization']
        assert [actual[i]['response']['sha256'] for i in (seat, 1-seat)] == [PARENT, EXAMINER]
        assert actual[1-seat]['response']['original'] == dict(_S_ABORT=.58, _TEMP=1., _SEED=None, _SEARCH_OK=True)
        book = ledger.replay(game, z, cfg)

    rows = book.pop('transitions')
    end = len(rows)
    margins = book['cash_margin_by_state']
    parts = [window(ledger, rows, seat, lo, min(lo+50,end)) for lo in range(0,end,50)]
    losses = {}
    for width in (3, 5, 10):
        if end < width:
            continue
        start = min(range(end-width+1), key=lambda t: margins[t+width]-margins[t])
        report = window(ledger, rows, seat, start, start+width)
        report['history'] = {n: window(ledger, rows, seat, max(0,start-n), start) for n in (3, 5, 10)}
        report['actions_and_context'] = rows[max(0,start-10):start+width]
        losses[width] = report
    early_tree_losses = []
    for start in range(0, min(end, 100) - 9):
        loss = 0
        for row in rows[start:start+10]:
            sides = row['signed_cash_by_seat']
            loss += sum(sides[seat].get(k, 0) - sides[1-seat].get(k, 0) for k in GROUPS['tree_net'])
        early_tree_losses.append(loss)
    tree_window = None
    if early_tree_losses:
        t0 = min(range(len(early_tree_losses)), key=early_tree_losses.__getitem__)
        tree_window = window(ledger, rows, seat, t0, t0+10)
        tree_window['actions_and_context'] = rows[max(0,t0-10):t0+10]
    # "Available cargo" is the visible inventory, separate from score.
    balances = [dict(turn=t, margin=margins[t], cash=r['pre']['cash'], cargo=r['pre']['carried_by_seat'],
                     trees=r['pre']['trees_by_seat'], workers=[sum(w[3]==s for w in r['pre']['workers'].values())
                                                               for s in (0,1)])
                for t,r in enumerate(rows)]
    balances.append(dict(turn=end, margin=margins[end], cash=book['physical_cash_by_seat'],
                         cargo=[p['carried'] for p in book['end_assets_by_physical_seat']],
                         trees=[p['alive_trees'] for p in book['end_assets_by_physical_seat']]))
    stable = next((i for i in range(len(margins)-9) if all(m<0 for m in margins[i:i+10])), None)
    out = dict(panel=t['panel'], seed=game['seed'], terrain=game['terrain'], seat=seat,
               archive=t['archive'], archive_sha256=t['archive_sha256'], receipt_sha256=game['receipt_sha256'],
               official=book['official_margin'], turns=end, end_scores=book['official_scores_by_seat'],
               net=grouped(ledger, book['signed_account_by_seat'], seat), bands=parts, first_ten_behind=stable,
               worst_windows=losses, early_tree_window=tree_window, balance=balances)
    ledger.check_equal(sum(out['net'].values()), out['official'], 'official margin must reconcile')
    return out


def main():
    spec = importlib.util.spec_from_file_location('cash_sources', LEDGER)
    ledger = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ledger)
    tickets = select()
    cfg = ledger.config()
    output = []
    for t in tickets:
        row = analyze(ledger, t, cfg)
        output.append(row)
        print(row['panel'], row['seed'], row['seat'], row['official'], flush=True)
    by_map = defaultdict(list)
    for row in output:
        by_map[row['terrain']].append(row)
    assert all({r['seat'] for r in group} == {0,1} for group in by_map.values())
    save('games.json.gz', dict(schema='v2-original-cash-paths/1', candidate=PARENT, original=EXAMINER,
                               ledger_sha256=sha(LEDGER.read_bytes()),
                               source_settlement_sha256={t['panel']:t['settlement_sha256'] for t in tickets}, games=output))


if __name__ == '__main__':
    main()

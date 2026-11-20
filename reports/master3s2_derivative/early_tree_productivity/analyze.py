"""Separate early tree count, lifetime, and income per owned turn in original-opponent games."""
from collections import Counter, defaultdict
import gzip
import importlib.util
import json
from pathlib import Path
from statistics import mean
import sys
import zipfile

HERE = Path(__file__).resolve().parent
FAMILY = HERE.parent
ROOT = FAMILY.parents[1]
sys.path.insert(0, str(FAMILY/'v2_original_cash_paths'))
from study import LEDGER, select, sha

WINDOW = 100


def save(name, obj):
    raw = (json.dumps(obj, allow_nan=False, ensure_ascii=False, separators=(',', ':'))+'\n').encode()
    if name.endswith('.gz'):
        raw = gzip.compress(raw, mtime=0)
    path = HERE/name
    tmp = path.with_suffix('.partial')
    tmp.write_bytes(raw)
    tmp.replace(path)


def read_game(ledger, ticket, cfg):
    game = ticket['game']
    archive = ROOT/ticket['archive']
    assert sha(archive.read_bytes()) == ticket['archive_sha256']
    with zipfile.ZipFile(archive) as z:
        book = ledger.replay(game, z, cfg)
    players = [Counter(), Counter()]
    cohorts = [{}, {}]
    trees = {}
    rows = book['transitions']
    assert len(rows) >= WINDOW
    for r in rows[:WINDOW]:
        step = r['t']
        snapshot = r['pre']
        for holder in (0, 1):
            assert sum(h == holder for _, h, _ in trees.values()) == snapshot['trees_by_seat'][holder]
            for name, w in snapshot['workers'].items():
                if w[3] == holder and not w[2]:
                    players[holder]['planter_turns'] += 1
                    order = r['commands_by_seat'][holder].get(name, 'NONE')
                    players[holder]['planter_stay_turns'] += order == 'NONE'
        for kind, tid, cell, previous, holder in r['tree_changes']:
            if kind == 'new':
                assert tid not in trees and previous is None
                lineage = dict(kind='plant', first_turn=step, cell=cell, owned_turns=0, receipts=0., land=0., tax=0.)
            else:
                pos, owner, old = trees.pop(tid)
                assert (pos, owner) == (cell, previous)
                old['lost_turn'] = step
                old['end_reason'] = kind
                if kind == 'expiry':
                    continue
                assert kind == 'seizure' and holder != previous
                lineage = dict(kind='seizure', first_turn=step, cell=cell, owned_turns=0, receipts=0., land=0., tax=0.)
            cohorts[holder][f'{tid}@{step}'] = lineage
            trees[tid] = (cell, holder, lineage)
            players[holder][kind] += 1
        for _, holder, lineage in trees.values():
            players[holder]['owned_turns'] += 1
            lineage['owned_turns'] += 1
        for kind, holder, value, *details in r['cash_events']:
            if kind in ('hire_planter','plant','seize'):
                players[holder][kind+'_cash'] += value
                if kind == 'hire_planter':
                    players[holder]['hire_planter_count'] += 1
            if kind == 'tree_absorption':
                tid, cell, tax = details
                pos, who, lineage = trees[tid]
                assert (pos, who) == (cell, holder)
                lineage['receipts'] += value
                lineage['land'] += value-tax
                lineage['tax'] += tax
                players[holder]['land'] += value-tax
                players[holder]['tax'] += tax
            elif kind == 'tree_collision_cash':
                players[holder]['collision'] += value
    components = ('land', 'tax', 'collision', 'plant_cash', 'seize_cash')
    for i in (0,1):
        x = players[i]
        assert x['owned_turns'] == sum(c['owned_turns'] for c in cohorts[i].values())
        x['tree_net'] = sum(x[k] for k in components)
        profile = ledger.profile(i, rows, 0, WINDOW)['by_physical_seat'][i]
        names = ('ground_rent','cargo_tax','tree_collision_cash','plant','seize')
        ledger.check_equal(x['tree_net'], sum(profile.get(k,0.) for k in names), 'first 100 tree cash')
    return dict(seed=game['seed'], terrain=game['terrain'], seat=game['seat'], margin=book['official_margin'],
                panel=ticket['panel'], archive_sha256=ticket['archive_sha256'], seats=players, cohorts=cohorts)


def summary(group):
    fields = ('hire_planter_count','planter_turns','planter_stay_turns','new','seizure','owned_turns','land',
              'tax','plant_cash','seize_cash','tree_net')
    pairs = [(row['seats'][row['seat']],row['seats'][1-row['seat']]) for row in group]
    ancestry = {}
    for side in (0,1):
        episodes = [item for r in group for item in r['cohorts'][r['seat']^side].values()]
        born = [c for c in episodes if c['kind']=='plant']
        captures = [c for c in born if c.get('end_reason')=='seizure']
        ancestry[side] = dict(initial_plants=len(born), captured=len(captures),
                              captured_within_12=sum(c['lost_turn']-c['first_turn']<=12 for c in captures),
                              observed_average_plant_turns=mean(c['owned_turns'] for c in born))
    return dict(games=len(group),
                accounts={k:[mean(a.get(k,0) for a in arm) for arm in zip(*pairs,strict=True)] for k in fields},
                land_decomposition={
                    'owned_turns':mean((a['owned_turns']-b['owned_turns'])*.5*
                        (a['land']/a['owned_turns']+b['land']/b['owned_turns']) for a,b in pairs),
                    'land_per_owned_turn':mean((a['land']/a['owned_turns']-b['land']/b['owned_turns'])*.5*
                        (a['owned_turns']+b['owned_turns']) for a,b in pairs)},
                cohorts=ancestry)


def main():
    spec = importlib.util.spec_from_file_location('replay_early_tree_ledger', LEDGER)
    ledger = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ledger)
    cfg = ledger.config()
    games = []
    for t in select():
        row = read_game(ledger,t,cfg)
        games.append(row)
        print(row['seed'],row['seat'],flush=True)
    assert len(games)==44
    counts = defaultdict(set)
    for r in games:
        counts[r['terrain']].add(r['seat'])
    assert len(counts)==22 and all(s=={0,1} for s in counts.values())
    groups = {k:summary(v) for k,v in (
        ('all',games),('loss',[r for r in games if r['margin']<0]),('win',[r for r in games if r['margin']>0]))}
    for group in groups.values():
        a,b=group['accounts']['land']
        ledger.check_equal(a-b,sum(group['land_decomposition'].values()),'land difference decomposition')
    save('groups.json', groups)
    save('cohorts.json.gz',dict(steps=WINDOW, ledger_sha256=sha(LEDGER.read_bytes()), games=games))
    print(groups,flush=True)


if __name__ == '__main__':
    main()

"""Reconcile official placement scorecards by registration and retain all settled games."""
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
from statistics import mean, median

HERE = Path(__file__).resolve().parent


def counts(rows):
    return {r: sum(row['result'] == r for row in rows) for r in ('W', 'D', 'L')}


def main():
    source = json.loads(gzip.decompress((HERE / 'cards.json.gz').read_bytes()))
    placements = {}
    report = {'recorded_utc': source['recorded_utc'], 'versions': {}, 'common_opponents': {}}
    for version, entry in source['versions'].items():
        ident = entry['id']
        scores = defaultdict(list)
        seen = set()
        for card in entry['cards']:
            assert ident in (card['a'], card['b'])
            own = 'a' if card['a'] == ident else 'b'
            foe = 'b' if own == 'a' else 'a'
            if card['state'] != 'done':
                continue
            for game in card['result']:
                key = (card['id'], game['game'])
                assert key not in seen
                seen.add(key)
                a, b = (game['scores'][k] for k in (own, foe))
                scores[card['kind']].append(dict(id=card['id'], game=game['game'], opponent=card['name_'+foe],
                                                opponent_id=card[foe], seat=game['order'].index(own), score=[a, b],
                                                margin=round(a-b, 2) if a is not None and b is not None else None,
                                                result={own: 'W', foe: 'L', 'draw': 'D'}[game['winner']]))
        per_kind = {}
        for kind, rows in scores.items():
            values = [row['margin'] for row in rows if row['margin'] is not None]
            per_kind[kind] = dict(counts=counts(rows), total=len(rows), scored=len(values),
                                  mean=mean(values), median=median(values), rows=rows,
                                  per_seat={str(s):counts([r for r in rows if r['seat']==s]) for s in (0,1)})
        report['versions'][version] = per_kind
        groups = defaultdict(list)
        for row in scores['newrank-placement']:
            groups[row['opponent_id']].append(row)
        placements[version] = groups
    for older in ('1','2'):
        a, b = placements['4'], placements[older]
        paired = []
        for ident in sorted(a.keys() & b.keys()):
            assert all(r['margin'] is not None for r in a[ident] + b[ident])
            diff = mean(r['margin'] for r in a[ident]) - mean(r['margin'] for r in b[ident])
            paired.append(dict(opponent_id=ident, opponent=a[ident][0]['opponent'], v4=a[ident], older=b[ident], difference=diff))
        report['common_opponents']['v4_minus_v'+older] = dict(
            opponents=len(paired), improved=sum(r['difference'] > 0 for r in paired),
            mean=mean(r['difference'] for r in paired), pairs=paired)
    (HERE/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


if __name__ == '__main__':
    main()

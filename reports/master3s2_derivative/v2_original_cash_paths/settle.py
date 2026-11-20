"""Calculate paired-map descriptive statistics from signed, replayed games."""
from collections import Counter, defaultdict
import gzip
import json
import math
from statistics import mean, median

from study import GROUPS, HERE, save, sha


def difference(row, index, name):
    here = row['balance'][index]
    return here[name][row['seat']] - here[name][1-row['seat']]


def head(rows):
    counts = Counter('win' if r['official'] > 0 else 'loss' if r['official'] < 0 else 'draw' for r in rows)
    def summarize(values):
        return dict(mean=mean(values), median=median(values), positive=sum(v > 0 for v in values))
    series = [0, 25, 50, 100, 150, 200, 250, 275, 299]
    assert all(len(r['balance']) == 300 for r in rows), 'State-aligned aggregate requires full length'
    return dict(games=len(rows), counts=counts, official=summarize([r['official'] for r in rows]),
                final_account={k:summarize([r['net'][k] for r in rows]) for k in GROUPS},
                early_tree_components={
                    field: summarize([sum(r['bands'][i]['by_seat'][r['seat']].get(field, 0)
                                          - r['bands'][i]['by_seat'][1-r['seat']].get(field, 0)
                                          for i in (0, 1)) for r in rows])
                    for field in GROUPS['tree_net']
                },
                early_tree_negative=sum(sum(b['components']['tree_net'] for b in r['bands'][:2]) < 0
                                        for r in rows),
                steps={t:{name:summarize([difference(r,t,name) for r in rows])
                          for name in ('cash','cargo','trees')} for t in series},
                bands={lo:{k:summarize([r['bands'][lo//50]['components'][k] for r in rows])
                           for k in GROUPS} for lo in range(0,300,50)},
                worst_10=dict(distribution=Counter(r['worst_windows']['10']['start']//50 for r in rows),
                              largest_component=Counter(min(r['worst_windows']['10']['components'],
                                                            key=r['worst_windows']['10']['components'].get) for r in rows),
                              preceding={name:summarize([difference(r,r['worst_windows']['10']['start'],name) for r in rows])
                                         for name in ('cash','cargo','trees')}))


def main():
    raw = (HERE/'games.json.gz').read_bytes()
    games = json.loads(gzip.decompress(raw))['games']
    by_map = defaultdict(list)
    for r in games:
        by_map[r['terrain']].append(r)
        assert math.isclose(sum(r['net'].values()),r['official'],abs_tol=1e-5)
    assert all({r['seat'] for r in group} == {0,1} for group in by_map.values())
    bins = {'all':games, 'loss':[r for r in games if r['official']<0],
            'win':[r for r in games if r['official']>0],
            **{f'seat{seat}':[r for r in games if r['seat']==seat] for seat in (0,1)}}
    summary = {label:head(rows) for label, rows in bins.items()}
    save('summary.json', dict(source_sha256=sha(raw), groups=summary, physical_maps=len(by_map),
                             by_map={r[0]['seed']:[{'seat':x['seat'],'margin':x['official']} for x in sorted(r,key=lambda x:x['seat'])]
                                     for r in by_map.values()}))
    for name, group in summary.items():
        print(name, group['counts'], round(group['official']['mean'],2),
              {k:round(v['mean'],2) for k,v in group['final_account'].items()})
        print('cash', {k: round(v['cash']['mean'],2) for k,v in group['steps'].items()})
        print('worst 10', group['worst_10'])
    print('Selected largest regrets:')
    for r in sorted(bins['loss'],key=lambda x:x['official'])[:4]:
        w=r['worst_windows']['10']; print(r['seed'],r['seat'],r['official'],w['start'],w['end'],w['components'],
                                        difference(r,w['start'],'cargo'))


if __name__ == '__main__':
    main()

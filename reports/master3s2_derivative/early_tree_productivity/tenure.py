"""Separate new-tree exposure into how many sites were planted and time held."""
import gzip
import json
from statistics import mean

from analyze import HERE, save, sha

raw = (HERE/'cohorts.json.gz').read_bytes()
games = json.loads(gzip.decompress(raw))['games']
groups = {}
for label, cases in (
        ('all',games), ('loss',[g for g in games if g['margin']<0]), ('win',[g for g in games if g['margin']>0])):
    first_moves = []
    changes = []
    for game in cases:
        mine, enemy = [[t for t in game['cohorts'][game['seat']^side].values() if t['kind']=='plant']
                       for side in (0,1)]
        assert mine and enemy
        m, e = (mean(t['owned_turns'] for t in episodes) for episodes in (mine,enemy))
        changes.append(dict(seed=game['seed'], seat=game['seat'],
                            planted_count=(len(mine)-len(enemy))*(m+e)/2,
                            mean_observed_tenure=(m-e)*(len(mine)+len(enemy))/2,
                            directly_observed=len(mine)*m-len(enemy)*e))
        first_moves.append([min(t['first_turn'] for t in branch) for branch in (mine,enemy)])
    agg = {k:mean(r[k] for r in changes) for k in ('planted_count','mean_observed_tenure','directly_observed')}
    assert abs(agg['planted_count']+agg['mean_observed_tenure']-agg['directly_observed'])<1e-8
    groups[label]=dict(first_plants=[mean(v) for v in zip(*first_moves,strict=True)], new_tree_turns=agg, games=changes)
save('tenure.json', dict(source_sha256=sha(raw),groups=groups))

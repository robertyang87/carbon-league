"""opening_trees -- per teacher side in dataset v10 games: trees owned at step 50 and whether that side won.
Writes data/opening_trees.json {game_key: {side: tree50}} for v18 opening weighting, and prints the per-teacher picture."""
import json, gzip, collections
meta = json.load(open('../data/cnet2_ds_v10/meta.json'))
out = {}; by = collections.defaultdict(list)
for g in meta['games']:
    try: st = json.load(gzip.open(g['file']))['steps']
    except Exception: continue
    if len(st) < 51: continue
    P = st[50][0]['observation']['players']
    for sd in g['sides']:
        n50 = len(P[sd][3]) if isinstance(P[sd][3], dict) else 0
        out.setdefault(g['key'], {})[sd] = n50
        by[g['names'][sd]].append((n50, g['scores'][sd] > g['scores'][1 - sd]))
json.dump(out, open('../data/opening_trees.json', 'w'))
for name, v in sorted(by.items(), key=lambda kv: -len(kv[1]))[:12]:
    hi = [w for n, w in v if n >= 6]; lo = [w for n, w in v if n < 4]
    print(f"{name[:30]:30s} games {len(v):4d} tree50 mean {sum(n for n,_ in v)/len(v):.1f} | win% when >=6 trees {100*sum(hi)/max(1,len(hi)):.0f} (n{len(hi)})  when <4 {100*sum(lo)/max(1,len(lo)):.0f} (n{len(lo)})")

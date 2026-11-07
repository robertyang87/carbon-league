"""tree_fate -- why our trees earn less: ownership length, seized-by-enemy rate, late plants, income per owned step,
and neighbourhood carbon at planting (from the side's own observation)."""
import gzip, json, os, sys, collections, statistics as stt
NB = lambda c: [((c//15+dr)%15)*15+(c%15+dc)%15 for dr in (-1,0,1) for dc in (-1,0,1) if dr or dc]
def fate(st, p):
    res = []; info = {}
    for t in range(len(st)):
        o = st[t][0]['observation']; T = o.get('trees') or {}
        mine = o['players'][p][3] if len(o['players'][p]) > 3 else {}
        theirs = o['players'][1-p][3] if len(o['players'][1-p]) > 3 else {}
        carbon = st[t][p]['observation'].get('carbon') or o.get('carbon')
        for k, v in mine.items():
            if k.startswith(f'player-{p}-'):
                d = info.setdefault(k, {'t0': t, 'own': 0, 'inc': 0.0, 'end': None, 'nb': None})
                if d['nb'] is None and carbon:
                    pos = v[0] if isinstance(v, (list, tuple)) else None
                    if pos is not None: d['nb'] = sum(carbon[c] for c in NB(int(pos)))
                d['own'] += 1; d['inc'] += (T.get(k) or [0, 0])[1] or 0
        for k, d in info.items():
            if d['end'] is None and k not in mine:
                d['end'] = 'seized' if k in theirs else 'died'
    for k, d in info.items():
        res.append(d)
    return res
ms = json.load(open(sys.argv[1])); rd = sys.argv[2]; ours = sys.argv[3]
import ladder
lb = ladder.request("/api/leaderboard"); rows = lb if isinstance(lb, list) else lb.get('strategies') or lb.get('rows')
rk = {r['name']: r.get('rank') or 999 for r in rows}
grp = collections.defaultdict(list)
for m in ms:
    fn = os.path.join(rd, f"{m['id']}-{m['game']}.json.gz")
    if not os.path.exists(fn): continue
    st = json.load(gzip.open(fn))['steps']
    me = 'a' if m['name_a'] == ours else 'b'; op = 'b' if me == 'a' else 'a'; p = m['order'].index(me)
    win = (m['scores'][me] or 0) > (m['scores'][op] or 0)
    key = ('W' if win else 'L')
    for side, q in (('us', p), ('them', 1-p)):
        grp[(key, side)].append(fate(st, q))
for key in (('W','us'),('W','them'),('L','us'),('L','them')):
    games = grp[key]; tr = [d for g in games for d in g]
    if not tr: continue
    nbv = [d['nb'] for d in tr if d['nb'] is not None]
    print(f"{key[0]} {key[1]:4s} games {len(games):2d} trees/g {len(tr)/len(games):5.1f}  owned steps {stt.mean(d['own'] for d in tr):5.1f}"
          f"  seized-by-enemy {sum(d['end']=='seized' for d in tr)/len(tr):5.1%}  planted>=250 {sum(d['t0']>=250 for d in tr)/len(tr):5.1%}"
          f"  income/owned-step {stt.mean(d['inc']/max(1,d['own']) for d in tr):5.2f}  nb-carbon@plant {stt.mean(nbv) if nbv else float('nan'):6.1f}")

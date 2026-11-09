import gzip, json, os, sys, collections
me, tag = sys.argv[1], sys.argv[2]
ms = [m for m in json.load(open(f'../data/{tag}_matches.json')) if m.get('scores')]
CK = (50, 75, 100, 150, 200, 250, 285)
def fam(o):
    for p, f in (('claude-AlphaCarbon-Master2s', 'Master2s'), ('claude-AlphaCarbon-Master2', 'Master2'), ('无始', '无始'),
                 ('1C-Zero', '1C本家'), ('WB-', '1C汤'), ('hy', '1C汤'), ('AgyCarbon-P2C', 'Godslayer')):
        if o.startswith(p): return f
    return '其他'
rows = []
for m in ms:
    a = m['name_a'] == me; s = m['scores']; d = (s['a'] - s['b']) * (1 if a else -1); o = m['name_b' if a else 'name_a']
    f = f"../data/{tag}_ladder/replays/{m['id']}-{m.get('game', 1)}.json.gz"
    if not os.path.exists(f): continue
    order = m.get('order') or ['a', 'b']; pm = order.index('a' if a else 'b'); po = 1 - pm
    stp = json.load(gzip.open(f))['steps']; prev = None
    R = {'opp': o, 'fam': fam(o), 'd': d, 'cd': {}, 'u': {}, 'lost': [0, 0], 'late': [0, 0], 'dep': [0, 0], 'plant': [0, 0], 'seize': [0, 0]}
    ptrees = None
    for t in range(len(stp)):
        P = stp[t][0]['observation']['players']
        if t in CK:
            R['cd'][t] = P[pm][0] - P[po][0]
            R['u'][t] = [(sum(v[2] == 'COLLECTOR' for v in P[q][2].values()), sum(v[2] == 'PLANTER' for v in P[q][2].values()), len(P[q][3])) for q in (pm, po)]
        if prev:
            for i, q in enumerate((pm, po)):
                bases = set(P[q][1].values())
                for k, w0 in prev[q].items():
                    w1 = P[q][2].get(k)
                    if w1 is None and w0[2] == 'COLLECTOR' and w0[1] > 0:
                        R['lost'][i] += w0[1]; R['late'][i] += w0[1] if t >= 250 else 0
                    elif w1 and w0[2] == 'COLLECTOR' and w0[1] > 0 and w1[1] == 0 and w1[0] in bases:
                        R['dep'][i] += w0[1]
                for tid in P[q][3]:
                    if tid not in ptrees[q]:
                        if tid.startswith(f'player-{q}-'): R['plant'][i] += 1
                        else: R['seize'][i] += 1
        prev = {q: {k: list(v) for k, v in P[q][2].items()} for q in (pm, po)}; ptrees = {q: set(P[q][3]) for q in (pm, po)}
    rows.append(R)
L = [r for r in rows if r['d'] < 0]; W = [r for r in rows if r['d'] > 0]
mean = lambda X, f: sum(f(r) for r in X) / max(1, len(X))
print(f"{tag}: {len(rows)} games with replays, {len(W)} wins, {len(L)} losses")
for nm, X in (('LOSSES', L), ('WINS', W)):
    print(f"\n== {nm} (n {len(X)}) mean margin {mean(X, lambda r: r['d']):+.0f}")
    print("  cash diff @ " + "  ".join(f"t{t}:{mean(X, lambda r, t=t: r['cd'].get(t, 0)):+.0f}" for t in CK))
    for t in (50, 75, 100, 150, 200, 250):
        print(f"  t{t}: me c/p/tree {mean(X, lambda r,t=t: r['u'][t][0][0]):.1f}/{mean(X, lambda r,t=t: r['u'][t][0][1]):.1f}/{mean(X, lambda r,t=t: r['u'][t][0][2]):.1f}"
              f" | opp {mean(X, lambda r,t=t: r['u'][t][1][0]):.1f}/{mean(X, lambda r,t=t: r['u'][t][1][1]):.1f}/{mean(X, lambda r,t=t: r['u'][t][1][2]):.1f}")
    print(f"  deliveries me/opp {mean(X, lambda r: r['dep'][0]):.0f}/{mean(X, lambda r: r['dep'][1]):.0f} | cargo lost me/opp {mean(X, lambda r: r['lost'][0]):.0f}/{mean(X, lambda r: r['lost'][1]):.0f}"
          f" (last 50: {mean(X, lambda r: r['late'][0]):.0f}/{mean(X, lambda r: r['late'][1]):.0f}) | planted me/opp {mean(X, lambda r: r['plant'][0]):.1f}/{mean(X, lambda r: r['plant'][1]):.1f} seized {mean(X, lambda r: r['seize'][0]):.1f}/{mean(X, lambda r: r['seize'][1]):.1f}")
print("\n== each loss")
for r in sorted(L, key=lambda r: r['d']):
    print(f"  {r['d']:+6.0f} {r['opp'][:28]:28s} | t75 {r['cd'][75]:+5.0f} t150 {r['cd'][150]:+5.0f} t250 {r['cd'][250]:+5.0f} | trees t75 {r['u'][75][0][2]}/{r['u'][75][1][2]} t150 {r['u'][150][0][2]}/{r['u'][150][1][2]}"
          f" | col t75 {r['u'][75][0][0]}/{r['u'][75][1][0]} | dep {r['dep'][0]:.0f}/{r['dep'][1]:.0f} lost {r['lost'][0]:.0f}/{r['lost'][1]:.0f} late {r['late'][0]:.0f} | plant {r['plant'][0]}/{r['plant'][1]} seize {r['seize'][0]}/{r['seize'][1]}")

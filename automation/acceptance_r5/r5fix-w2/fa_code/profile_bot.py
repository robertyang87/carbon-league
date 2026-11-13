"""profile_bot -- behavioural profile of a ladder bot from replays (data/ladder_top).
  python3 profile_bot.py <family: 1C|Master> [vs-family filter]
Units over time, recruit order, planting (step, distance to nearest own base, 8-neighbour carbon),
tree lifetime use, collector trips (cargo per deposit, steps per trip), endgame (last recruit, cargo held at end)."""
import json, gzip, collections, sys
IDX = '../data/ladder_top/index.json'
FAM = {'1C': lambda n: n.startswith(('1C-Zero-v7', '1C-Zero-v8', '1C-Zero-v9')),
       'Master': lambda n: n == 'claude-AlphaCarbon-Master'}
N = 15
def tdist(a, b):
    ar, ac, br, bc = a // N, a % N, b // N, b % N
    dr = abs(ar - br); dc = abs(ac - bc)
    return min(dr, N - dr) + min(dc, N - dc)
def nb8(p):
    r, c = p // N, p % N
    return [((r + dr) % N) * N + (c + dc) % N for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]
def prof(st, p):
    R = {}
    T = len(st)
    cnt = {t: [0, 0] for t in (20, 30, 50, 100, 150, 200, 250, 290)}
    order = []; prevw = {}; plants = []; seen = set()
    trip_start = {}; trips = []; deps = []
    last_rec = 0
    for t in range(T):
        o = st[t][0]['observation']; pl = o['players'][p]
        bases = list(pl[1].values()); ws = pl[2]
        full = o.get('full_carbon') or o.get('carbon')
        if t in cnt:
            cnt[t] = [sum(w[2] == 'COLLECTOR' for w in ws.values()), sum(w[2] == 'PLANTER' for w in ws.values())]
        for wid, w in ws.items():
            if wid not in prevw:
                order.append((t, w[2][0])); last_rec = t; trip_start[wid] = t
        mine = pl[3] if len(pl) > 3 else {}
        for k, v in mine.items():
            if k not in seen and k.startswith(f'player-{p}-'):
                pos = v[0]
                prevc = st[t - 1][0]['observation'].get('full_carbon') or full
                plants.append((t, min(tdist(pos, b) for b in bases) if bases else -1,
                               sum(prevc[q] for q in nb8(pos)) if prevc else 0))
        seen |= set(mine)
        bp = set(bases)
        for wid, w0 in prevw.items():
            w1 = ws.get(wid)
            if w1 and w0[2] == 'COLLECTOR' and w0[1] > 0 and w1[1] == 0 and w1[0] in bp:
                deps.append(w0[1]); trips.append(t - trip_start.get(wid, t)); trip_start[wid] = t
        prevw = {k: list(v) for k, v in ws.items()}
    end_cargo = sum(w[1] for w in st[-1][0]['observation']['players'][p][2].values())
    R.update(cnt=cnt, order=''.join(c for _, c in order[:12]), plants=plants, deps=deps, trips=trips,
             last_rec=last_rec, end_cargo=end_cargo, n_rec=len(order))
    return R
def main(fam, vs=None):
    idx = json.load(open(IDX)); f = FAM[fam]
    rows = []
    for g in idx:
        for s in ('a', 'b'):
            if not f(g['strategies'][s]['name']): continue
            o = 'b' if s == 'a' else 'a'; on = g['strategies'][o]['name']
            if f(on) or (vs and not FAM[vs](on)): continue
            st = json.load(gzip.open('../data/ladder_top/' + g['file']))['steps']
            r = prof(st, g['order'].index(s)); r['win'] = g['winner'] == s; rows.append(r)
    n = len(rows); print(f'== {fam}{" vs " + vs if vs else ""}: {n} games, {sum(r["win"] for r in rows)} wins')
    print(' units (collectors/planters):', '  '.join(f't{t}: {sum(r["cnt"][t][0] for r in rows)/n:.1f}/{sum(r["cnt"][t][1] for r in rows)/n:.1f}' for t in rows[0]['cnt']))
    oc = collections.Counter(r['order'][:6] for r in rows)
    print(' first-6 recruit orders:', oc.most_common(6))
    pls = [x for r in rows for x in r['plants']]
    for lo, hi in ((0, 50), (50, 100), (100, 200), (200, 300)):
        q = [x for x in pls if lo <= x[0] < hi]
        if q: print(f'  plants steps {lo}-{hi}: {len(q)/n:.1f}/game, dist-to-base {sum(x[1] for x in q)/len(q):.1f}, nb8 carbon {sum(x[2] for x in q)/len(q):.0f}')
    d = [x for r in rows for x in r['deps']]; tr = [x for r in rows for x in r['trips'] if x > 0]
    print(f' deposits/game {len(d)/n:.1f}, cargo/deposit {sum(d)/len(d):.0f}, steps/trip {sum(tr)/len(tr):.1f}')
    print(f' recruits/game {sum(r["n_rec"] for r in rows)/n:.1f}, last recruit step {sum(r["last_rec"] for r in rows)/n:.0f}, cargo held at end {sum(r["end_cargo"] for r in rows)/n:.0f}')
if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)

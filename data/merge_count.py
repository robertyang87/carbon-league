"""merge_count -- own-car merges (two own collectors meeting off a base cell) per game by phase, from ladder replays.
A vanished own collector is a MERGE if no enemy car grew by >= 90% of its cargo and some own car's cargo grew;
it is excluded when the absorbing own car stands on an own base cell (merged at home = deposited together).
Zero-cargo vanishings with no enemy gain are counted apart (empty cars colliding). Columns: ours, and each listed
opponent's own merges seen from our replays."""
import gzip, json, collections, os
PH = ((1, 100), (101, 200), (201, 270), (271, 300))
def ph(t): return next(i for i, (a, b) in enumerate(PH) if a <= t <= b)
def scan(stp, p):
    out = [0] * 4; zero = 0; cg = 0.0
    for t in range(len(stp) - 1):
        P0 = stp[t][0]['observation']['players']; P1 = stp[t + 1][0]['observation']['players']
        eg = [v[1] - P0[1 - p][2][k][1] for k, v in P1[1 - p][2].items() if k in P0[1 - p][2]]
        og = [(v[1] - P0[p][2][k][1], v[0]) for k, v in P1[p][2].items() if k in P0[p][2]]
        bases = set(int(b) for b in P1[p][1].values())
        for uid, w in P0[p][2].items():
            if w[2] != 'COLLECTOR' or uid in P1[p][2]: continue
            c = w[1]
            if c > 0 and any(g >= 0.9 * c for g in eg): continue          # enemy seizure
            if c <= 0:
                zero += 1; continue
            hit = [pos for g, pos in og if g >= 0.9 * c]
            if hit and all(int(pos) in bases for pos in hit): continue  # merged on a base cell
            out[ph(t + 1)] += 1; cg += c
    return out, zero, cg
rows = collections.defaultdict(lambda: [[0] * 4, 0, 0.0, 0])
for tag, me in (('v14', 'claude-opus55-cloud-v14'), ('v15', 'claude-opus55-cloud-v15'), ('v16b', 'claude-opus55-cloud-v16b'), ('v17', 'claude-opus55-cloud-v17')):
    for m in json.load(open(f'{tag}_matches.json')):
        f = f"{tag}_ladder/replays/{m['id']}-{m.get('game', 1)}.json.gz"
        if not m.get('scores') or not os.path.exists(f): continue
        a = m['name_a'] == me
        if not a and m['name_b'] != me: me2 = None; continue
        order = m.get('order') or ['a', 'b']; pm = order.index('a' if a else 'b'); o = m['name_b' if a else 'name_a']
        stp = json.load(gzip.open(f))['steps']
        for who, p in ((tag, pm), (o.replace('claude-AlphaCarbon-', ''), 1 - pm)):
            r = rows[who]; x, z, c = scan(stp, p)
            r[0] = [i + j for i, j in zip(r[0], x)]; r[1] += z; r[2] += c; r[3] += 1
print(f"{'player':28s} games  merges/game by step 1-100 | 101-200 | 201-270 | 271-300 | total | cargo merged/game | empty-car deaths/game")
for who, (x, z, c, n) in sorted(rows.items(), key=lambda kv: -kv[1][3]):
    if n < 5 and who not in ('v15',): continue
    print(f"{who[:28]:28s} {n:4d}   " + " | ".join(f"{v/n:5.2f}" for v in x) + f" | {sum(x)/n:5.2f} | {c/n:5.0f} | {z/n:4.2f}")

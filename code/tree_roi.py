"""tree_roi -- per-tree economics from ladder replays: price paid at planting vs lifetime income.
  python3 tree_roi.py MATCHES.json REPLAY_DIR OUR_NAME"""
import gzip, json, os, sys, collections, statistics as stt
def price(n_alive): return 20 + 5 * 1.235 ** n_alive
def game(st, p):
    out = []; born = {}; inc = collections.defaultdict(float); prev = set(); owner = {}
    for t in range(len(st)):
        o = st[t][0]['observation']; T = o.get('trees') or {}
        alive_prev = len(prev)
        mine = set(o['players'][p][3].keys()) if len(o['players'][p]) > 3 else set()
        for k in mine:
            if k not in prev and k.startswith(f'player-{p}-') and k not in born:
                born[k] = (t, price(alive_prev_all))
            v = T.get(k)
            if v: inc[k] += v[1] or 0
        alive_prev_all = sum(len(o['players'][q][3]) if len(o['players'][q]) > 3 else 0 for q in (0, 1))
        prev = set().union(*[set(o['players'][q][3].keys()) if len(o['players'][q]) > 3 else set() for q in (0, 1)])
    return [(t, pr, inc[k]) for k, (t, pr) in born.items()]
def main(mf, rd, ours):
    ms = json.load(open(mf)); agg = {True: [], False: []}; top = {True: [], False: []}
    import ladder
    lb = ladder.request("/api/leaderboard"); rows = lb if isinstance(lb, list) else lb.get('strategies') or lb.get('rows')
    rk = {r['name']: r.get('rank') or 999 for r in rows}
    for m in ms:
        fn = os.path.join(rd, f"{m['id']}-{m['game']}.json.gz")
        if not os.path.exists(fn): continue
        st = json.load(gzip.open(fn))['steps']
        me = 'a' if m['name_a'] == ours else 'b'; op = 'b' if me == 'a' else 'a'; p = m['order'].index(me)
        win = (m['scores'][me] or 0) > (m['scores'][op] or 0)
        for side, q in (('us', p), ('them', 1 - p)):
            trees = game(st, q)
            agg[win].append((side, trees))
            if rk.get(m['name_' + op], 999) <= 6: top[win].append((side, m['name_' + op], trees))
    def summ(lst, label):
        for side in ('us', 'them'):
            tr = [x for s, *rest in lst if s == side for x in rest[-1]]
            if not tr: continue
            buckets = [(0, 40), (40, 80), (80, 150), (150, 300), (300, 1e9)]
            print(f"  {label} {side:4s} trees/game {len(tr)/max(1,sum(1 for s,*_ in lst if s==side)):5.1f}  "
                  f"avg price {stt.mean(x[1] for x in tr):6.1f}  avg life income {stt.mean(x[2] for x in tr):6.1f}  net/tree {stt.mean(x[2]-x[1] for x in tr):+6.1f}")
            for lo, hi in buckets:
                b = [x for x in tr if lo <= x[1] < hi]
                if b: print(f"      price {lo:>3}-{'' if hi>1e8 else hi:<4} n {len(b):4d}  income {stt.mean(x[2] for x in b):6.1f}  net {stt.mean(x[2]-x[1] for x in b):+7.1f}  loss-making {sum(x[2]<x[1] for x in b)/len(b):.0%}  step(avg) {stt.mean(x[0] for x in b):5.0f}")
    print("== ALL WINS"); summ(agg[True], 'W')
    print("== ALL LOSSES"); summ(agg[False], 'L')
    print("== vs TOP-6: wins"); summ([(s, t) for s, n, t in top[True]], 'W6')
    print("== vs TOP-6: losses"); summ([(s, t) for s, n, t in top[False]], 'L6')
if __name__ == "__main__": main(*sys.argv[1:4])

"""ladder_compare NAME_A TAG_A NAME_B TAG_B [--top] -- side-by-side ladder-replay metrics of two of our entries
(e.g. v17d2 vs v17 over the same period). Per game: final margin, our stranded cargo (cargo on our cars in the
last observation), our tree count at steps 150/200/250, opponent tree income proxy (opponent tree-steps), and the
number of trees the OPPONENT planted (new tree ids of the opponent that were not seized from us).
--top restricts to the top-4 opponents (Master2, Master2s, 无始6.0, 无始7.0)."""
import gzip, json, os, statistics as S, sys

TOP = ('claude-AlphaCarbon-Master2s', 'claude-AlphaCarbon-Master2', '无始6.0', '无始7.0')


def scan(name, tag, top):
    out = []
    for m in json.load(open(f'{tag}_matches.json')):
        f = f"{tag}_ladder/replays/{m['id']}-{m.get('game', 1)}.json.gz"
        if not m.get('scores') or not os.path.exists(f):
            continue
        a = m['name_a'] == name; o = m['name_b' if a else 'name_a']
        if top and o not in TOP:
            continue
        s = m['scores']; d = (s['a'] - s['b']) * (1 if a else -1)
        order = m.get('order') or ['a', 'b']; p = order.index('a' if a else 'b'); q = 1 - p
        stp = json.load(gzip.open(f))['steps']
        seen_ours = set(); seen_theirs = set(); opp_planted = 0; opp_ts = 0; out_t = []
        for t in range(len(stp)):
            P = stp[t][0]['observation']['players']
            mine = P[p][3] if isinstance(P[p][3], dict) else {}
            theirs = P[q][3] if isinstance(P[q][3], dict) else {}
            seen_ours |= set(mine)
            for tid in theirs:
                if tid not in seen_theirs and tid not in seen_ours:
                    opp_planted += 1                          # a new opponent tree that was never ours: planted
            seen_theirs |= set(theirs)
            opp_ts += len(theirs)
            if t in (150, 200, 250):
                out_t.append(len(mine))
        last = stp[-1][0]['observation']['players']
        out.append(dict(d=d, strand=sum(w[1] for w in last[p][2].values()), trees=out_t, opp_planted=opp_planted, opp_ts=opp_ts))
    return out


def show(lbl, R):
    if not R:
        print(lbl, 'no games'); return
    T = [r['trees'] for r in R if len(r['trees']) == 3]
    print(f"{lbl:10s} n {len(R):3d} | mean margin {S.mean(r['d'] for r in R):+.0f} | strand {S.mean(r['strand'] for r in R):.0f} "
          f"| our trees @150/200/250 {[round(S.mean(t[i] for t in T), 1) for i in range(3)]} | opp planted {S.mean(r['opp_planted'] for r in R):.1f} "
          f"| opp tree-steps {S.mean(r['opp_ts'] for r in R):.0f}")


if __name__ == '__main__':
    na, ta, nb, tb = sys.argv[1:5]; top = '--top' in sys.argv
    show(ta, scan(na, ta, top)); show(tb, scan(nb, tb, top))

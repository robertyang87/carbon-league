"""unit_ledger -- realized return of every recruit and every tree, read from replays (nothing estimated).

  collector : carbon it deposited at our bases - 30
  planter   : carbon absorbed by the trees it planted / seized while we own them - tree prices - 30
  tree      : carbon absorbed while we own it - price paid (plant: 20, or 20 + 5 * 1.235^(living trees, both sides)
              once we own a tree; seize: 20)
  "tree business" per game = tree income - tree prices - 30 * planters recruited

Sources (both sides of every game are scored; the side named NAME is "self"):
  python3 unit_ledger.py ladder DIR NAME     DIR/rows.json (ladder.py matches) + DIR/rep/<id>-<game>.json.gz
  python3 unit_ledger.py corpus DIR NAME     DIR/index.json (ladder_corpus / awr_gen snapshot) + its replay files;
                                             NAME may be a prefix (e.g. RyanX-AIPowercore-M)
Options:  --json OUT   also write one row per unit / tree (labels for outcome-weighted training):
                       {game, side, kind, step, net, gross, cost, trees, life} and {game, side, tree, step, price, inc, held, fate}
Tree income comes from obs["trees"][tid][1] (the engine's last absorption), so replays must carry it (ladder ones do)."""
import collections
import gzip
import json
import os
import statistics as stt
import sys

BK = ((0, 50), (50, 100), (100, 150), (150, 200), (200, 240), (240, 256), (256, 276), (276, 300))
PRICE_BK = ((0, 26), (26, 35), (35, 50), (50, 80), (80, 9999))


def _xy(i):
    return (i // 15, i % 15)


def score_side(d, p):
    """-> (units, trees) for player p of replay d"""
    st = d["steps"]
    units, trees, by_tree = {}, {}, {}
    dep = collections.Counter(); tinc = collections.Counter(); tcost = collections.Counter(); ntree = collections.Counter()
    prev_w = None; prev_t = {}; pall = 0
    for t, s in enumerate(st):
        o = s[0]["observation"]; P = o["players"][p]; O = o["players"][1 - p]
        W, T = P[2], P[3]; tinfo = o.get("trees") or {}
        bases = {_xy(b) for b in P[1].values()}
        if prev_w is not None:
            for u in set(W) - set(prev_w):
                units[u] = dict(step=t, kind=W[u][2], died=None)
            for u in set(prev_w) - set(W):
                if u in units:
                    units[u]["died"] = t
            for u, w in W.items():
                if u in prev_w and w[2] == "COLLECTOR" and prev_w[u][1] > 5 and w[1] < 0.5 and _xy(w[0]) in bases:
                    dep[u] += prev_w[u][1]
            for tid, tv in T.items():
                if tid not in prev_t:
                    own = tid.startswith(f"player-{p}-")
                    price = (20.0 if not prev_t else 20.0 + 5.0 * 1.235 ** pall) if own else 20.0
                    who = [u for u, w in prev_w.items() if w[2] == "PLANTER" and w[0] == tv[0]]
                    trees[tid] = dict(step=t, kind="plant" if own else "seize", price=price, inc=0.0, end=None, fate=None)
                    by_tree[tid] = who[0] if who else None
                    if who:
                        tcost[who[0]] += price; ntree[who[0]] += 1
            for tid, r in trees.items():
                if r["end"] is None and tid not in T:
                    r["end"] = t; r["fate"] = "seized" if tid in O[3] else "withered"
        for tid in T:
            if tid in trees and tid in tinfo:
                a = float(tinfo[tid][1] or 0.0)
                trees[tid]["inc"] += a
                if by_tree.get(tid) is not None:
                    tinc[by_tree[tid]] += a
        prev_w = W; prev_t = dict(T); pall = sum(len(o["players"][q][3]) for q in (0, 1))
    for r in trees.values():
        if r["end"] is None:
            r["end"] = len(st) - 1; r["fate"] = "standing"
        r["held"] = r["end"] - r["step"]
    out = []
    for u, r in units.items():
        if r["kind"] == "COLLECTOR":
            gross, cost = dep[u], 0.0
        else:
            gross, cost = tinc[u], tcost[u]
        out.append(dict(kind=r["kind"], step=r["step"], gross=gross, cost=cost, net=gross - cost - 30.0, trees=ntree[u],
                        life=(r["died"] if r["died"] is not None else len(st)) - r["step"]))
    return out, list(trees.values())


def load(mode, D, name):
    """-> list of (self_units, self_trees, opp_units, opp_trees, opp_name, game_key)"""
    G = []
    if mode == "ladder":
        for m in json.load(open(os.path.join(D, "rows.json"))):
            f = os.path.join(D, "rep", f"{m['id']}-{m.get('game', 1)}.json.gz")
            if not os.path.exists(f) or not m.get("scores"):
                continue
            nm = {"a": m["name_a"], "b": m["name_b"]}; names = [nm[m["order"][0]], nm[m["order"][1]]]
            ps = [i for i, n in enumerate(names) if n.startswith(name)]
            if len(ps) != 1:
                continue
            d = json.load(gzip.open(f)); p = ps[0]
            G.append(score_side(d, p) + score_side(d, 1 - p) + (names[1 - p], f"{m['id']}-{m.get('game', 1)}"))
    else:
        for m in json.load(open(os.path.join(D, "index.json"))):
            nm = {k: v["name"] for k, v in m["strategies"].items()}; names = [nm[m["order"][0]], nm[m["order"][1]]]
            ps = [i for i, n in enumerate(names) if n.startswith(name)]
            f = os.path.join(D, m["file"])
            if len(ps) != 1 or not os.path.exists(f):
                continue
            d = json.load(gzip.open(f)); p = ps[0]
            G.append(score_side(d, p) + score_side(d, 1 - p) + (names[1 - p], m["file"]))
    return G


def report(label, sides):
    """sides: list of (units, trees)"""
    n = len(sides)
    if not n:
        return
    U = [u for us, _ in sides for u in us]; T = [t for _, ts in sides for t in ts]
    print(f"\n### {label}  ({n} games)")
    for kind, zh in (("COLLECTOR", "collectors"), ("PLANTER", "planters")):
        print(f"  {zh}: recruit step | per game | mean net | share losing | mean life" + (" | trees each" if kind == "PLANTER" else ""))
        for lo, hi in BK:
            x = [u for u in U if u["kind"] == kind and lo <= u["step"] < hi]
            if x:
                print(f"    {lo:3d}-{hi:<3d} | {len(x) / n:4.1f} | {stt.mean(u['net'] for u in x):+6.0f} | {sum(u['net'] < 0 for u in x) / len(x):4.0%} | "
                      f"{stt.mean(u['life'] for u in x):4.0f}" + (f" | {stt.mean(u['trees'] for u in x):.2f}" if kind == "PLANTER" else ""))
    PL = [u for u in U if u["kind"] == "PLANTER"]
    for kind in ("plant", "seize"):
        x = [t for t in T if t["kind"] == kind]
        if x:
            print(f"  {kind:5s} trees: {len(x) / n:4.1f}/game | price {stt.mean(t['price'] for t in x):3.0f} | income {stt.mean(t['inc'] for t in x):3.0f} | "
                  f"net {stt.mean(t['inc'] - t['price'] for t in x):+4.0f} | losing {sum(t['inc'] < t['price'] for t in x) / len(x):.0%} | "
                  f"held {stt.mean(t['held'] for t in x):.0f} steps | seized from us {sum(t['fate'] == 'seized' for t in x) / len(x):.0%}")
    x = [t for t in T if t["kind"] == "plant"]
    if x:
        print("  planted trees by price paid:  price | share | net | losing")
        for lo, hi in PRICE_BK:
            g = [t for t in x if lo <= t["price"] < hi]
            if g:
                print(f"    {lo:3d}-{min(hi, 999):<3d} | {len(g) / len(x):4.0%} | {stt.mean(t['inc'] - t['price'] for t in g):+4.0f} | {sum(t['inc'] < t['price'] for t in g) / len(g):.0%}")
    if PL:
        z = sum(1 for u in PL if u["trees"] == 0)
        inc = sum(t["inc"] for t in T) / n; cost = sum(t["price"] for t in T) / n; rec = 30.0 * len(PL) / n
        print(f"  planters {len(PL) / n:.1f}/game, zero-yield {z / len(PL):.0%} ({30.0 * z / n:.0f} wasted/game)")
        print(f"  TREE BUSINESS per game: income {inc:.0f} - tree prices {cost:.0f} - planter recruits {rec:.0f} = {inc - cost - rec:+.0f}")


def main():
    a = sys.argv[1:]; out = None
    if "--json" in a:
        i = a.index("--json"); out = a[i + 1]; del a[i:i + 2]
    mode, D, name = a[0], a[1], a[2]
    G = load(mode, D, name)
    report(f"{name} (all games)", [(g[0], g[1]) for g in G])
    report(f"opponents of {name} (all games)", [(g[2], g[3]) for g in G])
    by = collections.defaultdict(list)
    for g in G:
        by[g[4]].append(g)
    for opp, gs in sorted(by.items(), key=lambda kv: -len(kv[1])):
        if len(gs) >= 8:
            report(f"{name} vs {opp}", [(g[0], g[1]) for g in gs]); report(f"{opp} (in those games)", [(g[2], g[3]) for g in gs])
    if out:
        with open(out, "w") as f:
            for g in G:
                for side, (us, ts) in (("self", (g[0], g[1])), ("opp", (g[2], g[3]))):
                    for u in us:
                        f.write(json.dumps(dict(u, game=g[5], side=side, rec="unit")) + "\n")
                    for t in ts:
                        f.write(json.dumps(dict(t, game=g[5], side=side, rec="tree")) + "\n")
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

"""replay_ledger -- per-side economic/combat ledger from ladder replays.

  python3 replay_ledger.py MATCHES.json REPLAY_DIR OUR_NAME [OUT.json]

For every replay of a match in MATCHES.json (compact /api/matches rows with
name_a/name_b/order/scores), computes for us and the opponent:
  cash at steps 50/100/150/200/250/300, peak and step-100 collectors/planters,
  recruits, worker deaths (total, collectors, planters, in last 30 steps),
  cargo lost in deaths (total / last 30 steps), deposits (cargo delivered
  at own base), trees owned (step 100/200/300), trees planted, trees seized
  (now owned but planted by the other side), trees lost to seizure.
Prints per-game rows and per-opponent-family means (ours vs theirs).
"""
import collections
import glob
import gzip
import json
import os
import sys

CHK = (50, 100, 150, 200, 250, 299)


def side_stats(st, p):
    T = len(st)
    out = {f"cash{c}": None for c in CHK}
    prev_w = None
    deaths = dcol = dpl = dlate = 0
    lost = lost_late = 0.0
    deposits = 0.0
    recruits = 0
    planted = set(); seized_now = set()
    ncol = []; npl = []; ntree = {}
    seen_trees = set()
    for t in range(T):
        o = st[t][0]["observation"]
        pl = o["players"][p]
        cash, bases, workers, trees = pl[0], pl[1], pl[2], pl[3] if len(pl) > 3 else {}
        if t in CHK:
            out[f"cash{t}"] = round(cash, 1)
        ncol.append(sum(1 for w in workers.values() if w[2] == "COLLECTOR"))
        npl.append(sum(1 for w in workers.values() if w[2] == "PLANTER"))
        tids = set(trees.keys())
        if t in (100, 200, 299):
            ntree[t] = len(tids)
        for k in tids:
            if k.startswith(f"player-{p}-"):
                if k not in seen_trees:
                    planted.add(k)
            else:
                seized_now.add(k)
        seen_trees |= tids
        if prev_w is not None:
            base_pos = set(bases.values())
            for wid, w0 in prev_w.items():
                w1 = workers.get(wid)
                if w1 is None:
                    deaths += 1
                    if w0[2] == "COLLECTOR":
                        dcol += 1
                        lost += w0[1]
                        if t >= T - 30:
                            lost_late += w0[1]
                    else:
                        dpl += 1
                    if t >= T - 30:
                        dlate += 1
                elif w0[2] == "COLLECTOR" and w0[1] > 0 and w1[1] == 0 and w1[0] in base_pos:
                    deposits += w0[1]
            recruits += sum(1 for wid in workers if wid not in prev_w)
        prev_w = {k: list(v) for k, v in workers.items()}
    out.update(peak_col=max(ncol), peak_pl=max(npl), col100=ncol[100] if T > 100 else None,
               pl100=npl[100] if T > 100 else None, recruits=recruits, deaths=deaths, dcol=dcol, dpl=dpl,
               dlate=dlate, lost=round(lost), lost_late=round(lost_late), deposits=round(deposits),
               trees100=ntree.get(100), trees200=ntree.get(200), trees300=ntree.get(299),
               planted=len(planted), seized=len(seized_now))
    return out


def family(name):
    for f in ("1C-Zero", "AlphaCarbon-Master", "RyanX", "Godslayer", "claude-AlphaCarbon", "豆包", "dafeiyu"):
        if f in name:
            return f
    return name


def main(mfile, rdir, ours, out=None):
    ms = json.load(open(mfile))
    rows = []
    for m in ms:
        fn = os.path.join(rdir, f"{m['id']}-{m['game']}.json.gz")
        if not os.path.exists(fn):
            continue
        d = json.load(gzip.open(fn))
        st = d["steps"]
        me = "a" if m["name_a"] == ours else "b"
        op = "b" if me == "a" else "a"
        pme = m["order"].index(me); pop = 1 - pme
        a = side_stats(st, pme); b = side_stats(st, pop)
        rows.append({"id": m["id"], "opp": m["name_" + op], "fam": family(m["name_" + op]),
                     "score_me": m["scores"][me], "score_op": m["scores"][op],
                     "me": a, "op": b})
    if out:
        json.dump(rows, open(out, "w"), indent=1)
    keys = ["cash100", "cash200", "cash299", "deposits", "lost", "lost_late", "deaths", "dcol", "dpl",
            "peak_col", "peak_pl", "col100", "pl100", "recruits", "planted", "seized", "trees200", "trees300"]
    by = collections.defaultdict(list)
    for r in rows:
        by[r["fam"]].append(r); by["ALL"].append(r)
    for fam, rs in sorted(by.items(), key=lambda x: -len(x[1])):
        print(f"\n== vs {fam}: {len(rs)} games, score {sum(r['score_me'] for r in rs)/len(rs):.0f} : "
              f"{sum(r['score_op'] for r in rs)/len(rs):.0f}")
        for k in keys:
            vm = [r["me"][k] for r in rs if r["me"][k] is not None]
            vo = [r["op"][k] for r in rs if r["op"][k] is not None]
            if vm and vo:
                print(f"  {k:10s} us {sum(vm)/len(vm):8.1f}   them {sum(vo)/len(vo):8.1f}")
    return rows


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else None)

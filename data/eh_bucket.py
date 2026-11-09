"""eh_bucket TAG_TE TAG_BASE -- paired by (seed, side): v17te minus v17t, bucketed by how often each endgame rule
fired in the v17te game (eh_e = edge rule, d in [left-1, left]; eh_h = heavy-only, fired only via the wider slack).
Per bucket: n, paired margin diff, stranded cargo and cargo lost in the last 30 steps (both sides of the account)."""
import json, sys, statistics as S
L = "/Users/baggioustb/dev/carbon/carbon-league/data/bigeval_live/"


def rows(tag):
    out = {}
    for l in open(L + tag + ".jsonl"):
        g = json.loads(l)
        if g.get("margin") is not None and not g.get("err"):
            out[(g["seed"], g["side"])] = g
    return out


te, base = rows(sys.argv[1]), rows(sys.argv[2])
P = [(te[k], base[k]) for k in te if k in base]
print(f"paired games {len(P)}")


def show(name, f):
    B = {}
    for a, b in P:
        B.setdefault(f(a), []).append((a, b))
    for k in sorted(B):
        v = B[k]; d = [a["margin"] - b["margin"] for a, b in v]
        st = [(a.get("strand") or 0) - (b.get("strand") or 0) for a, b in v]
        l30 = [a["lx"]["pol"].get("late30", 0) - b["lx"]["pol"].get("late30", 0) for a, b in v]
        fe = [a["lx"]["pol"].get("late30_eh") or 0 for a, _ in v]; fb = [b["lx"]["pol"].get("late30_eh") or 0 for _, b in v]
        la = sum(a["lx"]["pol"].get("late30", 0) for a, _ in v); lb = sum(b["lx"]["pol"].get("late30", 0) for _, b in v)
        se = S.stdev(d) / len(d) ** .5 if len(d) > 1 else 0
        print(f"  {name} {k}: n {len(v)} | margin diff {S.mean(d):+.0f} ± {se:.0f} | strand diff {S.mean(st):+.0f} "
              f"| lost-last-30 diff {S.mean(l30):+.0f} | lost while under the rule (<=3 steps before death): te {100*sum(fe)/max(1,la):.0f}% vs base(shadow) {100*sum(fb)/max(1,lb):.0f}% | strand te/base {S.mean([a.get('strand') or 0 for a, _ in v]):.0f}/{S.mean([b.get('strand') or 0 for _, b in v]):.0f} "
              f"| enemy-seized last 30 te/base {S.mean([a['lx']['pol'].get('late30', 0) for a, _ in v]):.0f}/{S.mean([b['lx']['pol'].get('late30', 0) for _, b in v]):.0f} "
              f"| own merges last 30 te/base {S.mean([a['lx']['pol'].get('late30_self', 0) for a, _ in v]):.0f}/{S.mean([b['lx']['pol'].get('late30_self', 0) for _, b in v]):.0f}")


show("all", lambda a: "-")
show("heavy fired", lambda a: "0" if not a.get("eh_h") else ("1-2" if a["eh_h"] <= 2 else "3+"))
show("edge fired", lambda a: "0" if not a.get("eh_e") else ("1-2" if a["eh_e"] <= 2 else "3+"))

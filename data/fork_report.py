"""fork_report FORKS.jsonl [...] -- what fork_gen measured: per decision kind, the mean effect of making the change
(child - parent) at each horizon with its standard error and spread, split by game phase; and, for the forks played to
the end, how well the short-horizon effect predicts the final one."""
import collections
import json
import statistics as st
import sys


def ms(x):
    return (st.mean(x), st.stdev(x) / len(x) ** 0.5 if len(x) > 1 else 0.0, st.stdev(x) if len(x) > 1 else 0.0)


def corr(a, b):
    ma, mb = st.mean(a), st.mean(b)
    va = sum((x - ma) ** 2 for x in a); vb = sum((y - mb) ** 2 for y in b)
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (va * vb) ** 0.5 if va > 0 and vb > 0 else float("nan")


def main():
    R = [json.loads(l) for f in sys.argv[1:] for l in open(f) if l.strip()]
    live = [r for r in R if not r["same"]]
    hs = sorted({k for r in R for k in r["d"] if k != "end"}, key=int)
    print(f"{len(R)} forks from {len({(r['seed']) for r in R})} games; {len(R) - len(live)} changed nothing (ignored by the engine)")
    print("\neffect of making the change (child - parent), mean +- SE [spread]; + means the change was better than what the net did")
    PH = (("all", 0, 999), ("0-60", 0, 60), ("60-240", 60, 240), ("240+", 240, 999))
    for kind in sorted({r["kind"] for r in live}):
        for nm, lo, hi in PH:
            g = [r for r in live if r["kind"] == kind and lo <= r["t"] < hi]
            if len(g) < 8:
                continue
            row = f"  {kind:10s} {nm:7s} n{len(g):4d}"
            for h in hs:
                x = [r["d"][h] for r in g if h in r["d"]]
                if len(x) > 1:
                    m, se, sd = ms(x); row += f" | h{h}: {m:+6.0f} +-{se:3.0f} [{sd:4.0f}]"
            x = [r["d"]["end"] for r in g if "end" in r["d"] and r.get("to_end")]
            if len(x) > 3:
                m, se, sd = ms(x); row += f" | end: {m:+6.0f} +-{se:3.0f} [{sd:4.0f}] n{len(x)}"
            print(row)
    cars = [r for r in live if r["kind"] == "car>2nd" and "p2" in r]
    if len(cars) > 20:
        print("\ncar>2nd by how close the net's second choice was (p2 = its probability), and by cargo on board")
        def line(nm, g):
            if len(g) < 8:
                return
            row = f"  {nm:22s} n{len(g):5d}"
            for h in hs:
                x = [r["d"][h] for r in g if h in r["d"]]
                if len(x) > 1:
                    m, se, sd = ms(x); row += f" | h{h}: {m:+6.0f} +-{se:3.0f} [{sd:4.0f}] better {sum(v > 0 for v in x) / len(x):.0%}"
            print(row)
        for nm, lo, hi in (("p2 >= 0.30", 0.30, 2), ("p2 0.10-0.30", 0.10, 0.30), ("p2 0.02-0.10", 0.02, 0.10), ("p2 < 0.02", -1, 0.02)):
            line(nm, [r for r in cars if lo <= r["p2"] < hi])
        for nm, lo, hi in (("cargo 0", -1, 0.5), ("cargo 0-100", 0.5, 100), ("cargo 100+", 100, 1e9)):
            line(nm, [r for r in cars if lo <= r["cargo"] < hi])
        for nm, lo, hi in PH[1:]:
            line("step " + nm, [r for r in cars if lo <= r["t"] < hi])
    E_ = [r for r in live if r.get("to_end") and "end" in r["d"]]
    if len(E_) > 10:
        print(f"\nforks played to the end: {len(E_)}; spread of the final effect {st.stdev([r['d']['end'] for r in E_]):.0f}")
        for h in hs:
            g = [r for r in E_ if h in r["d"]]
            if len(g) > 10:
                print(f"  corr(effect at h{h}, final effect) = {corr([r['d'][h] for r in g], [r['d']['end'] for r in g]):+.2f}  (n {len(g)})")
        for nm, lo, hi in PH[1:]:
            g = [r for r in E_ if lo <= r["t"] < hi and hs[-1] in r["d"]]
            if len(g) > 10:
                print(f"  phase {nm}: corr(h{hs[-1]}, final) = {corr([r['d'][hs[-1]] for r in g], [r['d']['end'] for r in g]):+.2f}  (n {len(g)}), final spread {st.stdev([r['d']['end'] for r in g]):.0f}")


if __name__ == "__main__":
    main()

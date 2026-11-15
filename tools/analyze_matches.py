#!/usr/bin/env python3
"""Stats for one strategy's match list (117-endpoint compact format)."""
import json, statistics, sys
from datetime import datetime, timezone, timedelta

CST = timezone(timedelta(hours=8))

def analyze(path, target):
    ms = json.load(open(path))
    ms = sorted(ms, key=lambda m: m["created"])
    def is_target(m, side):
        return m[f"name_{side}"] == target
    def margin(m):
        # signed margin for target
        if is_target(m, "a"):
            return m["scores"]["a"] - m["scores"]["b"]
        return m["scores"]["b"] - m["scores"]["a"]
    def won(m):
        w = m["winner"]
        return (w == "a" and is_target(m, "a")) or (w == "b" and is_target(m, "b"))
    def first_hand(m):
        # order[0] plays first; order lists 'a'/'b'
        first = m.get("order", ["a", "b"])[0]
        return (first == "a" and is_target(m, "a")) or (first == "b" and is_target(m, "b"))
    n = len(ms)
    wins = sum(1 for m in ms if won(m))
    mg = [margin(m) for m in ms]
    fh = [m for m in ms if first_hand(m)]
    sh = [m for m in ms if not first_hand(m)]
    last10 = ms[-10:]
    opps = {}
    for m in ms:
        opp = m["name_b"] if is_target(m, "a") else m["name_a"]
        w, l = opps.get(opp, (0, 0))
        if won(m):
            w += 1
        else:
            l += 1
        opps[opp] = (w, l)
    t0 = datetime.fromtimestamp(ms[0]["created"], CST)
    t1 = datetime.fromtimestamp(ms[-1]["created"], CST)
    hours = max((t1 - t0).total_seconds() / 3600, 0.01)
    print(f"== {target} ({path}) ==")
    print(f"total={n} wins={wins} win%={100*wins/n:.1f} mean={statistics.mean(mg):+.1f} median={statistics.median(mg):+.1f}")
    print(f"first-hand {sum(1 for m in fh if won(m))}/{len(fh)}  second-hand {sum(1 for m in sh if won(m))}/{len(sh)}")
    print(f"last10 {sum(1 for m in last10 if won(m))}/10 mean={statistics.mean([margin(m) for m in last10]):+.1f}")
    print(f"first game {t0:%m-%d %H:%M}  last {t1:%m-%d %H:%M}  pace {n/hours:.2f}/h ({hours:.1f}h span)")
    top = sorted(opps.items(), key=lambda kv: -(kv[1][0] + kv[1][1]))[:12]
    print("opponents: " + "; ".join(f"{k} {w}/{w+l}" for k, (w, l) in top))

if __name__ == "__main__":
    analyze(sys.argv[1], sys.argv[2])

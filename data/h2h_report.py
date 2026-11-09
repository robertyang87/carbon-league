"""h2h_report TAG [TAG ...] -- head-to-head win rate (A = first exp in the tag, the candidate) and paired margin."""
import json, statistics as S, sys
for t in sys.argv[1:]:
    try:
        R = [g for g in map(json.loads, open(f"bigeval_live/{t}.jsonl")) if g.get("margin") is not None]
    except FileNotFoundError:
        print(t, "missing"); continue
    m = [g["margin"] for g in R]; w = sum(x > 0 for x in m) / max(1, len(m))
    print(f"{t}: n {len(m)}  win rate {w:.1%} (± {(w*(1-w)/max(1,len(m)))**.5:.1%})  margin {S.mean(m):+.0f} ± {S.stdev(m)/len(m)**.5:.0f}  cargo left ours {S.mean(g.get('strand') or 0 for g in R):.0f} theirs {S.mean(g.get('ostrand') or 0 for g in R):.0f}")

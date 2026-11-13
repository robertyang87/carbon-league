"""livecheck -- running summary of in-progress bigeval runs (data/bigeval_live/*.jsonl).
  python3 livecheck.py [TAG_SUBSTRING]"""
import glob, json, os, statistics as st, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sub = sys.argv[1] if len(sys.argv) > 1 else ""
for fn in sorted(glob.glob(os.path.join(HERE, "..", "data", "bigeval_live", "*.jsonl"))):
    tag = os.path.basename(fn)[:-6]
    if sub not in tag: continue
    rows = [json.loads(l) for l in open(fn) if l.strip()]; m = [r["margin"] for r in rows if r.get("margin") is not None]
    if not m: continue
    se = st.stdev(m) / len(m) ** .5 if len(m) > 1 else 0
    w = sum(x > 0 for x in m); d = sum(x != 0 for x in m)
    print(f"{tag:30s} n {len(m):3d} wins {w:3d} ({w/len(m):4.0%}) ties {len(m)-d} wins/decided {w/max(1,d):4.0%} mean {st.mean(m):+6.0f} ± {se:4.0f}"
          f"  <-600 {sum(x<-600 for x in m)}  <-1000 {sum(x<-1000 for x in m)}  worst {min(m):+6d}  last {rows[-1]['t']}")

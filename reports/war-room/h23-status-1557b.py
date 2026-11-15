"""H23 status 15:57 run part 2: verify Hakimi v3 four live files on sync/h31.
Fetch raw content, count lines, compute wins/mean/se/lo2se from margin,
report seed range and last-game time. Writes h23-status-1557b.log.
"""
import base64, json, math, sys, os
sys.path.insert(0, "/home/hatch/workspace")
from gh_api import req

OWNER, REPO = "robertyang87", "carbon-league"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "h23-status-1557b.log")
FILES = ["hakimi3r1_m2", "hakimi3r1_echo2", "hakimi3g1_m2", "hakimi3g1_echo2"]

lines = ["MARKER H23STATUS1557B"]
def emit(s):
    lines.append(s)
    print(s, flush=True)

for name in FILES:
    path = f"synced/h31/bigeval_live/{name}.jsonl"
    try:
        obj = req("GET", f"/repos/{OWNER}/{REPO}/contents/{path}?ref=sync/h31")
        raw = base64.b64decode(obj["content"]).decode("utf-8", "replace")
        rows = [l for l in raw.splitlines() if l.strip()]
        n = len(rows)
        first = json.loads(rows[0]); last = json.loads(rows[-1])
        emit(f"FILE {name} lines={n} sha={obj['sha'][:7]}")
        emit(f"  first_keys={sorted(first.keys())}")
        emit(f"  first={json.dumps(first)[:300]}")
        emit(f"  last={json.dumps(last)[:300]}")
        # find margin-like numeric field
        cand = None
        for k in ("margin", "score", "result", "margin_mean"):
            if isinstance(first.get(k), (int, float)):
                cand = k; break
        if cand:
            ms = [json.loads(r)[cand] for r in rows]
            wins = sum(1 for m in ms if m > 0)
            mean = sum(ms) / n
            var = sum((m - mean) ** 2 for m in ms) / (n - 1) if n > 1 else 0
            se = math.sqrt(var / n)
            emit(f"  {cand}: wins={wins} mean={mean:.1f} se={se:.1f} lo2se={mean-2*se:.1f}")
        seeds = [json.loads(r).get("seed") for r in rows if json.loads(r).get("seed") is not None]
        if seeds:
            emit(f"  seeds min={min(seeds)} max={max(seeds)}")
        # last-game time field hunt
        for k in ("time", "timestamp", "ended_at", "ts", "datetime"):
            if k in last:
                emit(f"  last[{k}]={last[k]}")
    except Exception as e:
        emit(f"FILE {name} ERROR {type(e).__name__}: {e}")

with open(OUT, "w") as f:
    f.write("\n".join(lines) + "\n")
print("WROTE", OUT)

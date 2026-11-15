"""H23 status probe 19:12 - read sync/h23 branch via GitHub API. Marker H23STAT1912Q7R3."""
import json, sys, base64
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER, REPO, BR = "robertyang87", "carbon-league", "sync/h23"

def get_json(path):
    return gh_api.req("GET", path)

def fetch_raw(path, ref):
    d = gh_api.get_file(OWNER, REPO, path, ref=ref)
    if isinstance(d, dict) and d.get("content"):
        return base64.b64decode(d["content"]).decode("utf-8", errors="replace")
    return None

# 1. latest commit on sync/h23
commits = get_json(f"/repos/{OWNER}/{REPO}/commits?sha={BR}&per_page=3")
for c in commits:
    print("COMMIT:", c["sha"][:7], c["commit"]["committer"]["date"], c["commit"]["message"][:100].replace("\n", " "))

# 2. list dirs
for d in ["synced/h23", "synced/h23/bigeval_live"]:
    items = gh_api.get_file(OWNER, REPO, d, ref=BR)
    if isinstance(items, list):
        print(f"\nDIR {d} ({len(items)} items):")
        for it in items:
            print(f"  {it['name']}  size={it.get('size')}  type={it['type']}")
    else:
        print(f"\nDIR {d}: {json.dumps(items, ensure_ascii=False)[:300]}")

# 3. m3s2g1_m2.jsonl live progress
raw = fetch_raw("synced/h23/bigeval_live/m3s2g1_m2.jsonl", BR)
if raw is None:
    print("\nLIVE m3s2g1_m2.jsonl: MISSING")
else:
    lines = [l for l in raw.splitlines() if l.strip()]
    print(f"\nLIVE m3s2g1_m2.jsonl: {len(lines)} lines")
    wins = 0
    margins = []
    errs = 0
    first_ts = last_ts = None
    for l in lines:
        try:
            j = json.loads(l)
        except Exception:
            continue
        # tolerate various key names
        w = j.get("win") or j.get("winner") or j.get("w")
        m = j.get("margin") or j.get("score_diff") or j.get("diff")
        t = j.get("ts") or j.get("time") or j.get("ended_at") or j.get("timestamp")
        e = j.get("err")
        if w is True or w == 1 or (isinstance(w, str) and w.lower() in ("a", "g1", "m3s2g1")):
            wins += 1
        if isinstance(m, (int, float)):
            margins.append(m)
        if e:
            errs += 1
        if t:
            if first_ts is None:
                first_ts = t
            last_ts = t
    print(f"  wins={wins} losses={len(lines)-wins} err={errs}")
    if margins:
        print(f"  mean_margin={sum(margins)/len(margins):.1f} n={len(margins)}")
    print(f"  first_ts={first_ts} last_ts={last_ts}")
    print("  first line:", lines[0][:300] if lines else None)
    print("  last line:", lines[-1][:300] if lines else None)

# 4. *.out files full text
for f in ["m3s2g1_m2.out", "e0_m2_base.out", "m3s2r1pre.out"]:
    raw = fetch_raw(f"synced/h23/{f}", BR)
    if raw is None:
        print(f"\nOUT {f}: MISSING")
    else:
        print(f"\nOUT {f} ({len(raw)} chars):\n{raw}")

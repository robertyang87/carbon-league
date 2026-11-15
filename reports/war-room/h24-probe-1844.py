import sys, json, base64
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

owner, repo = "robertyang87", "carbon-league"
branch = "sync/h24"

# 1. Latest commits on sync/h24
r = gh_api.req("GET", f"/repos/{owner}/{repo}/commits?sha={branch}&per_page=6")
print("== latest sync/h24 commits ==")
for c in r:
    print(c["commit"]["committer"]["date"], c["sha"][:8], c["commit"]["message"].splitlines()[0][:120])

# 2. Tree of synced/h24
r = gh_api.req("GET", f"/repos/{owner}/{repo}/git/trees/{branch}?recursive=1")
paths = [t["path"] for t in r.get("tree", [])]
print("\n== synced/h24 files ==")
for p in sorted(paths):
    if p.startswith("synced/h24"):
        print(p)

# 3. Live files: read each bigeval_live jsonl
print("\n== bigeval_live contents ==")
for p in sorted(paths):
    if p.startswith("synced/h24/bigeval_live/"):
        f = gh_api.get_file(owner, repo, p, ref=branch)
        raw = base64.b64decode(f["content"]).decode("utf-8", "replace")
        lines = [l for l in raw.strip().splitlines() if l.strip()]
        print(f"--- {p}: {len(lines)} lines ---")
        if lines:
            print("first:", lines[0][:300])
            print("last :", lines[-1][:300])
        # aggregate wins/margin if jsonl has fields
        try:
            wins = 0; tot = 0.0; err = 0
            for l in lines:
                d = json.loads(l)
                tot += d.get("margin", d.get("score", 0)) or 0
                if (d.get("margin", 0) or 0) > 0: wins += 1
                if d.get("err") or d.get("error"): err += 1
            print(f"agg: wins={wins}/{len(lines)} mean={tot/len(lines):.1f} err={err}")
        except Exception as e:
            print("agg failed:", e)

# 4. cal.out files
print("\n== *_cal.out ==")
for p in sorted(paths):
    if p.startswith("synced/h24/") and p.endswith("_cal.out"):
        f = gh_api.get_file(owner, repo, p, ref=branch)
        raw = base64.b64decode(f["content"]).decode("utf-8", "replace")
        print(f"--- {p} ({len(raw)} bytes) ---")
        print(raw[:4000])

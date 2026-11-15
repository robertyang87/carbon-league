"""H23 status probe 18:27 - read sync/h23 branch via GitHub API. Marker H23STAT1827K7M3."""
import json, sys, base64
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER, REPO, BR = "robertyang87", "carbon-league", "sync/h23"
print("MARKER: H23STAT1827K7M3")

def fetch_raw(path, ref):
    d = gh_api.get_file(OWNER, REPO, path, ref=ref)
    if isinstance(d, dict) and d.get("content"):
        return base64.b64decode(d["content"]).decode("utf-8", errors="replace")
    return f"<no content: {str(d)[:150]}>"

# 1. latest commit
commits = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/commits?sha={BR}&per_page=1")
c = commits[0]
print("LATEST COMMIT:", c["sha"][:7], c["commit"]["committer"]["date"], c["commit"]["message"][:100])

# 2. list dirs
for d in ["synced/h23", "synced/h23/bigeval_live"]:
    items = gh_api.get_file(OWNER, REPO, d, ref=BR)
    if isinstance(items, list):
        print(f"DIR {d} ({len(items)} items):")
        for it in items:
            print(f"  {it['name']}  size={it.get('size')}")
    else:
        print(f"DIR {d}: {str(items)[:200]}")

# 3. live file e0_m2_base.jsonl (one game per line)
raw = fetch_raw("synced/h23/bigeval_live/e0_m2_base.jsonl", BR)
lines = [l for l in raw.splitlines() if l.strip()]
wins = 0; tot = 0.0; n = 0; errs = 0; first_t = None; last_t = None
first3 = []; last3 = []
for l in lines:
    r = json.loads(l)
    n += 1
    m = r.get("margin", 0)
    tot += m
    if m > 0:
        wins += 1
    if r.get("err"):
        errs += 1
    if first_t is None:
        first_t = r.get("t")
    last_t = r.get("t")
    if len(first3) < 3:
        first3.append(m)
    last3.append(m)
    last3 = last3[-3:]
print(f"e0_m2_base: games={n} wins={wins} losses={n-wins} mean_margin={tot/n:.1f} err_games={errs} first_t={first_t} last_t={last_t} first3={first3} last3={last3}")

# 4. cal.out files
for name in ["e0_m2_base.out"]:
    raw = fetch_raw(f"synced/h23/{name}", BR)
    print(f"{name}: len={len(raw)} content={raw[:600]!r}")

# 5. bigeval.jsonl tail
raw = fetch_raw("synced/h23/bigeval.jsonl", BR)
blines = [l for l in raw.splitlines() if l.strip()]
print(f"bigeval.jsonl: {len(blines)} lines")
for l in blines[-5:]:
    print("  TAIL:", l[:400])

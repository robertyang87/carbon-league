"""H23 status probe 18:12 part 3 - fixed aggregation (one game per line)."""
import json, sys, base64
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER, REPO, BR = "robertyang87", "carbon-league", "sync/h23"

def fetch_raw(path, ref):
    d = gh_api.get_file(OWNER, REPO, path, ref=ref)
    if isinstance(d, dict) and d.get("content"):
        return base64.b64decode(d["content"]).decode("utf-8", errors="replace")
    return f"<no content: {str(d)[:150]}>"

# fresh latest commit
commits = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/commits?sha={BR}&per_page=1")
c = commits[0]
print("LATEST COMMIT:", c["sha"][:7], c["commit"]["committer"]["date"], c["commit"]["message"][:80])

raw = fetch_raw("synced/h23/bigeval_live/e0_m2_base.jsonl", BR)
lines = [l for l in raw.splitlines() if l.strip()]
wins = 0; tot = 0.0; n = 0; errs = 0; first_t = None; last_t = None
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
print(f"e0_m2_base: games={n} wins={wins} losses={n-wins} mean_margin={tot/n:.1f} err_games={errs} first_t={first_t} last_t={last_t}")

raw = fetch_raw("synced/h23/e0_m2_base.out", BR)
print(f"e0_m2_base.out: len={len(raw)} content={raw[:300]!r}")

raw = fetch_raw("synced/h23/bigeval.jsonl", BR)
blines = [l for l in raw.splitlines() if l.strip()]
print(f"bigeval.jsonl: {len(blines)} lines")
for l in blines[-4:]:
    print("  TAIL:", l[:350])

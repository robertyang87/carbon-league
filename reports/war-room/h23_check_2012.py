import json, sys
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER, REPO = "robertyang87", "carbon-league"

def branch(name):
    b = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/branches/{name}")
    c = b["commit"]["commit"]
    print(f"{name}: sha={b['commit']['sha'][:8]} date={c['committer']['date']} msg={c['message'].splitlines()[0][:80]}")

for br in ["sync/h23", "sync/h29", "sync/h31", "main"]:
    branch(br)

items = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/contents/synced/h23?ref=sync/h23")
print(f"synced/h23 items: {len(items)}")
for it in items:
    if it["name"].endswith("_cal.out") or it["name"] in ("bigeval_live", "bigeval.jsonl"):
        print(f"h23-item: {it['name']} size={it['size']} type={it['type']}")
live = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/contents/synced/h23/bigeval_live?ref=sync/h23")
if isinstance(live, list):
    print(f"bigeval_live items: {len(live)}")
    for it in live[:20]:
        print(f"live-item: {it['name']} size={it['size']} type={it['type']}")
else:
    print(f"bigeval_live: {live}")

rl = gh_api.req("GET", "/rate_limit")
core = rl["resources"]["core"]
print(f"rate: remaining={core['remaining']} reset={core['reset']}")

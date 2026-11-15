import sys, json
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

for br in ["sync/h23", "sync/h29", "sync/h31"]:
    try:
        b = gh_api.req("GET", f"/repos/robertyang87/carbon-league/branches/{br}")
        c = b["commit"]["commit"]
        print(f"{br}: sha={b['commit']['sha'][:7]} date={c['committer']['date']} msg={c['message'].splitlines()[0][:100]}")
    except Exception as e:
        print(f"{br}: ERROR {e}")

# list synced/h23 contents on sync/h23
try:
    items = gh_api.req("GET", "/repos/robertyang87/carbon-league/contents/synced/h23?ref=sync/h23")
    if isinstance(items, list):
        for it in items:
            print(f"h23-item: {it['name']} size={it['size']} type={it['type']}")
    else:
        print("h23-list:", json.dumps(items)[:500])
except Exception as e:
    print("h23-list ERROR", e)

# rate limit
try:
    rl = gh_api.req("GET", "/rate_limit")
    core = rl["resources"]["core"]
    print(f"rate: remaining={core['remaining']} reset={core['reset']}")
except Exception as e:
    print("rate ERROR", e)

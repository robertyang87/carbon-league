import sys, json, base64
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

O, R = "robertyang87", "carbon-league"

def branch_head(b):
    d = gh_api.req("GET", f"/repos/{O}/{R}/branches/{b}")
    c = d["commit"]
    return c["sha"][:8], c["commit"]["committer"]["date"], c["commit"]["message"].split("\n")[0]

for b in ["sync/h23", "sync/h29", "sync/h31", "main"]:
    try:
        sha, dt, msg = branch_head(b)
        print(f"{b}: sha={sha} date={dt} msg={msg}")
    except Exception as e:
        print(f"{b}: ERROR {e}")

def list_dir(path, ref):
    d = gh_api.get_file(O, R, path, ref=ref)
    if isinstance(d, dict):
        return [d]
    return d

try:
    items = list_dir("synced/h23", "sync/h23")
    print(f"synced/h23 items: {len(items)}")
    for it in items:
        if it["name"].endswith("_cal.out") or it["name"] == "bigeval.jsonl":
            print(f"h23-item: {it['name']} size={it['size']} type={it['type']}")
except Exception as e:
    print(f"synced/h23 list ERROR {e}")

try:
    live = list_dir("synced/h23/bigeval_live", "sync/h23")
    print(f"bigeval_live items: {len(live)}")
except Exception as e:
    print(f"bigeval_live ERROR {e}")

# m3s2r1pre_cal.out content — confirm unchanged
try:
    f = gh_api.get_file(O, R, "synced/h23/m3s2r1pre_cal.out", ref="sync/h23")
    txt = base64.b64decode(f["content"]).decode(errors="replace")
    print("m3s2r1pre_cal.out content:")
    print(txt)
except Exception as e:
    print(f"cal.out read ERROR {e}")

try:
    rl = gh_api.req("GET", "/rate_limit")
    core = rl["resources"]["core"]
    import datetime
    print(f"rate: remaining={core['remaining']} reset={core['reset']} "
          f"({datetime.datetime.fromtimestamp(core['reset'], datetime.timezone(datetime.timedelta(hours=8))).strftime('%H:%M CST')})")
except Exception as e:
    print(f"rate ERROR {e}")

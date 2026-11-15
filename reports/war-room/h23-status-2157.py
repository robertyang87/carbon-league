import sys, json, base64, datetime
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

# --- h23 silence check ---
try:
    items = list_dir("synced/h23", "sync/h23")
    print(f"synced/h23 items: {len(items)}")
    for it in items:
        if it["name"].endswith("_cal.out"):
            print(f"h23-cal: {it['name']} size={it['size']}")
except Exception as e:
    print(f"synced/h23 list ERROR {e}")

try:
    live = list_dir("synced/h23/bigeval_live", "sync/h23")
    print(f"bigeval_live items: {len(live)}")
except Exception as e:
    print(f"bigeval_live ERROR {e}")

try:
    f = gh_api.get_file(O, R, "synced/h23/m3s2r1pre_cal.out", ref="sync/h23")
    txt = base64.b64decode(f["content"]).decode(errors="replace")
    print("m3s2r1pre_cal.out content:")
    print(txt)
except Exception as e:
    print(f"cal.out read ERROR {e}")

# --- h29 waux1 exam progress ---
try:
    live29 = list_dir("synced/h29/bigeval_live", "sync/h29")
    print(f"h29 bigeval_live items: {len(live29)}")
    for it in live29:
        if "waux1" in it["name"]:
            print(f"h29-live: {it['name']} size={it['size']}")
except Exception as e:
    print(f"h29 bigeval_live ERROR {e}")

for tag in ["m9xj570waux1_m2", "m9xj570waux1_echo2"]:
    try:
        f = gh_api.get_file(O, R, f"synced/h29/bigeval_live/{tag}.jsonl", ref="sync/h29")
        raw = base64.b64decode(f["content"]).decode(errors="replace")
        lines = [ln for ln in raw.splitlines() if ln.strip()]
        print(f"{tag}: lines={len(lines)}")
        if lines:
            import json as _j
            last = _j.loads(lines[-1])
            print(f"  last: seed={last.get('seed')} margin={last.get('margin')} win={last.get('win')}")
            first = _j.loads(lines[0])
            print(f"  first: seed={first.get('seed')}")
    except Exception as e:
        print(f"{tag} read ERROR {e}")

# waux1 summaries / cal.out on h29
try:
    sums = list_dir("synced/h29/summaries", "sync/h29")
    print(f"h29 summaries items: {len(sums)}")
    for it in sums:
        if "waux1" in it["name"]:
            print(f"h29-summary: {it['name']} size={it['size']}")
except Exception as e:
    print(f"h29 summaries ERROR {e}")

try:
    items29 = list_dir("synced/h29", "sync/h29")
    for it in items29:
        if it["name"].endswith("_cal.out"):
            print(f"h29-cal: {it['name']} size={it['size']}")
except Exception as e:
    print(f"synced/h29 list ERROR {e}")

try:
    rl = gh_api.req("GET", "/rate_limit")
    core = rl["resources"]["core"]
    print(f"rate: remaining={core['remaining']} reset={core['reset']} "
          f"({datetime.datetime.fromtimestamp(core['reset'], datetime.timezone(datetime.timedelta(hours=8))).strftime('%H:%M CST')})")
except Exception as e:
    print(f"rate ERROR {e}")

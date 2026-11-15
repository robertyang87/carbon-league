import sys, json, base64
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

O, R = "robertyang87", "carbon-league"

d = gh_api.req("GET", f"/repos/{O}/{R}/branches/sync/h29")
c = d["commit"]
print(f"sync/h29: sha={c['sha'][:8]} date={c['commit']['committer']['date']} msg={c['commit']['message'].split(chr(10))[0]}")

for tag in ["m9xj570waux1_m2", "m9xj570waux1_echo2"]:
    try:
        f = gh_api.get_file(O, R, f"synced/h29/bigeval_live/{tag}.jsonl", ref="sync/h29")
        raw = base64.b64decode(f["content"]).decode(errors="replace")
        lines = [ln for ln in raw.splitlines() if ln.strip()]
        seeds = sorted(json.loads(ln).get("seed") for ln in lines)
        print(f"{tag}: lines={len(lines)} seed_min={seeds[0] if seeds else None} seed_max={seeds[-1] if seeds else None}")
    except Exception as e:
        print(f"{tag} read ERROR {e}")

try:
    sums = gh_api.get_file(O, R, "synced/h29/summaries", ref="sync/h29")
    names = [it["name"] for it in sums if "waux1" in it["name"]]
    print(f"h29 waux1 summaries: {names}")
except Exception as e:
    print(f"summaries ERROR {e}")

import json, sys
sys.path.insert(0, "/home/hatch/workspace")
import gh_api
out = {}
try:
    rl = gh_api.req("GET", "/rate_limit")
    out["rate"] = rl["resources"]["core"]
except Exception as e:
    out["rate_error"] = str(e)
for b in ["main", "sync/h23", "sync/h24", "sync/h28", "sync/h29", "sync/h31"]:
    try:
        r = gh_api.req("GET", f"/repos/robertyang87/carbon-league/branches/{b}")
        c = r["commit"]["commit"]
        out[b] = {"sha": r["commit"]["sha"][:8], "date": c["committer"]["date"], "msg": c["message"][:60]}
    except Exception as e:
        out[b] = {"error": str(e)[:120]}
print(json.dumps(out, indent=1))

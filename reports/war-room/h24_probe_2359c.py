import sys, json
sys.path.insert(0, "/home/hatch/workspace")
import gh_api
OWNER, REPO = "robertyang87", "carbon-league"
res = {}
for branch, sha in [("sync/h24","2a4f2e3e"),("sync/h31","c8d3c1d7"),("sync/h29","12e8a4ee")]:
    try:
        t = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/git/trees/{sha}?recursive=1")
        paths = [e["path"] for e in t.get("tree", [])]
        live = [p for p in paths if "/bigeval_live/" in p]
        outs = [p for p in paths if p.startswith("synced/") and p.endswith(".out")]
        cals = [p for p in outs if p.endswith("_cal.out")]
        res[branch] = {"tree_truncated": t.get("truncated"), "total": len(paths),
                       "live_count": len(live), "out_count": len(outs), "cal_outs": cals}
    except Exception as e:
        res[branch] = {"error": str(e)[:300]}
# read m3s2r1_cal.out content to confirm unchanged
try:
    f = gh_api.get_file(OWNER, REPO, "synced/h24/m3s2r1_cal.out", ref="sync/h24")
    res["cal_content"] = f.get("content","")[:1200] if isinstance(f, dict) else str(f)[:300]
except Exception as e:
    res["cal_content"] = {"error": str(e)[:200]}
print(json.dumps(res, ensure_ascii=False, indent=1))

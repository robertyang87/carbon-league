import sys, json
sys.path.insert(0, "/home/hatch/workspace")
import gh_api
OWNER, REPO = "robertyang87", "carbon-league"
res = {}
for b in ["sync/h24", "sync/h31", "sync/h29", "main"]:
    try:
        d = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/branches/{b}")
        c = d["commit"]["commit"]
        res[b] = {"sha": d["commit"]["sha"][:8], "date": c["committer"]["date"], "msg": c["message"][:80]}
    except Exception as e:
        res[b] = {"error": str(e)[:200]}
print(json.dumps(res, ensure_ascii=False, indent=1))

import sys
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

O, R = "robertyang87", "carbon-league"
tree = gh_api.req("GET", f"/repos/{O}/{R}/git/trees/sync/h29?recursive=1")
paths = [t["path"] for t in tree["tree"]]
print("total entries:", len(paths), "truncated:", tree.get("truncated"))
for p in paths:
    if "bigeval_live" in p or "summaries" in p:
        print(p)

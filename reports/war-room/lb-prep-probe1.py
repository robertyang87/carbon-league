import sys, json
sys.path.insert(0, "/home/hatch/workspace")
from gh_api import req, get_file

OWNER, REPO, REF = "robertyang87", "carbon-league", "cnet2-search-v12"

r = req("GET", f"/repos/{OWNER}/{REPO}/git/trees/{REF}?recursive=1")
tree = r.get("tree", [])
print("total entries:", len(tree), "truncated:", r.get("truncated"))
keep = []
for e in tree:
    p = e["path"]
    low = p.lower()
    if any(k in low for k in ["cnet2", "search", "k4", "ladder_corpus", "build_top", "set_uw", "replica_gap", "exams", "status"]):
        keep.append((p, e.get("size")))
for p, s in sorted(keep):
    print(p, s)

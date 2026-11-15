import sys, os
sys.path.insert(0, "/home/hatch/workspace")
from gh_api import req, get_file

OWNER, REPO, REF = "robertyang87", "carbon-league", "cnet2-search-v12"
OUT = "/tmp/lbprep"
os.makedirs(OUT, exist_ok=True)

r = req("GET", f"/repos/{OWNER}/{REPO}/git/trees/{REF}?recursive=1")
for e in r.get("tree", []):
    if "m3s2" in e["path"].lower() or e["path"].startswith("data/ds") or "ds_m3s2" in e["path"]:
        print(e["path"], e.get("size"))

for p in ["reports/live/replica_gap_plan.md", "reports/live/k4_probe.md",
          "code/cnet2_search.py", "code/cnet2_train.py", "code/cnet2_build.py"]:
    try:
        data = get_file(OWNER, REPO, p, ref=REF)
        fn = os.path.join(OUT, p.replace("/", "_"))
        with open(fn, "wb") as f:
            f.write(data)
        print("FETCHED", p, len(data), "->", fn)
    except Exception as ex:
        print("FAIL", p, ex)

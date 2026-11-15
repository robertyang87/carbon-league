import sys, os, base64
sys.path.insert(0, "/home/hatch/workspace")
from gh_api import get_file

OWNER, REPO, REF = "robertyang87", "carbon-league", "cnet2-search-v12"
OUT = "/tmp/lbprep"
os.makedirs(OUT, exist_ok=True)

for p in ["reports/live/replica_gap_plan.md", "reports/live/k4_probe.md",
          "code/cnet2_search.py", "code/cnet2_train.py", "code/cnet2_build.py",
          "code/build_top.py", "code/set_uw.py"]:
    try:
        d = get_file(OWNER, REPO, p, ref=REF)
        raw = base64.b64decode(d["content"])
        fn = os.path.join(OUT, p.replace("/", "_"))
        with open(fn, "wb") as f:
            f.write(raw)
        print("FETCHED", p, len(raw), "sha", d.get("sha"))
    except Exception as ex:
        print("FAIL", p, repr(ex)[:200])

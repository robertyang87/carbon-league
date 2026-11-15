import sys, os, base64, shutil
sys.path.insert(0, "/home/hatch/workspace")
from gh_api import req, get_file

OWNER, REPO, REF = "robertyang87", "carbon-league", "cnet2-search-v12"
SB = "/tmp/lbtest"
CODE = os.path.join(SB, "code")
WDIR = os.path.join(SB, "data", "weights")
os.makedirs(CODE, exist_ok=True)
os.makedirs(WDIR, exist_ok=True)

# code files already fetched to /tmp/lbprep with flattened names
flat = {
    "code_cnet2_search.py": "cnet2_search.py",
    "code_cnet2_feat.py": "cnet2_feat.py",
    "code_cnet2_np.py": "cnet2_np.py",
    "code_cnet2_agent.py": "cnet2_agent.py",
    "code_fastsim.py": "fastsim.py",
    "code_eval_v53.py": "eval_v53.py",
}
for src, dst in flat.items():
    shutil.copy(os.path.join("/tmp/lbprep", src), os.path.join(CODE, dst))
    print("code", dst, os.path.getsize(os.path.join(CODE, dst)))

# env config json for eval_v53 CFG
p = "code/env_src/carbon_challenge-master/zerosum_env/envs/carbon/carbon.json"
d = get_file(OWNER, REPO, p, ref=REF)
raw = base64.b64decode(d["content"])
dst = os.path.join(CODE, "env_src/carbon_challenge-master/zerosum_env/envs/carbon/carbon.json")
os.makedirs(os.path.dirname(dst), exist_ok=True)
open(dst, "wb").write(raw)
print("json", len(raw))

# weights via blobs API
tree = req("GET", f"/repos/{OWNER}/{REPO}/git/trees/{REF}?recursive=1")["tree"]
sha = {e["path"]: e["sha"] for e in tree}
for p in ["data/weights/cnet2_b128_q8.npz", "data/weights/cnet2_s64.npz"]:
    b = req("GET", f"/repos/{OWNER}/{REPO}/git/blobs/{sha[p]}")
    raw = base64.b64decode(b["content"])
    out = os.path.join(WDIR, os.path.basename(p))
    open(out, "wb").write(raw)
    print("weights", p, len(raw))
print("CFG check via eval_v53 next")

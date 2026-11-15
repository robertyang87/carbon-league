import gh_api, base64, os
OUT = "/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/recon_code"
def get(path):
    d = gh_api.get_file("robertyang87", "carbon-league", path, ref="main")
    return base64.b64decode(d["content"]).decode("utf-8", "replace")
s = get("code/cnet2_feat.py")
open(os.path.join(OUT, "cnet2_feat.py"), "w").write(s)
print("cnet2_feat.py chars", len(s), flush=True)
for i, l in enumerate(s.splitlines()):
    if l.startswith(("class ", "def ", "    def ")) or "aux" in l and ("=" in l or "def" in l):
        print(f"F {i+1}: {l.strip()[:150]}", flush=True)

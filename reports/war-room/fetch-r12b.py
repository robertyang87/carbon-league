import gh_api, base64, os
OUT = "/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/recon_code"
def get(path):
    d = gh_api.get_file("robertyang87", "carbon-league", path, ref="main")
    return base64.b64decode(d["content"]).decode("utf-8", "replace")
r = gh_api.req("GET", "/repos/robertyang87/carbon-league/contents/reports/review_evidence/20261112?ref=main&per_page=100")
names = [x["name"] for x in r] if isinstance(r, list) else r
print("EVIDENCE 20261112:", names, flush=True)
s = get("code/cnet2_search.py")
open(os.path.join(OUT, "cnet2_search.py"), "w").write(s)
print("cnet2_search.py chars", len(s), flush=True)
for i, l in enumerate(s.splitlines()):
    ll = l.lower()
    if "tta" in ll or "cachepolicy" in ll or "guard" in ll:
        print(f"S {i+1}: {l.strip()[:160]}", flush=True)

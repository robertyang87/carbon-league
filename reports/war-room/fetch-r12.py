import gh_api, base64, os, re
OUT = "/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/recon_code"
def get(path):
    d = gh_api.get_file("robertyang87", "carbon-league", path, ref="main")
    return base64.b64decode(d["content"]).decode("utf-8", "replace")
try:
    r = gh_api.req("GET", "/repos/robertyang87/carbon-league/contents/reports/review_evidence?ref=main&per_page=100")
    print("EVIDENCE DIR:", [x["name"] for x in r] if isinstance(r, list) else r, flush=True)
except Exception as e:
    print("EVIDENCE DIR FAIL", type(e).__name__, e, flush=True)
try:
    rev = get("reports/REVIEW_20261112.md")
    open(os.path.join(OUT, "REVIEW_20261112.md"), "w").write(rev)
    print("REVIEW chars", len(rev), flush=True)
    idx = [m.start() for m in re.finditer(r"R12", rev)]
    print("R12 mentions:", len(idx), flush=True)
    for i in idx[:6]:
        print("---- R12 ctx ----", flush=True)
        print(rev[max(0, i - 200): i + 2500], flush=True)
except Exception as e:
    print("REVIEW FAIL", type(e).__name__, e, flush=True)
try:
    tta = get("code/cnet2_tta_eval.py")
    open(os.path.join(OUT, "cnet2_tta_eval.py"), "w").write(tta)
    print("=== cnet2_tta_eval.py (first 5000) ===", flush=True)
    print(tta[:5000], flush=True)
except Exception as e:
    print("TTA FAIL", type(e).__name__, e, flush=True)

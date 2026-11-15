import sys, json
sys.path.insert(0, "/home/hatch/workspace")
import gh_api
OWNER, REPO = "robertyang87", "carbon-league"
out = {}
try:
    cs = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/commits?sha=main&per_page=5")
    out["main"] = [{"sha": c["sha"][:8], "date": c["commit"]["committer"]["date"],
                    "msg": c["commit"]["message"].split("\n")[0][:120]} for c in cs]
except Exception as e:
    out["main_error"] = repr(e)
try:
    cmp = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/compare/2523235bab4859102d837eecb0fec0d4de00b4c4...6e740aaeed07a4f0b5996da3cde80587b1e2b0f5")
    out["compare"] = {"behind_by": cmp.get("behind_by"), "ahead_by": cmp.get("ahead_by"),
                      "total_commits": cmp.get("total_commits"),
                      "files": [f["filename"] for f in cmp.get("files", [])][:20]}
except Exception as e:
    out["compare_error"] = repr(e)
try:
    rl = gh_api.req("GET", "/rate_limit")
    out["rate_core_remaining"] = rl["resources"]["core"]["remaining"]
    out["rate_core_reset"] = rl["resources"]["core"]["reset"]
except Exception as e:
    out["rate_error"] = repr(e)
print(json.dumps(out, indent=1, ensure_ascii=False))

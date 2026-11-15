"""h24 status probe h24-1859: direct read of sync/h24 branch."""
import sys, json, base64
sys.path.insert(0, "/home/hatch/workspace")
from gh_api import req

OWNER, REPO, BR = "robertyang87", "carbon-league", "sync/h24"
out = {"probe": "h24-1859"}

# branch head
br = req("GET", f"/repos/{OWNER}/{REPO}/branches/{BR}")
c = br["commit"]["commit"]
out["branch_head_sha"] = br["commit"]["sha"]
out["branch_head_time_utc"] = c["committer"]["date"]
out["branch_head_msg"] = c["message"][:120]

# synced/h24 top-level listing
top = req("GET", f"/repos/{OWNER}/{REPO}/contents/synced/h24?ref={BR}")
out["synced_h24_count"] = len(top)
out["synced_h24_names_sample"] = [t["name"] for t in top[:10]]
out["cal_outs"] = [t["name"] for t in top if t["name"].endswith("_cal.out")]

# bigeval_live listing
live = req("GET", f"/repos/{OWNER}/{REPO}/contents/synced/h24/bigeval_live?ref={BR}")
out["bigeval_live_count"] = len(live)
out["bigeval_live_names_sample"] = [t["name"] for t in live[:10]]

# m3s2r1_cal.out full text
f = req("GET", f"/repos/{OWNER}/{REPO}/contents/synced/h24/m3s2r1_cal.out?ref={BR}")
out["m3s2r1_cal_sha"] = f["sha"]
out["m3s2r1_cal_size"] = f["size"]
out["m3s2r1_cal_text"] = base64.b64decode(f["content"]).decode("utf-8", errors="replace")

# rate limit
rl = req("GET", "/rate_limit")
out["api_core_remaining"] = rl["resources"]["core"]["remaining"]
out["api_core_reset_utc"] = rl["resources"]["core"]["reset"]

print(json.dumps(out, indent=2, ensure_ascii=False))

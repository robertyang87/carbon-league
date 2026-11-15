"""h24 probe: read sync/h24 branch head, top listing, bigeval_live count,
m3s2r1_cal.out full text, m3s2r1_echo2.jsonl summary. Tag h24-1530."""
import json, sys, base64
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OUT = "/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/h24_probe_1530.json"
OWNER, REPO, BR = "robertyang87", "carbon-league", "sync/h24"

res = {"tag": "h24-1530"}

# branch head
br = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/branches/{BR}")
res["branch_head_sha"] = br["commit"]["sha"]
res["branch_head_date"] = br["commit"]["commit"]["committer"]["date"]
res["branch_head_msg"] = br["commit"]["commit"]["message"]

# rate limit
rl = gh_api.req("GET", "/rate_limit")
res["rate_core_remaining"] = rl["resources"]["core"]["remaining"]
res["rate_core_limit"] = rl["resources"]["core"]["limit"]

# top listing synced/h24
top = gh_api.get_file(OWNER, REPO, "synced/h24", ref=BR)
res["top_count"] = len(top)
res["top_names"] = [e["name"] for e in top]
res["cal_outs"] = [e["name"] for e in top if e["name"].endswith("_cal.out")]

# bigeval_live listing (count only)
live = gh_api.get_file(OWNER, REPO, "synced/h24/bigeval_live", ref=BR)
res["live_count"] = len(live)
res["live_names_sample"] = [e["name"] for e in live][:5]

# cal.out full text
cal = gh_api.get_file(OWNER, REPO, "synced/h24/m3s2r1_cal.out", ref=BR)
res["cal_sha"] = cal["sha"]
res["cal_size"] = cal["size"]
res["cal_text"] = base64.b64decode(cal["content"]).decode("utf-8", errors="replace")

# echo2 live jsonl summary
ej = gh_api.get_file(OWNER, REPO, "synced/h24/bigeval_live/m3s2r1_echo2.jsonl", ref=BR)
res["echo2_sha"] = ej["sha"]
res["echo2_size"] = ej["size"]
lines = base64.b64decode(ej["content"]).decode("utf-8", errors="replace").strip().splitlines()
res["echo2_n"] = len(lines)
wins = 0
margins = []
last = None
for ln in lines:
    try:
        d = json.loads(ln)
    except Exception:
        continue
    last = d
    m = d.get("margin")
    if m is not None:
        margins.append(m)
        if m > 0:
            wins += 1
res["echo2_wins"] = wins
res["echo2_mean"] = sum(margins) / len(margins) if margins else None
res["echo2_last_seed"] = last.get("seed") if last else None
res["echo2_last_margin"] = last.get("margin") if last else None

with open(OUT, "w") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)
print(json.dumps({k: v for k, v in res.items() if k != "cal_text"}, ensure_ascii=False, indent=2))
print("---CAL---")
print(res["cal_text"])

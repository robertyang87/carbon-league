import sys, json, base64
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER, REPO, BR = "robertyang87", "carbon-league", "sync/h24"
out = {"tag": "h24-1545"}

# branch head
br = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/branches/{BR}")
head = br["commit"]
out["branch_head_sha"] = head["sha"]
out["branch_head_date"] = head["commit"]["committer"]["date"]
out["branch_head_msg"] = head["commit"]["message"]

# rate limit
rl = gh_api.req("GET", "/rate_limit")
out["rate_core_remaining"] = rl["resources"]["core"]["remaining"]
out["rate_core_limit"] = rl["resources"]["core"]["limit"]

def get_json(path):
    return gh_api.get_file(OWNER, REPO, path, ref=BR)

def get_text(path):
    j = get_json(path)
    if isinstance(j, dict) and j.get("encoding") == "base64":
        return base64.b64decode(j["content"]).decode("utf-8", errors="replace"), j.get("sha"), j.get("size")
    return None, None, None

top = get_json("synced/h24")
out["top_count"] = len(top)
out["top_names"] = sorted(e["name"] for e in top)
out["cal_outs"] = sorted(e["name"] for e in top if e["name"].endswith("_cal.out"))

live = get_json("synced/h24/bigeval_live")
out["live_count"] = len(live)
out["live_names_sample"] = sorted(e["name"] for e in live)[:5]

cal_text, cal_sha, cal_size = get_text("synced/h24/m3s2r1_cal.out")
out["cal_sha"] = cal_sha
out["cal_size"] = cal_size
out["cal_text"] = cal_text

echo_text, echo_sha, echo_size = get_text("synced/h24/bigeval_live/m3s2r1_echo2.jsonl")
out["echo2_sha"] = echo_sha
out["echo2_size"] = echo_size
if echo_text:
    lines = [l for l in echo_text.splitlines() if l.strip()]
    out["echo2_n"] = len(lines)
    wins = 0; margins = []
    last = None
    for l in lines:
        try:
            d = json.loads(l)
        except Exception:
            continue
        m = d.get("margin")
        if m is not None:
            margins.append(m)
            if m > 0:
                wins += 1
        last = d
    out["echo2_wins"] = wins
    out["echo2_mean"] = round(sum(margins)/len(margins), 3) if margins else None
    out["echo2_last_seed"] = last.get("seed") if last else None
    out["echo2_last_margin"] = last.get("margin") if last else None

p = "/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/h24_probe_1545.json"
with open(p, "w") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print(json.dumps({k: out[k] for k in ["branch_head_sha","branch_head_date","rate_core_remaining","top_count","live_count","cal_sha","cal_size","echo2_sha","echo2_size","echo2_n","echo2_wins","echo2_mean","echo2_last_seed","echo2_last_margin"]}, ensure_ascii=False))

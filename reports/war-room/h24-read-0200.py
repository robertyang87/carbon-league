"""h24 0200 fresh read v2: fault-tolerant per file."""
import json, sys, base64, urllib.error
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER, REPO, BRANCH = "robertyang87", "carbon-league", "sync/h24"
out = {}

def get(path):
    return gh_api.req("GET", path)

commits = get(f"/repos/{OWNER}/{REPO}/commits?sha={BRANCH}&per_page=3")
out["head"] = [{"sha": c["sha"][:7], "date": c["commit"]["committer"]["date"],
                "msg": c["commit"]["message"].splitlines()[0][:100]} for c in commits]
head_sha = commits[0]["sha"][:7] if commits else None

live_commits = get(f"/repos/{OWNER}/{REPO}/commits?sha={BRANCH}&path=synced/h24/bigeval_live&per_page=2")
out["live_dir_head"] = [{"sha": c["sha"][:7], "date": c["commit"]["committer"]["date"],
                         "msg": c["commit"]["message"].splitlines()[0][:100]} for c in live_commits]

def read_text(path):
    try:
        f = gh_api.get_file(OWNER, REPO, path, ref=BRANCH)
    except urllib.error.HTTPError as e:
        return f"HTTP {e.code}"
    except Exception as e:
        return f"ERROR {e}"
    if isinstance(f, dict) and "content" in f:
        return base64.b64decode(f["content"]).decode("utf-8", errors="replace")
    if isinstance(f, list):
        return f"DIR {[e.get('name') for e in f]}"
    return f"UNEXPECTED {str(f)[:200]}"

for name in ["k4v2r1_e0.jsonl", "k4v2g1_e0.jsonl"]:
    txt = read_text(f"synced/h24/bigeval_live/{name}")
    if not isinstance(txt, str) or txt.startswith(("HTTP", "ERROR", "UNEXPECTED", "DIR")):
        out[name] = {"error": txt}
    else:
        lines = [l for l in txt.splitlines() if l.strip()]
        last = None
        if lines:
            try:
                j = json.loads(lines[-1])
                last = {k: j.get(k) for k in ("tag", "seed0", "n", "wins", "mean", "t") if isinstance(j, dict) and k in j} or str(j)[:200]
            except Exception:
                last = lines[-1][:200]
        out[name] = {"lines": len(lines), "bytes": len(txt), "last": last}

for name in ["k4v2_chain.log", "k4v2r1_e0.launcher.log"]:
    txt = read_text(f"synced/h24/{name}")
    out[name] = txt[-3000:] if isinstance(txt, str) and len(txt) > 3000 else txt

out["head_moved_since_0015"] = (head_sha != "2a4f2e3")
if out["head_moved_since_0015"]:
    out["m3s2r1_cal.out"] = read_text("synced/h24/m3s2r1_cal.out")
else:
    out["m3s2r1_cal.out"] = "UNCHANGED since 00:15 read (h24-read-0015.json: m3s2r1_m2 wins 183 mean 1197 se 63 lo2se 1072; m3s2r1_e0 wins 146 mean 409 se 53 lo2se 303; CAL_ALL_DONE)"

print(json.dumps(out, indent=1, ensure_ascii=False))
with open("/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/h24-read-0200.json", "w") as f:
    json.dump(out, f, indent=1, ensure_ascii=False)

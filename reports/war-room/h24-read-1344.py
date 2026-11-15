"""H24 trip read 13:44 CST 2026-11-13.
Reads sync/h24 (per task body) AND sync/h31 (spot-31, current exam host):
latest commits, synced/<h>/bigeval_live listing + live-file line counts,
*_cal.out full text. Exam: k4v2r1 vs Echo2, seed 999000, 200 games, ruling (a)."""
import json, sys, base64
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER, REPO = "robertyang87", "carbon-league"

def get(path):
    return gh_api.req("GET", path)

def read_file_text(path, branch):
    f = gh_api.get_file(OWNER, REPO, path, ref=branch)
    if isinstance(f, dict) and "content" in f:
        return base64.b64decode(f["content"]).decode("utf-8", errors="replace")
    return None

def branch_read(branch):
    out = {"branch": branch}
    try:
        commits = get(f"/repos/{OWNER}/{REPO}/commits?sha={branch}&per_page=4")
        out["commits"] = [
            {"sha": c["sha"][:7], "date": c["commit"]["committer"]["date"],
             "msg": c["commit"]["message"].splitlines()[0][:140]}
            for c in commits
        ]
    except Exception as e:
        out["commits_error"] = str(e)
    try:
        listing = gh_api.get_file(OWNER, REPO, f"synced/{branch.split('/')[-1]}", ref=branch)
        out["files"] = [e["name"] for e in listing] if isinstance(listing, list) else listing
    except Exception as e:
        out["files_error"] = str(e)
    try:
        live = gh_api.get_file(OWNER, REPO, f"synced/{branch.split('/')[-1]}/bigeval_live", ref=branch)
        out["live_files"] = [e["name"] for e in live] if isinstance(live, list) else live
    except Exception as e:
        out["live_files_error"] = str(e)
    return out

out = {}
for br in ["sync/h24", "sync/h31", "sync/h30"]:
    out[br] = branch_read(br)

# live stats for any k4v2r1 echo2 live file on h31 (and h30 for reference)
def live_stats(path, branch):
    try:
        txt = read_file_text(path, branch)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
    if txt is None:
        return {"error": "unreadable/missing"}
    lines = [l for l in txt.splitlines() if l.strip()]
    wins = 0; margins = []; last = None; first = None
    for l in lines:
        try:
            j = json.loads(l)
        except Exception:
            continue
        if first is None:
            first = j
        last = j
        m = j.get("margin", j.get("score_diff"))
        if isinstance(m, (int, float)):
            margins.append(m)
            if m > 0:
                wins += 1
    import statistics
    return {"lines": len(lines), "wins": wins,
            "mean_margin": round(statistics.mean(margins), 1) if margins else None,
            "first": first, "last": last}

for br, names in [("sync/h31", ["k4v2r1_echo2.jsonl", "k4v2r1_e0.jsonl"]),
                  ("sync/h30", ["k4v2r1_echo2.jsonl"])]:
    tag = br.split("/")[-1]
    for n in names:
        out[f"{tag}:{n}"] = live_stats(f"synced/{tag}/bigeval_live/{n}", br)

# cal.out files on h31 and h24
for br, names in [("sync/h31", ["k4v2r1_echo2_cal.out", "k4v2r1_cal.out"]),
                  ("sync/h24", ["m3s2r1_cal.out", "m3s2r1_e0_cal.out"])]:
    tag = br.split("/")[-1]
    for n in names:
        try:
            txt = read_file_text(f"synced/{tag}/{n}", br)
            out[f"{tag}:{n}"] = txt if txt is not None else "MISSING"
        except Exception as e:
            out[f"{tag}:{n}"] = f"ERROR {e}"

print(json.dumps(out, indent=1, ensure_ascii=False))

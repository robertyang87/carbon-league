import sys, json, base64
sys.path.insert(0, "/home/hatch/workspace")
from gh_api import req

OUT = "/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/h24-read-1429.json"
res = {"trip": "h24-status-report 14:29 CST", "read_at_cst": "2026-11-13 14:29"}

try:
    rl = req("GET", "/rate_limit")
    res["rate_limit"] = {"core_remaining": rl["resources"]["core"]["remaining"],
                         "core_reset_cst": rl["resources"]["core"]["reset"]}
except Exception as e:
    res["rate_limit"] = {"error": str(e)}

def branch(name):
    try:
        b = req("GET", f"/repos/robertyang87/carbon-league/branches/{name}")
        c = b["commit"]["commit"]
        return {"sha": b["commit"]["sha"][:8], "date": c["committer"]["date"],
                "message": c["message"][:120]}
    except Exception as e:
        return {"error": str(e)}

res["sync_h31_branch"] = branch("sync/h31")
res["sync_h24_branch"] = branch("sync/h24")

try:
    listing = req("GET", "/repos/robertyang87/carbon-league/contents/synced/h31?ref=sync/h31")
    res["synced_h31_listing"] = [{"name": x["name"], "size": x["size"], "type": x["type"]}
                                 for x in listing] if isinstance(listing, list) else {"raw": str(listing)[:300]}
except Exception as e:
    res["synced_h31_listing"] = {"error": str(e)}

# verify echo2 live file: line count + last line + md5 via content
try:
    f = req("GET", "/repos/robertyang87/carbon-league/contents/synced/h31/k4v2r1_echo2.jsonl?ref=sync/h31")
    raw = base64.b64decode(f["content"])
    lines = raw.decode().strip().split("\n")
    import hashlib
    last = json.loads(lines[-1])
    res["k4v2r1_echo2_jsonl"] = {"size": f["size"], "sha_api": f.get("sha"),
        "md5": hashlib.md5(raw).hexdigest(), "lines": len(lines),
        "last_seed": last.get("seed"), "last_margin": last.get("margin"),
        "wins_recomputed": sum(1 for ln in lines if json.loads(ln).get("margin", 0) > 0)}
except Exception as e:
    res["k4v2r1_echo2_jsonl"] = {"error": str(e)}

try:
    f2 = req("GET", "/repos/robertyang87/carbon-league/contents/synced/h31/k4v2r1_m2.jsonl?ref=sync/h31")
    raw2 = base64.b64decode(f2["content"])
    lines2 = raw2.decode().strip().split("\n")
    import hashlib
    res["k4v2r1_m2_jsonl"] = {"size": f2["size"], "md5": hashlib.md5(raw2).hexdigest(),
        "lines": len(lines2),
        "wins_recomputed": sum(1 for ln in lines2 if json.loads(ln).get("margin", 0) > 0)}
except Exception as e:
    res["k4v2r1_m2_jsonl"] = {"error": str(e)}

with open(OUT, "w") as fh:
    json.dump(res, fh, indent=2, ensure_ascii=False)
print(json.dumps(res, indent=2, ensure_ascii=False))

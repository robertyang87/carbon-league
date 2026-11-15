#!/usr/bin/env python3
"""h23 status check 09:27 trip: sync/h23 + main heads, synced/h23 dir, cal.out contents,
sync/h29 + sync/h31 heads (it85 dual-field verdict watch), h29/h31 live + cal.out,
main recent commits."""
import sys, json, base64
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER, REPO = "robertyang87", "carbon-league"

def head_commit(branch):
    cs = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/commits?sha={branch}&per_page=1")
    c = cs[0]
    return {"sha": c["sha"][:8], "date": c["commit"]["committer"]["date"],
            "msg": c["commit"]["message"][:100]}

def get_text(path, ref):
    d = gh_api.get_file(OWNER, REPO, path, ref=ref)
    if isinstance(d, dict) and d.get("encoding") == "base64":
        return base64.b64decode(d["content"]).decode("utf-8", errors="replace")
    return None

out = {}
try:
    rl = gh_api.req("GET", "/rate_limit")
    out["rate_remaining"] = rl["resources"]["core"]["remaining"]
    out["rate_reset"] = rl["resources"]["core"]["reset"]
except Exception as e:
    out["rate_error"] = str(e)

for br in ("main", "sync/h23", "sync/h29", "sync/h31"):
    try:
        out[br] = head_commit(br)
    except Exception as e:
        out[br] = {"error": str(e)[:200]}

# h23 own branch
try:
    d = gh_api.get_file(OWNER, REPO, "synced/h23", ref="sync/h23")
    if isinstance(d, list):
        out["h23_dir_count"] = len(d)
        cals = [e["name"] for e in d if e["name"].endswith("_cal.out")]
        out["h23_cal_out"] = cals
        for c in cals:
            try:
                out[f"cal:{c}"] = get_text(f"synced/h23/{c}", "sync/h23")
            except Exception as e:
                out[f"cal:{c}"] = {"error": str(e)[:200]}
    else:
        out["h23_dir"] = {"error": str(d)[:200]}
except Exception as e:
    out["h23_dir"] = {"error": str(e)[:200]}

try:
    d = gh_api.get_file(OWNER, REPO, "synced/h23/bigeval_live", ref="sync/h23")
    out["h23_live_count"] = len(d) if isinstance(d, list) else {"error": str(d)[:200]}
except Exception as e:
    out["h23_live"] = {"error": str(e)[:200]}

# h29 (C1 Master2 field) and h31 (C2 Echo2 field): live listings + cal.out + it85 file sizes
for br, tag in (("sync/h29", "h29"), ("sync/h31", "h31")):
    try:
        d = gh_api.get_file(OWNER, REPO, f"synced/{tag}/bigeval_live", ref=br)
        if isinstance(d, list):
            out[f"{tag}_live_count"] = len(d)
            out[f"{tag}_live_it85"] = [
                {"name": e["name"], "size": e.get("size")}
                for e in d if "it85" in e["name"] or "m3s2r1" in e["name"]]
        else:
            out[f"{tag}_live"] = {"error": str(d)[:200]}
    except Exception as e:
        out[f"{tag}_live"] = {"error": str(e)[:200]}
    try:
        d = gh_api.get_file(OWNER, REPO, f"synced/{tag}", ref=br)
        if isinstance(d, list):
            cals = [e["name"] for e in d if e["name"].endswith("_cal.out")]
            out[f"{tag}_cal_out"] = cals
            for c in cals:
                try:
                    out[f"{tag}cal:{c}"] = get_text(f"synced/{tag}/{c}", br)
                except Exception as e:
                    out[f"{tag}cal:{c}"] = {"error": str(e)[:200]}
            sums = [e["name"] for e in d if e["name"] == "summaries"]
            out[f"{tag}_has_summaries_dir"] = bool(sums)
        else:
            out[f"{tag}_dir"] = {"error": str(d)[:200]}
    except Exception as e:
        out[f"{tag}_dir"] = {"error": str(e)[:200]}

# summaries dir listing for h29/h31 (it85 summaries landed per C1/C2 receipts)
for br, tag in (("sync/h29", "h29"), ("sync/h31", "h31")):
    try:
        d = gh_api.get_file(OWNER, REPO, f"synced/{tag}/summaries", ref=br)
        if isinstance(d, list):
            out[f"{tag}_summaries"] = sorted(
                e["name"] for e in d if "it85" in e["name"])
        else:
            out[f"{tag}_summaries"] = {"error": str(d)[:150]}
    except Exception as e:
        out[f"{tag}_summaries"] = {"error": str(e)[:150]}

# recent main commits
try:
    cs = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/commits?sha=main&per_page=6")
    out["main_recent"] = [{"sha": c["sha"][:8], "date": c["commit"]["committer"]["date"],
                           "msg": c["commit"]["message"][:90]} for c in cs]
except Exception as e:
    out["main_recent"] = {"error": str(e)[:200]}

print(json.dumps(out, indent=1, ensure_ascii=False))

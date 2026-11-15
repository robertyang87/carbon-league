#!/usr/bin/env python3
"""h23 status check 04:42 CST 2026-11-15: sync/h23 head, synced/h23 dir + cal.out
full text, bigeval_live listing + last-5 tails, subdir counts. Minimal API calls."""
import sys, json, base64, hashlib
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER, REPO = "robertyang87", "carbon-league"

def get_text(path, ref):
    d = gh_api.get_file(OWNER, REPO, path, ref=ref)
    if isinstance(d, dict) and d.get("encoding") == "base64":
        return base64.b64decode(d["content"]).decode("utf-8", errors="replace")
    return None

out = {"trip": "2026-11-15 04:42 CST", "mark": "H23STATUS0442A"}
try:
    rl = gh_api.req("GET", "/rate_limit")
    out["rate_remaining"] = rl["resources"]["core"]["remaining"]
    out["rate_reset"] = rl["resources"]["core"]["reset"]
except Exception as e:
    out["rate_error"] = str(e)

try:
    cs = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/commits?sha=sync/h23&per_page=3")
    out["h23_head"] = [{"sha": c["sha"][:8], "date": c["commit"]["committer"]["date"],
                        "msg": c["commit"]["message"][:100]} for c in cs]
except Exception as e:
    out["h23_head"] = {"error": str(e)}

try:
    d = gh_api.get_file(OWNER, REPO, "synced/h23", ref="sync/h23")
    names = [e["name"] for e in d] if isinstance(d, list) else []
    out["top_level_items"] = len(names)
    out["top_level_names_md5"] = hashlib.md5("\n".join(sorted(names)).encode()).hexdigest()
    out["top_level_names"] = sorted(names)
    cals = [n for n in names if n.endswith("_cal.out")]
    out["cal_out_files"] = cals
    out["cal"] = {}
    for c in cals:
        try:
            out["cal"][c] = get_text(f"synced/h23/{c}", "sync/h23")
        except Exception as e:
            out["cal"][c] = {"error": str(e)[:200]}
except Exception as e:
    out["top_level"] = {"error": str(e)}

try:
    d = gh_api.get_file(OWNER, REPO, "synced/h23/bigeval_live", ref="sync/h23")
    names = [e["name"] for e in d] if isinstance(d, list) else []
    out["live_items"] = len(names)
    out["live_names_md5"] = hashlib.md5("\n".join(sorted(names)).encode()).hexdigest()
    out["live_last5"] = sorted(names)[-5:]
    out["live_tails"] = {}
    for f in sorted(names)[-5:]:
        try:
            txt = get_text(f"synced/h23/bigeval_live/{f}", "sync/h23")
            lines = [l for l in txt.splitlines() if l.strip()] if txt else []
            lastinfo = None
            if lines:
                try:
                    lj = json.loads(lines[-1])
                    lastinfo = {k: lj.get(k) for k in ("seed", "ts", "time", "margin", "winner", "n", "game") if k in lj}
                except Exception:
                    lastinfo = lines[-1][:150]
            out["live_tails"][f] = {"lines": len(lines), "last": lastinfo}
        except Exception as e:
            out["live_tails"][f] = {"error": str(e)[:200]}
except Exception as e:
    out["live"] = {"error": str(e)}

for sub in ("launch", "weights", "packs"):
    try:
        d = gh_api.get_file(OWNER, REPO, f"synced/h23/{sub}", ref="sync/h23")
        out[f"subdir_{sub}"] = len(d) if isinstance(d, list) else {"error": d}
    except Exception as e:
        out[f"subdir_{sub}"] = {"error": str(e)[:200]}

print(json.dumps(out, indent=1, ensure_ascii=False))

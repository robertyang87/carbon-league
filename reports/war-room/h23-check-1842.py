#!/usr/bin/env python3
"""h23 status check 18:42: sync/h23 head, synced/h23 dir + cal.out full text,
bigeval_live line counts + last-line time. Minimal API calls."""
import sys, json, base64
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER, REPO = "robertyang87", "carbon-league"

def get_text(path, ref):
    d = gh_api.get_file(OWNER, REPO, path, ref=ref)
    if isinstance(d, dict) and d.get("encoding") == "base64":
        return base64.b64decode(d["content"]).decode("utf-8", errors="replace")
    return None

out = {}
try:
    rl = gh_api.req("GET", "/rate_limit")
    out["rate_remaining"] = rl["resources"]["core"]["remaining"]
    out["rate_reset_utc"] = rl["resources"]["core"]["reset"]
except Exception as e:
    out["rate_error"] = str(e)

try:
    cs = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/commits?sha=sync/h23&per_page=3")
    out["h23_head"] = [{"sha": c["sha"][:8], "date": c["commit"]["committer"]["date"],
                        "msg": c["commit"]["message"][:100]} for c in cs]
except Exception as e:
    out["h23_head"] = {"error": str(e)}


try:
    for b in ("main", "sync/h29", "sync/h31"):
        cs = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/commits?sha={b}&per_page=1")
        c = cs[0]
        out[f"head:{b}"] = {"sha": c["sha"][:8], "date": c["commit"]["committer"]["date"],
                            "msg": c["commit"]["message"][:100]}
except Exception as e:
    out["head:others"] = {"error": str(e)}

try:
    d = gh_api.get_file(OWNER, REPO, "synced/h23", ref="sync/h23")
    names = [e["name"] for e in d] if isinstance(d, list) else {"error": d}
    out["h23_dir"] = names
    cals = [n for n in names if isinstance(n, str) and n.endswith("_cal.out")] if isinstance(names, list) else []
    out["h23_cal_out"] = cals
    for c in cals:
        try:
            out[f"cal:{c}"] = get_text(f"synced/h23/{c}", "sync/h23")
        except Exception as e:
            out[f"cal:{c}"] = {"error": str(e)[:200]}
except Exception as e:
    out["h23_dir"] = {"error": str(e)}

try:
    d = gh_api.get_file(OWNER, REPO, "synced/h23/bigeval_live", ref="sync/h23")
    names = [e["name"] for e in d] if isinstance(d, list) else {"error": d}
    out["h23_live_dir"] = names
    total = 0
    per = {}
    if isinstance(names, list):
        for f in names:
            try:
                txt = get_text(f"synced/h23/bigeval_live/{f}", "sync/h23")
                if txt is None:
                    per[f] = {"error": "not-base64-or-missing"}
                    continue
                lines = [l for l in txt.splitlines() if l.strip()]
                total += len(lines)
                last = lines[-1] if lines else ""
                try:
                    lj = json.loads(last)
                    lastinfo = {k: lj.get(k) for k in ("seed", "ts", "time", "margin", "winner", "n", "game") if k in lj}
                except Exception:
                    lastinfo = last[:150]
                per[f] = {"lines": len(lines), "last": lastinfo}
            except Exception as e:
                per[f] = {"error": str(e)[:200]}
    out["live_per_file"] = per
    out["live_total"] = total
except Exception as e:
    out["h23_live_dir"] = {"error": str(e)}

print(json.dumps(out, indent=1, ensure_ascii=False))

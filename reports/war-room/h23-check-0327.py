#!/usr/bin/env python3
"""h23 status check: sync/h23 + sync/h29 branch heads, live file line counts, cal.out scan."""
import sys, json, base64
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER, REPO = "robertyang87", "carbon-league"

def commits(branch, n=6):
    return gh_api.req("GET", f"/repos/{OWNER}/{REPO}/commits?sha={branch}&per_page={n}")

def listdir(path, ref):
    return gh_api.get_file(OWNER, REPO, path, ref=ref)

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

for br in ("sync/h23", "sync/h29"):
    try:
        cs = commits(br, 6)
        out[br] = [{"sha": c["sha"][:8], "date": c["commit"]["committer"]["date"],
                    "msg": c["commit"]["message"][:80]} for c in cs]
    except Exception as e:
        out[br] = {"error": str(e)}

# live files under sync/h29
try:
    d = listdir("synced/h29/bigeval_live", "sync/h29")
    names = [e["name"] for e in d] if isinstance(d, list) else {"error": d}
    out["h29_live_dir"] = names
except Exception as e:
    out["h29_live_dir"] = {"error": str(e)}

live_files = ["fabler1_m2.jsonl", "fabler1_echo2.jsonl", "m7bsv2g1_echo2.jsonl",
              "m7bsv2r1_m2.jsonl", "m7bsv2g1_m2.jsonl", "m7bsv2r1_echo2.jsonl",
              "fableg1_m2.jsonl", "fableg1_echo2.jsonl"]
total = 0
per = {}
for f in live_files:
    try:
        txt = get_text(f"synced/h29/bigeval_live/{f}", "sync/h29")
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
            lastinfo = last[:120]
        per[f] = {"lines": len(lines), "last": lastinfo}
    except Exception as e:
        per[f] = {"error": str(e)[:200]}
out["live_per_file"] = per
out["live_total"] = total

# cal.out scan on sync/h23
try:
    d = listdir("synced/h23", "sync/h23")
    names = [e["name"] for e in d] if isinstance(d, list) else {"error": d}
    out["h23_dir"] = names
    cals = [n for n in names if isinstance(n, str) and n.endswith("_cal.out")] if isinstance(names, list) else []
    out["h23_cal_out"] = cals
    for c in cals[:4]:
        try:
            out[f"cal:{c}"] = get_text(f"synced/h23/{c}", "sync/h23")
        except Exception as e:
            out[f"cal:{c}"] = {"error": str(e)[:200]}
except Exception as e:
    out["h23_dir"] = {"error": str(e)}

# also check sync/h29 dir for cal.out
try:
    d = listdir("synced/h29", "sync/h29")
    names = [e["name"] for e in d] if isinstance(d, list) else {"error": d}
    out["h29_dir"] = names
except Exception as e:
    out["h29_dir"] = {"error": str(e)}

print(json.dumps(out, indent=1, ensure_ascii=False))

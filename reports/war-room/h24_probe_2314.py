#!/usr/bin/env python3
"""h24 probe 23:14 CST: sync/h24 branch head, dir counts, cal.out full text,
live file sizes/shas, rate limit. Marker: h24-2314."""
import sys, json, base64
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER, REPO, BR = "robertyang87", "carbon-league", "sync/h24"

def show(label, obj):
    print(f"== {label} ==")
    print(json.dumps(obj, indent=1, ensure_ascii=False)[:4000])

try:
    br = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/branches/{BR}")
    print("BRANCH_HEAD_SHA:", br["commit"]["sha"])
    print("BRANCH_HEAD_DATE:", br["commit"]["commit"]["committer"]["date"])
    print("BRANCH_HEAD_MSG:", br["commit"]["commit"]["message"][:120])
except Exception as e:
    print("BRANCH_ERROR:", repr(e))

try:
    rl = gh_api.req("GET", "/rate_limit")
    print("RATE_CORE:", rl["resources"]["core"]["remaining"], "/", rl["resources"]["core"]["limit"],
          "reset:", rl["resources"]["core"]["reset"])
except Exception as e:
    print("RATE_ERROR:", repr(e))

try:
    d = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/contents/synced/h24?ref={BR}")
    print("SYNCED_H24_COUNT:", len(d))
    cal = [x for x in d if x["name"].endswith("_cal.out")]
    print("CAL_FILES:", [(x["name"], x["size"], x["sha"]) for x in cal])
except Exception as e:
    print("SYNCED_H24_ERROR:", repr(e))

try:
    live = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/contents/synced/h24/bigeval_live?ref={BR}")
    print("LIVE_COUNT:", len(live))
    for name in ("m3s2r1_m2.jsonl", "m3s2r1_e0.jsonl", "m3s2r1_echo2.jsonl"):
        m = [x for x in live if x["name"] == name]
        print("LIVE_FILE:", name, [(x["size"], x["sha"]) for x in m])
except Exception as e:
    print("LIVE_ERROR:", repr(e))

try:
    f = gh_api.get_file(OWNER, REPO, "synced/h24/m3s2r1_cal.out", ref=BR)
    print("CAL_OUT_SHA:", f.get("sha"), "SIZE:", f.get("size"))
    txt = base64.b64decode(f["content"]).decode("utf-8", errors="replace")
    print("CAL_OUT_TEXT_START")
    print(txt)
    print("CAL_OUT_TEXT_END")
except Exception as e:
    print("CAL_OUT_ERROR:", repr(e))

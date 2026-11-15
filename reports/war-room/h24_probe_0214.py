#!/usr/bin/env python3
"""h24 probe 02:14 CST: sync/h24 branch head, dir counts, cal.out full text,
newest live file lines/last line. Marker: h24-0214."""
import sys, json, base64
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER, REPO, BR = "robertyang87", "carbon-league", "sync/h24"

try:
    br = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/branches/{BR}")
    print("BRANCH_HEAD_SHA:", br["commit"]["sha"])
    print("BRANCH_HEAD_DATE:", br["commit"]["commit"]["committer"]["date"])
    print("BRANCH_HEAD_MSG:", br["commit"]["commit"]["message"][:120])
except Exception as e:
    print("BRANCH_ERROR:", repr(e))

try:
    d = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/contents/synced/h24?ref={BR}")
    print("SYNCED_H24_COUNT:", len(d))
    cal = [x for x in d if x["name"].endswith("_cal.out")]
    print("CAL_FILES:", [(x["name"], x["size"], x["sha"]) for x in cal])
    print("ROOT_NAMES_SAMPLE:", sorted(x["name"] for x in d)[:12])
except Exception as e:
    print("SYNCED_H24_ERROR:", repr(e))

try:
    live = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/contents/synced/h24/bigeval_live?ref={BR}")
    print("LIVE_COUNT:", len(live))
    names = sorted(x["name"] for x in live)
    print("LIVE_NAMES_TAIL:", names[-8:])
    newest = names[-1]
    print("NEWEST_LIVE_FILE:", newest)
except Exception as e:
    print("LIVE_ERROR:", repr(e))
    newest = None

if newest:
    try:
        f = gh_api.get_file(OWNER, REPO, f"synced/h24/bigeval_live/{newest}", ref=BR)
        txt = base64.b64decode(f["content"]).decode("utf-8", errors="replace")
        lines = [l for l in txt.splitlines() if l.strip()]
        print("NEWEST_LIVE_SIZE:", f.get("size"), "SHA:", f.get("sha"))
        print("NEWEST_LIVE_LINES:", len(lines))
        if lines:
            print("NEWEST_LIVE_FIRST:", lines[0][:200])
            print("NEWEST_LIVE_LAST:", lines[-1][:400])
    except Exception as e:
        print("NEWEST_LIVE_ERROR:", repr(e))

try:
    f = gh_api.get_file(OWNER, REPO, "synced/h24/m3s2r1_cal.out", ref=BR)
    print("CAL_OUT_SHA:", f.get("sha"), "SIZE:", f.get("size"))
    txt = base64.b64decode(f["content"]).decode("utf-8", errors="replace")
    print("CAL_OUT_TEXT_START")
    print(txt)
    print("CAL_OUT_TEXT_END")
except Exception as e:
    print("CAL_OUT_ERROR:", repr(e))

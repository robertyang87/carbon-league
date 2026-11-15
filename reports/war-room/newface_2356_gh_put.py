#!/usr/bin/env python3
"""Newface watch 23:56 run — put changed matches lists to main.
For each label: get_file (full sha + content) -> verify local 2356 pull is a
pure-append superset of repo content (missing ids must be 0) -> put_file ->
get_file read-back count. Prints one JSON line per label."""
import base64, json, sys
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER, REPO = "robertyang87", "carbon-league"
JOBS = [
    ("krypton", "/home/hatch/workspace/carbon-live/pull_krypton_2356.json"),
    ("lite-v8", "/home/hatch/workspace/carbon-live/pull_lite-v8_2356.json"),
    ("hakimi3pro", "/home/hatch/workspace/carbon-live/pull_hakimi3pro_2356.json"),
    ("hakimi31", "/home/hatch/workspace/carbon-live/pull_hakimi31_2356.json"),
    ("m3s3z", "/home/hatch/workspace/carbon-live/pull_m3s3z_2356.json"),
    ("fable", "/home/hatch/workspace/carbon-live/pull_fable_2356.json"),
]
for lab, local in JOBS:
    try:
        cur = gh_api.get_file(OWNER, REPO, f"data/{lab}_matches.json", ref="main")
        repo_arr = json.loads(base64.b64decode(cur["content"]))
        repo_ids = {m.get("id") for m in repo_arr}
        new_arr = json.load(open(local))
        new_ids = {m.get("id") for m in new_arr}
        missing = repo_ids - new_ids
        if missing:
            print(json.dumps({"label": lab, "status": "SKIP-missing-in-new",
                              "repo": len(repo_arr), "new": len(new_arr),
                              "missing": len(missing)}, ensure_ascii=False))
            continue
        r = gh_api.put_file(OWNER, REPO, f"data/{lab}_matches.json",
                            json.dumps(new_arr, ensure_ascii=False).encode(),
                            f"newface-watch 2356: {lab} matches {len(repo_arr)}->{len(new_arr)}",
                            sha=cur["sha"], branch="main")
        back = gh_api.get_file(OWNER, REPO, f"data/{lab}_matches.json", ref="main")
        back_arr = json.loads(base64.b64decode(back["content"]))
        print(json.dumps({"label": lab, "status": "ok" if len(back_arr) == len(new_arr) else "MISMATCH",
                          "repo_before": len(repo_arr), "put": len(new_arr),
                          "readback": len(back_arr), "sha": back["sha"][:8],
                          "commit": (r.get("commit", {}) or {}).get("sha", "")[:8]},
                         ensure_ascii=False))
    except Exception as e:
        print(json.dumps({"label": lab, "status": "ERROR", "error": str(e)[:300]}, ensure_ascii=False))

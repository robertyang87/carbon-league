import base64
import hashlib
import json
import os
import sys

sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER = "robertyang87"
REPO = "carbon-league"
SNAP = "/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/exam_snapshots/h29"
results = {}
for tag in ["it95_smoke", "it95base_smoke"]:
    local = os.path.join(SNAP, f"{tag}.jsonl")
    raw = open(local, "rb").read()
    path = f"synced/h29/bigeval_live/{tag}.jsonl"
    sha = None
    try:
        cur = gh_api.get_file(OWNER, REPO, path, ref="sync/h29")
        sha = cur["sha"]
    except Exception:
        sha = None
    r = gh_api.put_file(OWNER, REPO, path, raw,
                        f"snapshot {tag} smoke 2 lines (it95 snaploop aliveness check)",
                        sha=sha, branch="sync/h29")
    back = gh_api.get_file(OWNER, REPO, path, ref="sync/h29")
    back_raw = base64.b64decode(back["content"]) if back.get("content") else b""
    results[tag] = {
        "lines": raw.count(b"\n"),
        "bytes": len(raw),
        "md5": hashlib.md5(raw).hexdigest(),
        "commit": r["commit"]["sha"],
        "readback_md5": hashlib.md5(back_raw).hexdigest(),
        "readback_match": back_raw == raw,
    }
    print(json.dumps({tag: results[tag]}, indent=2))
with open(os.path.join(SNAP, "it95-smoke-push.json"), "w") as f:
    json.dump(results, f, indent=2)

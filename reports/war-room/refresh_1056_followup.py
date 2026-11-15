#!/usr/bin/env python3
"""newface-watch 10:56 follow-up: the 7 labels whose sid could not be
resolved from repo files (compact format has no strategy names).
Sids recovered from match_refresh_1656.py / refresh_2056.py.
hakimi31 / m3s3 / m9xj are VERIFY-ONLY (never put). Name guard enforced."""
import base64, json, os, subprocess, sys
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER, REPO = "robertyang87", "carbon-league"
TARGETS = [
    ("fable", "2b4b44fd148fa7fdaa53bb0d110ae705", "Fable5.2-Max-V5", False),
    ("m9xj", "e43b20c5d8d644b4145cb1e7ff198201", "RyanX-AIPowercore-M9XJ-it260", True),
    ("hakimi31", "29844f8f0c9b14b3d37902a551d26a1e", "哈基米 v3.1", True),
    ("m3s3", "f47e13ba624a85c5c2dc1b40caecc193", "claude-AlphaCarbon-Master3s3", True),
    ("krypton-search256", "2c57e7cd01dee46bef98643398ab4ca3", "Krypton-Search256-v1", False),
    ("krypton-prosearch", "fede68b79a2965889eeae0e226601139", "Krypton-ProSearch-v1", False),
    ("krypton-search256-v2", "484814899c333f8149e984eacdaf5275", "Krypton-Search256-v2", False),
]
TMP = "/tmp/m1056"
env = dict(os.environ, LADDER_BASE="https://117.72.210.134")
out = {}
for label, sid, name, verify_only in TARGETS:
    row = {"name": name, "verify_only": verify_only}
    try:
        r = gh_api.get_file(OWNER, REPO, f"data/{label}_matches.json")
        repo = json.loads(base64.b64decode(r["content"]))
        sha = r.get("sha")
    except Exception as e:
        row["error"] = f"get_file: {e}"
        out[label] = row
        continue
    row["repo_count"] = len(repo)
    row["repo_sha"] = (sha or "")[:12]
    f = f"{TMP}/{label}.json"
    p = subprocess.run(["python3", "ladder.py", "matches", sid, f],
                       cwd="/home/hatch/workspace/carbon-live",
                       env=env, capture_output=True, text=True, timeout=300)
    if p.returncode != 0 or not os.path.exists(f):
        row["error"] = f"117 pull failed rc={p.returncode}: {p.stderr[-300:]}"
        out[label] = row
        continue
    live = json.load(open(f))
    bad = [m for m in live if name not in (m.get("name_a"), m.get("name_b"))]
    row["live_count"] = len(live)
    row["name_guard_bad"] = len(bad)
    if bad:
        row["error"] = "NAME GUARD FAIL, skipped"
        out[label] = row
        continue
    wins = sum(1 for m in live if (m.get("winner") == "a" and m.get("name_a") == name) or
               (m.get("winner") == "b" and m.get("name_b") == name))
    row["live_wins"] = wins
    old_ids = {(m.get("id"), m.get("game")) for m in repo}
    new_ids = {(m.get("id"), m.get("game")) for m in live}
    row["added"] = len(new_ids - old_ids)
    row["removed"] = len(old_ids - new_ids)
    if verify_only:
        row["note"] = "verify-only, no put"
    elif old_ids != new_ids:
        try:
            put = gh_api.put_file(OWNER, REPO, f"data/{label}_matches.json",
                                  json.dumps(live, indent=1, ensure_ascii=False).encode(),
                                  f"newface-watch 1056: {label} {len(repo)}->{len(live)}",
                                  sha=sha, branch="main")
            row["commit"] = (put.get("commit") or {}).get("sha", "")[:12]
        except Exception as e:
            row["error"] = f"put_file failed: {e}"
            out[label] = row
            continue
        r2 = gh_api.get_file(OWNER, REPO, f"data/{label}_matches.json")
        row["readback_count"] = len(json.loads(base64.b64decode(r2["content"])))
        row["readback_sha"] = (r2.get("sha") or "")[:12]
    out[label] = row
print(json.dumps(out, ensure_ascii=False, indent=1))

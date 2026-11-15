#!/usr/bin/env python3
"""h24 status probe, marker h24-0129. Batched GitHub API reads only."""
import sys, json, base64
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER, REPO, BR = "robertyang87", "carbon-league", "sync/h24"
out = {"marker": "h24-0129"}

# 1. branch head
br = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/branches/{BR}")
out["head_sha"] = br["commit"]["sha"][:8]
out["head_commit_date"] = br["commit"]["commit"]["committer"]["date"]
out["head_msg"] = br["commit"]["commit"]["message"][:120]

# 2. synced/h24 listing
root = gh_api.get_file(OWNER, REPO, "synced/h24", ref=BR)
if isinstance(root, dict):
    root = [root]
out["synced_h24_count"] = len(root)
cal = [e for e in root if e["name"].endswith("_cal.out")]
out["cal_out_files"] = [(e["name"], e["size"]) for e in cal]
out["root_names_sample"] = [e["name"] for e in root[:10]]

# 3. bigeval_live listing
live_dir = gh_api.get_file(OWNER, REPO, "synced/h24/bigeval_live", ref=BR)
if isinstance(live_dir, dict):
    live_dir = [live_dir]
out["bigeval_live_count"] = len(live_dir)
names = sorted(e["name"] for e in live_dir)
out["live_names_tail"] = names[-8:]
out["live_names_head"] = names[:5]

# 4. fetch the newest live file by name (names sort chronologically by convention)
if names:
    newest = names[-1]
    f = gh_api.get_file(OWNER, REPO, f"synced/h24/bigeval_live/{newest}", ref=BR)
    raw = base64.b64decode(f["content"]).decode("utf-8", errors="replace")
    lines = [l for l in raw.splitlines() if l.strip()]
    out["newest_live_file"] = newest
    out["newest_live_size"] = f["size"]
    out["newest_live_lines"] = len(lines)
    out["newest_live_last_line"] = lines[-1][:300] if lines else None
    out["newest_live_first_line"] = lines[0][:200] if lines else None

# 5. fetch cal.out files (full text)
for name, size in out["cal_out_files"]:
    f = gh_api.get_file(OWNER, REPO, f"synced/h24/{name}", ref=BR)
    raw = base64.b64decode(f["content"]).decode("utf-8", errors="replace")
    out[f"cal_out_{name}"] = raw

print(json.dumps(out, ensure_ascii=False, indent=1))

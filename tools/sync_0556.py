#!/usr/bin/env python3
"""newface-watch 05:56 data-insurance sync: compare 117-end match lists with
robertyang87/carbon-league main data/<label>_matches.json by id set; update
changed files (except frozen m9xj570) and read back to verify counts."""
import base64, json, sys
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

LABELS = ["m3s3z", "hakimi3pro", "k4v3", "fable6", "lite-v8", "krypton", "fable"]
FROZEN = {"m9xj570": "9d69b8472d70"}

for lbl in LABELS + list(FROZEN):
    path = f"data/{lbl}_matches.json"
    try:
        cur = gh_api.get_file("robertyang87", "carbon-league", path, ref="main")
    except Exception as e:
        print(f"{lbl}: GET FAILED {e}")
        continue
    repo = json.loads(base64.b64decode(cur["content"]).decode())
    repo_ids = {m["id"] for m in repo}
    local = json.load(open(f"/tmp/nf0556/{lbl}_117.json"))
    local_ids = {m["id"] for m in local}
    new = local_ids - repo_ids
    gone = repo_ids - local_ids
    print(f"{lbl}: repo={len(repo)} local117={len(local)} new={len(new)} gone={len(gone)} repo_sha={cur['sha'][:12]}")
    if lbl in FROZEN:
        want = FROZEN[lbl]
        print(f"  FROZEN check: sha startswith {want[:12]}? {cur['sha'].startswith(want[:12])} (full {cur['sha'][:12]}) — NOT updated")
        continue
    if not new and not gone:
        print("  unchanged, skip")
        continue
    content = json.dumps(local, ensure_ascii=False).encode()
    r = gh_api.put_file("robertyang87", "carbon-league", path, content,
                        f"newface-watch 0556: sync {lbl} matches {len(repo)}->{len(local)}",
                        sha=cur["sha"], branch="main")
    print(f"  PUT -> commit {r.get('commit', {}).get('sha', '?')[:12]}")
    back = gh_api.get_file("robertyang87", "carbon-league", path, ref="main")
    back_n = len(json.loads(base64.b64decode(back["content"]).decode()))
    print(f"  READBACK: {back_n} entries, sha {back['sha'][:12]}")

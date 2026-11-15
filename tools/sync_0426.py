#!/usr/bin/env python3
"""Sync 117-end match lists to robertyang87/carbon-league main (data/<label>_matches.json).
Compares by match-id SET (117 order is unstable); only put_file when changed; get_file read-back to verify.
m9xj570 is FROZEN at 89 — verify sha only, never overwrite."""
import base64, json, sys
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

BASE = "/home/hatch/workspace/carbon-live"
JOBS = [
    ("krypton", "pull_krypton_0426.json", "data/krypton_matches.json"),
    ("lite-v8", "pull_lite-v8_0426.json", "data/lite-v8_matches.json"),
    ("m3s3z", "pull_m3s3z_0426.json", "data/m3s3z_matches.json"),
    ("hakimi3pro", "pull_hakimi3pro_0426.json", "data/hakimi3pro_matches.json"),
    ("k4v3", "pull_k4v3_0426.json", "data/k4v3_matches.json"),
    ("fable6", "pull_fable6_0426.json", "data/fable6_matches.json"),
    ("fable", "pull_fable5_0426.json", "data/fable_matches.json"),
]

for label, local, path in JOBS:
    local_list = json.load(open(f"{BASE}/{local}"))
    local_ids = {m.get("id") for m in local_list}
    try:
        cur = gh_api.get_file("robertyang87", "carbon-league", path, ref="main")
    except Exception as e:
        print(f"{label}: GET FAIL {e}")
        continue
    cur_list = json.loads(base64.b64decode(cur["content"]))
    cur_ids = {m.get("id") for m in cur_list}
    if local_ids == cur_ids and len(local_list) == len(cur_list):
        print(f"{label}: no change ({len(local_list)} matches, sha {cur['sha'][:12]}), skip")
        continue
    new_ids = local_ids - cur_ids
    gone_ids = cur_ids - local_ids
    print(f"{label}: {len(cur_list)} -> {len(local_list)} (+{len(new_ids)} new, -{len(gone_ids)} gone)")
    content = json.dumps(local_list, ensure_ascii=False).encode()
    r = gh_api.put_file("robertyang87", "carbon-league", path, content,
                        f"newface-watch 04:26: {label} matches {len(cur_list)}->{len(local_list)} (117-end)",
                        sha=cur["sha"], branch="main")
    print(f"{label}: put -> {r.get('commit', {}).get('sha', r)[:12] if isinstance(r, dict) else r}")
    back = gh_api.get_file("robertyang87", "carbon-league", path, ref="main")
    back_list = json.loads(base64.b64decode(back["content"]))
    back_ids = {m.get("id") for m in back_list}
    ok = back_ids == local_ids and len(back_list) == len(local_list)
    print(f"{label}: read-back {'OK' if ok else 'MISMATCH'} {len(back_list)} matches sha {back['sha'][:12]}")

# frozen check only
cur = gh_api.get_file("robertyang87", "carbon-league", "data/m9xj570_matches.json", ref="main")
cur_list = json.loads(base64.b64decode(cur["content"]))
print(f"m9xj570 FROZEN check: {len(cur_list)} matches sha {cur['sha'][:12]} (expect 89 / 9d69b8472d70) local 117 pull has 120 — NOT overwritten")

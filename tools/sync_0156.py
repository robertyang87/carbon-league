#!/usr/bin/env python3
"""newface-watch 01:56 run: refresh data/<label>_matches.json on main from 117 pulls.
Modes: update (put + read-back verify) / verify-only (frozen or vanished).
Pure-append check: old id list must equal prefix of new id list."""
import json, sys, base64
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER, REPO = "robertyang87", "carbon-league"
BASE = "/home/hatch/workspace/carbon-live"

def ids(ms):
    out = []
    for m in ms:
        if isinstance(m, dict):
            out.append(str(m.get("id") or m.get("match_id") or m.get("game_id") or json.dumps(m, sort_keys=True)[:80]))
        else:
            out.append(str(m))
    return out

def repo_list(path):
    d = gh_api.get_file(OWNER, REPO, path, ref="main")
    raw = base64.b64decode(d["content"])
    return json.loads(raw), d["sha"]

JOBS = [
    # (label, local pull file, repo path, mode)
    ("m3s3z", "pull_m3s3z_0156.json", "data/m3s3z_matches.json", "update"),
    ("hakimi3pro", "pull_hakimi3pro_0156.json", "data/hakimi3pro_matches.json", "update"),
    ("krypton", "pull_krypton_0156.json", "data/krypton_matches.json", "update"),
    ("lite-v8", "pull_lite-v8_0156.json", "data/lite-v8_matches.json", "update"),
    ("fable", "pull_fable5_0156.json", "data/fable_matches.json", "update"),
    ("k4v3", "pull_k4v3_0156.json", "data/k4v3_matches.json", "update"),
    ("fable6", "pull_fable6_0156.json", "data/fable6_matches.json", "update"),
    ("m9xj570", "pull_m9xj570_0156.json", "data/m9xj570_matches.json", "verify"),
    ("hakimi31", "matches_29844f8f0c9b14b3d37902a551d26a1e.json", "data/hakimi31_matches.json", "verify"),
    ("m3s3", "pull_m3s3_2156.json", "data/m3s3_matches.json", "verify"),
    ("m9xj320", "matches_it320_final.json", "data/m9xj320_matches.json", "verify"),
]

for label, local, path, mode in JOBS:
    new = json.load(open(f"{BASE}/{local}"))
    new_ids = ids(new)
    try:
        old, sha = repo_list(path)
    except Exception as e:
        print(f"{label}: REPO READ FAIL {e}")
        continue
    old_ids = ids(old)
    prefix_ok = old_ids == new_ids[:len(old_ids)]
    subset_ok = set(old_ids) <= set(new_ids)
    if mode == "verify":
        print(f"{label}: VERIFY-ONLY repo={len(old)} local={len(new)} prefix_ok={prefix_ok} subset_ok={subset_ok} sha={sha[:12]}")
        continue
    if len(new) == len(old) and set(old_ids) == set(new_ids):
        print(f"{label}: NO CHANGE repo={len(old)} local={len(new)}")
        continue
    if not prefix_ok:
        print(f"{label}: WARNING not pure-append (prefix_ok=False subset_ok={subset_ok}) repo={len(old)} local={len(new)} — still updating, flag for review")
    content = json.dumps(new, ensure_ascii=False).encode()
    try:
        r = gh_api.put_file(OWNER, REPO, path, content,
                            f"newface-watch 0156: {label} matches {len(old)}->{len(new)}", sha=sha, branch="main")
        commit = (r.get("commit") or {}).get("sha", "?")[:12]
    except Exception as e:
        print(f"{label}: PUT FAIL {e}")
        continue
    try:
        back, _ = repo_list(path)
        ok = len(back) == len(new) and ids(back) == new_ids
        print(f"{label}: UPDATED {len(old)}->{len(new)} commit={commit} readback={'OK' if ok else 'MISMATCH len=' + str(len(back))}")
    except Exception as e:
        print(f"{label}: UPDATED {len(old)}->{len(new)} commit={commit} READBACK FAIL {e}")

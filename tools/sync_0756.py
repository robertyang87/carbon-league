#!/usr/bin/env python3
"""Data-insurance sync run 07:56 (newface-watch 2026-11-14).
Compare by match-id SET (117-endpoint order is unstable), put_file only on change,
read back and verify count + id set. Also verify the it570 frozen file read-only
and count corpus files per label via the contents API.
Usage: python3 sync_0756.py
"""
import base64, json, sys
sys.path.insert(0, "/home/hatch/workspace")
from gh_api import get_file, put_file, req

OWNER, REPO = "robertyang87", "carbon-league"
LABELS = ["m3s3z", "hakimi3pro", "k4v3", "fable6", "lite-v8", "krypton", "k4v2", "fable"]

def repo_list(path):
    d = get_file(OWNER, REPO, path)
    raw = base64.b64decode(d["content"])
    data = json.loads(raw)
    return data, d.get("sha")

def idset(data):
    return {m["id"] for m in data}

results = {}
for label in LABELS:
    local = json.load(open(f"/home/hatch/workspace/carbon-live/pull_{label}_0756.json"))
    ln = len(local)
    lids = idset(local)
    try:
        repo_data, sha = repo_list(f"data/{label}_matches.json")
    except Exception as e:
        print(f"{label}: repo read FAILED ({e}); local={ln}; skipped")
        results[label] = {"local": ln, "status": "repo-read-failed"}
        continue
    rids = idset(repo_data)
    if lids == rids:
        print(f"{label}: unchanged repo={len(repo_data)} local={ln} (id sets identical)")
        results[label] = {"local": ln, "repo": len(repo_data), "status": "unchanged"}
        continue
    added = len(lids - rids); removed = len(rids - lids)
    print(f"{label}: repo={len(repo_data)} -> local={ln} (id-set +{added} -{removed}); updating")
    content = json.dumps(local, ensure_ascii=False).encode()
    try:
        r = put_file(OWNER, REPO, f"data/{label}_matches.json", content,
                     f"sync {label} matches: {len(repo_data)} -> {ln} (newface-watch 07:56)", sha=sha)
        commit = (r.get("commit") or {}).get("sha", "")[:12]
    except Exception as e:
        print(f"{label}: PUT FAILED: {e}")
        results[label] = {"local": ln, "repo": len(repo_data), "status": f"put-failed: {e}"}
        continue
    repo_data2, sha2 = repo_list(f"data/{label}_matches.json")
    ok = idset(repo_data2) == lids and len(repo_data2) == ln
    print(f"{label}: put commit={commit} readback={len(repo_data2)} sha={sha2[:12]} {'VERIFIED' if ok else 'MISMATCH'}")
    results[label] = {"local": ln, "repo_before": len(repo_data), "commit": commit,
                      "readback": len(repo_data2), "sha": sha2[:12],
                      "status": "updated-verified" if ok else "MISMATCH"}

# it570 frozen file: read-only verification
try:
    frozen, fsha = repo_list("data/m9xj570_matches.json")
    fids = idset(frozen)
    print(f"m9xj570(frozen): repo={len(frozen)} sha={fsha[:12]} "
          f"{'FROZEN-OK' if len(frozen)==89 and fsha.startswith('9d69b8472d70') else 'FROZEN-CHANGED!'}")
    results["m9xj570_frozen"] = {"repo": len(frozen), "sha": fsha[:12],
        "status": "frozen-ok" if len(frozen)==89 and fsha.startswith("9d69b8472d70") else "FROZEN-CHANGED"}
except Exception as e:
    print(f"m9xj570(frozen): read FAILED: {e}")
    results["m9xj570_frozen"] = {"status": f"read-failed: {e}"}

# corpus counts via contents API
for label in LABELS + ["m9xj570"]:
    try:
        items = req("GET", f"/repos/{OWNER}/{REPO}/contents/data/{label}_corpus")
        n = len(items) if isinstance(items, list) else None
        print(f"corpus {label}: {n} files")
        results.setdefault(label, {})["corpus"] = n
    except Exception as e:
        print(f"corpus {label}: read FAILED: {e}")
        results.setdefault(label, {})["corpus"] = f"error: {e}"

json.dump(results, open("/home/hatch/workspace/carbon-live/sync_0756_results.json", "w"), ensure_ascii=False, indent=1)
print("results -> sync_0756_results.json")

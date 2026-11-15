#!/usr/bin/env python3
"""Data-insurance sync: refresh data/<label>_matches.json on main when local list changed.
Usage: python3 sync_matches.py <label> <local.json> [<label> <local.json> ...]
Prints per-label: repo_count -> local_count, commit sha or 'unchanged'.
"""
import base64, json, sys
sys.path.insert(0, "/home/hatch/workspace")
from gh_api import get_file, put_file

OWNER, REPO = "robertyang87", "carbon-league"

def repo_count(path):
    try:
        d = get_file(OWNER, REPO, path)
    except Exception as e:
        return None, None, f"get error: {e}"
    if isinstance(d, dict) and "content" in d:
        raw = base64.b64decode(d["content"])
        try:
            data = json.loads(raw)
            return (len(data) if isinstance(data, list) else None), d.get("sha"), None
        except Exception:
            return None, d.get("sha"), "unparseable"
    return None, None, f"unexpected: {str(d)[:120]}"

def main():
    args = sys.argv[1:]
    for i in range(0, len(args), 2):
        label, local = args[i], args[i + 1]
        path = f"data/{label}_matches.json"
        local_data = json.load(open(local))
        ln = len(local_data)
        rc, sha, err = repo_count(path)
        if err:
            print(f"{label}: repo read FAILED ({err}); local={ln}; skipped")
            continue
        if rc == ln:
            print(f"{label}: unchanged repo={rc} local={ln}")
            continue
        if rc is None:
            print(f"{label}: repo has no list (new file); local={ln}")
        else:
            print(f"{label}: repo={rc} -> local={ln}; updating")
        content = json.dumps(local_data, ensure_ascii=False).encode()
        try:
            r = put_file(OWNER, REPO, path, content,
                         f"sync {label} matches: {rc} -> {ln} (newface-watch 04:26)", sha=sha)
            commit = (r.get("commit") or {}).get("sha", "")[:12]
        except Exception as e:
            print(f"{label}: PUT FAILED: {e}")
            continue
        rc2, _, err2 = repo_count(path)
        status = "VERIFIED" if rc2 == ln else f"MISMATCH readback={rc2}"
        print(f"{label}: put commit={commit} readback={rc2} {status}")

if __name__ == "__main__":
    main()

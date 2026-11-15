#!/usr/bin/env python3
"""h23-status 12:57 trip driver: direct read of sync/h23 (+ neighbors) via gh_api.
Marker: h23-status-1257. Read-only; batch calls, no polling loops."""
import sys, json
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER, REPO = "robertyang87", "carbon-league"
TAG = "h23-status-1257"

def branch_head(b):
    d = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/branches/{b}")
    c = d["commit"]["commit"]
    return d["commit"]["sha"][:7], c["committer"]["date"], c["message"].splitlines()[0] if c["message"] else ""

def main():
    print(f"[{TAG}] start")
    rl = gh_api.req("GET", "/rate_limit")
    print("rate core remaining:", rl["resources"]["core"]["remaining"],
          "reset:", rl["resources"]["core"]["reset"])

    sha, date, msg = branch_head("sync/h23")
    print(f"sync/h23 head: {sha} @ {date} :: {msg}")

    tree = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/git/trees/{sha}?recursive=1")
    blobs = [t for t in tree["tree"] if t["type"] == "blob"]
    print("tree blobs total:", len(blobs))
    live = [t for t in blobs if t["path"].startswith("synced/h23/bigeval_live/")]
    print("bigeval_live entries:", len(live))
    for t in sorted(live, key=lambda x: x["path"])[-5:]:
        print("  live:", t["path"], t["sha"][:7], t.get("size"))
    cals = [t for t in blobs if t["path"].startswith("synced/h23/") and t["path"].endswith("_cal.out")]
    print("cal.out files:", [(t["path"], t["sha"][:7], t.get("size")) for t in cals])

    for t in sorted(cals, key=lambda x: x["path"]):
        f = gh_api.get_file(OWNER, REPO, t["path"], ref="sync/h23")
        import base64
        content = base64.b64decode(f["content"]).decode("utf-8", errors="replace")
        print(f"--- {t['path']} sha={f['sha'][:7]} size={f['size']} ---")
        print(content if content.strip() else "(empty)")
        print(f"--- end {t['path']} ---")

    for b in ("sync/h29", "main"):
        s2, d2, m2 = branch_head(b)
        print(f"{b} head: {s2} @ {d2} :: {m2}")

    print(f"[{TAG}] done")

if __name__ == "__main__":
    main()

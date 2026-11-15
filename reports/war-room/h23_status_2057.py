#!/usr/bin/env python3
"""h23 status probe, run 20:57 CST. Reads sync/h23 branch directly via gh_api."""
import sys, json, base64
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER, REPO, BRANCH = "robertyang87", "carbon-league", "sync/h23"

def get_json(path):
    return gh_api.req("GET", path)

# 1. branch head
br = get_json(f"/repos/{OWNER}/{REPO}/branches/{BRANCH}")
c = br["commit"]["commit"]
print("BRANCH_HEAD", br["commit"]["sha"][:7], c["committer"]["date"], c["message"][:80])

# 2. list synced/h23
def list_dir(path):
    try:
        items = get_json(f"/repos/{OWNER}/{REPO}/contents/{path}?ref={BRANCH}")
    except Exception as e:
        print("LIST_FAIL", path, repr(e)); return []
    if isinstance(items, dict):
        print("LIST_SINGLE", path, items.get("name"), items.get("size")); return [items]
    return items

for d in ["synced/h23", "synced/h23/bigeval_live"]:
    items = list_dir(d)
    for it in items:
        print("ITEM", d, it["name"], it.get("size"), it.get("type"))

# 3. fetch a file's text
def fetch(path):
    try:
        f = get_json(f"/repos/{OWNER}/{REPO}/contents/{path}?ref={BRANCH}")
    except Exception as e:
        print("FETCH_FAIL", path, repr(e)); return None
    if isinstance(f, dict) and "content" in f:
        return base64.b64decode(f["content"]).decode("utf-8", "replace")
    print("FETCH_UNEXPECTED", path, type(f)); return None

# live files of interest: k4 probe
for name in ["k4v2r1_e0.jsonl", "k4v2g1_e0.jsonl"]:
    txt = fetch(f"synced/h23/bigeval_live/{name}")
    if txt is None:
        print("LIVE_MISSING", name); continue
    lines = [l for l in txt.splitlines() if l.strip()]
    print("LIVE", name, "lines=", len(lines))
    if lines:
        print("  first:", lines[0][:200])
        print("  last :", lines[-1][:200])

# cal.out files: check known names
for name in ["m7bsr1_cal.out", "m7bsr1_m2.out", "m7bsr1_echo2.out",
             "k4v2r1_e0_cal.out", "lineA_cal.out", "echo2_anchor.out",
             "m3s2g1_m2.out", "g1_chain.out"]:
    txt = fetch(f"synced/h23/{name}")
    if txt is None:
        print("CAL_MISSING", name); continue
    print("CAL", name, "bytes=", len(txt.encode()), "content=", repr(txt[:400]))

"""H24 direct read probe, trip 2026-11-15 ~00:14 CST. Marker: h24-0014b.

NOTE (2026-11-15 00:30): the contents API ignores `page` for directory
listings -- it returns the same first 100 items every time, so a paginated
list_dir loop never terminates (first h24-0014 attempt hung ~13 min on the
bigeval_live listing and had to be SIGTERM'd; the probe script shipped in
earlier trips, h24_probe_2359.py, carries the same bug). For large dirs use
the git trees API (recursive=1) instead: one call, full listing with sizes
and blob shas. Contents API is fine for small dirs (<100 items).
"""
import sys, json, base64
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER, REPO = "robertyang87", "carbon-league"
res = {}

d = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/branches/sync/h24")
c = d["commit"]["commit"]
res["sync_h24_head"] = {"sha": d["commit"]["sha"][:8], "date": c["committer"]["date"],
                        "msg": c["message"][:80]}

items = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/contents/synced/h24?ref=sync/h24")
cals = [i for i in items if i["name"].endswith("_cal.out")]
res["h24_synced"] = {"total": len(items),
                     "cal_outs": [(x["name"], x.get("size"), (x.get("sha") or "")[:8]) for x in cals]}
for x in cals:
    f = gh_api.get_file(OWNER, REPO, "synced/h24/" + x["name"], ref="sync/h24")
    res["h24_cal_" + x["name"]] = base64.b64decode(f["content"]).decode(errors="replace") if isinstance(f, dict) and f.get("encoding") == "base64" else str(f)

tree = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/git/trees/{d['commit']['sha']}?recursive=1")
live = [e for e in tree["tree"] if e["path"].startswith("synced/h24/bigeval_live/")]
res["h24_live"] = {"count": len(live), "truncated": tree.get("truncated")}
for name in ("m3s2r1_m2.jsonl", "m3s2r1_e0.jsonl", "m3s2r1_echo2.jsonl"):
    hits = [e for e in live if e["path"].endswith("/" + name)]
    res["h24_live_" + name] = [(e["path"].split("/")[-1], e.get("size"), (e.get("sha") or "")[:8]) for e in hits]

rl = gh_api.req("GET", "/rate_limit")
res["api_core_remaining"] = rl["resources"]["core"]["remaining"]

print(json.dumps(res, ensure_ascii=False, indent=1))

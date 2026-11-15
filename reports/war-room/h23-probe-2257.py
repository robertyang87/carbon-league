"""H23STATUS2257: direct read of sync/h23 + sync/h29 branches (GitHub API)."""
import sys, json, base64, urllib.error
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER, REPO = "robertyang87", "carbon-league"

def branch_head(name):
    try:
        b = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/branches/{name}")
        c = b["commit"]["commit"]
        return {"exists": True, "sha": b["commit"]["sha"][:7],
                "date": c["committer"]["date"], "msg": c["message"][:80]}
    except urllib.error.HTTPError as e:
        return {"exists": False, "http": e.code}

def list_dir(path, ref):
    try:
        r = gh_api.get_file(OWNER, REPO, path, ref=ref)
        if isinstance(r, list):
            return [{"name": x["name"], "size": x["size"], "type": x["type"]} for x in r]
        return {"file": r.get("name"), "size": r.get("size")}
    except urllib.error.HTTPError as e:
        return {"error": e.code}

def file_head(path, ref, n=400):
    try:
        r = gh_api.get_file(OWNER, REPO, path, ref=ref)
        raw = base64.b64decode(r["content"]).decode("utf-8", errors="replace")
        lines = raw.splitlines()
        return {"size": r["size"], "lines": len(lines),
                "last_lines": lines[-3:] if lines else [], "head": raw[:n]}
    except urllib.error.HTTPError as e:
        return {"error": e.code}

out = {}
out["sync_h23"] = branch_head("sync/h23")
out["sync_h29"] = branch_head("sync/h29")
out["h23_root"] = list_dir("synced/h23", "sync/h23")
out["h23_live"] = list_dir("synced/h23/bigeval_live", "sync/h23")
for cal in ["synced/h23/m7bsr1_cal.out", "synced/h23/lineA_cal.out",
            "synced/h23/k4probe.out"]:
    out[cal] = file_head(cal, "sync/h23")
for live in ["synced/h23/bigeval_live/m7bsr1_m2.jsonl",
             "synced/h23/bigeval_live/m7bsr1_echo2.jsonl"]:
    out[live] = file_head(live, "sync/h23")
print(json.dumps(out, indent=1, ensure_ascii=False))

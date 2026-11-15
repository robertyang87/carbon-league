"""H24 00:15 status read driver. Reads sync/h24 and sync/h28 branch heads,
listings, live jsonl line counts, and *_cal.out full text."""
import json, sys, base64
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER, REPO = "robertyang87", "carbon-league"

def get(path):
    return gh_api.req("GET", path)

def read_file_text(path, ref):
    f = gh_api.get_file(OWNER, REPO, path, ref=ref)
    if isinstance(f, dict) and "content" in f:
        return base64.b64decode(f["content"]).decode("utf-8", errors="replace")
    return None

out = {}
for branch in ["sync/h24", "sync/h28"]:
    b = {"branch": branch}
    try:
        commits = get(f"/repos/{OWNER}/{REPO}/commits?sha={branch}&per_page=3")
        b["commits"] = [
            {"sha": c["sha"][:7], "date": c["commit"]["committer"]["date"],
             "msg": c["commit"]["message"].splitlines()[0][:120]}
            for c in commits
        ]
    except Exception as e:
        b["commits_error"] = str(e)
    try:
        listing = gh_api.get_file(OWNER, REPO, "synced/h24" if branch == "sync/h24" else "synced/h28", ref=branch)
        b["files"] = [e["name"] for e in listing] if isinstance(listing, list) else listing
    except Exception as e:
        b["files_error"] = str(e)
    try:
        live = gh_api.get_file(OWNER, REPO, "synced/h24/bigeval_live" if branch == "sync/h24" else "synced/h28/bigeval_live", ref=branch)
        b["live_files"] = [e["name"] for e in live] if isinstance(live, list) else live
    except Exception as e:
        b["live_files_error"] = str(e)
    out[branch] = b

# h28 live jsonl line counts + last line
for name in ["k4v2r1_e0.jsonl", "k4v2g1_e0.jsonl"]:
    p = f"synced/h28/bigeval_live/{name}"
    try:
        txt = read_file_text(p, "sync/h28")
        if txt is None:
            out[name] = {"error": "unreadable"}
        else:
            lines = [l for l in txt.splitlines() if l.strip()]
            last = None
            try:
                last = json.loads(lines[-1]) if lines else None
            except Exception:
                last = lines[-1][:200] if lines else None
            out[name] = {"lines": len(lines), "bytes": len(txt), "last": last}
    except Exception as e:
        out[name] = {"error": str(e)}

# *_cal.out on h28 (root listing) and h24
for branch, prefix in [("sync/h28", "synced/h28"), ("sync/h24", "synced/h24")]:
    try:
        listing = gh_api.get_file(OWNER, REPO, prefix, ref=branch)
        names = [e["name"] for e in listing] if isinstance(listing, list) else []
        for n in names:
            if n.endswith("_cal.out") or n.endswith(".out"):
                try:
                    txt = read_file_text(f"{prefix}/{n}", branch)
                    out[f"{branch}:{n}"] = txt if txt is not None else "MISSING"
                except Exception as e:
                    out[f"{branch}:{n}"] = f"ERROR {e}"
    except Exception as e:
        out[f"{branch}:cal_list_error"] = str(e)

# bigeval.jsonl on h28 (merged summary) - size + tail
try:
    txt = read_file_text("synced/h28/bigeval.jsonl", "sync/h28")
    if txt is not None:
        lines = [l for l in txt.splitlines() if l.strip()]
        out["h28_bigeval.jsonl"] = {"lines": len(lines), "bytes": len(txt),
                                     "tail": lines[-2:] if lines else []}
    else:
        out["h28_bigeval.jsonl"] = "MISSING"
except Exception as e:
    out["h28_bigeval.jsonl"] = f"ERROR {e}"

print(json.dumps(out, indent=1, ensure_ascii=False))

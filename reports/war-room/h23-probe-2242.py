"""H23 probe 22:42 (marker H23STATUS2242) — direct read of sync/h23 + sync/h29 branches."""
import sys, json, base64
sys.path.insert(0, "/home/hatch/workspace")
from gh_api import req

OWNER, REPO = "robertyang87", "carbon-league"
out = ["H23 probe 22:42 (marker H23STATUS2242)"]

def list_dir(path, br):
    try:
        items = req("GET", f"/repos/{OWNER}/{REPO}/contents/{path}?ref={br}")
        if isinstance(items, dict):
            return [f"  {path}: <single file> size={items.get('size')}"]
        return [f"  {i['name']} {i['size']}B" for i in items]
    except Exception as e:
        return [f"  {path}: ERROR {e}"]

def get_text(path, br):
    try:
        f = req("GET", f"/repos/{OWNER}/{REPO}/contents/{path}?ref={br}")
        if f.get("size", 0) == 0:
            return f"- {path}: 0 bytes (empty)"
        txt = base64.b64decode(f["content"]).decode("utf-8", errors="replace")
        lines = txt.rstrip("\n").split("\n")
        return f"- {path}: {f['size']}B, {len(lines)} lines, last: {lines[-1][:200]}"
    except Exception as e:
        return f"- {path}: ERROR {e}"

# 1. latest commits on sync/h23
try:
    commits = req("GET", f"/repos/{OWNER}/{REPO}/commits?sha=sync/h23&per_page=3")
    for c in commits:
        out.append(f"- commit {c['sha'][:7]} {c['commit']['committer']['date']} msg={c['commit']['message'].splitlines()[0][:80]}")
except Exception as e:
    out.append(f"- sync/h23 commits ERROR {e}")

out.append("- synced/h23/:")
out.extend(list_dir("synced/h23", "sync/h23"))
out.append("- synced/h23/bigeval_live/:")
out.extend(list_dir("synced/h23/bigeval_live", "sync/h23"))

for p in ["synced/h23/m7bsr1_cal.out", "synced/h23/lineA_cal.out",
          "synced/h23/bigeval_live/m7bsr1_m2.jsonl",
          "synced/h23/bigeval_live/m7bsr1_echo2.jsonl"]:
    out.append(get_text(p, "sync/h23"))

# 2. sync/h29 branch existence (new branch for spot-29 live lines)
try:
    commits29 = req("GET", f"/repos/{OWNER}/{REPO}/commits?sha=sync/h29&per_page=3")
    out.append("- sync/h29 EXISTS:")
    for c in commits29:
        out.append(f"  commit {c['sha'][:7]} {c['commit']['committer']['date']} msg={c['commit']['message'].splitlines()[0][:80]}")
    out.append("- synced/h29/:")
    out.extend(list_dir("synced/h29", "sync/h29"))
    out.append("- synced/h29/bigeval_live/:")
    out.extend(list_dir("synced/h29/bigeval_live", "sync/h29"))
except Exception as e:
    out.append(f"- sync/h29: NOT FOUND / ERROR {e}")

print("\n".join(out))

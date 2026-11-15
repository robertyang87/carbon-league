"""H23 probe 22:12 (marker H23STATUS2212) — direct read of sync/h23 branch."""
import sys, json, base64
sys.path.insert(0, "/home/hatch/workspace")
from gh_api import req

OWNER, REPO, BR = "robertyang87", "carbon-league", "sync/h23"
out = ["H23 probe 22:12 (marker H23STATUS2212)"]

# 1. latest commits on sync/h23
commits = req("GET", f"/repos/{OWNER}/{REPO}/commits?sha={BR}&per_page=3")
for c in commits:
    out.append(f"- commit {c['sha'][:7]} {c['commit']['committer']['date']} msg={c['commit']['message'].splitlines()[0][:80]}")

# 2. tree listings
def list_dir(path):
    try:
        items = req("GET", f"/repos/{OWNER}/{REPO}/contents/{path}?ref={BR}")
        if isinstance(items, dict):
            return [f"  {path}: <single file> size={items.get('size')}"]
        return [f"  {i['name']} {i['size']}B" for i in items]
    except Exception as e:
        return [f"  {path}: ERROR {e}"]

out.append("- synced/h23/:")
out.extend(list_dir("synced/h23"))
out.append("- synced/h23/bigeval_live/:")
out.extend(list_dir("synced/h23/bigeval_live"))

# 3. cal.out contents (full text)
for p in ["synced/h23/m7bsr1_cal.out", "synced/h23/lineA_cal.out"]:
    try:
        f = req("GET", f"/repos/{OWNER}/{REPO}/contents/{p}?ref={BR}")
        if f.get("size", 0) == 0:
            out.append(f"- {p}: 0 bytes (empty)")
        else:
            txt = base64.b64decode(f["content"]).decode("utf-8", errors="replace")
            out.append(f"- {p}: {f['size']}B:\n{txt}")
    except Exception as e:
        out.append(f"- {p}: ERROR {e}")

# 4. last lines of the two m7bsr1 live files (tail check via blob)
for p in ["synced/h23/bigeval_live/m7bsr1_m2.jsonl", "synced/h23/bigeval_live/m7bsr1_echo2.jsonl"]:
    try:
        f = req("GET", f"/repos/{OWNER}/{REPO}/contents/{p}?ref={BR}")
        if f.get("size", 0) == 0:
            out.append(f"- {p}: 0 bytes")
        else:
            txt = base64.b64decode(f["content"]).decode("utf-8", errors="replace")
            lines = txt.rstrip("\n").split("\n")
            out.append(f"- {p}: {f['size']}B, {len(lines)} lines, last: {lines[-1][:200]}")
    except Exception as e:
        out.append(f"- {p}: ERROR {e}")

print("\n".join(out))

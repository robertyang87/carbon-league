"""h23-status 10:57 probe: sync/h23 branch state vs 10:42 baseline.
Run marker: H23STAT1057B4 (this run only; claim results by this marker)."""
import sys, json, base64, hashlib
sys.path.insert(0, "/home/hatch/workspace")
from gh_api import req, get_file

OWNER, REPO, BRANCH = "robertyang87", "carbon-league", "sync/h23"
print("MARKER H23STAT1057B4")

try:
    rate = req("GET", "/rate_limit")
    core = rate["resources"]["core"]
    print("RATE core remaining", core["remaining"], "reset", core["reset"])
except Exception as e:
    print("RATE ERROR", repr(e))

commits = req("GET", f"/repos/{OWNER}/{REPO}/commits?sha={BRANCH}&per_page=3")
head = []
for c in commits:
    head.append({"sha": c["sha"][:8], "date": c["commit"]["committer"]["date"],
                 "msg": c["commit"]["message"].splitlines()[0][:100]})
    print("COMMIT", c["sha"][:8], c["commit"]["committer"]["date"],
          c["commit"]["message"].splitlines()[0][:100])

def fetch_text(path):
    d = get_file(OWNER, REPO, path, ref=BRANCH)
    if isinstance(d, dict) and d.get("encoding") == "base64":
        return base64.b64decode(d["content"]).decode("utf-8", "replace")
    return None

top = get_file(OWNER, REPO, "synced/h23", ref=BRANCH)
top_names = sorted(e["name"] for e in top) if isinstance(top, list) else []
print("TOP_ITEMS", len(top_names))
print("TOP_NAMES_MD5", hashlib.md5("\n".join(top_names).encode()).hexdigest())
print("TOP_NAMES", json.dumps(top_names))

live_dir = get_file(OWNER, REPO, "synced/h23/bigeval_live", ref=BRANCH)
live_names = sorted(e["name"] for e in live_dir) if isinstance(live_dir, list) else []
print("LIVE_ITEMS", len(live_names))
print("LIVE_NAMES_MD5", hashlib.md5("\n".join(live_names).encode()).hexdigest())

cal = {}
for name in ["lineA_cal.out", "m3s2r1pre_cal.out", "m7bsr1_cal.out"]:
    txt = fetch_text(f"synced/h23/{name}")
    cal[name] = txt if txt is not None else "MISSING"
    print(f"--- {name} (len {len(txt) if txt is not None else 'NONE'}) ---")
    print(txt if txt is not None else "MISSING")

tails = {}
for name in live_names[-5:]:
    txt = fetch_text(f"synced/h23/bigeval_live/{name}")
    if txt is None:
        tails[name] = {"lines": None, "last": None}
        print("TAIL", name, "MISSING")
        continue
    lines = [l for l in txt.splitlines() if l.strip()]
    last = None
    if lines:
        try:
            o = json.loads(lines[-1])
            last = {"seed": o.get("seed"), "margin": o.get("margin")}
        except Exception:
            last = {"raw": lines[-1][:200]}
    tails[name] = {"lines": len(lines), "last": last}
    print("TAIL", name, len(lines), last)

out = {"trip": "2026-11-15 10:57 CST", "mark": "H23STATUS1057B4",
       "h23_head": head, "top_level_items": len(top_names),
       "top_level_names_md5": hashlib.md5("\n".join(top_names).encode()).hexdigest(),
       "top_level_names": top_names,
       "cal": cal, "live_items": len(live_names),
       "live_names_md5": hashlib.md5("\n".join(live_names).encode()).hexdigest(),
       "live_last5": live_names[-5:], "live_tails": tails}
with open("/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/h23-check-1057-1115.json", "w") as f:
    json.dump(out, f, indent=1, ensure_ascii=False)
print("WROTE h23-check-1057-1115.json")

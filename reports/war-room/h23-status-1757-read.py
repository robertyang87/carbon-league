"""h23-status 17:57 probe: sync/h23 branch state + e0_m2_base live + cal.out files.
Run marker: H23STAT1757K2Q8 (this run only; claim results by this marker)."""
import sys, json, base64
sys.path.insert(0, "/home/hatch/workspace")
from gh_api import req, get_file

OWNER, REPO, BRANCH = "robertyang87", "carbon-league", "sync/h23"
print("MARKER H23STAT1757K2Q8")

commits = req("GET", f"/repos/{OWNER}/{REPO}/commits?sha={BRANCH}&per_page=3")
for c in commits:
    print("COMMIT", c["sha"][:7], c["commit"]["committer"]["date"], c["commit"]["message"].splitlines()[0][:100])

def fetch_text(path):
    d = get_file(OWNER, REPO, path, ref=BRANCH)
    if isinstance(d, dict) and d.get("encoding") == "base64":
        return base64.b64decode(d["content"]).decode("utf-8", "replace")
    return None

live = fetch_text("synced/h23/bigeval_live/e0_m2_base.jsonl")
if live is None:
    print("E0_M2_BASE_LIVE MISSING")
else:
    lines = [l for l in live.splitlines() if l.strip()]
    print("E0_M2_BASE_LINES", len(lines))
    wins = loss = err = 0
    margins = []
    first_t = last_t = None
    bad = 0
    for l in lines:
        try:
            o = json.loads(l)
        except Exception:
            bad += 1
            continue
        m = o.get("margin")
        if m is not None:
            margins.append(m)
            if m > 0: wins += 1
            elif m < 0: loss += 1
        if o.get("err"): err += 1
        t = o.get("t") or o.get("time") or o.get("ts")
        if t:
            if first_t is None: first_t = t
            last_t = t
    print("WINS", wins, "LOSS", loss, "ERR", err, "BAD", bad,
          "MEAN", round(sum(margins)/len(margins), 1) if margins else None,
          "FIRST_T", first_t, "LAST_T", last_t)
    print("FIRST3", [json.loads(l).get("margin") for l in lines[:3]])
    print("LAST3", [json.loads(l).get("margin") for l in lines[-3:]])
    print("SAMPLE_LINE", lines[0][:400] if lines else None)

top = get_file(OWNER, REPO, "synced/h23", ref=BRANCH)
print("TOP_DIR:")
for e in top if isinstance(top, list) else []:
    print(" ", e["name"], e["size"], e["type"])

live_dir = get_file(OWNER, REPO, "synced/h23/bigeval_live", ref=BRANCH)
print("LIVE_DIR:")
for e in live_dir if isinstance(live_dir, list) else []:
    print(" ", e["name"], e["size"])

for name in ["m3s2r1pre_cal.out", "e0_m2_base.out"]:
    txt = fetch_text(f"synced/h23/{name}")
    print(f"--- {name} (len {len(txt) if txt is not None else 'NONE'}) ---")
    print(txt if txt is not None else "MISSING")

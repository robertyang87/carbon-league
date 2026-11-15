"""h23-status 17:27 probe step 2: fetch e0_m2_base live lines + cal.out files."""
import sys, json, base64
sys.path.insert(0, "/home/hatch/workspace")
from gh_api import get_file

OWNER, REPO, BRANCH = "robertyang87", "carbon-league", "sync/h23"

def fetch_text(path):
    d = get_file(OWNER, REPO, path, ref=BRANCH)
    if isinstance(d, dict) and d.get("encoding") == "base64":
        return base64.b64decode(d["content"]).decode("utf-8", "replace")
    return None

live = fetch_text("synced/h23/bigeval_live/e0_m2_base.jsonl")
lines = [l for l in live.splitlines() if l.strip()]
print("E0_M2_BASE_LINES", len(lines))
wins = loss = err = 0
margins = []
first_t = last_t = None
for l in lines:
    try:
        o = json.loads(l)
    except Exception:
        print("UNPARSEABLE:", l[:200]); continue
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
    # print first/last few raw
print("WINS", wins, "LOSS", loss, "ERR", err,
      "MEAN", round(sum(margins)/len(margins), 1) if margins else None,
      "FIRST_T", first_t, "LAST_T", last_t)
print("FIRST3", [json.loads(l).get("margin") for l in lines[:3]])
print("LAST3", [json.loads(l).get("margin") for l in lines[-3:]])
print("SAMPLE_LINE", lines[0][:400] if lines else None)

for name in ["m3s2r1pre_cal.out", "e0_m2_base.out"]:
    txt = fetch_text(f"synced/h23/{name}")
    print(f"--- {name} (len {len(txt) if txt is not None else 'NONE'}) ---")
    print(txt if txt is not None else "MISSING")

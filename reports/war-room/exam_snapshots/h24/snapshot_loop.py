"""2-min snapshot loop for h24 m3s2r1 vs Echo2 field (owner hard rule 2026-11-12).
Pulls synced/h24/bigeval_live/m3s2r1_echo2.jsonl from the sync/h24 branch every
120 s and overwrites the local copy; appends a line per trip to snapshot.log.
Stops when the field reaches 200 lines, when the summary for tag m3s2r1_echo2
lands in synced/h24/bigeval.jsonl, or after 6 h. Never falls back to a slower
cadence: on transient errors it retries at the next 2-min tick.
Run: PYTHONPATH=/home/hatch/workspace nohup python3 snapshot_loop.py >> snapshot_loop.out 2>&1 &
"""
import sys, time, json, base64, os
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

BASE = "/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/exam_snapshots/h24"
os.makedirs(BASE, exist_ok=True)
SNAP = os.path.join(BASE, "m3s2r1_echo2.jsonl")
LOG = os.path.join(BASE, "snapshot.log")
FINAL = os.path.join(BASE, "snapshot_final.txt")
INTERVAL = 120
MAX_TRIPS = 180  # 6 h cap
OWNER, REPO, BRANCH = "robertyang87", "carbon-league", "sync/h24"

def fetch(path):
    f = gh_api.get_file(OWNER, REPO, path, ref=BRANCH)
    if isinstance(f, dict) and "content" in f:
        return base64.b64decode(f["content"]).decode("utf-8", errors="replace"), f.get("sha", "")[:8]
    return None, ""

def logline(s):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG, "a") as fh:
        fh.write(f"{ts} {s}\n")

trips = 0
last_lines = -1
last_head = None
while trips < MAX_TRIPS:
    trips += 1
    try:
        commits = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/commits?sha={BRANCH}&per_page=1")
        head = commits[0]["sha"][:7] if commits else "?"
        txt, _ = fetch("synced/h24/bigeval_live/m3s2r1_echo2.jsonl")
        if txt is None:
            logline(f"trip={trips} NO-FILE head={head}")
        else:
            lines = [l for l in txt.splitlines() if l.strip()]
            with open(SNAP, "w") as fh:
                fh.write(txt)
            last_lines = len(lines)
            flag = " STALE" if head == last_head else ""
            logline(f"trip={trips} lines={len(lines)} head={head}{flag}")
            last_head = head
            if len(lines) >= 200:
                logline(f"DONE lines=200 trips={trips}")
                break
        # completion via summary row in bigeval.jsonl
        try:
            summ, _ = fetch("synced/h24/bigeval.jsonl")
            if summ and '"tag": "m3s2r1_echo2"' in summ:
                logline(f"DONE summary-landed trips={trips} lines={last_lines}")
                break
        except Exception:
            pass
    except Exception as e:
        logline(f"trip={trips} ERROR {repr(e)[:200]}")
    time.sleep(INTERVAL)

with open(FINAL, "w") as fh:
    fh.write(f"trips={trips} last_lines={last_lines} finished={time.strftime('%Y-%m-%d %H:%M:%S %Z')}\n")
print(f"snapshot loop ended: trips={trips} last_lines={last_lines}", flush=True)

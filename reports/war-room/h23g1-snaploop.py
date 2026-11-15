"""2-minute snapshot loop for the h23 g1 chained exams.

Pulls the per-game lines from the sync/h23 branch (the on-box sync chain
pushes every ~2 min) and saves them locally under exam_snapshots/h23/.
Falls back to nothing automatic on staleness except a STALE warning in the
log; the operator intervenes with a direct pull if the branch goes stale.
"""
import base64, json, os, sys, time, urllib.request

sys.path.insert(0, "/home/hatch/workspace")
import gh_api

SNAP_DIR = os.path.expanduser("~/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/exam_snapshots/h23")
os.makedirs(SNAP_DIR, exist_ok=True)
LOG = os.path.join(SNAP_DIR, "snapshot_log.txt")
FILES = {
    "m3s2g1_m2r.jsonl": "synced/h23/bigeval_live/m3s2g1_m2r.jsonl",
    "m3s2g1_e0.jsonl": "synced/h23/bigeval_live/m3s2g1_e0.jsonl",
}
SUMMARY_PATH = "synced/h23/bigeval.jsonl"
INTERVAL = 120
MAX_TRIPS = 60

def log_line(s):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {s}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")

def fetch_file(path):
    """Return (text, sha) or (None, None) on 404/error."""
    try:
        r = gh_api.get_file("robertyang87", "carbon-league", path, ref="sync/h23")
        return base64.b64decode(r["content"]).decode(), r.get("sha")
    except Exception as e:
        return None, str(e)

def count_lines(text):
    return len([l for l in text.split("\n") if l.strip()]) if text else 0

def main():
    log_line("SNAPLOOP_START interval=120s source=sync/h23 branch")
    last_sha = None
    stale_trips = 0
    for trip in range(1, MAX_TRIPS + 1):
        t0 = time.time()
        try:
            b = gh_api.req("GET", "/repos/robertyang87/carbon-league/branches/sync/h23")
            sha = b["commit"]["sha"][:8]
            cdate = b["commit"]["commit"]["author"]["date"]
        except Exception as e:
            log_line(f"trip={trip} BRANCH_HEAD_ERROR {e}")
            sha, cdate = None, None
        if sha and sha != last_sha:
            last_sha = sha
            stale_trips = 0
            counts = {}
            for name, path in FILES.items():
                text, _ = fetch_file(path)
                if text is None:
                    counts[name] = "MISSING"
                    continue
                tmp = os.path.join(SNAP_DIR, name + ".tmp")
                with open(tmp, "w") as f:
                    f.write(text)
                os.replace(tmp, os.path.join(SNAP_DIR, name))
                counts[name] = count_lines(text)
            # summary file (for completion detection)
            stext, _ = fetch_file(SUMMARY_PATH)
            summ = ""
            if stext:
                tmp = os.path.join(SNAP_DIR, "bigeval.jsonl.tmp")
                with open(tmp, "w") as f:
                    f.write(stext)
                os.replace(tmp, os.path.join(SNAP_DIR, "bigeval.jsonl"))
                summ = " m2r_summary=%s e0_summary=%s" % (
                    ("m3s2g1_m2r" in stext), ("m3s2g1_e0" in stext))
            log_line(f"trip={trip} sha={sha} branch_time={cdate} m2r={counts.get('m3s2g1_m2r.jsonl')} e0={counts.get('m3s2g1_e0.jsonl')}{summ}")
            m2r = counts.get("m3s2g1_m2r.jsonl")
            e0 = counts.get("m3s2g1_e0.jsonl")
            if m2r == 89 and e0 == 200:
                log_line(f"trip={trip} BOTH_FIELDS_COMPLETE m2r=89 e0=200 SNAPLOOP_DONE")
                return
        else:
            stale_trips += 1
            log_line(f"trip={trip} sha={sha or 'ERR'} NO_CHANGE stale_trips={stale_trips}")
            if stale_trips >= 3:
                log_line(f"trip={trip} STALE_WARNING branch head unchanged for {stale_trips} trips - operator: direct pull needed")
        dt = time.time() - t0
        time.sleep(max(0, INTERVAL - dt))
    log_line("SNAPLOOP_MAX_TRIPS_REACHED")

if __name__ == "__main__":
    main()

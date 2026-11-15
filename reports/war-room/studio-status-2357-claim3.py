import sys
sys.path.insert(0, "/home/hatch/workspace")
import importlib.util
spec = importlib.util.spec_from_file_location("c2", "studio-status-2357-claim2.py")
# reuse fetch via exec of defs only: simpler - import functions by exec
src = open("studio-status-2357-claim2.py").read()
head = src.split("r = gh_api.req")[0]
exec(head)
jobs = [131310082082, 131309496294, 131305996774, 131306014902]
for jid in jobs:
    best = ""
    for attempt in range(5):
        try:
            log, n, how = fetch_log_tolerant("robertyang87", "carbon-league", jid)
        except Exception as e:
            print(jid, "attempt", attempt, "error", e, flush=True); continue
        if len(log) > len(best):
            best = log
        print(jid, "attempt", attempt, "bytes", n, how, "marker" if MARK in log else "no-marker", flush=True)
        if MARK in log:
            i0 = log.find(MARK)
            print(log[max(0, i0-200):i0+16000], flush=True)
            break
    print(jid, "best bytes", len(best), "marker" if MARK in best else "no-marker", flush=True)

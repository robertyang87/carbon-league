import sys, json
sys.path.insert(0, "/home/hatch/workspace")
import gh_api
MARK = "STUDIOSTAT2342IT320"
log = gh_api.get_job_logs("robertyang87", "carbon-league", 131297575732)
open("studio-status-2342-full.log", "w").write(log if isinstance(log, str) else str(log))
i0 = log.find(MARK)
seg = log[max(0, i0-200):i0+16000]
open("studio-status-2342.log", "w").write(seg)
print("saved full len", len(log), "marker at", i0)
state = {
  "run": "studio-status-report 2026-11-12 23:42",
  "direct_read_cst": "2026-11-12 23:46:37",
  "marker": MARK,
  "claimed_run": 43251320363,
  "claimed_job": 131297575732,
  "conclusion": "success",
  "studio": "mac-studio.local, uptime 14d 7:38, load 2.40/2.02/1.88, disk free 423G",
  "train_proc": "none (only mds_stores)",
  "ship_watcher": "none",
  "it320_wr1": "done 22:34, best val 0.8673 @ep8, ep19 22:34, ckpt ep19/last 22:34, unchanged",
  "new_logs": "none since 22:34",
  "line_b_lb": "no lb_* files on Studio",
  "fable_logs": "stale 17:08/17:20, unchanged",
  "verdict": "still normally idle, no change vs 22:44 report; war room already notified 22:44 and confirmed standby 22:45; no repeat to war room"
}
json.dump(state, open("studio-status-last.json", "w"), ensure_ascii=False, indent=2)
print("state written")

"""Start S3 lanes 2-7 on Studio (pilot passed), update lanes.json, (re)start watchdog."""
import json
import os
import subprocess
import time

BASE = "/Users/arm/junyang/carbon/data/s3_enum_studio"
SPECS = [(2, "278-281"), (3, "282-285"), (4, "286-289"),
         (5, "290-293"), (6, "294-297"), (7, "298-299")]

with open(os.path.join(BASE, "lanes.json"), encoding="utf-8") as fh:
    reg = json.load(fh)
lanes = {l["lane"]: l for l in reg["lanes"]}

for lane, games in SPECS:
    r = subprocess.run(["pgrep", "-f", f"shard-id lane{lane}"],
                       capture_output=True, text=True)
    if r.stdout.strip():
        print(f"lane{lane} already running: {r.stdout.strip()}; skip")
        continue
    out = os.path.join(BASE, "out", f"lane{lane}")
    os.makedirs(out, exist_ok=True)
    p = subprocess.Popen(
        ["/Users/arm/junyang/carbon/.venv/bin/python",
         os.path.join(BASE, "s3_launch_lane.py"),
         str(lane), games, out, f"lane{lane}"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    lanes[lane] = {"lane": lane, "pgid": p.pid, "games": games, "out": out,
                   "shard": f"lane{lane}", "peak_rss_kb": 0}
    print(f"lane{lane} started pid/pgid {p.pid} games {games}")

reg = {"lanes": [lanes[k] for k in sorted(lanes)]}
with open(os.path.join(BASE, "lanes.json"), "w", encoding="utf-8") as fh:
    json.dump(reg, fh, indent=1)

r = subprocess.run(["pgrep", "-f", "s3_lane_watch.py"],
                   capture_output=True, text=True)
if r.stdout.strip():
    print("watchdog already running:", r.stdout.strip())
else:
    subprocess.Popen(
        ["/Users/arm/junyang/carbon/.venv/bin/python",
         os.path.join(BASE, "s3_lane_watch.py")],
        stdout=open(os.path.join(BASE, "watch_stdout.log"), "ab"),
        stderr=subprocess.STDOUT, start_new_session=True)
    print("watchdog (re)started")

time.sleep(5)
out = subprocess.run(
    ["ps", "-ax", "-o", "pid=", "-o", "pgid=", "-o", "etime=", "-o", "rss=",
     "-o", "command="], capture_output=True, text=True).stdout
for line in out.splitlines():
    if "enumerate_roots" in line or "s3_lane_watch" in line:
        print(line.strip())

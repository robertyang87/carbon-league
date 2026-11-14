"""Post-launch verification for one arm (runs ON L20).

Usage: remote_verify_arm.py <m|c> <pid>
Prints one JSON object: alive, gpu_mib (nvidia-smi for the pid, -1 if absent),
league_keys (keys in league.json, -1 if missing), spawn_main (all spawn_main
cmdline matches; with a single arm running this must be 15 = 14 workers + 1 tracker).
"""
import json, subprocess, sys

arm, pid = sys.argv[1], sys.argv[2]
alive = subprocess.run(["kill", "-0", pid], capture_output=True).returncode == 0
gpu_mib = -1
out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                      "--format=csv,noheader"], capture_output=True, text=True).stdout
for line in out.strip().splitlines():
    parts = [p.strip() for p in line.split(",")]
    if len(parts) == 2 and parts[0] == pid:
        gpu_mib = int(parts[1].split()[0])
league_keys = -1
try:
    with open(f"/root/carbon/data/rl3/ppou2_{arm}/league.json") as f:
        league_keys = len(json.load(f))
except Exception:
    pass
n = subprocess.run(["pgrep", "-f", "spawn_mai[n]"], capture_output=True, text=True).stdout
spawn_main = len([x for x in n.strip().splitlines() if x.strip()])
print(json.dumps({"arm": arm, "pid": int(pid), "alive": alive, "gpu_mib": gpu_mib,
                  "league_keys": league_keys, "spawn_main": spawn_main}))

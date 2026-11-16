"""S3 lane watchdog (Mac scope equivalent, runs on Studio).

Polls every 30 s and writes status to <BASE>/watch.json:
- per-lane RSS (sum over its process group); >512 MB -> SIGKILL the group,
  log KILLED_512M (protocol SS10.2 MemoryMax 512M equivalent).
- any HALT_<shard>.json (protocol SS1.4 trigger) -> SIGTERM all other lanes,
  SIGKILL after 10 s if still alive, log HALTED; the whole run stops.
- lane done: process group gone AND report_<shard>.json exists.
- tracks peak RSS per lane.

Lane registry: <BASE>/lanes.json, written by the launcher driver:
  {"lanes": [{"lane": 0, "pgid": N, "games": "270-273",
              "out": ".../out/lane0", "shard": "lane0"}, ...]}
"""
import json
import os
import signal
import subprocess
import time

BASE = "/Users/arm/junyang/carbon/data/s3_enum_studio"
REG = os.path.join(BASE, "lanes.json")
STATUS = os.path.join(BASE, "watch.json")
LOG = os.path.join(BASE, "watch.log")
MEM_LIMIT_KB = 512 * 1024


def log(msg):
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")


def group_rss(pgid):
    """Sum RSS (KB) of all processes in the group; None if group gone."""
    try:
        out = subprocess.run(
            ["ps", "-ax", "-o", "pgid=", "-o", "rss="],
            capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return None
    total, seen = 0, False
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] == str(pgid):
            total += int(parts[1])
            seen = True
    return total if seen else None


def kill_group(pgid, sig):
    try:
        os.killpg(pgid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def main():
    while True:
        try:
            with open(REG, encoding="utf-8") as fh:
                reg = json.load(fh)
        except FileNotFoundError:
            time.sleep(30)
            continue
        lanes = reg.get("lanes", [])
        status = {"t": time.strftime("%Y-%m-%d %H:%M:%S"),
                  "lanes": {}, "halted": False}
        halt_found = None
        for ln in lanes:
            out, shard = ln["out"], ln["shard"]
            halt_path = os.path.join(out, f"HALT_{shard}.json")
            if os.path.exists(halt_path):
                halt_found = (ln["lane"], halt_path)
            rss = group_rss(ln["pgid"]) if ln.get("pgid") else None
            peak = ln.get("peak_rss_kb", 0)
            if rss is not None and rss > peak:
                peak = rss
                ln["peak_rss_kb"] = peak
            report = os.path.join(out, f"report_{shard}.json")
            done = rss is None and os.path.exists(report)
            entry = {"rss_kb": rss, "peak_rss_kb": peak,
                     "done": done, "halt": os.path.exists(halt_path)}
            if rss is not None and rss > MEM_LIMIT_KB:
                kill_group(ln["pgid"], signal.SIGKILL)
                log(f"lane {ln['lane']} KILLED_512M rss={rss}KB peak={peak}KB")
                entry["killed_512m"] = True
            status["lanes"][str(ln["lane"])] = entry
        if halt_found:
            status["halted"] = True
            log(f"HALT in lane {halt_found[0]} ({halt_found[1]}): "
                "stopping all other lanes per SS1.4")
            for ln in lanes:
                if ln["lane"] == halt_found[0]:
                    continue
                if group_rss(ln.get("pgid")) is not None:
                    kill_group(ln["pgid"], signal.SIGTERM)
            time.sleep(10)
            for ln in lanes:
                if group_rss(ln.get("pgid")) is not None:
                    kill_group(ln["pgid"], signal.SIGKILL)
            status["lanes"] = {k: dict(v, done=False)
                               for k, v in status["lanes"].items()}
        with open(REG, "w", encoding="utf-8") as fh:
            json.dump(reg, fh, indent=1)
        with open(STATUS, "w", encoding="utf-8") as fh:
            json.dump(status, fh, indent=1)
        if halt_found:
            log("watchdog exiting after HALT stop")
            break
        if lanes and all(status["lanes"][str(ln["lane"])]["done"]
                         for ln in lanes):
            log("all registered lanes done; watchdog exiting")
            break
        time.sleep(30)


if __name__ == "__main__":
    main()

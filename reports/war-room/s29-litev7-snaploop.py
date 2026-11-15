#!/usr/bin/env python3
"""Lite-V7 dual-arm exam snapshot loop (spot-29, direct Tailscale SSH).

Every 120s: fetch each tag's bigeval_live file from spot-29, validate seed
domain, sort by seed, mirror locally, and push to sync/h29 branch only when
content changed. Full-file snapshots: a dead loop loses nothing while the
machine lives; restart and the next cycle re-syncs everything.
Exits when all four exam tags reach 200 lines (writes litev7-snaploop.done).
"""
import base64, hashlib, json, os, subprocess, sys, time

sys.path.insert(0, "/home/hatch/workspace")
import gh_api

BASE = "/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/exam_snapshots/h29"
STATE = os.path.join(BASE, "litev7-snap-state.json")
LOG = os.path.join(BASE, "snaploop-litev7.log")
DONE = os.path.join(BASE, "litev7-snaploop.done")
HOST = "root@100.65.8.27"
SSH = ["ssh", "-i", "/home/hatch/.ssh/id_ed25519", "-o", "IdentitiesOnly=yes",
       "-o", "ConnectTimeout=30", "-o", "BatchMode=yes",
       "-o", "UserKnownHostsFile=/home/hatch/.ssh/known_hosts",
       "-o", "GlobalKnownHostsFile=/dev/null",
       "-o", "ProxyCommand=nc -X connect -x hatch-egress-proxy:3130 %h %p", HOST]
TAGS = {
    "lite_v7r1_smoke": (999901, 999902),
    "lite_v7r1_m2": (996000, 996199),
    "lite_v7r1_echo2": (999000, 999199),
    "lite_v7g1_m2": (996000, 996199),
    "lite_v7g1_echo2": (999000, 999199),
}
EXAM_TAGS = ["lite_v7r1_m2", "lite_v7r1_echo2", "lite_v7g1_m2", "lite_v7g1_echo2"]
INTERVAL = 120
MAX_RUNTIME = 6 * 3600


def log(msg):
    line = f"{time.strftime('%F %T')} {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def fetch(tag):
    r = subprocess.run(["timeout", "-k", "10", "75"] + SSH +
                       [f"cat /root/carbon/data/bigeval_live/{tag}.jsonl 2>/dev/null"],
                       capture_output=True, timeout=90)
    if r.returncode != 0 or not r.stdout:
        return None
    return r.stdout


def process(tag, raw, state):
    lo, hi = TAGS[tag]
    rows = {}
    bad = 0
    for line in raw.decode("utf-8", "replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            seed = json.loads(line)["seed"]
        except Exception:
            bad += 1
            continue
        if lo <= seed <= hi:
            rows[seed] = line
        else:
            bad += 1
    seeds = sorted(rows)
    content = ("\n".join(rows[s] for s in seeds) + "\n").encode() if seeds else b""
    md5 = hashlib.md5(content).hexdigest()
    mirror = os.path.join(BASE, tag + ".jsonl")
    old = None
    if os.path.exists(mirror):
        with open(mirror, "rb") as f:
            old = hashlib.md5(f.read()).hexdigest()
    if old != md5:
        with open(mirror, "wb") as f:
            f.write(content)
    st = state.setdefault(tag, {})
    if st.get("pushed_md5") != md5 and content:
        path = f"synced/h29/bigeval_live/{tag}.jsonl"
        sha = None
        try:
            cur = gh_api.get_file("robertyang87", "carbon-league", path, ref="sync/h29")
            if isinstance(cur, dict):
                sha = cur.get("sha")
        except Exception:
            sha = None
        gh_api.put_file("robertyang87", "carbon-league", path, content,
                        f"snap {tag}: {len(seeds)} lines", sha=sha, branch="sync/h29")
        st["pushed_md5"] = md5
        log(f"PUSH {tag} lines={len(seeds)} bad={bad}")
    st["count"] = len(seeds)
    return len(seeds)


def main():
    os.makedirs(BASE, exist_ok=True)
    state = {}
    if os.path.exists(STATE):
        with open(STATE) as f:
            state = json.load(f)
    t0 = time.time()
    log("snaploop start tags=" + ",".join(TAGS))
    while time.time() - t0 < MAX_RUNTIME:
        counts = {}
        for tag in TAGS:
            try:
                raw = fetch(tag)
                counts[tag] = process(tag, raw, state) if raw else state.get(tag, {}).get("count", 0)
            except Exception as e:
                log(f"ERR {tag}: {type(e).__name__} {e}")
                counts[tag] = state.get(tag, {}).get("count", 0)
        with open(STATE, "w") as f:
            json.dump(state, f)
        log("cycle counts=" + json.dumps(counts))
        if all(counts.get(t, 0) >= 200 for t in EXAM_TAGS):
            with open(DONE, "w") as f:
                f.write(time.strftime("%F %T") + " all four exam tags at 200\n")
            log("DONE all four exam tags at 200, loop exits")
            return
        time.sleep(INTERVAL)
    log("MAX_RUNTIME reached, loop exits")


if __name__ == "__main__":
    main()

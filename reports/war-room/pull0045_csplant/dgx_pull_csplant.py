#!/usr/bin/env python3
"""DGX-host pull for pull0045 Krypton-CS-Plant-v1, driven by C2 (sole executor).
Reason for host: replay /data/ endpoint requires a logged-in ladder session;
the live login (claude-opus55-cloud) exists on DGX, not on the C2 VM
(verified 2026-11-15 00:46 CST: this host whoami OK; C2 VM /data/ = 401).
Frozen list (uploaded by C2, NOT re-pulled here): csplant_matches_0045.json,
192 games pulled from the 117 endpoint by C2 at 00:41 CST,
strategy hex adaf275d51a177b7a9cf8224ab66c467. Label: krypton-csplant.
Does NOT touch the repo clone or any *_matches.json inside it.
"""
import gzip, hashlib, json, os, subprocess, sys, time

OUT = "/home/arm/junyang/pull0045_csplant"
REPO = "/home/arm/junyang/carbon-repo"
MF = os.path.join(OUT, "csplant_matches_0045.json")
REPLAYS = os.path.join(OUT, "replays")
LABEL = "krypton-csplant"

os.makedirs(REPLAYS, exist_ok=True)
env = dict(os.environ, LADDER_BASE="https://117.72.210.134")
print(f"=== {LABEL} pull start {time.strftime('%H:%M:%S')} ===", flush=True)
r = subprocess.run([sys.executable, "code/ladder.py", "replays", MF, REPLAYS],
                   cwd=REPO, env=env)
print(f"replays rc={r.returncode}", flush=True)
if r.returncode != 0:
    print("PULL0045_FAIL replays", flush=True)
    sys.exit(1)

ms = json.load(open(MF))
want = {f"{m['id']}-{m['game']}.json.gz" for m in ms
        if m.get("winner") is not None or m.get("scores")}
have = set(os.listdir(REPLAYS))
bad, missing = [], []
for fn in sorted(want):
    p = os.path.join(REPLAYS, fn)
    if not os.path.exists(p):
        missing.append(fn)
        continue
    try:
        d = json.load(gzip.open(p))
        assert "steps" in d
    except Exception as e:
        bad.append((fn, str(e)))
print(f"want={len(want)} have={len(have & want)} missing={len(missing)} bad={len(bad)}",
      flush=True)
if missing:
    print("missing sample:", missing[:5], flush=True)
if bad:
    print("bad sample:", bad[:5], flush=True)

man = os.path.join(OUT, "pull0045_krypton-csplant.manifest")
n = 0
with open(man, "w") as mf:
    for fn in sorted(os.listdir(REPLAYS)):
        p = os.path.join(REPLAYS, fn)
        h = hashlib.md5(open(p, "rb").read()).hexdigest()
        mf.write(f"{h}  data/{LABEL}_corpus/{fn}\n")
        n += 1
print(f"manifest entries={n} -> {man}", flush=True)
if missing or bad:
    print("PULL0045_INCOMPLETE", flush=True)
    sys.exit(2)
print("PULL0045_CSPLANT_ALL_DONE", flush=True)

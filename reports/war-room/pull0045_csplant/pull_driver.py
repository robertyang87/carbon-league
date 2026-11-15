#!/usr/bin/env python3
"""C2 pull0045: Krypton-CS-Plant-v1 replay pull (frozen list) + verify + manifest.
Local VM execution; delivery to DGX is a separate step.
Frozen list: csplant_matches_0045.json (192 games, pulled from 117 endpoint
2026-11-15 00:41 CST via pull_by_hex.py, strategy hex
adaf275d51a177b7a9cf8224ab66c467). Label: krypton-csplant.
"""
import gzip, hashlib, json, os, subprocess, sys, time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MF = os.path.join(BASE_DIR, "csplant_matches_0045.json")
OUT = os.path.join(BASE_DIR, "replays")
LABEL = "krypton-csplant"
LADDER_DIR = "/home/hatch/workspace/carbon-live"

os.makedirs(OUT, exist_ok=True)
env = dict(os.environ, LADDER_BASE="https://117.72.210.134")
print(f"pull start {time.strftime('%H:%M:%S')}", flush=True)
r = subprocess.run([sys.executable, "ladder.py", "replays", MF, OUT],
                   cwd=LADDER_DIR, env=env)
print(f"replays rc={r.returncode}", flush=True)
if r.returncode != 0:
    print("PULL0045_FAIL replays", flush=True)
    sys.exit(1)

ms = json.load(open(MF))
want = {f"{m['id']}-{m['game']}.json.gz" for m in ms
        if m.get("winner") is not None or m.get("scores")}
have = set(os.listdir(OUT))
bad, missing = [], []
for fn in sorted(want):
    p = os.path.join(OUT, fn)
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

man = os.path.join(BASE_DIR, "pull0045_krypton-csplant.manifest")
n = 0
with open(man, "w") as mf:
    for fn in sorted(os.listdir(OUT)):
        p = os.path.join(OUT, fn)
        h = hashlib.md5(open(p, "rb").read()).hexdigest()
        mf.write(f"{h}  data/{LABEL}_corpus/{fn}\n")
        n += 1
print(f"manifest {man} entries={n}", flush=True)
if missing or bad:
    print("PULL0045_INCOMPLETE", flush=True)
    sys.exit(2)
print("PULL0045_CSPLANT_ALL_DONE", flush=True)

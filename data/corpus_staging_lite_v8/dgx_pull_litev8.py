#!/usr/bin/env python3
"""DGX-side lite-v8 pull: fresh 117 match list + replay downloads (skip existing)
into the clone's data/lite-v8_corpus dir, then full-dir md5 manifest.
Runs ON DGX. Does NOT git add/commit/push (warehousing is the war-room API line).
Does NOT move the clone (pinned at 1d57e671 by war-room order)."""
import json, os, subprocess, sys, hashlib, gzip, time
REPO = "/home/arm/junyang/carbon-repo"
os.chdir(REPO)
ENV = dict(os.environ, LADDER_BASE="https://117.72.210.134")
OUT = "/home/arm/junyang/litev8"
os.makedirs(OUT, exist_ok=True)
TARGETS = [
    ("lite-v8", "3a4d267d9fcea50794f48d38d3e1360d", "data/lite-v8_corpus"),
]
def run(cmd, **kw):
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, **kw)
for label, hexid, outdir in TARGETS:
    print(f"=== {label} start {time.strftime('%H:%M:%S')} ===", flush=True)
    mfile = os.path.join(OUT, f"{label}_matches.json")
    r = run([sys.executable, "code/ladder.py", "matches", hexid, mfile], env=ENV)
    if r.returncode != 0:
        print(f"BATCH-FAIL {label} matches rc={r.returncode}", flush=True); continue
    ms = json.load(open(mfile))
    print(f"{label}: list {len(ms)} matches", flush=True)
    r = run([sys.executable, "code/ladder.py", "replays", mfile, outdir], env=ENV)
    if r.returncode != 0:
        print(f"BATCH-FAIL {label} replays rc={r.returncode}", flush=True); continue
    want = {f"{m['id']}-{m['game']}.json.gz" for m in ms if m.get("winner") is not None or m.get("scores")}
    have = set(os.listdir(outdir)) if os.path.isdir(outdir) else set()
    bad, missing = [], []
    for fn in sorted(want):
        p = os.path.join(outdir, fn)
        if not os.path.exists(p):
            missing.append(fn); continue
        try:
            d = json.load(gzip.open(p))
            assert "steps" in d
        except Exception as e:
            bad.append((fn, str(e)))
    print(f"{label}: want={len(want)} have={len(have & want)} missing={len(missing)} bad={len(bad)}", flush=True)
    if missing: print(f"  missing sample: {missing[:5]}", flush=True)
    if bad: print(f"  bad sample: {bad[:5]}", flush=True)
    man_path = os.path.join(OUT, f"pull_lite-v8.manifest")
    with open(man_path, "w") as mf:
        for fn in sorted(os.listdir(outdir)):
            p = os.path.join(outdir, fn)
            h = hashlib.md5(open(p, "rb").read()).hexdigest()
            mf.write(f"{h}  data/{label}_corpus/{fn}\n")
    print(f"{label}: manifest {man_path} entries={len(os.listdir(outdir))}", flush=True)
print("PULL_LITEV8_ALL_DONE", flush=True)

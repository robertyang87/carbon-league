#!/usr/bin/env python3
"""DGX-side gap3 pull: fresh 117 match lists + replay downloads (skip existing)
into the clone's data/<label>_corpus dirs, then full-dir md5 manifests.
Runs ON DGX. Does NOT git add/commit/push (warehousing is the war-room API line)."""
import json, os, subprocess, sys, hashlib, gzip, time
REPO = "/home/arm/junyang/carbon-repo"
os.chdir(REPO)
ENV = dict(os.environ, LADDER_BASE="https://117.72.210.134")
OUT = "/home/arm/junyang/gap3"
os.makedirs(OUT, exist_ok=True)
TARGETS = [
    ("m3s3z", "f9fdbce38def1a78eb9c77f8711a1103", "data/m3s3z_corpus"),
    ("lite-v8", "3a4d267d9fcea50794f48d38d3e1360d", "data/lite-v8_corpus"),
    ("k4v3", "9d596068f39e0b8cca317c3bb27f8ae5", "data/k4v3_corpus"),
    ("krypton", "0f2c23daf04aaf650576575d2482b940", "data/krypton_corpus"),
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
    man_path = os.path.join(OUT, f"pull0031_{label}_gap3.manifest")
    with open(man_path, "w") as mf:
        for fn in sorted(os.listdir(outdir)):
            p = os.path.join(outdir, fn)
            h = hashlib.md5(open(p, "rb").read()).hexdigest()
            mf.write(f"{h}  data/{label}_corpus/{fn}\n")
    print(f"{label}: manifest {man_path} entries={len(os.listdir(outdir))}", flush=True)
print("PULL_GAP3_ALL_DONE", flush=True)

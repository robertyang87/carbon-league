#!/usr/bin/env python3
"""DGX-side krypton-search256 insurance pull: fresh 117 match list + replay
downloads into a NEW clone dir data/krypton-search256_corpus, then full-dir
md5 manifest.
Runs ON DGX. Does NOT git add/commit/push (warehousing is the war-room API
line). Does NOT move the clone (pinned at 1d57e671 by war-room order).
Does NOT touch data/*_matches.json inside the clone and does NOT touch the
it570 frozen file; the fresh list snapshot is written to OUT only.
Identity (verified by the war room before dispatch):
- existing data/krypton_corpus belongs to Krypton-Cloud500
  (hex 0f2c23daf04aaf650576575d2482b940), NOT to Krypton-Search256-v1;
- this is a NEW corpus, label krypton-search256, hex
  2c57e7cd01dee46bef98643398ab4ca3 (board #16, 66/111 at dispatch time).
Insurance only: pull + manifest. No analysis, no trigger fact pack, no
training/replication dispatch."""
import json, os, subprocess, sys, hashlib, gzip, time
REPO = "/home/arm/junyang/carbon-repo"
os.chdir(REPO)
ENV = dict(os.environ, LADDER_BASE="https://117.72.210.134")
OUT = "/home/arm/junyang/krypton_s256"
os.makedirs(OUT, exist_ok=True)
TARGETS = [
    ("krypton-search256", "2c57e7cd01dee46bef98643398ab4ca3", "data/krypton-search256_corpus"),
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
    man_path = os.path.join(OUT, "pull0033_krypton-search256.manifest")
    with open(man_path, "w") as mf:
        for fn in sorted(os.listdir(outdir)):
            p = os.path.join(outdir, fn)
            h = hashlib.md5(open(p, "rb").read()).hexdigest()
            mf.write(f"{h}  data/{label}_corpus/{fn}\n")
    print(f"{label}: manifest {man_path} entries={len(os.listdir(outdir))}", flush=True)
print("PULL_KRYPTON_S256_ALL_DONE", flush=True)

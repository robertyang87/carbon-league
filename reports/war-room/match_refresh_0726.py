#!/usr/bin/env python3
"""2b match-list refresh, run 07:26 2026-11-14.
Pull 117-end full match lists via ladder.py (LADDER_BASE=117), compare id
sets with warehouse main data/<label>_matches.json, put_file only on change,
get_file read-back to verify. it570 is FROZEN at 89 (sha 9d69b8472d70):
pull for reporting only, never update."""
import json, os, subprocess, sys
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

BASE = "/home/hatch/workspace/carbon-live"
WORK = "/tmp/newface_0726"
os.makedirs(WORK, exist_ok=True)
OWNER, REPO = "robertyang87", "carbon-league"

# (label, hex strategy id, warehouse path, frozen?)
TARGETS = [
    ("m3s3z",     "f9fdbce38def1a78eb9c77f8711a1103", "data/m3s3z_matches.json", False),
    ("hakimi3pro","aff6e57c5eba8857d5546783c7d39a46", "data/hakimi3pro_matches.json", False),
    ("k4v3",      "9d596068f39e0b8cca317c3bb27f8ae5", "data/k4v3_matches.json", False),
    ("fable6",    "5b453cb4f7c49439673deda697d46a9b", "data/fable6_matches.json", False),
    ("lite-v8",   "3a4d267d9fcea50794f48d38d3e1360d", "data/lite-v8_matches.json", False),
    ("krypton",   "0f2c23daf04aaf650576575d2482b940", "data/krypton_matches.json", False),
    ("k4v2",      "aaa02959b15d57e4e28f80d665ae3482", "data/k4v2_matches.json", False),
    ("fable",     "2b4b44fd148fa7fdaa53bb0d110ae705", "data/fable_matches.json", False),
    ("m9xj570",   "28c19bba51a260bbae29ea5fba59951c", "data/m9xj570_matches.json", True),
]

def ids(ms):
    return {(m.get("id"), m.get("game")) for m in ms}

env = dict(os.environ, LADDER_BASE="https://117.72.210.134")
report = []
for label, sid, path, frozen in TARGETS:
    out = f"{WORK}/{label}_matches_117.json"
    r = subprocess.run(["python3", "ladder.py", "matches", sid, out],
                       cwd=BASE, env=env, capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        report.append(f"{label}: PULL FAILED rc={r.returncode} {r.stderr.strip()[-200:]}")
        continue
    new = json.load(open(out))
    new_ids = ids(new)
    wins = sum(1 for m in new if m.get("winner") == "a" or m.get("winner") == 0)
    # winner field semantics vary; count wins by matching name instead
    cur = gh_api.get_file(OWNER, REPO, path, ref="main")
    if cur is None:
        report.append(f"{label}: warehouse file MISSING ({len(new)} pulled)")
        continue
    import base64
    old = json.loads(base64.b64decode(cur["content"]).decode())
    old_ids = ids(old)
    sha = cur["sha"]
    if frozen:
        report.append(f"{label}: FROZEN pull={len(new)} ware={len(old)} sha={sha[:12]} "
                      f"{'UNCHANGED' if sha.startswith('9d69b8472d70') else 'SHA CHANGED!'}")
        continue
    if new_ids == old_ids:
        report.append(f"{label}: no change ({len(new)} games, id set identical)")
        continue
    added = new_ids - old_ids
    removed = old_ids - new_ids
    content = open(out, "rb").read()
    put = gh_api.put_file(OWNER, REPO, path, content,
                          f"newface-watch 07:26: {label} matches {len(old)}->{len(new)}",
                          sha=sha, branch="main")
    back = gh_api.get_file(OWNER, REPO, path, ref="main")
    back_n = len(json.loads(base64.b64decode(back["content"]).decode())) if back else -1
    report.append(f"{label}: {len(old)}->{len(new)} (+{len(added)} -{len(removed)}) "
                  f"commit={put.get('commit',{}).get('sha','?')[:12]} readback={back_n} "
                  f"sha={back['sha'][:12] if back else '?'} "
                  f"{'OK' if back_n == len(new) else 'READBACK MISMATCH!'}")

print("\n".join(report))
json.dump(report, open(f"{WORK}/refresh_report.json", "w"), ensure_ascii=False, indent=1)

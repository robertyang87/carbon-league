#!/usr/bin/env python3
"""adj2 league identity assertion (PREREG §8): every file-bearing key's path
exists on this machine and its md5 matches the pinned member table.
Runs ON L20 against /root/carbon/data/rl3/ppou2_adj2/league.json."""
import hashlib
import json
import os
import sys

LEAGUE = "/root/carbon/data/rl3/ppou2_adj2/league.json"
EXPECT = {
    "cnet2_m3s2r1L.npz": "b6db08c9",
    "cnet2_m3s2g1L.npz": "975c518c",
    "cnet2_m7bsr1L.npz": "0835830f",
    "cnet2_m9xj260wr1_fromstudio.npz": "f0f81835",
    "cnet2_m9xj260r1.npz": "afc7aedc",
    "cnet2_m6r1L.npz": "0cdb80cc",
    "cnet2_m7bsv2r1.npz": "84254f2a",
    "cnet2_m9xj260g1.npz": "c9f10637",
    "cnet2_m7bsv2g1.npz": "304a5041",
    "cnet2_caragor1_ep19.npz": "82027e53",
    "cnet2_m3w5L.npz": "82fde46b",
    "snap_10.npz": "8b338a93",
    "snap_20.npz": "43ec2a25",
    "snap_30.npz": "21082c47",
    "snap_40.npz": "72e2338f",
    "snap_50.npz": "ed604ac8",
}


def md5_of(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


lg = json.load(open(LEAGUE))
raw = open(LEAGUE, "rb").read()
print(f"league file md5: {hashlib.md5(raw).hexdigest()}  keys: {len(lg)}")
fails = 0
for key, v in lg.items():
    if key == "SELF":
        print(f"SELF prior={v['prior']} (no file)")
        continue
    assert key.startswith("SNAP:cnet2:"), key
    path = key[len("SNAP:cnet2:"):].split("#")[0]
    base = os.path.basename(path)
    if not os.path.exists(path):
        print(f"FAIL {base}: MISSING {path}")
        fails += 1
        continue
    got = md5_of(path)[:8]
    want = EXPECT[base]
    ok = got == want
    print(f"{'OK  ' if ok else 'FAIL'} {base}: md5 {got} (want {want}) prior={v['prior']}")
    fails += 0 if ok else 1
total = sum(v["prior"] for v in lg.values())
print(f"total prior: {total:.2f}")
print("ASSERTION:", "ALL OK" if fails == 0 else f"{fails} FAILURES")
sys.exit(1 if fails else 0)

#!/usr/bin/env python3
import os, sys, hashlib, urllib.request, urllib.error
sys.path.insert(0, "/opt/muse/skills/custom-connector-setup/bin")
from dynamic_credentials import add_surrogate_to_request

DST = os.path.expanduser("~/workspace/carbon-league-repo")
API = "https://api.github.com/repos/robertyang87/carbon-league"

def blob_sha(content):
    return hashlib.sha1(b"blob " + str(len(content)).encode() + b"\x00" + content).hexdigest()

samples = [
    "README.md",
    "MANIFEST.md",
    "code/base_v53.py",
    "code/distilled_v1_weights.npz",
    "code/caseb_rl_ckpt/aux_seg2_zero.npz",
    "data/weights/pred_data_deepseek.npz",
    "data/traces/INDEX.md",
    "reports/casec_report.md",
]
# add 3 arbitrary trace files
troot = os.path.join(DST, "data/traces")
tnames = sorted(f for f in os.listdir(troot) if f.endswith(".json.gz"))[:3]
samples += [f"data/traces/{t}" for t in tnames]

for rel in samples:
    with open(os.path.join(DST, rel), "rb") as f:
        sha = blob_sha(f.read())
    r = urllib.request.Request(f"{API}/git/blobs/{sha}", method="GET",
        headers={"Accept": "application/vnd.github+json", "User-Agent": "muse-agent"})
    add_surrogate_to_request(r, "custom.github", allowed_hosts=["api.github.com"])
    try:
        with urllib.request.urlopen(r, timeout=60) as resp:
            print(f"OK   {resp.status} {rel}")
    except urllib.error.HTTPError as e:
        print(f"MISS {e.code} {rel}")

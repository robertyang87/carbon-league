"""Three-way md5 check for the probe per-game lines:
machine file vs local snapshot vs sync/h29 archive readback.
"""
import base64
import hashlib
import json
import os
import subprocess
import sys

sys.path.insert(0, "/home/hatch/workspace")
import gh_api

SNAP = "/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/exam_snapshots/h29"
HOST = "root@100.65.8.27"
SSH_OPTS = [
    "-i", "/home/hatch/.ssh/id_ed25519",
    "-o", "IdentitiesOnly=yes",
    "-o", "UserKnownHostsFile=/home/hatch/.ssh/known_hosts",
    "-o", "GlobalKnownHostsFile=/dev/null",
    "-o", "ProxyCommand=nc -X connect -x hatch-egress-proxy:3130 %h %p",
]

rc = subprocess.run(
    ["timeout", "-k", "30", "60", "ssh"] + SSH_OPTS +
    [HOST, "md5sum /root/carbon/data/bigeval_live/probe_it95_g1.jsonl /root/carbon/data/bigeval_live/probe_base_g1.jsonl"],
    capture_output=True, text=True, timeout=120)
assert rc.returncode == 0, rc.stderr
machine = {}
for ln in rc.stdout.strip().splitlines():
    md5, path = ln.split()
    machine[os.path.basename(path)] = md5

out = {}
for tag in ("probe_it95_g1", "probe_base_g1"):
    local_raw = open(os.path.join(SNAP, f"{tag}.jsonl"), "rb").read()
    local_md5 = hashlib.md5(local_raw).hexdigest()
    back = gh_api.get_file("robertyang87", "carbon-league", f"synced/h29/bigeval_live/{tag}.jsonl", ref="sync/h29")
    back_raw = base64.b64decode("".join(back["content"].split()))
    back_md5 = hashlib.md5(back_raw).hexdigest()
    out[tag] = {
        "machine_md5": machine[f"{tag}.jsonl"],
        "local_md5": local_md5,
        "archive_md5": back_md5,
        "archive_sha": back["sha"],
        "lines": len([l for l in local_raw.decode().splitlines() if l.strip()]),
        "match": machine[f"{tag}.jsonl"] == local_md5 == back_md5,
    }
    print(json.dumps({tag: out[tag]}, indent=2))
    assert out[tag]["match"], f"{tag} md5 mismatch"
print("ARCHIVE VERIFY OK")
with open(os.path.join(SNAP, "probe-archive.json"), "w") as f:
    json.dump(out, f, indent=2)

"""Probe smoke verification: pull 2-line smoke files off spot-29, check
structure, push to sync/h29 as a snapshot-path liveness check, read back
and compare md5. Tags: probe_smoke / probebase_smoke, seeds 990000-990001.
"""
import hashlib
import json
import os
import subprocess
import sys

sys.path.insert(0, "/home/hatch/workspace")
import gh_api

HERE = "/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/exam_snapshots/h29"
os.makedirs(HERE, exist_ok=True)
HOST = "root@100.65.8.27"
SSH_OPTS = [
    "-i", "/home/hatch/.ssh/id_ed25519",
    "-o", "IdentitiesOnly=yes",
    "-o", "UserKnownHostsFile=/home/hatch/.ssh/known_hosts",
    "-o", "GlobalKnownHostsFile=/dev/null",
    "-o", "ProxyCommand=nc -X connect -x hatch-egress-proxy:3130 %h %p",
]


def pull(tag):
    local = os.path.join(HERE, f"{tag}.jsonl")
    tmp = local + ".part"
    rc = subprocess.run(
        ["timeout", "-k", "30", "120", "scp"] + SSH_OPTS +
        [f"{HOST}:/root/carbon/data/bigeval_live/{tag}.jsonl", tmp],
        capture_output=True, timeout=180)
    assert rc.returncode == 0 and os.path.exists(tmp) and os.path.getsize(tmp) > 0, f"scp {tag} failed rc={rc.returncode}"
    os.replace(tmp, local)
    return local


for tag in ("probe_smoke", "probebase_smoke"):
    local = pull(tag)
    raw = open(local, "rb").read()
    rows = [json.loads(ln) for ln in raw.decode().splitlines() if ln.strip()]
    assert len(rows) == 2, f"{tag}: {len(rows)} rows, want 2"
    assert sorted(r["seed"] for r in rows) == [990000, 990001], f"{tag}: seeds"
    assert all(r["err"] is None for r in rows), f"{tag}: err not null"
    assert all(isinstance(r["u"], float) and "ah" in r and "bh" in r and "hh" in r for r in rows), f"{tag}: missing u/ah/bh/hh"
    md5 = hashlib.md5(raw).hexdigest()
    path = f"synced/h29/bigeval_live/{tag}.jsonl"
    try:
        cur = gh_api.get_file("robertyang87", "carbon-league", path, ref="sync/h29")
        sha = cur["sha"]
    except Exception:
        sha = None
    r = gh_api.put_file("robertyang87", "carbon-league", path, raw,
                        f"smoke {tag} 2 lines (probe snaploop liveness)", sha=sha, branch="sync/h29")
    commit = r["commit"]["sha"]
    back = gh_api.get_file("robertyang87", "carbon-league", path, ref="sync/h29")
    import base64
    back_raw = base64.b64decode("".join(back["content"].split()))
    back_md5 = hashlib.md5(back_raw).hexdigest()
    print(f"{tag}: rows=2 md5={md5} push_commit={commit[:8]} readback_md5={back_md5} MATCH={back_md5 == md5}")
    assert back_md5 == md5
print("SMOKE VERIFY OK")

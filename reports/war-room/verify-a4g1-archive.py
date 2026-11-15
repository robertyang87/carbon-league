import base64
import hashlib
import json
import math
import os
import sys

sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER = "robertyang87"
REPO = "carbon-league"
SNAP = "/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/exam_snapshots/h29"


def fetch(path):
    meta = gh_api.get_file(OWNER, REPO, path, ref="sync/h29")
    if meta.get("content"):
        return base64.b64decode(meta["content"]), meta["sha"]
    blob = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/git/blobs/{meta['sha']}")
    return base64.b64decode(blob["content"]), meta["sha"]


def u_of(margin):
    return 0.5 * (1.0 if margin > 0 else 0.0) + 0.5 / (1.0 + math.exp(-margin / 1000.0))


out = {}
for tag in ["a4g1_a4_m2", "a4g1_base_m2"]:
    local = open(os.path.join(SNAP, f"{tag}.jsonl"), "rb").read()
    remote, sha = fetch(f"synced/h29/bigeval_live/{tag}.jsonl")
    rows = [json.loads(ln) for ln in local.decode().splitlines() if ln.strip()]
    seeds = [r["seed"] for r in rows]
    errs = sum(1 for r in rows if r.get("err") is not None)
    wins = sum(1 for r in rows if r.get("win"))
    mean = sum(r["margin"] for r in rows) / len(rows)
    max_u_dev = max(abs(r["u"] - u_of(r["margin"])) for r in rows)
    max_m_dev = max(abs(r["margin"] - (r["sa"] - r["sb"])) for r in rows)
    out[tag] = {
        "local_md5": hashlib.md5(local).hexdigest(),
        "remote_md5": hashlib.md5(remote).hexdigest(),
        "byte_match": local == remote,
        "blob_sha": sha,
        "lines": len(rows),
        "seed_min": min(seeds), "seed_max": max(seeds),
        "seeds_unique": len(set(seeds)) == len(seeds),
        "seed_set_ok": set(seeds) == set(range(996000, 996200)),
        "err_rows": errs,
        "wins": wins,
        "mean_margin": round(mean, 2),
        "max_u_recompute_dev": max_u_dev,
        "max_margin_recompute_dev": max_m_dev,
        "ah_set": sorted(set(r["ah"] for r in rows)),
        "bh_set": sorted(set(r["bh"] for r in rows)),
        "hh_set": sorted(set(r["hh"] for r in rows)),
    }
    print(json.dumps({tag: out[tag]}, indent=2))
with open(os.path.join(SNAP, "a4g1-archive-verify.json"), "w") as f:
    json.dump(out, f, indent=2)

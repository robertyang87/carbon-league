"""Select old, physically distinct tree-training replays using map identity alone."""
from __future__ import annotations

import gzip
import hashlib
import json
import io
import os
import tarfile
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SNAPSHOT = "b1f3d98689635990772d07a37ed0dc20c0cea15b"
TOP = "reports/sapling_1200k/all_top5"
DERIVATIVE = "reports/master3s2_derivative"
LADDER = DERIVATIVE + "/ladder"
NAMESPACE = b"ds-pro-tree-estimated-payback-v1/"
CACHE = Path("/workspace/sapling-all-top5-20261119/replays")


def git(*args):
    return subprocess.check_output(["git", "-C", ROOT, *args], env={**os.environ, "GIT_NO_LAZY_FETCH": "1"})


def blob(path):
    return git("show", f"{SNAPSHOT}:{path}")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def register():
    paths = git("ls-tree", "--name-only", "-r", SNAPSHOT).decode().splitlines()
    tables = [p for p in paths if re.fullmatch(TOP + r"/(?:appendix_[0-9a-f]+/)?validation_maps\.jsonl\.gz", p)]
    pool = {}
    for p in sorted(tables):
        for line in gzip.decompress(blob(p)).splitlines():
            r = json.loads(line)
            key = r["initial_terrain_u8_sha256"]
            if key in pool:
                raise ValueError(f"Repeated public map identity {key}")
            pool[key] = (p, r)
    exposure_path = LADDER + "/exposed_ladder_maps.json"
    exposure = json.loads(blob(exposure_path))
    excluded_terrain = {r["opening_carbon_bytes_sha256"] for r in exposure}
    excluded_seed = {r[k] for r in exposure for k in ("scorecard_seed", "replay_random_seed")}
    excluded_id = {r["match_id"] for r in exposure}
    # Replay custody can precede the ladder exclusion receipt. Read only opening
    # identity and replay ID here. Match rewards, orders and tree outcomes do not
    # participate in registration.
    live_paths = [p for p in paths if p.startswith(LADDER+"/replays/") and p.endswith(".json.gz")]
    for p in live_paths:
        obj = json.loads(gzip.decompress(blob(p)))
        excluded_terrain.add(sha(bytes(obj["steps"][0][0]["observation"]["full_carbon"])))
        excluded_seed.add(obj["configuration"]["randomSeed"])
        excluded_id.add(Path(p).name.split("-")[0])
    clearance_paths = [p for p in paths if p.startswith(DERIVATIVE+"/") and p.endswith("/clearance.json")]
    protected_ranges = set()
    for p in clearance_paths:
        cleared = json.loads(blob(p))
        excluded_seed.update(map(int, cleared["by_seed"]))
        excluded_terrain.update(cleared["by_seed"].values())
        protected_ranges.add(tuple(cleared["formal_seed_half_open"]))
    # Previous original-policy and derivative training, validation and held
    # protocol cards can also carry a literal terrain ID and game seed.
    protocols = [p for p in paths if p.startswith(DERIVATIVE+"/") and p.endswith("protocol.json")]
    for p in protocols:
        if not p.startswith(HERE.relative_to(ROOT).as_posix()+"/"):
            text = blob(p).decode()
            excluded_terrain.update(re.findall(r'"([0-9a-f]{64})"', text))
            excluded_seed.update(int(n) for n in re.findall(r'"(?:seed|map_seed)"\s*:\s*([0-9]+)', text))
    def eligible(k):
        row = pool[k][1]
        return (k not in excluded_terrain and row["seed"] not in excluded_seed
                and row["id"].split("-")[0] not in excluded_id
                and not any(lo <= row["seed"] < hi for lo, hi in protected_ranges))
    allowed = sorted((k for k in pool if eligible(k)), key=lambda k: sha(NAMESPACE + bytes.fromhex(k)))
    selected = allowed[:32]
    assert len(selected) == len(set(selected)) == 32
    manifests = {}
    tickets = []
    for ordinal, terrain in enumerate(selected):
        registry, row = pool[terrain]
        parent = registry.rsplit("/",1)[0]
        if parent not in manifests:
            manifests[parent] = {g["id"]:g for g in json.loads(blob(parent+"/manifest.json"))["games"]}
        identity = manifests[parent][row["id"]]
        assert identity["seed"] == row["seed"] and identity["frame_verified"]
        member = identity["archive_location"]
        if "archive" in member:
            assert member["archive"] in paths and member["member"] == identity["path"]
        else:
            assert member["repo_path"] in paths
            assert git("rev-parse", SNAPSHOT+":"+member["repo_path"]).decode().strip()==member["git_blob"]
        cached = CACHE / (row["id"]+".json.gz")
        if cached.exists():
            raw = cached.read_bytes()
        elif "repo_path" in member:
            raw = blob(member["repo_path"])
        else:
            with tarfile.open(fileobj=io.BytesIO(blob(member["archive"]))) as archive:
                raw = archive.extractfile(member["member"]).read()
        assert sha(raw) == identity["sha256"]
        tickets.append(dict(ordinal=ordinal, split="train" if ordinal<24 else "held", id=row["id"], seed=row["seed"],
                    terrain=terrain, compressed_sha256=identity["sha256"], source=member, cache_member=identity["path"], identity_registry=registry))
    return dict(source_snapshot=SNAPSHOT, name="tree-payback-24-training-8-holdout", hash_namespace=NAMESPACE.decode(),
                selection="Ascending SHA256(namespace UTF-8 + raw 32-byte opening SHA256); 0..23 fit, 24..31 held. Each game both physical seats. Whole map assigned together.",
                total_unique_scored_maps=len(pool), source_registries=len(tables), eligible_maps=len(allowed),
                exclusions=dict(custodied_ladder_replays=len(live_paths), excluded_ladder_match_ids=sorted(excluded_id),
                                explicit_formal_ranges=sorted(protected_ranges), derivative_clearance_sources=clearance_paths),
                replay_contract="The source is either a direct repo_path/Git blob or a Git tar archive/member at the frozen revision. Verify compressed_sha256. Source-snapshot context.py Game.snapshot(t, seat) projects shared public keys but reads private carbon from exactly that seat and masks hidden squares with None. Own collectors see own square; own planters see four wrapped neighbours only. Both seat 0/1 supported. Keep actions at t+1 and referee full_carbon outside model inputs. Tree labels are excluded from this register.",
                tickets=tickets)


if __name__ == "__main__":
    output=HERE/"maps.json"
    raw=(json.dumps(register(), indent=2, ensure_ascii=False, allow_nan=False)+"\n").encode()
    if output.exists() and output.read_bytes()!=raw:
        raise ValueError("Registration differs from frozen file")
    output.write_bytes(raw)
    print(sha(raw))

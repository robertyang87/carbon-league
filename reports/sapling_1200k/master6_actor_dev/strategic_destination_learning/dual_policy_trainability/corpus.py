"""Recover two short training-tape windows in the real physical expert's visible frame."""
import copy
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
EXACT = HERE.parents[1] / "master3s2_policy_distill_dev/postsearch/exact"
ROOT = next(p for p in HERE.parents if (p / ".git").exists())
sys.path.insert(0, str(ROOT / "code"))
import cnet2_feat as F
from fastsim import FastCarbon

GIDS = (0, 2)
SIGNED_MANIFEST = "0b9c563f84f8aef9e3dbf7db15222d5915b62b699b842b55e6f080437e7c58b0"
OBS_FIELDS = ("maps", "scal", "tok_pos", "tok_kind", "tok_mask", "tok_cargo", "hist_cell", "hist_cargo", "hist_move")


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def checked(path, digest):
    raw = path.read_bytes() if path.is_file() else subprocess.check_output(
        ["git", "-C", str(ROOT), "show", "HEAD:" + str(path.relative_to(ROOT))])
    require(hashlib.sha256(raw).hexdigest() == digest, f"data bytes changed {path}")
    return raw


def json_file(path, digest):
    data = checked(path, digest)
    return json.loads(gzip.decompress(data) if path.suffix == ".gz" else data)


def sources():
    m = json_file(EXACT / "manifest.json", SIGNED_MANIFEST)
    protocol = json_file(EXACT / "protocol.json", m["protocol"])
    checked(EXACT / "build.py", m["builder"])
    cards = [c for c in m["cards"] if c["gid"] in GIDS]
    require([c["gid"] for c in cards] == [0, 2] and [c["seat"] for c in cards] == [1, 0],
            "incorrect physical match or expert seat")
    require(len({c["terrain"] for c in cards}) == 2 and all(
        c["split"] == "fit" and c["seed"] in protocol["fit"] and c["steps"] >= 32 for c in cards),
        "tape is not part of the frozen training partition")
    return dict(manifest_sha256=SIGNED_MANIFEST, protocol_sha256=m["protocol"], builder_sha256=m["builder"],
                original_sha256=m["master"], cards=cards,
                implementation={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in (ROOT/"code"/"cnet2_feat.py", ROOT/"code"/"fastsim.py")})


def physical(root, labels):
    """Convert an OWN-slot canonical action vector to a sparse physical joint command."""
    require(len(labels) == len(root["uids"]) == F.OWN, "bad unit layout")
    command = {}
    for slot, uid in enumerate(root["uids"]):
        if uid is None:
            continue
        action = int(labels[slot])
        if slot < 10:
            require(0 <= action < len(F.MOVES), "invalid active worker action")
            name = F.MOVES[F.CANON_MOVE[action] if root["seat"] else action]
        else:
            require(0 <= action < len(F.BASE_ACTS), "invalid active base action")
            name = F.BASE_ACTS[action]
        if action:
            command[uid] = name
    return command


def clean(order):
    return {uid: a for uid, a in order.items() if a and a != "NONE"}


def roots():
    registry = sources()
    selected = []
    for c in registry["cards"]:
        path = ROOT / c["source"]
        source = json_file(path, c["source_sha256"])
        response = json_file(ROOT / c["response"], c["response_sha256"])
        seat = c["seat"]
        first = c["first_step"]
        require(source["seed"] == c["seed"] and source["candidate_seat"] == 1-seat and not source["errors"],
                "wrong source map, physical seat, or failed source")
        require(response["case"]["source"] == c["source"] and response["case"]["t"] == first-1 and
                response["case"]["seat"] == 1-seat and response["case"]["seed"] == c["seed"] and
                response["binding"]["source"] == c["source_sha256"] and response["arm"] == c["arm"] and
                response["binding"]["original"] == registry["original_sha256"] and
                response["statuses"] == ["DONE", "DONE"] and len(response["tape"]) == len(response["calls"])+1,
                "mismatched signed branch")
        require(clean(response["tape"][0][seat]) == clean(source["actions"][first-1][seat]),
                "incorrect source boundary action")
        sim = FastCarbon(source["game_configuration"]).reset(seed=c["seed"])
        require(hashlib.sha256(np.asarray(sim.carbon, np.uint8).tobytes()).hexdigest() ==
                source["initial_terrain_u8_sha256"] == c["terrain"], "mismatched physical opening")
        feature = F.Feat(seat)
        tape = source["actions"][:first-1] + response["tape"]
        with np.load(io.BytesIO(checked(EXACT / c["file"], c["sha256"])), allow_pickle=False) as src:
            keys = (*OBS_FIELDS, "gid", "step", "pre_act", "src_act", "y", "legal", "uf", "mf",
                    "source_uid_number", "eligible", "unit_changed", "team_changed", "prefix_repaired")
            shard = {key: src[key][:32] for key in keys}
        for step, pair in enumerate(tape):
            obs = sim.obs(seat)
            require(obs["player"] == seat and obs["step"] == step and "full_carbon" not in obs, "wrong public view")
            if step < first + 32:
                x = feature.observe(obs)
            if first <= step < first + 32:
                j = step-first
                require(j < len(shard["step"]) and step == shard["step"][j] and c["gid"] == shard["gid"][j],
                        "shard step or gid mismatch")
                clock = response["calls"][j]
                require(clock["step"] == step and bool(clock["changed"]) == bool(shard["team_changed"][j]),
                        "source search changed flag mismatch")
                for key in OBS_FIELDS:
                    archived = shard[key][j]
                    require(np.array_equal(np.asarray(x[key], dtype=archived.dtype), archived),
                            f"public {key} at {c['gid']}/{step} disagrees with archive")
                actual = feature.labels(pair[seat])
                require(np.array_equal(actual, shard["src_act"][j]) and np.array_equal(actual, shard["y"][j])
                        and not shard["prefix_repaired"][j].any(), "non-native or repaired expert prefix")
                uid = tuple(feature.own_uid)
                numbers = [int(u.rsplit("-", 1)[1]) if u else -1 for u in uid]
                require(np.array_equal(numbers, shard["source_uid_number"][j]), "changed physical unit identity")
                prefix = F.Prefix(x, feature.aux)
                eligible = np.zeros(F.OWN, bool)
                for slot in np.flatnonzero(actual >= 0):
                    uf, mf, legal = prefix.feats(slot)
                    require(np.array_equal(legal, shard["legal"][j,slot]) and legal[actual[slot]]
                            and np.array_equal(uf.astype(shard["uf"].dtype), shard["uf"][j,slot])
                            and np.array_equal(mf.astype(shard["mf"].dtype), shard["mf"][j,slot]),
                            "teacher-prefix feasibility or physical feature mismatch")
                    eligible[slot] = legal.sum() > 1
                    prefix.advance(slot, int(actual[slot]))
                changed = actual != shard["pre_act"][j]
                require(np.array_equal(eligible, shard["eligible"][j]) and
                        np.array_equal(changed, shard["unit_changed"][j]) and changed.any() == clock["changed"],
                        "stored changed/eligible unit tags disagree with source")
                root = dict(gid=c["gid"], seed=c["seed"], seat=seat, step=step, split="fit" if j < 16 else "probe",
                            x=copy.deepcopy(x), aux=copy.deepcopy(feature.aux), actual=actual.copy(),
                            changed=changed.copy(), eligible=eligible, uids=uid)
                require(physical(root, actual) == clean(pair[seat]), "physical output failed to round trip")
                selected.append(root)
            sim.step(*pair)
        require(np.allclose(sim.cash, response["score"], rtol=0, atol=1.e-5),
                "complete official branch cash disagrees with its signed receipt")
    require(len(selected) == 64, "required all temporal windows on both physical maps")
    return selected

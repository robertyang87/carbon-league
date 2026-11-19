"""Build a small, game-held-out BC dataset from archived Master5s/Master6 ladder play."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

import cnet2_build as builder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reports/master_family/deep_review"))
from recover_public import key_of, validate

REVISION = "499a8555c16d21eb56a78de1b8db1a1362b58f60"
MODELS = ("Master5s", "Master6")
STYLE_SLOT = 2
SPLIT_SEED = "sapling-elite-full-v1-20261119"


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args])


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(tmp, path)


def build(out, revision):
    if any(key.startswith("CNET2_") for key in os.environ):
        raise ValueError("unset CNET2_* environment switches to reproduce the fixed data contract")
    revision = git("rev-parse", "--verify", revision + "^{commit}").decode().strip()
    out.mkdir(parents=True, exist_ok=False)
    used_matches, used_seeds = set(), set()
    selected = []
    for model in MODELS:
        prefix = "data/" + model.lower()
        corpus = prefix + "_corpus"
        available = set(git("ls-tree", "--name-only", revision + ":" + corpus).decode().splitlines())
        scores = json.loads(git("show", revision + ":" + prefix + "_matches.json"))
        name = "claude-AlphaCarbon-" + model
        for seat in (0, 1):
            candidates = [r for r in scores if r.get("state") == "done"
                          and r["name_" + r["order"][seat]] == name
                          and key_of(r) + ".json.gz" in available]
            candidates.sort(key=lambda r: digest((SPLIT_SEED + "|" + key_of(r)).encode()))
            count = 0
            for row in candidates:
                if row["id"] in used_matches:
                    continue
                key = key_of(row)
                src = corpus + "/" + key + ".json.gz"
                data = git("show", revision + ":" + src)
                env = json.loads(gzip.decompress(data))
                check = validate(env, row)
                seed = check["random_seed"]
                if not isinstance(seed, int) or seed in used_seeds or 20291101 <= seed <= 20292100:
                    continue
                if check["statuses"] != ["DONE", "DONE"]:
                    continue
                if check["seat_model_names"][seat] != name:
                    raise ValueError("strategy identity disagrees with replay")
                selected.append(dict(id=key, split="val" if count == 0 else "train",
                                     shard=key + ".npz", model=model, seat=seat,
                                     source=src, source_sha256=digest(data), seed=seed,
                                     end_step=check["last_step"], rewards=env["rewards"]))
                used_matches.add(row["id"])
                used_seeds.add(seed)
                count += 1
                if count == 4:
                    break
            if count != 4:
                raise ValueError("insufficient distinct games for " + name + " seat " + str(seat))
    manifest = dict(schema="sapling-elite-bc/1", revision=revision, selection_seed=SPLIT_SEED,
                    style_slot=STYLE_SLOT, games=selected)
    write_json(out / "selection.json", manifest)
    for gid, item in enumerate(selected):
        data = git("show", revision + ":" + item["source"])
        if digest(data) != item["source_sha256"]:
            raise ValueError("source digest changed")
        seat = item["seat"]
        margin = item["rewards"][seat] - item["rewards"][1-seat]
        path = out / item["shard"]
        with tempfile.NamedTemporaryFile(dir=out, suffix=".json.gz") as replay:
            replay.write(data)
            replay.flush()
            _, rows, error = builder.build_game((replay.name, [seat], gid, [STYLE_SLOT], [margin], str(path)))
        if error or rows != item["end_step"] or not rows:
            raise ValueError(f"failed or partial replay {item['id']}: {rows} rows, {error}")
        with np.load(path, allow_pickle=False) as arrays:
            if not np.array_equal(arrays["step"], np.arange(rows)) or not (arrays["teacher"] == STYLE_SLOT).all():
                raise ValueError("a replay lost its history, endgame, or style slot")
            y = arrays["y"]
            active = y >= 0
            target_legal = np.take_along_axis(arrays["legal"], y.clip(min=0)[:, :, None], axis=2)[:, :, 0]
            if not target_legal[active].all():
                raise ValueError("an expert target is outside the legal mask")
            item.update(rows=rows, labeled_actions=int(active.sum()), sha256=digest(path.read_bytes()))
        print(f"{item['split']} {item['model']} seat {seat} {item['id']} {rows} rows", flush=True)
    write_json(out / "manifest.json", manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path, help="new output directory outside the checkout")
    parser.add_argument("--revision", default=REVISION, help="frozen Git source revision")
    args = parser.parse_args()
    build(args.out, args.revision)

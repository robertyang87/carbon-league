"""Identify the played opponent bytes. Only compile or build old packages, never call an agent."""
import collections
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile

ACTOR = "reports/sapling_1200k/master6_actor_dev"
FIT = ACTOR + "/terminal_return_fit"
OPENING = ACTOR + "/blended_actor_dev/opening_training"
ONPOLICY = ACTOR + "/cargo_tree_action_learning_dev/onpolicy_native"


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def lineups(snapshot, games):
    seen = collections.defaultdict(list)
    for row in games:
        seen[row["opponent_pack_sha256"]].append(row)
    paths = collections.defaultdict(list)
    for path, (obj, size) in snapshot.tree.items():
        if path.endswith(".py") and size >= 175000:
            paths[obj].append(path)
    result = {}
    for obj, aliases in sorted(paths.items()):
        data = snapshot.bytes(aliases[0])
        digest = hashlib.sha256(data).hexdigest()
        if digest in seen:
            result[digest] = {"identification": "frozen_git_package_bytes", "paths": sorted(aliases)}

    def rebuilt(data, recipe, arm, inputs):
        digest = hashlib.sha256(data).hexdigest()
        if digest not in seen:
            raise ValueError(("rebuilt opponent differs from played bytes", recipe, arm, digest))
        result[digest] = {"identification": "rebuilt_frozen_package_bytes", "builder": recipe, "arm": arm,
                          "inputs": {p: snapshot.prove(p) for p in sorted(set(inputs) | {recipe})}}

    build_path = FIT + "/build.py"
    recipe = load_module(snapshot.local(build_path), "expansion_return_builder")
    check = json.loads(snapshot.bytes(FIT + "/initial_checkpoint.json"))
    for arm in ("plus", "minus"):
        config = check["arms"][arm]
        source = recipe.source(config["theta"])
        if hashlib.sha256(source).hexdigest() != config["pack_sha256"]:
            raise ValueError("frozen terminal-return package mismatch")
        rebuilt(source, build_path, arm, [FIT + "/" + f for f in ("initial_checkpoint.json", "runtime.py", "protocol.json")]
                + [str(recipe.PARENT.relative_to(snapshot.repo))])

    # Invoke the archived packager without invoking its CLI or its native match driver.
    multimap_path = FIT + "/multimap/model.py"
    sys.path.insert(0, str(snapshot.repo / FIT / "multimap"))
    model = load_module(snapshot.local(multimap_path), "expansion_multimap_builder")
    for arm, sign in (("plus", 1), ("minus", -1)):
        rebuilt(model.package(sign * model.PLAN["epsilon"]), multimap_path, arm,
                [FIT + "/multimap/protocol.json", str(model.V1.relative_to(snapshot.repo))])

    with tempfile.TemporaryDirectory(prefix="native-corpus-opponents-") as out:
        scratch = Path(out)
        opening_path = OPENING + "/build.py"
        for name in ("protocol.json", "fit.json", "opening_weights.npz"):
            shutil.copyfile(snapshot.local(OPENING + "/" + name), scratch / name)
        opening = load_module(snapshot.local(opening_path), "expansion_opening_builder")
        opening.HERE = scratch
        destination = scratch / "opening.py"
        release = opening.build(destination)
        if release != json.loads(snapshot.bytes(OPENING + "/release.json")):
            raise ValueError("opening package differs from archived release")
        rebuilt(destination.read_bytes(), opening_path, "opening_adapter",
                [str(opening.BASE.relative_to(snapshot.repo))] +
                [OPENING + "/" + n for n in ("protocol.json", "fit.json", "opening_weights.npz", "release.json")])

        onpolicy_path = ONPOLICY + "/build.py"
        sampler = load_module(snapshot.local(onpolicy_path), "expansion_sampling_builder")
        shutil.copyfile(snapshot.local(ONPOLICY + "/sampler.py"), scratch / "sampler.py")
        sampler.HERE = scratch
        rebuilt(sampler.build().read_bytes(), onpolicy_path, "onpolicy_training_opponent",
                [ONPOLICY + "/sampler.py", str(sampler.PARENT.relative_to(snapshot.repo))])

    if set(result) != set(seen):
        raise ValueError(("unresolved played opponent packages", set(seen) - set(result)))
    for digest, rows in seen.items():
        result[digest].update({"weight_identities": sorted({r["opponent_weight_identity"] for r in rows}),
                               "raw_files": len(rows), "seeds": sorted({r["seed"] for r in rows}),
                               "match_directories": sorted({str(Path(r["path"]).parent) for r in rows})})
    return result

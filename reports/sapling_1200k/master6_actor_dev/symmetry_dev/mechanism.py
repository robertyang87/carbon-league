"""Compare decoded legal physical commands on exposed, independently replayed native trajectories."""
from collections import Counter
import gzip
import importlib.util
import json
from pathlib import Path
import sys
from time import perf_counter

from build import HERE, ROOT, build, digest, seal

sys.path.insert(0, str(ROOT / "code"))
from fastsim import FastCarbon

STEPS = [*range(0, 300, 30), 298]


def unpack():
    spec = importlib.util.spec_from_file_location("sapling_four_frame_frozen", HERE / "candidate.py")
    package = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(package)
    return package


def compare(package, obs, config, root):
    if obs["step"] == 0:
        package.agent(obs, config)
        package._FEAT = sys.modules["cnet2_feat"].Feat(obs["player"])
    x = package._FEAT.observe(obs)
    if not root:
        return None
    aux = package._FEAT.aux
    actions = package._ONE_VIEW_PARENT(x, aux)
    one = package._ensemble_actions(x, aux, groups=(0,))
    before = perf_counter()
    four = package._ensemble_actions(x, aux)
    elapsed = perf_counter() - before
    ablated = package._ensemble_actions(x, aux, student_weight=0)
    return {
        "identity_slot_errors": int((one != actions).sum()),
        "parent_worker_changes": int((four[:10] != actions[:10]).sum()),
        "student_worker_changes": int((four[:10] != ablated[:10]).sum()),
        "elapsed_seconds": elapsed,
        "actions": {k: package._FEAT.to_actions(v) for k, v in
                    [("v1", actions), ("one", one), ("four", four), ("expert_only", ablated)]}
    }


def read_tape(path):
    with gzip.open(path, "rt") as stream:
        return json.load(stream)


def main():
    release = build()
    tree = HERE.parent / "tree_response_training"
    source = json.loads((tree / "source_receipt.json").read_bytes())
    rows = []
    for seat in (0, 1):
        item = next(g for g in source["games"] if g["seed"] == 621471120 and g["seat"] == seat)
        path = ROOT / item["raw_path"]
        if digest(path) != item["raw_sha256"] or item["steps"] != 299:
            raise ValueError("source replay identity changed")
        tape = read_tape(path)
        package = unpack()
        game = FastCarbon(tape["game_configuration"]).reset(seed=621471120)
        for t, joint in enumerate(tape["actions"]):
            row = compare(package, game.obs(seat), game.config, t in STEPS)
            if row is not None:
                row.update({"step":t, "seat":seat, "source_sha256":item["raw_sha256"]})
                row["archived_v1_match"] = row["actions"]["v1"] == joint[seat]
                rows.append(row)
            game.step(*joint)
    totals = {field:sum(row[field] for row in rows) for field in
              ("identity_slot_errors", "parent_worker_changes", "student_worker_changes", "archived_v1_match")}
    passed = (len(rows) == 22 and totals["identity_slot_errors"] == 0 and totals["archived_v1_match"] == 22
              and totals["parent_worker_changes"] > 0 and totals["student_worker_changes"] > 0)
    result = {"passed":passed, **{k:release[k] for k in ("protocol_sha256","candidate_pack_sha256")},
              "source_receipt_sha256":digest(tree / "source_receipt.json"),
              "test_code_sha256":digest(Path(__file__)), "root_count":len(rows), "totals":totals,
              "max_local_seconds":max(r["elapsed_seconds"] for r in rows),"roots":rows}
    name = "mechanism.json" if passed else "mechanism_skip.json"
    seal(HERE / name, (json.dumps(result, indent=2) + "\n").encode())
    print(passed, totals, flush=True)


if __name__ == "__main__":
    main()

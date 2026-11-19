"""Summarize fixed-dose policy imitation with and without tactical loss emphasis."""
import argparse
import json
from pathlib import Path


def actionable(metric, forced):
    # In CNet2 a forced live slot is a base with only NONE legal. Its
    # prediction agrees, and masked softmax gives this target exactly zero CE.
    n = metric["slots"]
    correct = round(n * metric["agreement"])
    count = n - forced
    return dict(slots=count, agreement=(correct - forced) / count, ce=n * metric["ce"] / count)


def compact(validation):
    contexts = validation["contexts"]
    keep = ("all", "other", "attack", "defense", "cargo_defense", "tree_defense")
    result = {k: {s: contexts[k][s] for s in ("slots", "games", "agreement", "ce")} for k in keep}
    forced = validation["roles"]["base_forced_none_n"]
    result["actionable_all"] = actionable(contexts["all"], forced)
    result["actionable_other"] = actionable(contexts["other"], forced)
    result["forced_base_none"] = forced
    return result


def compare(a, b):
    same = ("parent_sha256", "dataset_manifest_sha256", "runner_sha256", "train_games", "val_games",
            "train_rows", "val_rows", "train_rows_seen", "train_games_seen", "tactical_slot_exposures", "before")
    if any(a[k] != b[k] for k in same):
        raise ValueError("the parent, dataset, exposure, or baseline metrics differ")
    settings = {k: v for k, v in a["settings"].items() if k != "tactical_factor"}
    if settings != {k: v for k, v in b["settings"].items() if k != "tactical_factor"}:
        raise ValueError("another training setting differs")
    if b["settings"]["tactical_factor"] != 1 or a["settings"]["tactical_factor"] <= 1:
        raise ValueError("expected emphasized and neutral arms respectively")
    return dict(protocol="same init, dataset, sampled batches, optimizer and final update; no checkpoint selection",
                interpretation="Teacher-forced policy prediction on held-out whole games; no gameplay claim.",
                parent_sha256=a["parent_sha256"], dataset_manifest_sha256=a["dataset_manifest_sha256"],
                training_code_sha256=a["runner_sha256"], matched_settings=settings,
                tactical_factors=[a["settings"]["tactical_factor"], 1],
                weighted_sha256=a["trained_sha256"], neutral_sha256=b["trained_sha256"],
                unique_train_rows_seen=a["train_rows_seen"], train_games_seen=a["train_games_seen"],
                train_slot_exposures=a["tactical_slot_exposures"],
                initial=compact(a["before"]["val"]), weighted=compact(a["after"]["val"]),
                neutral=compact(b["after"]["val"]))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("weighted", "neutral", "out"):
        p.add_argument("--" + key, required=True, type=Path)
    args = p.parse_args()
    result = compare(json.loads(args.weighted.read_bytes()), json.loads(args.neutral.read_bytes()))
    with args.out.open("x") as f:
        f.write(json.dumps(result, indent=2, allow_nan=False) + "\n")

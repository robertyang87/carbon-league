"""Audit the four supervision groups and separate chosen actions from observed tree events."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


def support(path):
    manifest_bytes = path.read_bytes()
    manifest = json.loads(manifest_bytes)
    totals = defaultdict(Counter)
    echo_games = []
    for g in manifest["games"]:
        counts = totals[g["split"]]
        counts["games"] += 1
        raw = path.parent / g["shard"]
        if hashlib.sha256(raw.read_bytes()).hexdigest() != g["sha256"]:
            raise ValueError("shard hash mismatch")
        with np.load(raw, allow_pickle=False) as a:
            y, actual = a["y"], a["execution_y"]
            for name in ("planter", "return", "recruit", "defense"):
                mask = a[name + "_mask"]
                for action in range(5):
                    counts[f"{name}/teacher_action_{action}"] += int((mask & (y == action)).sum())
                    counts[f"{name}/executed_action_{action}"] += int((mask & (actual == action)).sum())
            # Prefix-decoder move features for staying on the current cell:
            # columns 5/6 indicate bases and 8/9 indicate own/enemy trees.
            stay = a["mf"][:, :, 0]
            open_cell = ~stay[:, :, [5, 6, 8, 9]].any(-1)
            affordable = a["scal"][:, 13, None] > 0
            for label, eligible in {
                "public_affordable_empty_cell": a["planter_mask"] & open_cell & affordable,
                "enemy_tree_cell": a["planter_mask"] & (stay[:, :, 9] > 0),
            }.items():
                counts[label + "/opportunities"] += int(eligible.sum())
                counts[label + "/teacher_stay"] += int((eligible & (y == 0)).sum())
                counts[label + "/executed_stay"] += int((eligible & (actual == 0)).sum())
        if g["opponent"] == "Echo4 bare style 2":
            p = 1 - g["seat"]
            echo_games.append(dict(seed=g["seed"], split=g["split"], teacher_seat=p,
                                   steps=g["transitions"], new_tree_cells=g["occupied_new_tree_cells_by_seat"][p],
                                   tree_cells_captured=g["occupied_tree_cells_swapping_owner_by_seat"][p],
                                   tree_absorption=g["tree_absorption_by_seat"][p]))
    return dict(manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(), by_split=totals,
                echo4_actually_played_games=echo_games,
                interpretation="Masks overlap. Action 0 = STAY/NONE; worker 1/2/3/4 = up/right/down/left; base 1/2 = recruit collector/planter. Opportunity filters use public root affordability before either side acts. A teacher stay does not establish a successful planting or capture. The Echo4 opponent-side outcomes below come from real transitions; labels on V3's seat were never executed. Tree location additions and ownership changes compare consecutive simulator states, absorption sums simulator absorption on each step.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    with args.out.open("x") as fp:
        json.dump(support(args.manifest), fp, indent=2, allow_nan=False)
        fp.write("\n")

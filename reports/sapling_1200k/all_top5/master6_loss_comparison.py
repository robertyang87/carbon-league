"""Within-opponent, within-seat contrasts for Master6 wins and losses."""
from collections import defaultdict
from pathlib import Path
from statistics import mean
import gzip
import json

SOURCE = Path(__file__).resolve().parent / "appendix_4bfdabf5/master6_pressure.json.gz"


def main():
    with gzip.open(SOURCE, "rt") as stream:
        games = json.load(stream)["rows"]
    result = {}
    for limit in (None, 300):
        groups = defaultdict(lambda: defaultdict(list))
        for row in games:
            if row["result"] == "draw" or "100_200" not in row["windows"]:
                continue
            if limit is not None and abs(row["landmarks"]["60"]["cash_gap"]) > limit:
                continue
            groups[row["opponent"], row["seat"]][row["result"]].append(row)
        cells = []
        for (opponent, seat), arms in sorted(groups.items()):
            if "win" not in arms or "loss" not in arms:
                continue
            differences = {}
            for period in ("60_100", "100_200"):
                for field in ("bank", "tree"):
                    arm_means = {}
                    for outcome, sample in arms.items():
                        values = [r["windows"][period] for r in sample]
                        arm_means[outcome] = mean(v[0][field] - v[1][field] for v in values)
                    differences[period + "_" + field] = round(arm_means["loss"] - arm_means["win"], 5)
            cells.append({"opponent": opponent, "master6_seat": seat, "difference": differences,
                          "games": {arm: [r["id"] for r in sample] for arm, sample in arms.items()}})
        metrics = {field: {"mean": round(mean(c["difference"][field] for c in cells), 2),
                           "negative_cells": sum(c["difference"][field] < 0 for c in cells)}
                   for field in cells[0]["difference"]} if cells else {}
        result[str(limit)] = {"max_absolute_t60_cash_gap": limit, "contrast": "loss minus win, each physical-seat/opponent cell equally weighted",
                              "metrics": metrics, "cells": cells}
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

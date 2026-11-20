"""Recalculate stratified held behavior and paired public-search receipts without replaying a match."""
import gzip
import json
import math
from collections import Counter
from pathlib import Path
from statistics import fmean

HERE = Path(__file__).resolve().parent


def read(filename):
    with gzip.open(HERE / filename, "rt") as stream:
        return json.load(stream)


def loss(rows, use_advantage):
    def weight(row):
        return row["weight"] * (math.exp(1.2 * row["A"]) if use_advantage else 1.)
    return sum(weight(row)*row["nll"] for row in rows) / sum(map(weight, rows))


def by_group(units, select):
    result = {}
    for method in ("zero", "bc", "awr"):
        groups = [[r for r in units[f"{role}_{method}"] if select(r)] for role in ("worker","base")]
        result[method] = {
            "roots_of_measurement": [len({(r["seed"],r["seat"],r["t"]) for r in rs}) for rs in groups],
            "entities": [len(rs) for rs in groups],
            "nll": fmean(loss(rs, False) for rs in groups),
            "weighted_nll": fmean(loss(rs, True) for rs in groups),
            "raw_top1_hits": [sum(r["top"] for r in rs) for rs in groups],
        }
    return result


def compare_baseline(rows, mode):
    counts = Counter()
    changed_maps = set()
    strict_maps = set()
    for r in rows:
        z, *_, repeated = [v for v in r["runs"] if v["mode"] == "zero"]
        candidate = next(v for v in r["runs"] if v["mode"] == mode)
        stable = all(z[key] == repeated[key] for key in ("issued","menu","enemy","K","H","M","depths"))
        fixed_enemy = z["enemy"] == candidate["enemy"]
        work_total = all(z[k] == candidate[k] for k in ("K","H","M")) and sorted(z["depths"]) == sorted(candidate["depths"])
        work_vector = work_total and z["depths"] == candidate["depths"]
        changed = z["issued"] != candidate["issued"]
        credited = changed and stable and fixed_enemy
        fields = dict(menu=z["menu"] != candidate["menu"], root0=z["first_menu"] != candidate["first_menu"],
                      issued=changed, stable_parent=stable, enemy_fixed=fixed_enemy, same_work_total=work_total,
                      same_work_vector=work_vector, change_with_vector=credited and work_vector,
                      change_with_total=credited and work_total)
        for k,v in fields.items():
            counts[k] += int(v)
        if credited and work_total: changed_maps.add(r["seed"])
        if credited and work_vector: strict_maps.add(r["seed"])
    return dict(counts=counts,maps_equal_work=sorted(changed_maps),maps_equal_vector=sorted(strict_maps))


def main():
    rows = read("held_unit_predictions.json.gz")
    groups = {"all": by_group(rows, lambda r: True)}
    for s in range(2):
        groups[f"seat{s}"] = by_group(rows, lambda r: r["seat"] == s)
    for p in range(3):
        groups[f"phase{p}"] = by_group(rows, lambda r: r["t"] // 100 == p)
    for name,positive in [("positive",True),("nonpositive",False)]:
        groups[name] = by_group(rows, lambda r: (r["A"] > 0) == positive)
    advantages = {}
    for method in ("bc","awr"):
        gains = {}
        for sign in ("positive","nonpositive"):
            g = groups[sign]
            gains[sign] = g["zero"]["weighted_nll"] - g[method]["weighted_nll"]
        advantages[method] = gains | {"positive_minus_nonpositive":gains["positive"] - gains["nonpositive"]}
    source_rows = read("native.json.gz")["source_roots"]
    result = dict(groups=groups,positive_vs_nonpositive_logprob_gain=advantages,
                  physical={k:compare_baseline(source_rows,k) for k in ("bc","awr")})
    (HERE/"behavior_summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")


if __name__ == "__main__":
    main()

"""Score the frozen Master6 freight predictor on the exact public risk alerts."""
from collections import Counter, defaultdict
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
ACTOR = HERE.parents[2]
OLD = ACTOR / "freight_horizon"
MODEL_SHA = "376fdde5bba0ea816c597f9d2aa88d20354edec7e4b641466ad8879400d353ff"
PREDICTOR_SHA = "764075bd986a33379500cde2ae43294bd4822b449c20db2b2377a03f881e3d69"
PROTOCOL_SHA = "dfeb10b6d289a434620d8ae255f7bbacee7b3612402a646dd2aeb6119da3c670"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def frozen_model():
    if sha(OLD / "predictor.py") != PREDICTOR_SHA or sha(OLD / "model.json") != MODEL_SHA:
        raise ValueError("legacy model or inference implementation changed")
    spec = importlib.util.spec_from_file_location("legacy_freight_predictor", OLD / "predictor.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.FreightPredictor(OLD / "model.json")


def training_overlap(plan):
    cards_path = ACTOR / "data_split.json"
    if sha(cards_path) != read(OLD / "protocol.json")["selection_sha256"]:
        raise ValueError("legacy selection provenance changed")
    cards = {r["id"]: r for r in read(cards_path)["games"]}
    maps = defaultdict(set)
    items = read(OLD / "dataset.json")["rows"]
    for item in items:
        card = cards[item["id"]]
        if (card["sha256"], card["seat"], card["split"]) != (
            item["replay_sha256"], item["observer_seat"], item["split"]
        ):
            raise ValueError("legacy replay/seat/split mismatch")
        maps[card["split"]].add(card["terrain_sha256"])
    observed = {k: len(v) for k, v in maps.items()}
    if observed != {"train": 218, "val": 10, "holdout": 18} or sum(observed.values()) != len(items):
        raise ValueError("legacy corpus cardinality changed")
    overlap = {s: {k: sorted(v.intersection(plan[s])) for k, v in maps.items()}
               for s in ("training_terrains", "held_terrains")}
    return observed, overlap


def predict_target(model, history, target):
    current = history[-1]
    step, seat = current["step"], current["player"]
    if len(history) < 6 or [v["step"] for v in history[-6:]] != list(range(step - 5, step + 1)):
        raise ValueError("missing prior public history")
    if any(v["player"] != seat or "full_carbon" in v for v in history):
        raise ValueError("wrong observer or private truth")
    pred = model.predict(current, history[-6:-1])[target["target_uid"]]
    if pred["cargo"] != target["target_cargo"] or pred["home_distance"] != target["target_home_distance"]:
        raise ValueError("physical target mismatch")
    return pred


def one_metrics(rows):
    return {
        "n": len(rows),
        "brier": mean((r["p"] - r["y"]) ** 2 for r in rows),
        "nll": mean(-math.log(r["p"] if r["y"] else 1 - r["p"]) for r in rows),
        "accuracy": mean((r["p"] >= .5) == r["y"] for r in rows),
        "positive_rate": mean(r["y"] for r in rows),
        "mean_p": mean(r["p"] for r in rows),
    }


def metrics(rows):
    if not rows:
        return None
    maps = defaultdict(list)
    bins = [[] for _ in range(5)]
    for row in rows:
        maps[row["terrain"]].append(row)
        bins[min(4, int(5 * row["p"]))].append(row)
    bymap = {k: one_metrics(v) for k, v in sorted(maps.items())}
    return {
        "n": len(rows), "maps": len(maps), "per_terrain": bymap,
        "map_equal": {k: mean(v[k] for v in bymap.values()) for k in ("brier", "nll", "accuracy", "positive_rate", "mean_p")},
        "pooled": one_metrics(rows),
        "bins_pooled": [dict(lower=i / 5, upper=(i+1) / 5, n=len(b),
                         mean_p=mean(r["p"] for r in b) if b else None,
                         observed=mean(r["y"] for r in b) if b else None) for i, b in enumerate(bins)]
    }


def main():
    from dataset import histories
    if sha(HERE / "protocol.json") != PROTOCOL_SHA:
        raise ValueError("comparison protocol changed")
    plan = read(HERE / "protocol.json")
    model = frozen_model()
    counts, overlap = training_overlap(plan)
    held = set(plan["held_terrains"])
    held_clean = held.difference(*(v for v in overlap["held_terrains"].values()))
    rows = []
    seen = set()
    games = set()
    for game, root, hist in histories():
        target = root["target"]
        pred = predict_target(model, hist, target)
        actual = root["actual"]
        amount = actual["target_positive_own_deposit"]
        signed = sum(v[3] for v in actual["banking_events"] if v[1] == "own_delivery" and v[2] == 1-game["seat"])
        if abs(signed - amount) > 1e-7 or not 0 < pred["p_deposit_10"] < 1:
            raise ValueError("invalid settlement or predicted probability")
        key = (game["terrain"], game["seat"], root["step"], root["prefix_sha256"], root["public_sha256"])
        if key in seen:
            raise ValueError("duplicate public alert")
        seen.add(key)
        games.add((game["source"],game["arm"]))
        rows.append(dict(terrain=key[0],seat=key[1],step=key[2],prefix_sha256=key[3],public_sha256=key[4],
                         seed=game["seed"],source=game["source"],arm=game["arm"],target_uid=target["target_uid"],
                         split="held" if key[0] in held else "train", observed_joint_turns=actual["observed_joint_turns"],
                         p=pred["p_deposit_10"], y=int(amount>0), gross_first_trip_amount=amount,
                         legacy_expected_root_principal=pred["expected_root_cash_10"]))
    if len(rows)!=885 or len(games)!=52 or {r["terrain"] for r in rows}!=held.union(plan["training_terrains"]):
        raise ValueError("cohort changed")
    path = HERE / "old_predictions.json.gz"
    payload = json.dumps(rows, sort_keys=True, separators=(",",":"), allow_nan=False).encode()
    path.write_bytes(gzip.compress(payload, compresslevel=9, mtime=0))
    info = dict(protocol_sha256=PROTOCOL_SHA, source_cases_sha256=plan["source_case_sha256"], model_sha256=MODEL_SHA,
                predictor_sha256=PREDICTOR_SHA, program_sha256=sha(Path(__file__)), history_reader_sha256=sha(HERE/"dataset.py"),
                source_split_sha256=sha(ACTOR/"data_split.json"), legacy_manifest_sha256=sha(OLD/"dataset.json"),
                legacy_protocol_sha256=sha(OLD/"protocol.json"), predictions_sha256=sha(path), old_terrain_counts=counts,
                overlapping_physical_terrains=overlap,new_roots=len(rows),games=len(games),distinct_public_prefixes=len(seen),
                held=metrics([r for r in rows if r["terrain"] in held]),
                held_excluding_all_legacy_splits=metrics([r for r in rows if r["terrain"] in held_clean]),
                train=metrics([r for r in rows if r["terrain"] not in held]),
                amount_comparability="The old cash output attributes only the root's preexisting cargo. Its Bernoulli event is the same-ID positive home deposit by the tenth transition. Compare its Bernoulli scores here; gross first-trip cash and root-principal cash have different accounting.",
                new_matches=0)
    (HERE/"old_summary.json").write_text(json.dumps(info,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print(info["held"]["n"], info["held"]["map_equal"], path, flush=True)


if __name__ == "__main__":
    main()

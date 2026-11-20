"""Score symmetry-averaged frozen values against official outcomes on archived legal histories."""

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
FAMILY = HERE.parent.parent
for root in (
    HERE.parent,
    FAMILY / "own_search_prior",
    FAMILY / "opponent_policy",
    FAMILY / "freight_forecast",
):
    sys.path.insert(0, str(root))
import dev_tapes
from original_policy import PROTOCOL as SOURCES, original, module, tape
from study import original_rows, STEPS
from rotate import rotate_features

PROTOCOL_SHA = "6a91a658eaa72538149353c26263d5fc19219e62b359f038a78bb4135619a02b"
CONVENTIONS = ("leaf_all_known", "observer_known")
ESTIMATORS = ("old", "rotated", "mean", "visible_cash")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def cards():
    for card in SOURCES["games"]:
        if card["split"] != "val":
            continue
        metadata = {
            "domain": "old_held",
            "seed": card["seed"],
            "expert_seat": 1 - card["candidate_seat"],
            "terrain": card["terrain"],
            "source_sha256": card["sha256"],
            "official_scores": tape(card)["official_score_by_seat"],
        }
        yield metadata, original_rows(card)
    for g, path, receipt in dev_tapes.inventory():
        if g["status"] != "complete":
            continue
        seat = 1 - g["seat"]
        yield (
            {
                "domain": "dev1",
                "seed": g["seed"],
                "expert_seat": seat,
                "terrain": g["terrain"],
                "source_sha256": g["receipt_sha256"],
                "official_scores": receipt["official_scores_by_seat"],
            },
            dev_tapes.rows(path, seat),
        )


def evaluate(game, stream, m, net):
    seat = game["expert_seat"]
    score = game["official_scores"]
    if len(score) != 2 or any(v is None for v in score):
        raise ValueError("Unscored trajectory")
    margin = float(score[seat]) - float(score[1 - seat])
    target = m._terminal(margin)
    tracker = m.BeliefTracker(m._Bel, seat)
    previous = None
    rows = []
    for record in stream:
        t = record["t"]
        obs = record["obs"]
        p = obs["players"]
        if set(obs) != dev_tapes.PUBLIC_KEYS or (obs["player"], obs["step"]) != (
            seat,
            t,
        ):
            raise ValueError("Illegal information or incorrect seat")
        est, known = tracker.step(p, t, obs["carbon"])
        if t in STEPS:
            sc = m._scalars(p, t, seat)
            xs = []
            for mask in (np.ones(m.N, dtype=np.float32), known):
                x = m._quant(m.encode(p, t, seat, est, mask, previous))
                r = rotate_features(x)
                if not np.array_equal(rotate_features(r), x):
                    raise ValueError("Rotation not involutive")
                xs.extend((x, r))
            _, _, v = net(
                np.ascontiguousarray(xs, dtype=np.float32),
                m._SID,
                np.repeat(sc[None], len(xs), axis=0),
            )
            if v.shape != (4,) or not np.isfinite(v).all():
                raise ValueError("Nonfinite or missing value")
            pred = {}
            cash = float(p[seat][0]) - float(p[1 - seat][0])
            for k, mask in enumerate(CONVENTIONS):
                old, rot = map(float, v[2 * k : 2 * k + 2])
                pred[mask] = {
                    "old": old,
                    "rotated": rot,
                    "mean": (old + rot) * 0.5,
                    "visible_cash": m._terminal(cash),
                }
            rows.append(
                dict(
                    step=t,
                    cash_margin=cash,
                    unknown_cells=int(np.count_nonzero(np.asarray(known) < 1)),
                    scalar=list(map(float, sc)),
                    values=pred,
                )
            )
        tracker.set_actions(p, record["own"])
        previous = p
    if len(rows) != 23:
        raise ValueError("Incomplete frozen position set")
    return {**game, "official_margin": margin, "terminal": target, "rows": rows}


def score(records, mask, keep=lambda g, r: True):
    """Each observed terrain has equal weight, then each game, then selected positions."""
    by_map = defaultdict(list)
    counter = []
    for game in records:
        target = game["terminal"]
        sel = [r for r in game["rows"] if keep(game, r)]
        if not sel:
            continue
        stats = {}
        for estimator in ESTIMATORS:
            errors = np.array([r["values"][mask][estimator] - target for r in sel])
            stats[estimator] = [
                float((errors**2).mean()),
                float(np.abs(errors).mean()),
                float(errors.mean()),
            ]
        by_map[game["terrain"]].append(stats)
        for row in sel:
            old, rot, avg = (row["values"][mask][k] for k in ("old", "rotated", "mean"))
            counter.append(
                dict(
                    seed=game["seed"],
                    seat=game["expert_seat"],
                    step=row["step"],
                    target=target,
                    old=old,
                    rotated=rot,
                    mean=avg,
                    delta_squared_error=(avg - target) ** 2 - (old - target) ** 2,
                )
            )
    maps = {}
    for terrain, games in by_map.items():
        vals = {k: np.mean([g[k] for g in games], axis=0).tolist() for k in ESTIMATORS}
        maps[terrain] = dict(
            games=len(games),
            estimates=vals,
            mean_minus_old_mse=vals["mean"][0] - vals["old"][0],
        )
    if not maps:
        return None
    top = {}
    for estimator in ESTIMATORS:
        mse, mae, bias = np.mean(
            [v["estimates"][estimator] for v in maps.values()], axis=0
        ).tolist()
        top[estimator] = dict(
            mse=mse, rmse=math.sqrt(mse), mae=mae, bias_prediction_minus_terminal=bias
        )
    mean_d = top["mean"]["mse"] - top["old"]["mse"]
    counter.sort(key=lambda d: d["delta_squared_error"])
    return dict(
        estimates=top,
        map_metrics_array_order=["MSE", "MAE", "signed_bias"],
        maps=maps,
        n_roots=len(counter),
        mean_minus_old_mse=mean_d,
        maps_improved=sum(v["mean_minus_old_mse"] < 0 for v in maps.values()),
        most_improved=counter[0],
        most_harmed=counter[-1],
    )


def summarize(games):
    summary = {}
    for dom in ("old_held", "dev1"):
        data = [g for g in games if g["domain"] == dom]
        if (
            len(data) != (8 if dom == "old_held" else 7)
            or len({g["terrain"] for g in data}) != 4
        ):
            raise ValueError("Changed corpus")
        out = {}
        for mask in CONVENTIONS:
            out[mask] = dict(
                all=score(data, mask),
                phases={
                    str(i): score(data, mask, lambda g, r: r["step"] // 100 == i)
                    for i in range(3)
                },
                seats={
                    str(i): score(data, mask, lambda g, r: g["expert_seat"] == i)
                    for i in (0, 1)
                },
            )
        out["wins_draws_losses"] = [
            sum(
                (g["official_margin"] > 0) - (g["official_margin"] < 0) == i
                for g in data
            )
            for i in (1, 0, -1)
        ]
        out["games_saturated_at_abs_3"] = sum(abs(g["terminal"]) == 3 for g in data)
        summary[dom] = out
    return summary


def write(path, value):
    blob = (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()
    if path.suffix == ".gz":
        blob = gzip.compress(blob, mtime=0)
    path.write_bytes(blob)
    return sha(blob)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if sha((HERE / "protocol.json").read_bytes()) != PROTOCOL_SHA:
        raise ValueError("Frozen contract changed")
    src = original()
    m = module(src)
    net = m._net()
    inputs = list(cards())
    maps = {(g["domain"], g["terrain"]) for g, _ in inputs}
    if (
        len({k[1] for k in maps}) != 8
        or len(inputs) != 15
        or any(g["seed"] in (621758590, 621758591) for g, _ in inputs)
    ):
        raise ValueError("Cross-domain or reserved maps")
    games = [evaluate(game, stream, m, net) for game, stream in inputs]
    results = summarize(games)
    main_results = [r["leaf_all_known"]["all"] for r in results.values()]
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    rows_sha = write(out / "positions.json.gz", games)
    summary = dict(
        protocol_sha256=PROTOCOL_SHA,
        original_sha256=sha(src),
        backend=type(net).__name__,
        rotation_module_sha256=sha((HERE.parent / "rotate.py").read_bytes()),
        positions_sha256=rows_sha,
        held=results,
        zero_step_archive={
            "seed": 621758582,
            "derivative_seat": 1,
            "original_seat": 0,
            "cause": "bilateral_init_timeout",
            "steps": 0,
        },
        favorable_observational_calibration=all(
            r["mean_minus_old_mse"] < 0 and r["maps_improved"] >= 3
            for r in main_results
        ),
    )
    print("results", write(out / "results.json", summary), "positions", rows_sha)
    for name, r in results.items():
        p = r["leaf_all_known"]["all"]
        print(name, p["estimates"], p["maps_improved"])


if __name__ == "__main__":
    main()

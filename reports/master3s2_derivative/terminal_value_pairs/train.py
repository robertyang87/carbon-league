"""Train the frozen neural value's two-parameter residual on paired original-opponent continuations."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import lsq_linear

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "action_rank_data"
sys.path.insert(0, str(HERE.parent / "opponent_policy"))
from original_policy import original, module

sys.path.insert(0, str(HERE))
from physical import public_features

FROZEN = "b81ee4baef396e42782d5b46df84cf3adb2373aaef409041e51c912e71fec447"
ROWS = ("first", "second")
MODELS = ("native", "full", "one", "cash")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def require(ok, msg):
    if not ok:
        raise ValueError(msg)


def read(path):
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)


def immutable(path, value):
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    if path.exists():
        require(raw == path.read_bytes(), f"Different frozen artifact {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(raw)
        tmp.replace(path)
    return sha(raw)


def setup():
    contract = read(HERE / "protocol.json")
    source = DATA / "protocol.json"
    require(
        sha((HERE / "protocol.json").read_bytes()) == FROZEN
        and sha(source.read_bytes()) == contract["selection_sha256"],
        "Changed freeze",
    )
    book = read(source)
    tickets = book["tickets"]
    require(
        len(tickets) == 12 and len({r["terrain"] for r in tickets}) == 12,
        "Duplicate terrain",
    )
    require(
        [t["ordinal"] for t in tickets] == list(range(12))
        and [t["split"] for t in tickets] == ["train"] * 8 + ["held"] * 4,
        "Changed split",
    )
    return tickets, module(original())


def paths(ticket):
    base = (
        DATA / ticket["split"] / f"{ticket['seed']}_{ticket['seat']}_{ticket['step']}"
    )
    return {
        key: base.with_name(base.name + "_" + ext)
        for key, ext in [
            ("root", "root.json.gz"),
            ("pair", "pair.json"),
            ("leaves", "leaf.npz"),
        ]
    }


def check_ticket(actual, expected):
    require(
        all(
            actual.get(k) == expected[k]
            for k in ("seed", "seat", "step", "terrain", "split")
        ),
        "Ticket identity mismatch",
    )


def inputs(t, m, net):
    files = paths(t)
    root = read(files["root"])
    require(
        root["protocol_sha256"] == read(HERE / "protocol.json")["selection_sha256"],
        "Wrong root selection binding",
    )
    check_ticket(root["ticket"], t)
    require(
        root["status"] == "paired_eligible",
        "Root was not eligible for a paired comparison",
    )
    arms = root["arms"]
    require(
        [a["name"] for a in arms] == ["first", "second"]
        and arms[0]["order"] != arms[1]["order"],
        "Duplicate/order mismatch",
    )
    with np.load(files["leaves"], allow_pickle=False) as z:
        X = z["X"]
        sc = z["scalars"]
        vv = z["critic"]
        indices = z["candidate_index"]
        roll = z["roll_index"]
        step = z["leaf_step"]
        depth = z["depth"]
        zero = z["exact_terminal"]
        require(
            roll.ndim == 1 and len(roll) > 0 and len(set(roll.tolist())) == len(roll),
            "Bad rollout IDs",
        )
        require(
            X.dtype == np.uint8 and X.shape == (2, len(roll), 54, 15, 15),
            "Missing physical feature grid",
        )
        require(
            sc.dtype == np.float32
            and sc.shape == (2, len(roll), 5)
            and vv.shape == (2, len(roll)),
            "Missing scalars/score cells",
        )
        require(
            all(a.shape == (2, len(roll)) for a in (step, depth, zero))
            and np.all(depth == 8),
            "Missing shared depth 8",
        )
        require(
            indices.shape == (2,) and indices[0] != indices[1],
            "Invalid alternative actions",
        )
        require(
            np.all(np.isfinite(sc)) and np.all(np.isfinite(vv)), "Invalid leaf score"
        )
        means = []
        check_errors = []
        for a, arm in enumerate(arms):
            require(
                arm["candidate_index"] == indices[a]
                and len(arm["leaves"]) == len(roll),
                "Arm binding mismatch",
            )
            feats = []
            cash_values = []
            leaves = {int(l["roll_index"]): l for l in arm["leaves"]}
            require(set(leaves) == set(roll.tolist()), "Distinct or missing rollout")
            for r, rid in enumerate(roll):
                l = leaves[int(rid)]
                seat = t["seat"]
                p = l["players"]
                tt = int(step[a, r])
                require(
                    tt == t["step"] + 8
                    and l["leaf_step"] == tt
                    and bool(l["exact_terminal"]) == bool(zero[a, r]),
                    "Mismatched leaf chronology",
                )
                cells = np.ascontiguousarray(
                    X[a, r].astype(np.float32) * np.float32(0.01)
                )
                require(
                    sha(cells.tobytes()) == l["x_float32_sha256"],
                    "Rotated/mistaken board encoding or float conversion",
                )
                expected = m._scalars(p, tt, seat)
                require(
                    np.allclose(expected, sc[a, r], rtol=0, atol=1e-6),
                    "Leaf roster/scalars mismatch",
                )
                if zero[a, r]:
                    feats.append([0.0, 0.0])
                    cash_values.append(float(vv[a, r]))
                    continue
                observed = float(net(cells[None], m._SID, sc[a, r][None])[2][0])
                delta = abs(observed - float(vv[a, r]))
                check_errors.append(delta)
                require(
                    np.isclose(observed, vv[a, r], rtol=0.0003, atol=0.0003),
                    "Archived critic differs from original on its actual leaf inputs",
                )
                feats.append(public_features(p, tt, seat))
                cash_values.append(
                    m._terminal(float(p[seat][0]) - float(p[1 - seat][0]))
                )
            value = float(np.mean(vv[a], dtype=np.float64))
            require(
                np.isclose(value, arm["original_value"], atol=0.0001, rtol=0.0001),
                "Original aggregated score is not the same common-roll score",
            )
            means.append(
                dict(
                    index=int(indices[a]),
                    original_value=value,
                    features=np.mean(feats, axis=0).tolist(),
                    cash_value=float(np.mean(cash_values)),
                )
            )
    native = means[0]["original_value"] - means[1]["original_value"]
    require(native >= -1e-6, "First candidate was not first by native value")
    return dict(
        terrain=t["terrain"],
        seat=t["seat"],
        seed=t["seed"],
        step=t["step"],
        native=native,
        cash=means[0]["cash_value"] - means[1]["cash_value"],
        x=(np.array(means[0]["features"]) - means[1]["features"]).tolist(),
        arms=means,
        n_rolls=len(roll),
        max_abs_critic_check=max(check_errors, default=0.0),
        root_sha256=sha(files["root"].read_bytes()),
        leaf_sha256=sha(files["leaves"].read_bytes()),
    )


def outcomes(t, m):
    pairpath = paths(t)["pair"]
    data = read(pairpath)
    check_ticket(data["ticket"], t)
    files = paths(t)
    require(
        data["protocol_sha256"] == read(HERE / "protocol.json")["selection_sha256"]
        and data["root_sha256"] == sha(files["root"].read_bytes()),
        "Broken pair/root provenance",
    )
    status = data["status"]
    if status == "complete":
        require(
            data["leaf_sha256"] == sha(files["leaves"].read_bytes()),
            "Outcome bound to different neural leaf",
        )
    if status != "complete":
        require(
            data.get("reason")
            and status in ("skipped", "failed", "invalid", "ineligible"),
            "Unknown/in-progress status",
        )
        return None, dict(
            terrain=t["terrain"],
            status=status,
            reason=data["reason"],
            pair_sha256=sha(pairpath.read_bytes()),
        )
    margins = []
    terminal = []
    scores = []
    for arm in ROWS:
        a = data["arms"][arm]
        v = a["scores"]
        seat = t["seat"]
        require(len(v) == 2 and np.isfinite(v).all(), "Missing terminal scores")
        margin = float(v[seat]) - float(v[1 - seat])
        value = m._terminal(margin)
        require(
            np.isclose(margin, a["margin"], atol=1e-5, rtol=0.0)
            and abs(value - a["terminal_value"]) < 1e-7
            and bool(a["win"]) == (margin > 0),
            "Official terminal or perspective mismatch",
        )
        margins.append(margin)
        terminal.append(value)
        scores.append(v)
    target = terminal[0] - terminal[1]
    require(abs(float(data["target"]) - target) < 1e-7, "Reversed target")
    return dict(
        margins=margins,
        terminal_values=terminal,
        scores=scores,
        target=target,
        pair_sha256=sha(pairpath.read_bytes()),
    ), None


def fit(rows, dims):
    X = np.asarray([r["x"][:dims] for r in rows], dtype=np.float64)
    y = np.asarray([r["target"] - r["native"] for r in rows])
    n = len(y)
    xa = np.vstack((X / np.sqrt(n), np.sqrt(0.01) * np.eye(dims)))
    ya = np.concatenate((y / np.sqrt(n), np.zeros(dims)))
    result = lsq_linear(xa, ya, bounds=(-1.0, 1.0), method="bvls", tol=1e-12)
    require(result.success, "Bounded ridge failed")
    return np.pad(result.x.astype(np.float32), (0, 2 - dims)).tolist()


def forecast(row, heads):
    x = np.array(row["x"], dtype=np.float64)
    return dict(
        native=row["native"],
        cash=row["cash"],
        **{s: row["native"] + float(x @ heads[s]) for s in ("full", "one")},
    )


def train(tickets, m, net, out):
    rows = []
    excluded = []
    for t in tickets[:8]:
        y, skip = outcomes(t, m)
        if skip:
            excluded.append(skip)
            continue
        rows.append({**inputs(t, m, net), **y})
    design = np.array([r["x"] for r in rows], dtype=float).reshape(-1, 2)
    support = dict(
        eligible=len(rows),
        physically_distinct=len(set(r["terrain"] for r in rows)),
        nonzero_feature_maps=sum(np.max(np.abs(r["x"])) > 1e-7 for r in rows),
        nonzero_target_maps=sum(abs(r["target"]) > 0 for r in rows),
        rank=int(np.linalg.matrix_rank(design)) if len(rows) else 0,
        feature_abs_max=np.max(np.abs(design), axis=0).tolist()
        if len(rows)
        else [0.0, 0.0],
    )
    support["nonzero_feature_maps"] = int(support["nonzero_feature_maps"])
    ok = (
        len(rows) >= 6
        and support["nonzero_feature_maps"] >= 3
        and support["nonzero_target_maps"] >= 2
    )
    heads = (
        {s: fit(rows, dims) for s, dims in [("full", 2), ("one", 1)]} if ok else None
    )
    descriptive = None
    if heads:
        from metrics import compare

        for r in rows:
            r["preds"] = forecast(r, heads)
        descriptive = compare(rows)
    result = dict(
        contract_sha256=FROZEN,
        train_eligible=ok,
        train_descriptive=descriptive,
        support=support,
        heads=heads,
        excluded=excluded,
        training=rows,
    )
    print("heads", immutable(out / "heads.json", result), support, heads, flush=True)


def load_heads(out):
    h = read(out / "heads.json")
    require(
        h["contract_sha256"] == FROZEN and h["train_eligible"] and h["heads"],
        "No frozen eligible model",
    )
    return h["heads"]


def predict(t, m, net, out):
    heads = load_heads(out)
    p = paths(t)
    root = read(p["root"])
    check_ticket(root["ticket"], t)
    if root["status"] != "paired_eligible":
        require(root.get("reason"), "Excluded root lacks reason")
        data = dict(
            terrain=t["terrain"],
            seed=t["seed"],
            seat=t["seat"],
            step=t["step"],
            excluded=root["reason"],
            root_sha256=sha(p["root"].read_bytes()),
        )
    else:
        data = inputs(t, m, net)
        data["preds"] = forecast(data, heads)
    data["heads_sha256"] = sha((out / "heads.json").read_bytes())
    print(
        "forecast",
        t["ordinal"],
        immutable(out / f"held_forecast_{t['ordinal']}.json", data),
        flush=True,
    )


def settle(tickets, m, out):
    from metrics import compare

    load_heads(out)
    rows = []
    skip = []
    for t in tickets[8:]:
        path = out / f"held_forecast_{t['ordinal']}.json"
        pred = read(path)
        require(
            pred["terrain"] == t["terrain"]
            and pred["heads_sha256"] == sha((out / "heads.json").read_bytes()),
            "Mismatched forecast or weights",
        )
        receipt = read(paths(t)["pair"])
        require(
            receipt["blind_forecast_sha256"] == sha(path.read_bytes())
            and receipt["root_sha256"] == pred["root_sha256"],
            "Held response not bound to the frozen pre-response prediction and root",
        )
        if "preds" in pred:
            require(
                receipt.get("leaf_sha256") == pred["leaf_sha256"],
                "Held prediction compared to another leaf",
            )
        y, missing = outcomes(t, m)
        if missing:
            skip.append(missing)
            continue
        require("preds" in pred, "Complete trial without blind forecast")
        rows.append({**pred, **y})
    print(
        "held",
        immutable(
            out / "held.json", dict(evaluation=compare(rows), excluded=skip, rows=rows)
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["train", "predict", "settle"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ordinal", type=int, choices=range(8, 12))
    args = parser.parse_args()
    tickets, m = setup()
    if args.mode == "train":
        train(tickets, m, m._net(), args.out)
    elif args.mode == "predict":
        require(args.ordinal is not None, "Select the registered held ticket")
        predict(tickets[args.ordinal], m, m._net(), args.out)
    else:
        settle(tickets, m, args.out)


if __name__ == "__main__":
    main()

"""Fit map-balanced enemy response forecasts and validate protected cargo on all registered orders."""

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
import runtime

HERE = Path(__file__).resolve().parent
FROZEN = "f70d742b7376fa40efab1e0988c5d8ad3d68ed47d3b2c97066ed5d86645d455a"


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def source(card):
    data = (HERE.parent / card["source"]).read_bytes()
    if sha(data) != card["sha256"]:
        raise ValueError("Wrong replay artifact")
    g = json.loads(gzip.decompress(data))
    if (
        len(g["steps"]) != 300
        or g["configuration"]["moveCost"] != 0
        or g["configuration"]["workerLimit"] != 10
        or g["configuration"]["recCollectorCost"] != 30
    ):
        raise ValueError("Unregistered arena configuration")
    return g


def write(path, value):
    data = (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode()
    if path.suffix == ".gz":
        data = gzip.compress(data, mtime=0)
    path.write_bytes(data)
    return sha(data)


def roots(g, seat):
    # The replay stores globally public unit rosters in seat 0. Hidden carbon is never passed to runtime.
    frames = g["steps"]
    previous = None
    for t in range(299):
        frame = frames[t]
        next_frame = frames[t + 1]
        p = frame[0]["observation"]["players"]
        nxt = next_frame[0]["observation"]["players"]
        if t != frame[0]["observation"]["step"] or any(
            s["status"] != "ACTIVE" for s in frame
        ):
            raise ValueError("Broken chronology")
        state = runtime.prepare(p, seat, t, previous)
        orders = [dict(s["action"] or {}) for s in next_frame]
        yield t, state, p, nxt, orders
        previous = p


def train(cards, out):
    X = []
    Y = []
    game_id = []
    recruits = available = 0
    for i, card in enumerate(cards):
        seat = card["seat"]
        enemy = 1 - seat
        g = source(card)
        for _, s, p, nxt, orders in roots(g, seat):
            # Each eligible recruiting centre is one opportunity before joint resolution.
            if s["can_enemy_recruit"]:
                for bid, cell in p[enemy][1].items():
                    available += 1
                    recruits += orders[enemy].get(bid) == "RECCOLLECTOR"
            for uid, (_, load, role) in s["our"].items():
                if role != "COLLECTOR":
                    continue
                for eid, x in runtime.pairs(s, uid):
                    X.append(x)
                    Y.append(runtime.ACTIONS.index(orders[enemy].get(eid) or "NONE"))
                    game_id.append(i)
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y)
    rows = np.arange(len(Y))
    game_id = np.array(game_id)
    counts = np.bincount(game_id, minlength=4)
    if np.min(counts) == 0:
        raise ValueError("No data on a registered training map")
    mass = 1 / counts[game_id]
    mass /= mass.sum()

    def loss(w):
        z = X @ w
        lp = z - logsumexp(z, axis=1)[:, None]
        objective = -mass @ lp[rows, Y] + 0.025 * w @ w
        res = np.exp(lp)
        res[rows, Y] -= 1
        return objective, np.einsum("i,ij,ijk->k", mass, res, X) + 0.05 * w

    opt = minimize(
        loss, np.zeros(X.shape[2]), jac=True, method="L-BFGS-B", options={"maxiter": 80}
    )
    if not opt.success:
        raise ValueError(opt.message)
    w = opt.x.astype(np.float32)
    b = (recruits + 1) / (available + 2)
    result = dict(
        protocol=FROZEN,
        weights=w.tolist(),
        features=runtime.FEATURES,
        recruit_probability=b,
        train_recruit_orders=recruits,
        train_recruit_slots=available,
        optimizer_iterations=opt.nit,
        softmax_rows_by_map={c["match_id"]: int(n) for c, n in zip(cards, counts)},
        ridge=0.025,
    )
    print(
        "Head", write(out / "weights.json", result), "rows", counts.tolist(), flush=True
    )
    return result


def evaluate(card, head, out, mode):
    from labels import label_actions

    g = source(card)
    seat = card["seat"]
    rows = []
    cases = []
    errors = []
    w = np.asarray(head["weights"], dtype=np.float32)
    for t, s, p, nxt, orders in roots(g, seat):
        for uid, info in p[seat][2].items():
            if info[2] != "COLLECTOR":
                continue
            est = runtime.forecast(s, uid, w, head["recruit_probability"])
            y = label_actions(p, seat, uid, orders, nxt)
            action = runtime.ACTIONS.index(orders[seat].get(uid) or "NONE")
            row = dict(
                match=card["match_id"],
                opponent=card["opponent"],
                step=t,
                seat=seat,
                uid=uid,
                load=info[1],
                action=action,
                **{k: est[k] for k in ("risk", "baseline", "friend", "home")},
                **{k: y[k] for k in ("labels", "loss", "occupancy", "alive_real")},
            )
            rows.append(row)
            if mode == "exposed" and (t, uid) in [
                (138, "player-1-worker-3"),
                (140, "player-1-worker-11"),
            ]:
                cases.append(
                    {
                        **row,
                        "source_distributions": est["source_distributions"],
                        "directions": runtime.ACTIONS,
                        "source_cell": info[0],
                    }
                )
    return rows, cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["train", "held", "exposed"])
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    protocol = HERE / "protocol.json"
    if sha(protocol.read_bytes()) != FROZEN:
        raise ValueError("Wrong frozen contract")
    cards = json.loads(protocol.read_bytes())["maps"]
    if args.mode == "train":
        return train(cards["train"], out)
    from metrics import summarize

    head = json.loads((out / "weights.json").read_bytes())
    if head["protocol"] != FROZEN or list(head["features"]) != list(runtime.FEATURES):
        raise ValueError("Altered data feature definition")
    rows = []
    cases = []
    for c in cards[args.mode]:
        r, k = evaluate(c, head, out, args.mode)
        rows += r
        cases += k
    scores = summarize(rows)
    m = list(scores["by_match"].values())
    grouped = [v["groups"]["cargo_ge_200"]["geometric_actions"] for v in m]
    briers = [v["risk_minus_baseline"]["brier"] for v in grouped]
    ranking = [v["risk_minus_baseline"]["mean_root_auc"] for v in grouped]
    strict = lambda vals: len(vals) == len(grouped) and all(i is not None for i in vals)
    rk = [v for v in ranking if v is not None]
    cond = dict(
        mean_brier_lower=strict(briers) and np.mean(briers) < 0,
        improved_brier_maps=sum(x < 0 for x in briers if x is not None),
        ranking_eligible_maps=len(rk),
        mean_root_rank_delta=float(np.mean(rk)) if rk else None,
    )
    decision = (
        "DIAGNOSTIC"
        if args.mode == "exposed"
        else "INDETERMINATE"
        if (len(rk) < 3 or not strict(briers))
        else "PASS"
        if cond["mean_brier_lower"]
        and cond["improved_brier_maps"] >= 3
        and cond["mean_root_rank_delta"] >= 0
        else "FAIL"
    )
    report = dict(
        weight_sha256=sha((out / "weights.json").read_bytes()),
        protocol=FROZEN,
        verdict=decision,
        primary=dict(
            mean_delta_brier=float(np.mean(briers)) if strict(briers) else None,
            **{
                k: (v.item() if isinstance(v, np.generic) else v)
                for k, v in cond.items()
            },
        ),
        scores=scores,
        cases=cases,
    )
    report["orders_sha256"] = write(out / (args.mode + "_orders.json.gz"), rows)
    print(
        args.mode,
        len(rows),
        write(out / (args.mode + ".json"), report),
        report["primary"],
        decision,
        flush=True,
    )


if __name__ == "__main__":
    main()

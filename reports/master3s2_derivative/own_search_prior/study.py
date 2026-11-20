"""Distill the unchanged examiner's own executed commands from its own public view."""

import argparse
import gzip
import io
import json
import os
from pathlib import Path
import sys
import zipfile
import hashlib
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "opponent_policy"))
sys.path.insert(0, str(HERE.parent / "freight_forecast"))
from original_policy import PROTOCOL as HISTORY, free_entry, original, module, tape
import dev_tapes

FROZEN = "84c5f39a46f9dd74e1d6325c8e2399ae4cc107055e5b746dd23b48735ce68e4a"
STEPS = frozenset(range(7, 294, 13))
ORIGINAL = "65f268b2919ce30fe19355115a928eada032a6beaa018d801662d434c083f2b5"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def original_rows(card):
    expert = 1 - card["candidate_seat"]
    joint = tape(card)["actions"]
    for step, obs, _, own in free_entry.public_history({**card, "expert_seat": expert}):
        if joint[step][expert] != own:
            raise ValueError("Wrong expert seat or issued action")
        yield {"t": step, "obs": obs, "own": own, "other": joint[step][1 - expert]}


def records():
    games = [
        (
            {
                "seed": c["seed"],
                "terrain": c["terrain"],
                "seat": 1 - c["candidate_seat"],
                "split": c["split"],
            },
            original_rows(c),
        )
        for c in HISTORY["games"]
    ]
    for game, path, receipt in dev_tapes.inventory():
        if game["status"] == "complete":
            expert = 1 - game["seat"]
            games.append(
                (
                    {
                        "seed": game["seed"],
                        "terrain": game["terrain"],
                        "seat": expert,
                        "split": "dev1",
                    },
                    dev_tapes.rows(path, expert),
                )
            )
    terrain = {
        s: {g["terrain"] for g, _ in games if g["split"] == s}
        for s in ["train", "val", "dev1"]
    }
    if (
        [len(terrain[s]) for s in terrain] != [24, 4, 4]
        or len(set.union(*terrain.values())) != 32
        or len(games) != 63
    ):
        raise ValueError("Overlapping terrain or wrong game count")
    if set(g["seed"] for g, _ in games) & {621758590, 621758591}:
        raise ValueError("Exposed reserved new maps")
    return games


def aggregate():
    baseline = original()
    if (
        sha(baseline) != ORIGINAL
        or sha((HERE / "protocol.json").read_bytes()) != FROZEN
    ):
        raise ValueError("Code or registration changed")
    reference = module(baseline)
    marker = b"        return lw.T, lb.T"
    if baseline.count(marker) != 1:
        raise ValueError("Ambiguous root forward")
    features = module(baseline.replace(marker, b"        return hf.T, lw.T, lb.T"))
    training = {r: [] for r in ["base", "worker"]}
    testing = {"val": [], "dev1": []}
    checked = 0
    for identity, stream in records():
        expert = identity["seat"]
        tracker = reference.BeliefTracker(reference._Bel, expert)
        previous = None
        data = []
        collected = 0
        for record in stream:
            t = record["t"]
            obs = record["obs"]
            players = obs["players"]
            if (
                set(obs) != dev_tapes.PUBLIC_KEYS
                or obs["player"] != expert
                or obs["step"] != t
            ):
                raise ValueError("Wrong information or original point of view")
            est, known = tracker.step(players, t, obs["carbon"])
            if t in STEPS:
                x = reference._quant(
                    reference.encode(players, t, expert, est, known, previous)
                )
                hidden, lw, lb = features._POLICY.forward(x, features._SID)
                if collected == 0:
                    old = reference._POLICY.forward(x, reference._SID)
                    if not (
                        np.array_equal(lw, old[0])
                        and np.array_equal(lb, old[1])
                        and hidden.shape == (225, 96)
                    ):
                        raise ValueError("Changed original forward")
                    checked += 1
                if identity["split"] == "train":
                    cash, bases, workers, _ = players[expert]
                    for kind, things, actions, z, index in [
                        ("worker", workers, reference.WIDX, lw, True),
                        ("base", bases, reference.BIDX, lb, False),
                    ]:
                        if kind == "base" and (cash < 30 or len(workers) >= 10):
                            continue
                        for uid, v in things.items():
                            position = v[0] if index else v
                            y = actions[record["own"].get(uid) or "NONE"]
                            training[kind].append(
                                (
                                    identity["terrain"],
                                    hidden[position].copy(),
                                    z[position].copy(),
                                    y,
                                )
                            )
                else:
                    data.append(
                        dict(
                            step=t,
                            obs=obs,
                            own=record["own"],
                            other=record["other"],
                            x=x,
                            h=hidden,
                            lw=lw,
                            lb=lb,
                        )
                    )
                collected += 1
            tracker.set_actions(players, record["own"])
            previous = players
        if collected != 23:
            raise ValueError("Incomplete source game")
        if data:
            testing[identity["split"]].append(dict(**identity, rows=data))
    if checked != 63:
        raise ValueError("Missing original numeric parity")
    return training, testing, reference


def logprob(z):
    return z - logsumexp(z, axis=1)[:, None]


def fit_head(samples):
    gids, raw, z, y = map(np.asarray, zip(*samples))
    _, inv, count = np.unique(gids, return_inverse=True, return_counts=True)
    mass = 1.0 / count[inv]
    mass /= mass.sum()
    raw = raw.astype(np.float64)
    z = z.astype(np.float64)
    prior = logprob(z)
    center = mass @ raw
    scale = np.maximum(0.1, np.sqrt(mass @ np.square(raw - center)))
    center, scale = [a.astype("<f2") for a in (center, scale)]
    x = np.column_stack(((raw - center) / scale, np.ones(len(raw))))
    groups = z.shape[1]
    rows = np.arange(len(y))

    def objective(v):
        w = v.reshape(97, groups)
        lp = logprob(z + x @ w)
        ce = -lp[rows, y]
        kl = np.sum(np.exp(prior) * (prior - lp), axis=1)
        dz = 2 * np.exp(lp) - np.exp(prior)
        dz[rows, y] -= 1
        return mass @ (ce + kl) + 0.1 * np.sum(w * w), (
            x.T @ (mass[:, None] * dz) + 0.2 * w
        ).ravel()

    result = minimize(
        objective,
        np.zeros(97 * groups),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 70},
    )
    if not result.success:
        raise ValueError(result.message)
    for n in range(18):
        w = (result.x.reshape(97, groups) * 2.0 ** (-n)).astype("<f2")
        dx = x @ w.astype(np.float32)
        lp = logprob(z + dx)
        kl = mass @ np.sum(np.exp(prior) * (prior - lp), axis=1)
        if kl <= 0.025 and np.max(np.abs(dx)) <= 2.0:
            break
    else:
        raise ValueError("Failed training-only budget on logit change")
    info = dict(
        rows=len(raw),
        maps=len(count),
        classes=groups,
        iterations=result.nit,
        halvings=n,
        kl=float(kl),
        max_delta=float(np.max(abs(dx))),
        mean_base_nll=float(-mass @ prior[rows, y]),
        mean_new_nll=float(-mass @ lp[rows, y]),
    )
    return dict(center=center, scale=scale, w=w), info


def correct(weights, h, lw, lb):
    logits = []
    for name, z in [("worker", lw), ("base", lb)]:
        p = weights[name]
        x = (h.astype(np.float32) - p["center"].astype(np.float32)) / p["scale"].astype(
            np.float32
        )
        w = p["w"].astype(np.float32)
        logits.append(z + (x @ w[:-1] + w[-1]))
    return tuple(logits)


def save_weights(path, heads):
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as out:
        for role, head in sorted(heads.items()):
            for key, value in sorted(head.items()):
                buf = io.BytesIO()
                np.save(buf, value, allow_pickle=False)
                item = zipfile.ZipInfo(f"{role}_{key}.npy", (1980, 1, 1, 0, 0, 0))
                item.compress_type = zipfile.ZIP_DEFLATED
                out.writestr(item, buf.getvalue())


def save_json(path, value):
    raw = (json.dumps(value, allow_nan=False, sort_keys=True, indent=2) + "\n").encode()
    if path.suffix == ".gz":
        raw = gzip.compress(raw, mtime=0)
    path.write_bytes(raw)


def score_splits(testing, reference, weights, out):
    from held_metrics import grade

    records = {}
    details = {}
    for split, games in testing.items():
        result = grade(games, reference, weights, correct)
        details[split] = result.pop("roots")
        records[split] = result
    summary = dict(
        protocol_sha256=FROZEN,
        source_sha256=ORIGINAL,
        weights_sha256=sha((out / "weights.npz").read_bytes()),
        zero_step_ticket={
            "map": 621758582,
            "derivative_seat": 1,
            "original_seat": 0,
            "status": "bilateral_initialization_timeout",
            "steps": 0,
        },
        eligible_for_full_search_probe=all(v["pass_gate"] for v in records.values()),
        panels=records,
    )
    save_json(out / "held.json", summary)
    save_json(out / "roots.json.gz", details)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out
    if out.resolve() == HERE:
        raise ValueError("Choose a separate output folder to reproduce these results")
    out.mkdir(parents=True, exist_ok=True)
    data, testing, reference = aggregate()
    weights = {}
    fit = {}
    for name, rows in data.items():
        weights[name], fit[name] = fit_head(rows)
    save_weights(out / "weights.npz", weights)
    save_json(out / "fit.json", fit)
    summary = score_splits(testing, reference, weights, out)
    print(
        "Eligibility for a separate matched full-search check:",
        summary["eligible_for_full_search_probe"],
    )
    for name in ["weights.npz", "fit.json", "held.json", "roots.json.gz"]:
        print(name, sha((out / name).read_bytes()))
        if (HERE / name).exists() and (out / name).read_bytes() != (
            HERE / name
        ).read_bytes():
            raise ValueError("Frozen artifact differs: " + name)


if __name__ == "__main__":
    main()

"""Fit one equal-map logistic risk head on split train. Validation labels are not opened here."""
from collections import Counter
import json

import numpy as np

from extract import HERE, load, sha, write
from model import NAMES


def shards(split):
    spec = load(HERE/"protocol.json")
    paths = [HERE/f"{split}_{seed}_{s}.json.gz" for seed in spec["split"][split]["maps"] for s in (0,1)]
    docs = [load(p) for p in paths]
    rows = [r for d in docs for r in d["rows"]]
    if len({(r["seed"],r["seat"],r["t"],r["uid"]) for r in rows}) != len(rows):
        raise ValueError("repeated root")
    return rows, docs, paths


def fit(rows):
    maps = Counter(r["seed"] for r in rows)
    x, y, a = [], [], []
    for row in rows:
        for action in row["actions"]:
            if action["geometric"]:
                x.append([1., *action["x"]])
                y.append(action["died"])
                a.append(1. / (len(maps)*maps[row["seed"]]*2))
            elif action["died"]:
                raise ValueError("excluded a death from the public envelope")
    x = np.asarray(x)
    y = np.asarray(y, dtype=float)
    weights = np.asarray(a)
    p_geo = float(weights@y/weights.sum())
    beta = np.zeros(x.shape[1])
    beta[0] = np.log(p_geo/(1-p_geo))
    penalty = np.diag([0., *[.02]*len(NAMES)])
    for i in range(30):
        p = 1. / (1. + np.exp(-np.clip(x@beta, -60., 60.)))
        gradient = x.T@(weights*(p-y)) + penalty@beta
        hessian = x.T@((weights*p*(1.-p))[:,None]*x) + penalty
        delta = np.linalg.solve(hessian,gradient)
        beta -= delta
        if np.max(np.abs(delta)) < 1e-10:
            break
    return dict(features=NAMES,weights=beta[1:].tolist(),intercept=float(beta[0]),contact_constant=p_geo,
                fit_seeds=sorted(maps),map_root_counts=dict(maps),contact_arms=len(y),iterations=i+1,
                training_loss_gradient_inf=float(np.max(np.abs(gradient))),regularization=.02)


if __name__ == "__main__":
    rows, docs, paths = shards("train")
    model = fit(rows)
    model["protocol_sha256"] = sha(HERE/"protocol.json")
    model["probability_baseline_protocol_sha256"] = sha(HERE/"probability_baseline.json")
    model["fit_shards_sha256"] = {p.name:sha(p) for p in paths}
    write(HERE/"weights.json",model)
    print("frozen",sha(HERE/"weights.json"), model["map_root_counts"], model["contact_arms"], flush=True)

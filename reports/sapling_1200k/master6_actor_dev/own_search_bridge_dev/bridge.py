"""Cheap self-policy rollout and native paired-label gate. No original examiner invocation."""

import argparse
import copy
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import mean, median
import sys
import time
from collections import defaultdict

import numpy as np

HERE = Path(__file__).resolve().parent
ACTOR = HERE.parent
ROOT = ACTOR.parents[2]
sys.path.insert(0, str(ROOT / "code"))
import cnet2_feat as F
from fastsim import FastCarbon

PLAN = json.loads((HERE / "protocol.json").read_bytes())
ONEHOT = np.eye(5, dtype="f")


def load(p):
    p = Path(p)
    b = p.read_bytes()
    return json.loads(gzip.decompress(b) if p.suffix == ".gz" else b)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p, obj):
    b = (
        json.dumps(obj, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    if p.suffix == ".gz":
        b = gzip.compress(b, mtime=0)
    if p.exists():
        if p.read_bytes() != b:
            raise RuntimeError("output differs " + str(p))
        return
    tmp = p.with_suffix(".part")
    tmp.write_bytes(b)
    os.replace(tmp, p)


def require(v, msg):
    if not v:
        raise ValueError(msg)


def features(x, prefix, s):
    """One row for each action, sharing logits across workers through the real decoder prefix."""
    unit, moves, legal = prefix.feats(s)
    pos = x["tok_pos"][s]
    dest = F.DEST[pos] if s < 10 else np.full(5, pos)
    local = x["maps"].reshape(F.P, 225)[:, dest].T.astype("f") / 255.0
    history = np.array([x["hist_move"][s, -1] == a for a in range(5)], dtype="f")[:, None]
    shared = np.tile(np.concatenate([unit, x["scal"]])[None, :], (5, 1))
    return np.concatenate([shared, moves, local, ONEHOT, history], axis=1), legal


def predict(weights, feats):
    return (np.tanh(feats @ weights["hidden"] + weights["bias"]) @ weights["head"]).reshape(-1)


def decode(feat, x, weights, noise=None, command=None):
    pref = F.Prefix(x, feat.aux)
    out = np.full(F.OWN, -1, dtype=np.int64)
    y = feat.labels(command) if command is not None else None
    training = []
    for s in np.flatnonzero(x["tok_mask"][: F.OWN]):
        mat, legal = features(x, pref, s)
        if command is None:
            score = predict(weights, mat)
            if noise is not None:
                score = score + noise(feat.own_uid[s])
            move = int(np.argmax(np.where(legal, score, -np.inf)))
        else:
            move = int(y[s])
            require(legal[move], "logged own policy issued illegal prefix command")
            if np.count_nonzero(legal) > 1:
                training.append((mat, legal, move, int(x["tok_kind"][s])))
        out[s] = move
        pref.advance(s, move)
    return feat.to_actions(out), training


def fit():
    import torch

    torch.set_num_threads(1)
    torch.manual_seed(9046)
    groups = [[], []]
    identity = []
    for i, d in enumerate(PLAN["training_tapes"]):
        paths = sorted((ACTOR / d).glob("raw_*.json.gz"))
        require(paths, "missing own trajectory " + d)
        for p in paths:
            raw = load(p)
            s = raw["candidate_seat"]
            feat = F.Feat(s)
            sim = FastCarbon(raw["game_configuration"]).reset(seed=raw["seed"])
            require(list(sim.carbon) == raw["initial_full_carbon"], "initial identity")
            for t, pair in enumerate(raw["actions"]):
                obs = sim.obs(s)
                x = feat.observe(obs)
                _, pts = decode(feat, x, None, command=pair[s])
                groups[int(i >= len(PLAN["training_tapes"]) - 2)].extend(pts)
                sim.step(*pair)
            identity.append(
                {
                    "raw": str(p.relative_to(ROOT)),
                    "raw_sha256": sha(p),
                    "seed": raw["seed"],
                    "terrain": raw["initial_terrain_u8_sha256"],
                    "pack_sha256": raw["pack_sha256"],
                    "side": s,
                    "split": "rollout_hold" if i >= len(PLAN["training_tapes"]) - 2 else "fit",
                }
            )

    def tensors(rows):
        return tuple(torch.from_numpy(np.asarray(col)) for col in zip(*rows))

    train, held = map(tensors, groups)
    X, L, Y, _ = train
    layer = torch.nn.Linear(X.shape[-1], 18)
    output = torch.nn.Linear(18, 1, bias=False)
    mlp = torch.nn.Sequential(layer, torch.nn.Tanh(), output)
    opt = torch.optim.Adam(mlp.parameters(), lr=0.002)
    for epoch in range(25):
        order = torch.randperm(len(X))
        for ix in order.split(384):
            z = mlp(X[ix]).squeeze(-1).masked_fill(~L[ix], -1e9)
            loss = torch.nn.functional.cross_entropy(z, Y[ix])
            opt.zero_grad()
            loss.backward()
            opt.step()
    result = {}
    for label, (xx, ll, yy, rr) in (("fit", train), ("rollout_hold", held)):
        with torch.no_grad():
            logits = torch.cat([mlp(q).squeeze(-1) for q in xx.split(1024)]).masked_fill(~ll, -1e9)
            result[label] = {
                str(role): dict(
                    n=int((rr == role).sum()),
                    top1=float((logits[rr == role].argmax(-1) == yy[rr == role]).float().mean()),
                    ce=float(torch.nn.functional.cross_entropy(logits[rr == role], yy[rr == role])),
                )
                for role in sorted(set(rr.tolist()))
            }
    arrays = dict(
        hidden=layer.weight.detach().numpy().T,
        bias=layer.bias.detach().numpy(),
        head=output.weight.detach().numpy().T,
    )
    with torch.no_grad():
        require(
            np.max(np.abs(predict(arrays, held[0][0].numpy()) - mlp(held[0][0]).squeeze(-1).numpy())) < 2e-5,
            "numpy serve parity",
        )
    np.savez_compressed(HERE / "rollout.npz", **arrays)
    save(
        HERE / "fit.json",
        dict(
            trajectories=identity,
            parameter_count=sum(v.size for v in arrays.values()),
            metrics=result,
            weights_sha256=sha(HERE / "rollout.npz"),
        ),
    )
    print(result, flush=True)


def public_enemy(obs, belief):
    other = 1 - obs["player"]
    seen = np.zeros(225, dtype=bool)
    for cell, _, worker in obs["players"][other][2].values():
        seen[[cell] if worker == "COLLECTOR" else F.DEST[cell, 1:]] = True
    return dict(
        player=other,
        step=obs["step"],
        players=obs["players"],
        trees=obs["trees"],
        carbon=np.where(seen, belief, 0.0).tolist(),
    )


def estimate_view(cfg, obs, feat):
    sim = FastCarbon(cfg)
    state = dict(obs, carbon=feat.est.tolist())
    sim.load_observation(state)
    # Observations omit retired IDs. Continue on a disjoint synthetic range in each paired arm.
    for p in (0, 1):
        sim._wid_counter[p] += 500
        sim._tid_counter[p] += 500
    return sim


def jitter(stream, t, s):
    def noise(uid):
        key = int.from_bytes(
            hashlib.blake2s(f"{stream}/{t}/{s}/{uid}".encode(), digest_size=8).digest(), "big"
        )
        return 0.55 * np.random.default_rng(key).gumbel(size=5)

    return noise


def leaf(sim, seat, pre):
    alive = []
    v = []
    for s in (0, 1):
        alive.append(
            bool(
                sim.player_workers[s]
                or sim.player_trees[s]
                or (
                    sim.bases[s]
                    and sim.cash[s] >= min(sim.config["recCollectorCost"], sim.config["recPlanterCost"])
                )
            )
        )
        bases = [c for _, c in sim.bases[s]]
        pos = sim.cash[s]
        for uid in sim.player_workers[s]:
            w = sim.workers[uid]
            if w.is_collector and bases:
                pos += float(w.carbon) / (1 + min(F.DIST[w.pos, c] for c in bases))
        v.append(pos)
    if not all(alive):
        vals = [
            float(pre[s]) if alive[s] else sim.step_count - sim.config["episodeSteps"] - 1 for s in (0, 1)
        ]
        return vals[seat] - vals[1 - seat], True
    if sim.step_count >= sim.config["episodeSteps"] - 1:
        return float(sim.cash[seat] - sim.cash[1 - seat]), True
    return float(v[seat] - v[1 - seat]), False


def play_public_pair(cfg, obs, views, root_enemy_x, candidates, weights):
    """The caller supplies two candidate commands and pre-action information only."""
    seat = obs["player"]
    op = 1 - seat
    initial = estimate_view(cfg, obs, views[seat]).get_state()
    samples = []
    wall = time.perf_counter()
    for stream in range(2):
        their, _ = decode(views[op], root_enemy_x, weights, jitter(stream, obs["step"], op))
        out = []
        for orders in candidates:
            sim = FastCarbon(cfg)
            sim.set_state(initial)
            track = copy.deepcopy(views)
            pair = [None, None]
            pair[seat] = orders
            pair[op] = their
            scores = []
            closed = False
            for d in range(8):
                if d:
                    pair = []
                    for s in (0, 1):
                        x = track[s].observe(sim.obs(s))
                        chosen, _ = decode(track[s], x, weights, jitter(stream, sim.step_count, s))
                        pair.append(chosen)
                pre = list(sim.cash)
                sim.step(*pair)
                value, closed = leaf(sim, seat, pre)
                scores.append(value)
                if closed:
                    break
            out.append({"h1": scores[0], "h8": scores[-1], "depth": len(scores), "terminal": closed})
        samples.append({"stream": stream, "root_enemy_command": their, "arm": out})
    return dict(
        samples=samples,
        mean_delta={k: mean(row["arm"][1][k] - row["arm"][0][k] for row in samples) for k in ("h1", "h8")},
        wall_s=time.perf_counter() - wall,
    )


def registered_cases():
    tree = ACTOR / "tree_response_training"
    dirs = (
        [tree]
        + [tree / "batches" / f"intake_{i:02d}" for i in range(2, 6)]
        + [tree / "supplement_01/batches" / f"intake_{i:02d}" for i in range(6, 9)]
    )
    enroll = defaultdict(list)
    for idx, base in enumerate(dirs, 1):
        cases = sorted(base.glob("case_*/live_protocol.json"))
        require(len(cases) == 8, "lost pre-existing branch")
        for p in cases:
            reg = load(p)
            enroll[reg["record"]["source"]].append((reg["root_t"], idx, reg, p.parent))
    return enroll


def screen():
    fit_receipt = load(HERE / "fit.json")
    path = HERE / "rollout.npz"
    require(sha(path) == fit_receipt["weights_sha256"], "wrong self weights")
    train_terrain = {r["terrain"] for r in fit_receipt["trajectories"]}
    train_seeds = {r["seed"] for r in fit_receipt["trajectories"]}
    weights = dict(np.load(path))
    cases = registered_cases()
    predictions = []
    for source, roots in sorted(cases.items()):
        raw = load(ROOT / source)
        seed = raw["seed"]
        s = raw["candidate_seat"]
        cfg = raw["game_configuration"]
        require(
            raw["initial_terrain_u8_sha256"] not in train_terrain and seed not in train_seeds,
            "rollout/native maps overlap",
        )
        require(
            all(sha(ROOT / source) == reg["record"]["source_sha256"] for _, _, reg, _ in roots),
            "source mismatch",
        )
        bytime = {t: (batch, reg, folder) for t, batch, reg, folder in roots}
        require(len(bytime) == len(roots), "duplicate root in trajectory")
        observed = FastCarbon(cfg).reset(seed=seed)
        track = [F.Feat(i) for i in (0, 1)]
        for t in range(max(bytime) + 1):
            obs = observed.obs(s)
            track[s].observe(obs)
            enemy_simulated_view = public_enemy(obs, track[s].est)
            enemy_x = track[1 - s].observe(enemy_simulated_view)
            if t in bytime:
                batch, reg, folder = bytime[t]
                root = load(folder / "root.json.gz")
                archive = root["public_at_root"]
                require(
                    all(archive[k] == obs[k] for k in ("step", "carbon", "players", "trees", "player")),
                    "root public frame mismatch",
                )
                alternatives = [root["v1"], root["alternative"]]
                require(alternatives[0] != alternatives[1], "same candidates")
                val = play_public_pair(cfg, obs, track, enemy_x, alternatives, weights)
                armfiles = [load(folder / f"live_{label}.json") for label in ("parent", "alternative")]
                verify = []
                for i, a in enumerate(armfiles):
                    require(
                        a["pack_sha256"] == raw["pack_sha256"]
                        and a["source_sha256"]
                        == "65f268b2919ce30fe19355115a928eada032a6beaa018d801662d434c083f2b5",
                        "native identity",
                    )
                    require(
                        a["root_actions"][s] == alternatives[i]
                        and a["action_sha256"] == sha(folder / a["action_tape"]),
                        "different native command",
                    )
                    verify.append(
                        dict(
                            terminal_scores=a["score"],
                            margin=a["margin"],
                            search_calls=a["search_calls"],
                            tape_sha256=a["action_sha256"],
                        )
                    )
                require(
                    armfiles[0]["root_actions"][1 - s] == armfiles[1]["root_actions"][1 - s],
                    "native shared root response",
                )
                true = verify[1]["margin"] - verify[0]["margin"]
                rank = load(folder / "training_label.json")["target_delta_official_margin"]
                require(
                    abs(true - (rank if root["paid_arm"] == "alternative" else -rank)) < 1e-7,
                    "official margin mismatch",
                )
                preds = dict(
                    seed=seed,
                    physical_terrain=raw["initial_terrain_u8_sha256"],
                    seat=s,
                    action_t=t,
                    batch=batch,
                    case=str(folder.relative_to(ROOT)),
                    root_sha256=sha(folder / "root.json.gz"),
                    actual_native=verify,
                    realized_alternative_minus_parent=true,
                    **val,
                )
                predictions.append(preds)
            if t < max(bytime):
                observed.step(*raw["actions"][t])
    require(len(predictions) == 64, "not all roots")
    results = {}
    for label, rows in (
        ("discovery", [x for x in predictions if x["batch"] <= 6]),
        ("held", [x for x in predictions if x["batch"] > 6]),
    ):
        bymap = {}
        for seed in sorted({x["seed"] for x in rows}):
            gs = [x for x in rows if x["seed"] == seed]
            bymap[str(seed)] = {}
            for key in ("h1", "h8"):
                selected = [x for x in gs if x["mean_delta"][key] > 0]
                bymap[str(seed)][key] = {
                    "roots": len(gs),
                    "switched": len(selected),
                    "official_delta_vs_parent": sum(x["realized_alternative_minus_parent"] for x in selected)
                    / len(gs),
                }
        results[label] = dict(maps=bymap)
        for key in ("h1", "h8"):
            vals = [m[key]["official_delta_vs_parent"] for m in bymap.values()]
            results[label][key] = {
                "switched": sum(m[key]["switched"] for m in bymap.values()),
                "official_equal_map_delta": mean(vals),
                "positive_maps": sum(v > 0 for v in vals),
                "map_count": len(vals),
            }
    held = results["held"]
    a = held["h8"]
    d = results["discovery"]["h8"]
    gate = bool(
        d["official_equal_map_delta"] > 0
        and a["official_equal_map_delta"] > 0
        and a["positive_maps"] >= 3
        and a["official_equal_map_delta"] > held["h1"]["official_equal_map_delta"]
        and a["switched"] >= 4
    )
    costs = [x["wall_s"] for x in predictions]
    summary = {
        "estimator": "two modeled joint moves from the focal visible history, sharing Gumbel noise by UID; after a branch the opponent recomputes its own commands",
        "results": results,
        "gate_pass": gate,
        "elapsed_per_pair_seconds": {
            "median": median(costs),
            "p95": float(np.percentile(costs, 95)),
            "max": max(costs),
        },
        "post_root_original_games_started": 0,
        "formal_games": 0,
        "protocol_sha256": sha(HERE / "protocol.json"),
        "source_sha256": sha(Path(__file__)),
        "weights_sha256": sha(path),
        "native_roots": len(predictions),
    }
    save(HERE / "root_scores.json.gz", predictions)
    summary["evidence_sha256"] = sha(HERE / "root_scores.json.gz")
    save(HERE / "screen.json", summary)
    print(json.dumps({k: {z: v[z] for z in ("h1", "h8")} for k, v in results.items()}), gate, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["fit", "screen"])
    args = parser.parse_args()
    if args.stage == "fit":
        fit()
    else:
        screen()

"""Fit the frozen map-balanced return/evade readout and grade the actual opponent orders."""

import argparse, json, sys
from collections import Counter, defaultdict

from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

HERE = Path(__file__).resolve().parent
CORE = HERE.parent / "opponent_policy"
sys.path.insert(0, str(CORE))
from original_policy import PROTOCOL, free_entry, tape, module, sha

TURNS = set(range(163, 296, 4)) | {296, 297, 298}


def state_features(mod, players, me, t):
    op = 1 - me
    bases = list(players[op][1].values())
    units = players[op][2]
    other = players[me][2]
    eligible = {}
    for uid, (p, c, role) in units.items():
        if role != "COLLECTOR" or c < 50:
            continue
        d = min(mod.dist(p, b) for b in bases)
        left = 299 - t
        slack = left - d
        if d < 1 or (d > 4 and not 0 <= slack <= 4):
            continue
        light = [q for q, cc, rr in other.values() if rr == "COLLECTOR" and cc < c]
        tie = [q for q, cc, rr in other.values() if rr == "COLLECTOR" and cc == c]
        near = bool(any(mod.dist(p, q) <= 2 for q in light))
        descriptors = np.array(
            [
                1.0,
                np.clip(np.log1p(c) / 6.0, 0, 1.5),
                min(1.0, d / 5),
                np.clip(slack / 5.0, 0, 1.0),
                float(0 <= slack <= 1),
            ]
        )
        A = []
        for a in range(5):
            q = mod.move(p, a)
            dnext = min(mod.dist(q, b) for b in bases)
            switch = np.array(
                [
                    d - dnext,
                    float(a == 0),
                    float(q in bases),
                    float(any(mod.dist(q, r) <= 1 for r in light)),
                    float(any(mod.dist(q, r) <= 1 for r in tie)),
                ]
            )
            A.append(np.outer(descriptors, switch).ravel())
        eligible[uid] = dict(
            x=np.array(A), distance=d, cargo=c, light=light, near=near, slack=slack
        )
    return eligible


def materialize_game(c, records, mod):
    me = c["seat"]
    op = 1 - me
    tr = mod.BeliefTracker(mod._Bel, me)
    prev = None
    result = []
    for row in records:
        t = row["t"]
        obs = row["obs"]
        players = obs["players"]
        own = row["own"]
        other = row["other"]
        if obs["player"] != me or obs["step"] != t or "full_carbon" in obs:
            raise ValueError("Private or wrong observation")
        est, _ = tr.step(players, t, obs["carbon"])
        if t in TURNS:
            units = players[op][2]
            bases = players[op][1]
            eligible = state_features(mod, players, me, t)
            entry = dict(t=t, eligible=eligible)
            if eligible:
                xo = mod._quant(mod.encode(players, t, op, est, [1.0] * mod.N, prev))[
                    None
                ]
                lw, lb, _ = mod._net()(xo, mod._SID)
                lw, lb = mod._opponent_root_readout(xo, lw, lb, players, op)
                for uid, v in eligible.items():
                    p = units[uid][0]
                    v.update(
                        z=lw[0, p].copy(), label=mod.WIDX[other.get(uid) or "NONE"]
                    )
                entry.update(
                    players=players,
                    own=own,
                    other=other,
                    w={u: lw[0, v[0]].copy() for u, v in units.items()},
                    b={u: lb[0, p].copy() for u, p in bases.items()},
                )
            result.append(entry)
        tr.set_actions(players, own)
        prev = players
    if len(result) != len(TURNS):
        raise ValueError(
            "Early or incomplete: preregister all original source denominator"
        )
    return dict(**c, rows=result)


def archive_rows(c):
    raw = tape(c)["actions"]
    seat = c["candidate_seat"]
    for t, obs, cfg, a in free_entry.public_history({**c, "expert_seat": seat}):
        if a != raw[t][seat]:
            raise ValueError("Source order or seat mismatch")
        yield dict(t=t, obs=obs, own=a, other=raw[t][1 - seat])


def evidence(mod):
    historical = [
        (
            {
                "seed": c["seed"],
                "terrain": c["terrain"],
                "seat": c["candidate_seat"],
                "split": c["split"],
            },
            archive_rows(c),
        )
        for c in PROTOCOL["games"]
    ]
    import dev_tapes

    development = [
        ({k: g[k] for k in ("seed", "terrain", "seat")} | {"split": "dev1"}, g["rows"])
        for g in dev_tapes.games()
    ]
    if len(development) != 7:
        raise ValueError("Expected seven played development games")
    return [materialize_game(c, stream, mod) for c, stream in historical + development]


def sm(z):
    lp = z - logsumexp(z, axis=1)[:, None]
    return lp, np.exp(lp)


def fit(records):
    rows = defaultdict(list)
    for g in records:
        for r in g["rows"]:
            rows[g["terrain"]].extend(r["eligible"].values())
    data = [
        ((u["x"], u["z"], u["label"]), 1 / len(v)) for v in rows.values() for u in v
    ]
    x, z, y = map(np.asarray, zip(*(k for k, _ in data)))
    mass = np.array([s for _, s in data])
    mass /= mass.sum()
    z = z.astype(np.float64)
    lp0, prior = sm(z)

    def func(w):
        dz = x @ w
        lp, p = sm(z + dz)
        loss = (
            mass @ (-lp[np.arange(len(y)), y] + 0.5 * np.sum(prior * (lp0 - lp), 1))
            + 0.05 * w @ w
        )
        d = 1.5 * p - 0.5 * prior
        d[np.arange(len(y)), y] -= 1
        return loss, np.einsum("ij,ijk,i->k", d, x, mass) + 0.1 * w

    opt = minimize(
        func, np.zeros(x.shape[2]), method="L-BFGS-B", jac=True, options={"maxiter": 50}
    )
    for n in range(14):
        w = (opt.x * 2.0**-n).astype(np.float32)
        delta = x @ w
        lp, p = sm(z + delta)
        kl = mass @ np.sum(prior * (lp0 - lp), axis=1)
        if kl <= 0.025 and np.max(abs(delta)) <= 1.5:
            break
    else:
        raise ValueError("Cannot bound model readout")
    return w, dict(
        n=int(len(y)),
        maps=len(rows),
        halves=n,
        kl=float(kl),
        max_abs_delta=float(np.max(abs(delta))),
        status=int(opt.status),
    )


def paired_forecasts(row, seat, weights, model):
    """Run native team decoding twice with exactly the same root RNG and budget."""
    people = row["players"]
    opponent = 1 - seat
    step = row["t"]
    old = np.zeros((225, 5), np.float32)
    b = np.zeros((225, 3), np.float32)
    for uid, v in row["w"].items():
        old[people[opponent][2][uid][0]] = v
    for uid, v in row["b"].items():
        b[people[opponent][1][uid]] = v
    new = old.copy()
    for uid, u in row["eligible"].items():
        new[people[opponent][2][uid][0]] += u["x"] @ weights

    def sample(z):
        return model._sample(
            people,
            opponent,
            z,
            b,
            np.random.RandomState(
                (int(1000003 * (step + 1) + 17 * seat) % (2**31)) + 7
            ),
            step,
        )

    ca, cb = sample(old), sample(new)
    return ca, cb, old, new


def grade(records, w, mod):
    count = Counter()
    by_map = defaultdict(Counter)
    loss = []
    cases = []
    for g in records:
        me = g["seat"]
        op = 1 - me
        st = by_map[g["terrain"]]
        for r in g["rows"]:
            t = r["t"]
            st["all_roots"] += 1
            if not r["eligible"]:
                continue
            st["opportunity_roots"] += 1
            people = r["players"]
            ca, cb, old, new = paired_forecasts(r, me, w, mod)
            bases = set(people[op][1].values())
            for uid, u in r["eligible"].items():
                p = people[op][2][uid][0]
                y = u["label"]
                a = mod.WIDX[ca.get(uid) or "NONE"]
                k = mod.WIDX[cb.get(uid) or "NONE"]
                truth = mod.move(p, y)
                first = mod.move(p, a)
                second = mod.move(p, k)
                st["eligible_units"] += 1
                st["old_hit"] += int(y == a)
                st["new_hit"] += int(y == k)
                st["changed"] += int(a != k)
                st["newly_correct"] += int(a != y and k == y)
                st["newly_wrong"] += int(a == y and k != y)
                lp, _ = sm(old[p][None])
                ll, _ = sm(new[p][None])
                loss.append((g["terrain"], float(lp[0, y] - ll[0, y])))
                if u["distance"] == 1:
                    st["adjacent_to_home"] += 1
                    st["actual_home"] += int(truth in bases)
                    for n, q in [("old", first), ("new", second)]:
                        st[n + "_home_tp"] += int(q in bases and truth in bases)
                        st[n + "_home_fp"] += int(q in bases and truth not in bases)
                if u["near"]:
                    st["lighter_rival_within_2"] += 1
                    arrivals = {
                        mod.move(info[0], mod.WIDX[r["own"].get(v) or "NONE"])
                        for v, info in people[me][2].items()
                        if info[2] == "COLLECTOR" and info[1] < u["cargo"]
                    }
                    st["actual_lighter_collision"] += int(truth in arrivals)
                    for n, q in [("old", first), ("new", second)]:
                        st[n + "_collision_tp"] += int(
                            q in arrivals and truth in arrivals
                        )
                        st[n + "_collision_fp"] += int(
                            q in arrivals and truth not in arrivals
                        )
                if a != k:
                    cases.append(
                        dict(
                            seed=g["seed"],
                            terrain=g["terrain"],
                            seat=me,
                            step=t,
                            uid=uid,
                            cargo=u["cargo"],
                            distance=u["distance"],
                            actual=y,
                            old=a,
                            new=k,
                            near_light=u["near"],
                        )
                    )
    for m, d in by_map.items():
        count.update(d)
    return dict(
        counts=count,
        maps={key: dict(d) for key, d in by_map.items()},
        mean_equal_map_delta_nll=np.mean(
            [np.mean([v for h, v in loss if h == key]) for key in by_map]
        ),
        improving_maps=sum(d["new_hit"] > d["old_hit"] for d in by_map.values()),
        nonworsening_maps=sum(d["new_hit"] >= d["old_hit"] for d in by_map.values()),
        cases=cases,
    )


def collateral(g, w, m):
    me = g["seat"]
    op = 1 - me
    by = Counter()
    for r in g["rows"]:
        if not r["eligible"]:
            continue
        people = r["players"]
        wu = people[op][2]
        base = people[op][1]
        a, c, _, _ = paired_forecasts(r, me, w, m)
        real = r["other"]
        by["opportunity_roots"] += 1
        by["changed_team_roots"] += a != c
        for uid in [*base, *wu]:
            role = (
                "base"
                if uid in base
                else ("eligible" if uid in r["eligible"] else "other_worker")
            )
            aa = a.get(uid) or "NONE"
            bb = c.get(uid) or "NONE"
            yy = real.get(uid) or "NONE"
            by[role + "_orders"] += 1
            by[role + "_changed"] += aa != bb
            by[role + "_old_hit"] += aa == yy
            by[role + "_new_hit"] += bb == yy
    return by


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Directory for independently reconstructed outputs.",
    )
    args = parser.parse_args()
    if args.out.resolve() == HERE:
        raise ValueError("Choose a separate folder to reproduce the frozen evidence")
    args.out.mkdir(parents=True, exist_ok=True)
    baseline = (CORE / "candidate.py").read_bytes()
    if (
        sha(baseline)
        != "245b5814d1855c5449435ee6bd4a53560017a8cc7a46d08729fa9dfdc13e9654"
    ):
        raise ValueError("Different native opponent forecast")
    if (
        sha((HERE / "protocol.json").read_bytes())
        != "dc4f5edd25a8af8222964e5489f81e479cfca46d8549c9f581e2ebcf3abdbf76"
    ):
        raise ValueError("Different study contract")
    mod = module(baseline)
    games = evidence(mod)
    # Match the originally registered deterministic group and seat ordering.
    games.sort(key=lambda g: (g["split"], g["seed"], g["seat"]))
    parts = {
        key: [g for g in games if g["split"] == key] for key in ("train", "val", "dev1")
    }
    if [len(parts[k]) for k in parts] != [48, 8, 7]:
        raise ValueError("Wrong game denominator")
    maps = {s: {g["terrain"] for g in p} for s, p in parts.items()}
    if (
        [len(maps[s]) for s in parts] != [24, 4, 4]
        or len(set.union(*maps.values())) != 32
        or len({(g["terrain"], g["seat"]) for g in games}) != 63
    ):
        raise ValueError("Overlapping or duplicated physical terrain or seat")
    w, train = fit(parts["train"])
    np.save(args.out / "head.npy", w)
    val = grade(parts["val"], w, mod)
    additional = grade(parts["dev1"], w, mod)
    passes = lambda r: (
        r["counts"]["new_hit"] > r["counts"]["old_hit"] and r["nonworsening_maps"] >= 3
    )
    result = dict(
        protocol=sha((HERE / "protocol.json").read_bytes()),
        train=train,
        original_held=val,
        exposed_seven=additional,
        eligible_as_next_experiment=bool(passes(val) and passes(additional)),
        weights=w.tolist(),
    )
    (args.out / "result.json").write_text(
        json.dumps(result, allow_nan=False, sort_keys=True, indent=2) + "\n"
    )
    counts = {}
    for kind in ["val", "dev1"]:
        total = Counter()
        for g in parts[kind]:
            total.update(collateral(g, w, mod))
        counts[kind] = total
    (args.out / "secondary.json").write_text(
        json.dumps(counts, indent=2, sort_keys=True) + "\n"
    )
    for name in ["head.npy", "result.json", "secondary.json"]:
        observed = (args.out / name).read_bytes()
        expected = (HERE / name).read_bytes()
        print(
            name,
            "identical" if observed == expected else "different",
            sha(observed),
            flush=True,
        )
        if observed != expected:
            raise ValueError("Frozen file differs: " + name)


if __name__ == "__main__":
    main()

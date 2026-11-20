"""Fit exactly two 96-channel linear heads to source logits; freeze before reading held examples."""

import argparse, gzip, hashlib, io, json, os, sys, zipfile
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL = HERE.parent / "opponent_policy"
sys.path.insert(0, str(MODEL))


def sha(b):
    return hashlib.sha256(b).hexdigest()


def encode(d):
    return (json.dumps(d, ensure_ascii=False, sort_keys=True, allow_nan=False, indent=2) + "\n").encode()


def save(p, d):
    raw = encode(d)
    if p.suffix == ".gz":
        raw = gzip.compress(raw, mtime=0)
    tmp = p.with_suffix(".tmp")
    tmp.write_bytes(raw)
    tmp.replace(p)


def archive(cache, plan):
    out = HERE / "simulated_rows.zip"
    tmp = out.with_suffix(".tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED) as z:
        for name in ["version.json", *[c["sha256"] + ".npz" for c in plan["games"]]]:
            z.writestr(
                zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0)), (cache / name).read_bytes()
            )
    tmp.replace(out)


def load_split(zipfile_path, split, plan):
    import numpy as np

    rows = []
    digests = set()
    seen = defaultdict(set)
    source = []
    redundant = 0
    with zipfile.ZipFile(zipfile_path) as z:
        version = json.loads(z.read("version.json"))
        assert version["protocol"] == sha((HERE / "protocol.json").read_bytes()) and version[
            "collector"
        ] == sha((HERE / "collect.py").read_bytes())
        assert version["feature_code"] == sha((HERE / "timing/forward.py").read_bytes())
        for c in plan["games"]:
            if c["split"] != split:
                continue
            raw = z.read(c["sha256"] + ".npz")
            with np.load(io.BytesIO(raw), allow_pickle=False) as data:
                card = json.loads(gzip.decompress(data["meta"].tobytes()))
                assert (
                    str(data["source"]) == c["sha256"]
                    and card["split"] == split
                    and card["seat"] == c["candidate_seat"]
                    and c["terrain"] == card["terrain"]
                )
                source.append(
                    dict(
                        seed=c["seed"],
                        terrain=c["terrain"],
                        seat=c["candidate_seat"],
                        source_sha256=c["sha256"],
                        captured_sha256=sha(raw),
                        roots=card["roots"],
                    )
                )
                counters = Counter()
                for r in card["rows"]:
                    blocks = {}
                    for k in ("worker", "base"):
                        n = sum(slot[1] == k for slot in r["slots"])
                        i = counters[k]
                        counters[k] += n
                        blocks[k] = (data[k][i : i + n], data[k + "_z"][i : i + n])
                    if r["hash"] in seen[c["terrain"]]:
                        redundant += 1
                        continue
                    seen[c["terrain"]].add(r["hash"])
                    digests.add(r["hash"])
                    rows.append((c["terrain"], c["seed"], c["candidate_seat"], r, blocks))
                assert all(
                    counters[k] == len(data[k]) == len(data[k + "_z"]) for k in ("worker", "base")
                )
    return rows, digests, source, redundant


def soft(z):
    import numpy as np

    q = np.exp(z - z.max(axis=-1, keepdims=True))
    return q / q.sum(axis=-1, keepdims=True)


def divergence(y, z):
    from scipy.special import logsumexp

    p = soft(y)
    return (p * (y - z)).sum(axis=-1) - logsumexp(y, axis=-1) + logsumexp(z, axis=-1)


def train(rows, k):
    import numpy as np
    from scipy.optimize import minimize
    from scipy.special import logsumexp

    counts = Counter()
    keys = defaultdict(set)
    for terrain, seed, seat, r, b in rows:
        root = (terrain, seat, r["root"])
        n = len(b[k][0])
        if n:
            keys[terrain].add(root)
            counts[root] += n
    batch = []
    target = []
    weights = []
    for terrain, seed, seat, r, b in rows:
        x, y = b[k]
        root = (terrain, seat, r["root"])
        n = len(x)
        if n:
            batch.extend(x)
            target.extend(y)
            weights.extend([1 / (len(keys) * len(keys[terrain]) * counts[root])] * n)
    x = np.asarray(batch, dtype="float64")
    y = np.asarray(target, dtype="float64")
    v = np.asarray(weights)
    assert np.isclose(v.sum(), 1) and np.isfinite(x).all() and np.isfinite(y).all()
    avg = v @ x
    scale = np.sqrt(v @ (x - avg) ** 2).clip(1.0e-4)
    x = np.column_stack(((x - avg) / scale, np.ones(len(x))))
    p = soft(y)
    classes = p.shape[1]

    def objective(flat):
        w = flat.reshape(x.shape[1], classes)
        z = x @ w
        loss = v @ (logsumexp(z, axis=1) - (p * z).sum(axis=1)) + 0.5e-4 * (w[:-1] ** 2).sum()
        grad = x.T @ (v[:, None] * (soft(z) - p))
        grad[:-1] += 1.0e-4 * w[:-1]
        return loss, grad.ravel()

    fit = minimize(
        objective,
        np.zeros(x.shape[1] * classes),
        method="L-BFGS-B",
        jac=True,
        options=dict(maxiter=100, ftol=1.0e-11, gtol=1.0e-7),
    )
    w = fit.x.reshape(x.shape[1], classes)
    coef = w[:-1] / scale[:, None]
    intercept = w[-1] - avg @ coef
    return (
        coef.astype("float32"),
        intercept.astype("float32"),
        dict(
            iterations=int(fit.nit),
            status=int(fit.status),
            slots=len(x),
            objective=float(fit.fun),
            maps=len(keys),
            roots=len(counts),
            max_abs_gradient=float(abs(fit.jac).max()),
        ),
    )


def evaluate(rows, heads, m, include_details):
    import numpy as np

    stats = defaultdict(Counter)
    kl = defaultdict(lambda: defaultdict(list))
    detail = []
    for terrain, game, seat, r, blocks in rows:
        p = r["players"]
        op = r["enemy"]
        units = p[op][2]
        bases = p[op][1]
        workers = p[1 - op][2]
        teacher = []
        student = []
        ablation = []
        for k, c in (("worker", 5), ("base", 3)):
            x, z = blocks[k]
            w, b = heads[k]
            pred = x @ w + b
            hw, hb = (m._NET.hw, m._NET.hb)[k == "base"]
            old = x @ hw.numpy() + hb.numpy()
            a = np.zeros((225, c), "float32")
            v = a.copy()
            q = a.copy()
            positions = [slot[2] for slot in r["slots"] if slot[1] == k]
            a[positions] = z
            v[positions] = pred
            q[positions] = old
            teacher.append(a)
            student.append(v)
            ablation.append(q)
            for model, pr in [("trained", pred), ("untrained", old)]:
                kl[terrain][model + "_" + k].extend(
                    divergence(z.astype("float64"), pr.astype("float64")).tolist()
                )
        native = lambda h: m._sample(p, op, h[0], h[1], np.random.RandomState(r["seed"]), r["t"])
        actual, trained, prior = [native(h) for h in (teacher, student, ablation)]
        assert actual == r["teacher"]
        result = {}
        for name, cmd in [("trained", trained), ("untrained", prior)]:
            totals = stats[(terrain, name)]
            totals["rows"] += 1
            totals["joint_agree"] += int(cmd == actual)
            own_dst = {m.move(v[0], m.WIDX[r["own"].get(uid) or "NONE"]) for uid, v in workers.items()}
            recruited = sum(cmd.get(uid) is not None for uid in bases)
            assert recruited <= min(len(bases), max(0, 10 - len(units)), int(p[op][0] // 30))
            assert set(cmd) <= units.keys() | bases.keys() and all(
                a in (m.WIDX if u in units else m.BIDX) for u, a in cmd.items()
            )
            for uid in bases:
                totals["base_n"] += 1
                totals["base_agree"] += int(cmd.get(uid) == actual.get(uid))
            for uid, v in units.items():
                pos, load, role = v
                groups = ["worker"]
                if role == "COLLECTOR" and load >= 50:
                    groups.append("loaded")
                if role == "COLLECTOR" and min(m.dist(pos, a) for a in bases.values()) <= 2:
                    groups.append("near_home")
                if any(
                    m.dist(pos, vv[0]) <= 2
                    for u, vv in list(units.items()) + list(workers.items())
                    if u != uid
                ):
                    groups.append("crowded")
                if any(
                    w[2] == "COLLECTOR" and w[1] >= 50 and m.dist(pos, w[0]) <= 2
                    for w in workers.values()
                ):
                    groups.append("near_enemy_cargo")
                target = m.move(pos, m.WIDX[actual.get(uid) or "NONE"])
                forecast = m.move(pos, m.WIDX[cmd.get(uid) or "NONE"])
                for group in groups:
                    totals[group + "_n"] += 1
                    totals[group + "_agree"] += int(target == forecast)
                totals["teacher_contact"] += int(target in own_dst)
                totals["contact_hit"] += int(forecast in own_dst and target in own_dst)
                totals["contact_false_positive"] += int(forecast in own_dst and target not in own_dst)
            if include_details:
                result[name] = cmd
        if include_details:
            detail.append(
                dict(
                    terrain=terrain,
                    seed=game,
                    seat=seat,
                    root=r["root"],
                    depth=r["depth"],
                    branch=r["branch"],
                    enemy=op,
                    hash=r["hash"],
                    teacher=actual,
                    own=r["own"],
                    **result
                )
            )
    bymap = {}
    for terrain in sorted(kl):
        bymap[terrain] = dict(
            counts={name: dict(stats[(terrain, name)]) for name in ("trained", "untrained")},
            teacher_kl={k: float(np.mean(v)) for k, v in kl[terrain].items()},
        )
    total = {
        name: sum((stats[(g, name)] for g in bymap), Counter()) for name in ["trained", "untrained"]
    }
    mapmean = {
        k: float(np.mean([v["teacher_kl"][k] for v in bymap.values()]))
        for k in next(iter(bymap.values()))["teacher_kl"]
    }
    return dict(counts=total, map_equal_teacher_kl=mapmean, by_map=bymap), detail


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path)
    ap.add_argument("--cpu", type=int, required=True)
    args = ap.parse_args()
    os.sched_setaffinity(0, {args.cpu})
    for k in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
        os.environ[k] = "1"
    import numpy as np
    from original_policy import PROTOCOL, original, module
    from collect import PLAN

    if args.cache is not None:
        archive(args.cache, PROTOCOL)
    data = HERE / "simulated_rows.zip"
    trainset, trainhash, traincards, td = load_split(data, "train", PROTOCOL)
    m = module(original())
    heads = {}
    log = {}
    params = {}
    for k in ("worker", "base"):
        w, b, log[k] = train(trainset, k)
        heads[k] = (w, b)
        params[k + "_weight"] = w
        params[k + "_bias"] = b
    assert sum(v.size for v in params.values()) == 776
    np.savez_compressed(HERE / "weights.npz", **params)
    train_result, _ = evaluate(trainset, heads, m, False)
    held, hhash, heldcards, hd = load_split(data, "val", PROTOCOL)
    assert len({g["terrain"] for g in traincards}) == 24 and len({g["terrain"] for g in heldcards}) == 4
    assert not hhash & trainhash and not {g["terrain"] for g in traincards} & {
        g["terrain"] for g in heldcards
    }
    assert len({(c["terrain"], c["seat"]) for c in traincards + heldcards}) == 56
    val, details = evaluate(held, heads, m, True)
    t = val["counts"]["trained"]
    kl = val["map_equal_teacher_kl"]
    checks = dict(
        coverage=all(r["captured_rows"] > 0 for c in heldcards for r in c["roots"])
        and len(heldcards) == 8,
        worker_kl=kl["trained_worker"] <= 0.04 and kl["trained_worker"] < kl["untrained_worker"],
        base_kl=kl["trained_base"] <= 0.04 and kl["trained_base"] < kl["untrained_base"],
        joint_agree=t["joint_agree"] >= 0.8 * t["rows"],
        worker_agree=t["worker_agree"] >= 0.94 * t["worker_n"],
        base_agree=t["base_agree"] >= 0.97 * t["base_n"],
    )
    for k in ("loaded", "near_home", "crowded"):
        checks[k] = t[k + "_n"] >= 30 and t[k + "_agree"] >= 0.9 * t[k + "_n"]
    save(HERE / "held_orders.json.gz", details)
    report = dict(
        schema="native-enemy-776-distill/1",
        trainer_sha256=sha(Path(__file__).read_bytes()),
        protocol_sha256=sha((HERE / "protocol.json").read_bytes()),
        data_sha256=sha(data.read_bytes()),
        weights_sha256=sha((HERE / "weights.npz").read_bytes()),
        held_orders_sha256=sha((HERE / "held_orders.json.gz").read_bytes()),
        new_parameters=776,
        original_source=PLAN["source"],
        fit=log,
        train=train_result,
        validation=val,
        checks=checks,
        eligible=all(checks.values()),
        source_fingerprints_overlap=0,
        train_rows=len(trainset),
        val_rows=len(held),
        cross_tape_duplicates_removed=[td, hd],
        cards=traincards + heldcards,
    )
    save(HERE / "fit.json", report)
    print(report["eligible"], checks, val["map_equal_teacher_kl"], val["counts"]["trained"])


if __name__ == "__main__":
    main()

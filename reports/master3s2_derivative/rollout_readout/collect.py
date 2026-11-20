"""Tap the actual original's enemy _sample calls inside native simulations on preassigned historical roots."""

import argparse, copy, gzip, hashlib, json, os, sys, time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
POLICY = HERE.parent / "opponent_policy"
sys.path[:0] = [str(POLICY), str(HERE / "timing")]
PLAN = json.loads((HERE / "protocol.json").read_bytes())


def sha(x):
    return hashlib.sha256(x).hexdigest()


def capture(m, tr, prev, obs, cfg, card, initial):
    import numpy as np

    n = obs["step"]
    me = obs["player"]
    seen = Counter()
    taken = []
    boards = {}
    own = {}
    turn_native, sample_native = m._turn, m._sample

    def turn(src, configuration, orders):
        after = turn_native(src, configuration, orders)
        boards[id(after["players"])] = (after, src["players"])
        return after

    def sample(players, side, lw, lb, rng, t, fixed=None):
        result = sample_native(players, side, lw, lb, rng, t, fixed)
        depth = t - n
        if depth in PLAN["depths"]:
            ref = id(players)
            if side == me:
                own[ref] = result
            else:
                seen[depth] += 1
                if seen[depth] <= PLAN["limit_per_depth"]:
                    state, before = boards[ref]
                    taken.append((state, before, lw, lb, result, depth, seen[depth], own[ref]))
        return result

    m._ST = dict(
        tr=copy.deepcopy(tr),
        prev=prev,
        rng=np.random.RandomState(
            int.from_bytes(
                hashlib.sha256(f"enemy-rollout-readout/{card['terrain']}/{me}/{n}".encode()).digest()[
                    :4
                ],
                "big",
            )
        ),
    )
    m._S_COST[:] = initial
    m._S_STATS.pop("last", None)
    m._turn, m._sample = turn, sample
    try:
        issued = m.agent(obs, {k: v for k, v in cfg.items() if k != "randomSeed"})
    finally:
        m._turn, m._sample = turn_native, sample_native
    last = m._S_STATS.get("last")
    return taken, dict(
        step=n,
        issued=issued,
        seen=seen,
        candidates=last[0] if last else 1,
        completed_depth=last[1] if last else 0,
        rollouts=last[2] if last else 0,
        wall=last[3] if last else None,
        initial_costs=initial,
    )


def collect_game(card, m, initial, cache):
    import numpy as np
    from original_policy import free_entry
    from forward import frozen_targets

    side = card["candidate_seat"]
    tr = m.BeliefTracker(m._Bel, side)
    prev = None
    rows = []
    roots = []
    hashed = set()
    for n, obs, cfg, issued in free_entry.public_history({**card, "expert_seat": side}):
        if n in PLAN["root_steps"]:
            sampled, res = capture(m, tr, prev, obs, cfg, card, initial)
            res["captured_rows"] = len(sampled)
            res["duplicate_within_tape"] = 0
            roots.append(res)
            for board, before, lw, lb, order, depth, idx, own in sampled:
                t = board["step"]
                assert t == n + depth
                p = board["players"]
                op = 1 - side
                x = m._quant(m.encode(p, t, op, board["carbon"], [1.0] * m.N, before)).astype(np.float32)
                key = sha(x.tobytes())
                if key in hashed:
                    res["duplicate_within_tape"] += 1
                    continue
                hashed.add(key)
                h, w, b = frozen_targets(m._NET, x[None], m._SID)
                # Net microbatch can change float summation order. The actual traced logits remain the supervisory targets.
                err = max(float(np.abs(w[0] - lw).max()), float(np.abs(b[0] - lb).max()))
                assert err < 0.006
                teacher_seed = int(1000003 * (n + 1) + 17 * side) % (2**31) + 7919 * depth + 2
                assert (
                    m._S_M == m._S_MMAX == 1
                    and m._sample(p, op, lw, lb, np.random.RandomState(teacher_seed), t) == order
                )
                pieces = []
                for units, kind, logits in ((p[op][2], "worker", lw), (p[op][1], "base", lb)):
                    for uid, item in units.items():
                        pos = item[0] if kind == "worker" else item
                        pieces.append((uid, kind, pos, h[0, pos].copy(), logits[pos].copy()))
                rows.append(
                    (
                        dict(
                            hash=key,
                            root=n,
                            t=t,
                            depth=depth,
                            branch=idx,
                            enemy=op,
                            players=p,
                            teacher=order,
                            own=own,
                            seed=teacher_seed,
                            teacher_check_max=err,
                            slots=[[u, k, pos] for u, k, pos, hf, z in pieces],
                        ),
                        [(hf, z) for u, k, p0, hf, z in pieces],
                    )
                )
        tr.step(obs["players"], n, obs["carbon"])
        tr.set_actions(obs["players"], issued)
        prev = obs["players"]
    heads = {k: [] for k in ("worker", "base")}
    targets = {k: [] for k in heads}
    for meta, values in rows:
        for (uid, kind, p0), (hf, z) in zip(meta["slots"], values):
            heads[kind].append(hf)
            targets[kind].append(z)
    inp = {k: np.stack(val) for k, val in heads.items() if val}
    dest = cache / (card["sha256"] + ".npz")
    np.savez_compressed(
        dest,
        source=card["sha256"],
        **inp,
        **{k + "_z": np.stack(v) for k, v in targets.items() if v},
        meta=np.frombuffer(
            gzip.compress(
                json.dumps(
                    dict(
                        seed=card["seed"],
                        terrain=card["terrain"],
                        seat=side,
                        split=card["split"],
                        rows=[r for r, _ in rows],
                        roots=roots,
                    )
                ).encode(),
                mtime=0,
            ),
            dtype=np.uint8,
        ),
    )
    print(card["split"], card["seed"], side, len(rows), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--cpu", type=int, required=True)
    args = ap.parse_args()
    os.sched_setaffinity(0, {args.cpu})
    for k in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
        os.environ[k] = "1"
    os.environ["CARBON_DEBUG"] = "1"
    from original_policy import original, module, PROTOCOL

    assert sha((POLICY / "protocol.json").read_bytes()) == PLAN["source_protocol_sha256"]
    src = original()
    assert sha(src) == PLAN["source"]
    m = module(src)
    initial = list(m._S_COST)
    assert m._S_ABORT == 0.58 and len(m._NET.blocks) == 6 and type(m._NET).__name__ == "_WinoNet"
    args.cache.mkdir(exist_ok=True, parents=True)
    groups = {x: {c["terrain"] for c in PROTOCOL["games"] if c["split"] == x} for x in ("train", "val")}
    assert not groups["train"] & groups["val"] and [len(g) for g in groups.values()] == [24, 4]
    markers = dict(
        protocol=sha((HERE / "protocol.json").read_bytes()),
        collector=sha(Path(__file__).read_bytes()),
        feature_code=sha((HERE / "timing/forward.py").read_bytes()),
    )
    version = args.cache / "version.json"
    if version.exists():
        assert json.loads(version.read_bytes()) == markers
    else:
        version.write_text(json.dumps(markers))
    for c in PROTOCOL["games"]:
        out = args.cache / (c["sha256"] + ".npz")
        if not out.exists():
            collect_game(c, m, initial, args.cache)


if __name__ == "__main__":
    main()

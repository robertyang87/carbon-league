"""Compare both frozen enemy-root forecasts on the public requests of the registered real games."""

import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import zipfile

HERE = Path(__file__).resolve().parent
HOSTED = HERE.parent / "hosted"
MODEL = HERE.parents[1] / "opponent_policy"
sys.path.insert(0, str(MODEL))
STEPS = tuple(range(7, 294, 13))
ARM = "opponent_policy"
SOURCE = "65f268b2919ce30fe19355115a928eada032a6beaa018d801662d434c083f2b5"
NEW = "245b5814d1855c5449435ee6bd4a53560017a8cc7a46d08729fa9dfdc13e9654"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def need(ok, msg):
    if not ok:
        raise ValueError(msg)


def compact(value):
    return (json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n").encode()


def read(path):
    return json.loads(path.read_bytes())


def write(path, value):
    raw = compact(value)
    if path.suffix == ".gz":
        raw = gzip.compress(raw, mtime=0)
    tmp = path.with_name("." + path.name + ".tmp")
    tmp.write_bytes(raw)
    tmp.replace(path)


def measure(acc, kind, old, new, truth):
    a, b, c = old == truth, new == truth, old != new
    acc[kind + "_n"] += 1
    acc[kind + "_old"] += int(a)
    acc[kind + "_new"] += int(b)
    if c:
        acc[kind + "_changed"] += 1
        acc[kind + ("_fix" if b else "_harm" if a else "_both_wrong")] += 1


def compare(base, new, obs, previous, estimated, net):
    import numpy as np

    me, t, p = obs["player"], obs["step"], obs["players"]
    enemy = 1 - me
    xo = base._quant(base.encode(p, t, enemy, estimated, [1.0] * base.N, previous))[None]
    lw, lb, _ = net(xo.astype(np.float32), base._SID)
    nw, nb = new._opponent_root_readout(xo, lw.copy(), lb.copy(), p, enemy)
    seed = int(1000003 * (t + 1) + 17 * me) % (2**31) + 7

    def predict(w, b):
        return base._sample(p, enemy, w[0], b[0], np.random.RandomState(seed), t)

    return predict(lw, lb), predict(nw, nb), ((lw, lb), (nw, nb))


def tally(base, obs, own, truth, old, new, logits, agg):
    us = obs["player"]
    op = 1 - us
    players = obs["players"]
    cash, bases, workers, _ = players[op]
    initial_slots = max(0, min(10 - len(workers), int(float(cash) // 30)))
    remain = initial_slots
    own_arr = {base.move(w[0], base.WIDX[own.get(uid) or "NONE"]) for uid, w in players[us][2].items()}
    measure(agg, "joint", old, new, truth)
    changed = []
    base_detail = []
    for k, (units, typ, idx) in enumerate(((workers, "worker", base.WIDX), (bases, "base", base.BIDX))):
        for uid, entry in units.items():
            pos = entry[0] if k == 0 else entry
            actual, a, b = [v.get(uid) or "NONE" for v in (truth, old, new)]
            measure(agg, typ, a, b, actual)
            measure(agg, "entity", a, b, actual)
            access = remain > 0
            if k == 1:
                if initial_slots:
                    measure(agg, "base_before_joint", a, b, actual)
                if access:
                    measure(agg, "base_available", a, b, actual)
                    measure(agg, "hire_available", a != "NONE", b != "NONE", actual != "NONE")
                    agg["base_available_actual_hires"] += int(actual != "NONE")
                    agg["base_available_old_hires"] += int(a != "NONE")
                    agg["base_available_new_hires"] += int(b != "NONE")
                else:
                    measure(agg, "base_unavailable", a, b, actual)
                base_detail.append(
                    dict(
                        uid=uid,
                        pos=pos,
                        pre_cash=cash,
                        pre_workers=len(workers),
                        joint_slots=initial_slots,
                        slots_after_actual_predecessors=remain,
                        original=a,
                        trained=b,
                        actual=actual,
                    )
                )
                if actual != "NONE":
                    need(remain > 0, "Examiner hired outside registered cash and worker cap")
                    remain -= 1
            else:
                role = "collector" if entry[2] == "COLLECTOR" else "planter"
                measure(agg, role, a, b, actual)
                dst = [base.move(pos, idx[v]) for v in (actual, a, b)]
                agg["contact_actual"] += int(dst[0] in own_arr)
                for dstk, side in [(dst[1], "old"), (dst[2], "new")]:
                    agg["contact_" + side + "_hit"] += int(dstk in own_arr and dst[0] in own_arr)
                    agg["contact_" + side + "_fp"] += int(dstk in own_arr and dst[0] not in own_arr)
            if a != b:
                changed.append(
                    dict(
                        uid=uid,
                        kind=typ,
                        position=pos,
                        actual=actual,
                        original=a,
                        trained=b,
                        destination=(
                            [base.move(pos, idx[v]) for v in (actual, a, b)] if k == 0 else None
                        ),
                    )
                )
            if k == 0 or access:
                group = "worker" if k == 0 else "base_available"
                for heads, which in zip(logits, ("old", "new")):
                    z = heads[k][0, pos].astype("float64")
                    agg[group + "_raw_nll_" + which] += float(
                        math.log(sum(math.exp(float(v - z.max())) for v in z))
                        + z.max()
                        - z[idx[actual]]
                    )
    return dict(
        step=obs["step"],
        changed=changed,
        bases=base_detail,
        own=own,
        enemy_truth=truth,
        original=old,
        trained=new,
    )


def archive_game(p, models, reg):
    base, new = models
    with zipfile.ZipFile(p) as z:
        get = lambda name: z.read(ARM + "/" + name)
        receipt_raw = get("receipt.json")
        r = json.loads(receipt_raw)
        sign = json.loads(get("registration.json"))
        tpath = ARM + "/turns.jsonl"
        turns = z.read(tpath) if tpath in z.namelist() else b""
        observations = get("observations.jsonl.gz")
    seed, s = r["seed"], r["candidate_seat"]
    need(
        sign["matrix_sha256"] == sha((HERE.parent / "protocol.json").read_bytes())
        and reg["physical_boards"][str(seed)] == sign["physical_map_sha256"]
        and sign["candidate_seat"] == s
        and sign["files"]["candidate"]["sha256"] == NEW
        and sign["files"]["opponent"]["sha256"] == SOURCE,
        "Game source mismatch",
    )
    need(not turns or sha(turns) == r["turns_sha256"], "Captured orders differ")
    tracker = base.BeliefTracker(base._Bel, s)
    prev = None
    agg = Counter()
    roots = []
    n = 0
    commands = [json.loads(line) for line in turns.splitlines()]
    public = [json.loads(line) for line in gzip.decompress(observations).splitlines()]
    need(len(commands) == len(public), "Archive length mismatch")
    for t, (turn, frame) in enumerate(zip(commands, public)):
        obs = frame["observations"][s]
        need(
            t == turn["step"] == obs["step"] == frame["step"]
            and obs["player"] == s
            and "full_carbon" not in obs
            and sha(compact(obs))
            == turn["public_observation_sha256"][s]
            == frame["public_observation_sha256"][s],
            "Bad candidate-side transport input",
        )
        if not all(v["status"] == "ok" for v in turn["calls"]):
            need(t == len(public) - 1, "Replay continued after failure")
            break
        players = obs["players"]
        est, _ = tracker.step(players, t, obs["carbon"])
        if t in STEPS:
            old, trained, logits = compare(base, new, obs, prev, est, base._net())
        # Opponent and same-step own command are read after the network outputs.
        joint = [v["response"]["action"] for v in turn["calls"]]
        if t in STEPS:
            roots.append(tally(base, obs, joint[s], joint[1 - s], old, trained, logits, agg))
        tracker.set_actions(players, joint[s])
        prev = players
        n += 1
    need(
        n == r["official_transitions"] and [v["step"] for v in roots] == [t for t in STEPS if t < n],
        "Partial physical root coverage",
    )
    for group in [
        "joint",
        "worker",
        "base",
        "base_before_joint",
        "base_available",
        "base_unavailable",
        "hire_available",
        "entity",
        "collector",
        "planter",
    ]:
        need(
            agg[group + "_new"] - agg[group + "_old"] == agg[group + "_fix"] - agg[group + "_harm"]
            and agg[group + "_changed"]
            == sum(agg[group + a] for a in ("_fix", "_harm", "_both_wrong")),
            "Changed prediction conservation",
        )
    return dict(
        seed=seed,
        seat=s,
        terrain=sign["physical_map_sha256"],
        archive_sha256=sha(p.read_bytes()),
        receipt_sha256=sha(receipt_raw),
        public_observations_sha256=sha(observations),
        status=r["status"],
        official_steps=n,
        counts=agg,
        roots=roots,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument(
        "--prepare",
        action="store_true",
        help="Process archives already received; publish only after every registered slot is settled",
    )
    ap.add_argument("--cpu", type=int, default=4)
    args = ap.parse_args()
    os.sched_setaffinity(0, {args.cpu})
    for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from original_policy import original, module

    reg = read(HERE.parent / "protocol.json")
    src = original()
    candidate = (MODEL / "candidate.py").read_bytes()
    need(sha(src) == SOURCE and sha(candidate) == NEW, "Model identity")
    args.cache.mkdir(exist_ok=True, parents=True)
    models = None
    results = {}
    program = sha(Path(__file__).read_bytes())
    for p in sorted(HOSTED.glob("*.zip")):
        key = sha(p.read_bytes())
        cached = args.cache / (key + "-" + program + ".json.gz")
        if cached.exists():
            game = json.loads(gzip.decompress(cached.read_bytes()))
        else:
            if models is None:
                models = (module(src), module(candidate))
            game = archive_game(p, models, reg)
            write(cached, game)
        need(game["archive_sha256"] == key, "Cache mismatch")
        ident = (game["seed"], game["seat"])
        need(ident not in results, "Duplicate game")
        results[ident] = game
    if args.prepare:
        print("source archives read", len(results), flush=True)
        return
    settlement = HOSTED / "settlement.json"
    settled = read(settlement)
    need(
        settled["records_reconciled"]
        and settled["protocol_sha256"] == sha((HERE.parent / "protocol.json").read_bytes()),
        "Unresolved official verification",
    )
    cards = {(g["seed"], g["seat"]): g for g in settled["games"]}
    tickets = {(t["seed"], t["seat"]) for t in reg["schedule"]}
    archives = {(g["seed"], g["seat"]): g for g in settled["archives"]}
    need(
        set(results) == set(cards) == set(archives) == tickets and len(tickets) == 8, "Incorrect cohort"
    )
    totals = Counter()
    for slot, g in results.items():
        c = cards[slot]
        a = archives[slot]
        need(
            g["archive_sha256"] == a["sha256"]
            and g["receipt_sha256"] == c["receipt_sha256"]
            and g["public_observations_sha256"] == c["public_observations_sha256"]
            and g["status"] == c["status"],
            "Settlement mismatch",
        )
        totals.update(g["counts"])
    dest = HERE / "forecast.json.gz"
    write(
        dest,
        dict(
            schema="dev1-real-opponent-forecast/1",
            program_sha256=program,
            root_steps=STEPS,
            candidate_sha256=NEW,
            original_sha256=SOURCE,
            dev_protocol_sha256=settled["protocol_sha256"],
            settlement_sha256=sha(settlement.read_bytes()),
            run_id=settled["run_id"],
            counts=totals,
            games=[results[key] for key in sorted(results)],
        ),
    )
    print(sha(dest.read_bytes()), len(results), totals, flush=True)


if __name__ == "__main__":
    main()

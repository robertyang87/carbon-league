"""Replay late enemy-root forecasts with the native A0 decoder and verify actual worker deposits."""

import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import types
import zipfile
from collections import Counter, defaultdict

HERE = Path(__file__).resolve().parent
DEV = HERE.parent
MODEL = DEV.parent / "opponent_policy"
SELECTION_START = 283
OLD = "65f268b2919ce30fe19355115a928eada032a6beaa018d801662d434c083f2b5"
NEW = "245b5814d1855c5449435ee6bd4a53560017a8cc7a46d08729fa9dfdc13e9654"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def decode(native, obs, x, lw, lb, seed, enable_return):
    """Execute native _act with the already computed enemy-view features, logits and paired RNG."""
    import numpy as np

    side = 1 - obs["player"]
    view = dict(obs, player=side)
    zero_history = types.SimpleNamespace(step=lambda *args: (None, None), set_actions=lambda *args: None)
    stub = types.SimpleNamespace(forward=lambda *args: (lw, lb))
    protected = {}
    by_cell = {w[0]: uid for uid, w in view["players"][side][2].items()}

    def home(position, load, probs, feature, bases, enemies, t, profile):
        answer = native.go_home(position, load, probs, feature, bases, enemies, t, profile)
        if answer is not None:
            protected[by_cell[position]] = native.WACT[answer]
        return answer

    ctx = dict(
        native.__dict__,
        _ST=dict(tr=zero_history, prev=None, rng=np.random.RandomState(seed)),
        _POLICY=stub,
        encode=lambda *args: x,
        _HOME=native._HOME if enable_return else None,
        go_home=home,
    )
    # Configuration=None returns exactly the native own action sampler/decoder before _search.
    wrapped = types.FunctionType(native._act.__code__, ctx, argdefs=native._act.__defaults__)
    return wrapped(view, None), protected


def forecast(native, model, obs, previous, estimated):
    import numpy as np

    me, t = obs["player"], obs["step"]
    op = 1 - me
    p = obs["players"]
    x = native._quant(native.encode(p, t, op, estimated, [1.0] * native.N, previous))
    assert np.array_equal(x, native._quant(x))
    lw, lb, _ = native._net()(x[None].astype(np.float32), native._SID)
    lw, lb = model._opponent_root_readout(x[None], lw, lb, p, op)
    seed = int(1000003 * (t + 1) + 17 * me) % (2**31) + 7
    old = native._sample(p, op, lw[0], lb[0], np.random.RandomState(seed), t)
    disabled, none = decode(native, obs, x, lw[0], lb[0], seed, False)
    guarded, protected = decode(native, obs, x, lw[0], lb[0], seed, True)
    assert old == disabled and not none
    return old, guarded, protected


def game(card, archive, native, model, ledger):
    from numpy import isfinite

    with zipfile.ZipFile(archive) as z:
        name = card["arm"]
        payload = z.read(name + "/observations.jsonl.gz")
        assert sha(payload) == card["public_observations_sha256"]
        assert sha(z.read(name + "/receipt.json")) == card["receipt_sha256"]
        book = ledger.replay(card, z, ledger.config())
    frames = gzip.decompress(payload).splitlines()
    turns = book["transitions"]
    me = card["seat"]
    op = 1 - me
    payments = defaultdict(list)
    for turn in turns:
        for event in turn["cash_events"]:
            if event[0] in ("own_delivery", "opponent_worker_delivery_to_this_base"):
                payments[event[3]].append(
                    dict(
                        t=turn["t"], account_seat=event[1], kind=event[0], amount=event[2], cell=event[4]
                    )
                )
    counts = Counter()
    samples = []
    prev = None
    tracker = native.BeliefTracker(native._Bel, me)
    for t, row in enumerate(turns):
        frame = json.loads(frames[t])
        obs = frame["observations"][me]
        orders = row["commands_by_seat"]
        p = obs["players"]
        assert (obs["step"], obs["player"]) == (t, me) and "full_carbon" not in obs
        encoded = (json.dumps(obs, separators=(",", ":"), allow_nan=False) + "\n").encode()
        assert (
            sha(encoded)
            == frame["public_observation_sha256"][me]
            == row["public_observation_sha256"][me]
        )
        est, _ = tracker.step(p, t, obs["carbon"])
        for j, lbl in ((me, "own"), (op, "enemy")):
            cash, bases, workers, _ = p[j]
            hires = {cell for uid, cell in bases.items() if orders[j].get(uid)}
            arrived = {
                native.move(w[0], native.WIDX[orders[j].get(uid) or "NONE"])
                for uid, w in workers.items()
            }
            counts[lbl + "_actual_hires"] += len(hires)
            counts[lbl + "_hired_and_friendly_arrived"] += len(hires & arrived)
        if t >= SELECTION_START:
            old, guard, protected = forecast(native, model, obs, prev, est)
            trigger = []
            for uid in protected:
                pos, load, kind = p[op][2][uid]
                assert kind == "COLLECTOR" and isfinite(load)
                a, b, real = [cmd.get(uid) or "NONE" for cmd in (old, guard, orders[op])]
                counts["triggered_worker_turns"] += 1
                counts["old_hit"] += int(a == real)
                counts["guarded_hit"] += int(b == real)
                if a != b:
                    counts["changed_triggered_worker"] += 1
                    counts["fixed" if b == real else "harmed" if a == real else "both_wrong"] += 1
                next_payment = next((e for e in payments[uid] if e["t"] >= t), None)
                trigger.append(
                    dict(
                        uid=uid,
                        cell=pos,
                        carried=load,
                        base_distance=min(native.dist(pos, y) for y in p[op][1].values()),
                        forced_pre_occupancy=protected[uid],
                        original=a,
                        aligned=b,
                        actual=real,
                        observed_next_payment=next_payment,
                    )
                )
            counts["checked_roots"] += 1
            counts["triggered_roots"] += bool(protected)
            counts["changed_joint"] += old != guard
            samples.append(dict(t=t, workers=trigger, original=old, aligned=guard, actual=orders[op]))
        tracker.set_actions(p, orders[me])
        prev = p
    assert len(turns) == card.get("official_transitions", 0) and len(samples) == max(
        0, len(turns) - SELECTION_START
    )
    return dict(
        seed=card["seed"],
        seat=me,
        archive_sha256=sha(archive.read_bytes()),
        status=card["status"],
        official_steps=len(turns),
        counts=counts,
        samples=samples,
        end_assets=book["end_assets_by_physical_seat"],
    )


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cpu", type=int, required=True)
    os.sched_setaffinity(0, {ap.parse_args().cpu})
    for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = "1"
    sys.path.insert(0, str(DEV))
    import economy

    sys.path.insert(0, str(MODEL))
    from original_policy import original, module

    src = original()
    blob = (MODEL / "candidate.py").read_bytes()
    assert sha(src) == OLD and sha(blob) == NEW
    m = module(src)
    v = module(blob)
    reach = 2 * (m.SIZE // 2)
    spare, minimum = m._HOME
    assert m.LAST_TURN == 298 and m.SIZE == 15 and (spare, minimum, m._RETIRE) == (1, 50.0, 0)
    earliest_possible_t = m.LAST_TURN + 1 - reach - spare
    assert SELECTION_START <= earliest_possible_t
    settle_path = DEV / "hosted/settlement.json"
    settled = json.loads(settle_path.read_bytes())
    assert settled["records_reconciled"] and len(settled["games"]) == 8
    archives = {p["id"]: p for p in settled["archives"]}
    out = []
    totals = Counter()
    for card in settled["games"]:
        archive = DEV / "hosted" / card["archive_path"]
        assert sha(archive.read_bytes()) == archives[card["archive_id"]]["sha256"]
        row = game(card, archive, m, v, economy.ledger)
        totals.update(row["counts"])
        out.append(row)
    result = dict(
        schema="master3s2-dev1-enemy-home-sampler/1",
        source=OLD,
        candidate=NEW,
        run=settled["run_id"],
        geometry=dict(
            torus_diameter=reach,
            spare=spare,
            min_cargo=minimum,
            first_mathematically_possible=earliest_possible_t,
            first_checked=SELECTION_START,
        ),
        program_sha256=sha(Path(__file__).read_bytes()),
        settlement_sha256=sha(settle_path.read_bytes()),
        totals=totals,
        games=out,
    )
    path = HERE / "roots.json.gz"
    raw = json.dumps(result, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(gzip.compress(raw, mtime=0))
    tmp.replace(path)
    print(totals, sha(path.read_bytes()))


if __name__ == "__main__":
    main()

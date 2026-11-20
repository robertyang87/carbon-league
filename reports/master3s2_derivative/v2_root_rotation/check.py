"""Check the released byte stream, public feature symmetry, and both action permutations."""
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import zipfile
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
STEPS = {0, 49, 149, 249, 298}


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def probability(logits):
    work = np.asarray(logits, dtype=np.float64)
    exp = np.exp(work - work.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def main():
    candidate = load(HERE / "candidate.py", "rotation_release")
    parent = load(ROOT / 'ladder/prediction_feedback/package/candidate.py', 'v2_reference')
    geo = load(ROOT / 'symmetry_value/rotate.py', 'official_board_rotation')
    times = []
    errors = []
    sources = []
    direction = list(geo.WORKER_INVERSE)
    for archive in sorted((ROOT / "active_embedding_dev/hosted").glob("*.zip")):
        with zipfile.ZipFile(archive) as z:
            prefix = "parent/"
            receipt = json.loads(z.read(prefix + "receipt.json"))
            seat = receipt["candidate_seat"]
            public = gzip.decompress(z.read(prefix + "observations.jsonl.gz")).splitlines()
            turns = z.read(prefix + "turns.jsonl").splitlines()
        tracker = parent.BeliefTracker(parent._Bel, seat)
        previous = None
        steps = []
        for step, (public_row, turn_row) in enumerate(zip(public, turns, strict=True)):
            obs = json.loads(public_row)["observations"][seat]
            assert obs["step"] == step and "full_carbon" not in obs
            players = obs["players"]
            est, known = tracker.step(players, step, obs["carbon"])
            if step in STEPS:
                x = parent._quant(parent.encode(players, step, seat, est, known, previous))
                y = parent._quant(geo.encode_rotated(parent, players, step, seat, est, known, previous))
                assert np.array_equal(geo.rotate_features(x), y)
                a = parent._POLICY.forward(x, parent._SID)
                b = parent._POLICY.forward(y, parent._SID)
                start = time.perf_counter()
                actual = candidate._root_rotation_policy(x, candidate._SID)
                times.append(time.perf_counter() - start)
                reverse = candidate._root_rotation_policy(y, candidate._SID)
                for head, action_map in enumerate((direction, [0, 1, 2])):
                    oracle = (probability(a[head]) + probability(b[head])[::-1, :][:, action_map]) * 0.5
                    got = np.exp(actual[head])
                    reverse_got = np.exp(reverse[head])[::-1, :][:, action_map]
                    assert np.isfinite(actual[head]).all()
                    error = float(np.max(np.abs(got - oracle)))
                    eq_error = float(np.max(np.abs(got - reverse_got)))
                    mass_error = float(np.max(np.abs(got.sum(axis=1) - 1.0)))
                    assert max(error, eq_error, mass_error) < 0.00002
                    errors.append([error, eq_error, mass_error])
                steps.append(step)
            tracker.set_actions(players, json.loads(turn_row)["calls"][seat]["response"]["action"])
            previous = players
        assert set(steps) == STEPS
        sources.append(dict(zip_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                            seed=receipt['seed'], seat=seat, steps=steps))
    assert len(sources) == 8
    result = dict(sources=sources, max_probability_error=np.max(errors, axis=0).tolist(),
                  error_columns=["independent_float64_mixture", "half_turn_equivariance", "unit_mass"],
                  independent_feature_encodings=len(times), positions_per_head=225,
                  own_double_forward_local_seconds={"median": float(np.median(times)), "max": float(max(times))},
                  candidate_sha256=hashlib.sha256((HERE/'candidate.py').read_bytes()).hexdigest())
    (HERE / "checks.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(result['max_probability_error'], result['own_double_forward_local_seconds'])


if __name__ == "__main__":
    main()

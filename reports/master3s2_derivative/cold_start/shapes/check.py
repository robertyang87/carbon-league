"""Measure first real batch of each size in a fresh Python process, then compare all network outputs."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import types

HERE = Path(__file__).resolve().parent
V1 = HERE.parents[1] / "opponent_policy"
PACKAGES = {"v1": V1 / "candidate.py", "v2": HERE.parent / "candidate.py"}
SIZES = (3, 5, 7, 9, 11, 15, 23, 31, 32)
SAMPLES = (7, 20, 33, 46, 59, 72, 85, 98)
NAMES = ("worker_logits", "base_logits", "critic")


def sha(x):
    return hashlib.sha256(x).hexdigest()


def module(p):
    m = types.ModuleType("one_native_probe")
    exec(compile(p.read_bytes(), str(p), "exec"), m.__dict__)
    return m


def input_panel(folder):
    import numpy as np

    sys.path.insert(0, str(V1))
    from original_policy import PROTOCOL, stream

    net = module(PACKAGES["v1"])
    cards = []
    for c in PROTOCOL["games"]:
        if c["split"] != "train" or c["terrain"] in {g["terrain"] for g in cards}:
            continue
        cards.append(c)
        if len(cards) == 4:
            break
    assert len(cards) == 4
    all_x = []
    all_s = []
    rows = []
    for c in cards:
        for t, obs, cfg, xo, own, enemy in stream(c, net):
            if t not in SAMPLES:
                continue
            side = 1 - obs["player"]
            s = net._scalars(obs["players"], t, side).astype(np.float32)
            all_x.append(xo[0].astype(np.float32))
            all_s.append(s)
            rows.append(
                dict(
                    t=t,
                    own_seat=obs["player"],
                    value_seat=side,
                    seed=c["seed"],
                    terrain=c["terrain"],
                    source_sha256=c["sha256"],
                )
            )
    x = np.stack(all_x)
    s = np.stack(all_s)
    assert (
        len(rows) == 32
        and x.shape[0] == s.shape[0] == 32
        and np.isfinite(x).all()
        and np.isfinite(s).all()
    )
    np.savez(folder / "inputs.npz", x=x, s=s)
    return dict(
        rows=rows,
        features_sha256=sha(x.tobytes()),
        critic_inputs_sha256=sha(s.tobytes()),
        x_shape=x.shape,
        s_shape=s.shape,
    )


def run_one(folder, version, n):
    import numpy as np

    inputs = np.load(folder / "inputs.npz", allow_pickle=False)
    # Distinct, real, visible seat encodings; no shape of this size is sent to a net before the package init.
    x = inputs["x"][:n]
    s = inputs["s"][:n]
    pkg = PACKAGES[version]
    started = time.perf_counter()
    m = module(pkg)
    seconds_to_loaded = time.perf_counter() - started
    net = m._net()
    assert net is not None and net is m._RNET
    records = []
    expected = None
    for _ in range(3):
        start_cpu = time.process_time()
        start = time.perf_counter()
        answer = net(x, m._SID, s)
        wall = time.perf_counter() - start
        cpu = time.process_time() - start_cpu
        assert all(np.isfinite(a).all() for a in answer)
        if expected is None:
            expected = [a.copy() for a in answer]
        else:
            assert all(np.array_equal(a, b) for a, b in zip(answer, expected))
        records.append(dict(wall_s=wall, cpu_s=cpu))
    key = f"{version}-{n}"
    np.savez(folder / (key + ".npz"), **dict(zip(NAMES, expected)))
    (folder / (key + ".json")).write_text(
        json.dumps(
            dict(
                times=records,
                first_after_import=True,
                native_torch_threads=m._torch.get_num_threads(),
                affinity=sorted(os.sched_getaffinity(0)),
                native_impl=type(net).__name__,
                seconds_to_loaded=seconds_to_loaded,
                numpy=np.__version__,
                torch=str(m._torch.__version__),
                package_sha256=sha(pkg.read_bytes()),
            )
        )
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cpu", type=int, required=True)
    ap.add_argument("--child", choices=("v1", "v2"))
    ap.add_argument("--folder", type=Path)
    ap.add_argument("--n", type=int)
    args = ap.parse_args()
    os.sched_setaffinity(0, {args.cpu})
    for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[k] = "1"
    if args.child:
        run_one(args.folder, args.child, args.n)
        return
    import numpy as np

    release = json.loads((HERE.parent / "release.json").read_bytes())
    assert (
        sha(PACKAGES["v1"].read_bytes()) == release["parent"]
        and sha(PACKAGES["v2"].read_bytes()) == release["candidate"]
    )
    with tempfile.TemporaryDirectory() as name:
        folder = Path(name)
        panel = input_panel(folder)
        results = []
        for i, n in enumerate(SIZES):
            order = ("v1", "v2") if i % 2 == 0 else ("v2", "v1")
            for version in order:
                subprocess.run(
                    [
                        sys.executable,
                        __file__,
                        "--cpu",
                        str(args.cpu),
                        "--child",
                        version,
                        "--folder",
                        str(folder),
                        "--n",
                        str(n),
                    ],
                    check=True,
                )
            diff = {}
            with np.load(folder / f"v1-{n}.npz") as a, np.load(folder / f"v2-{n}.npz") as b:
                for label in NAMES:
                    x, y = a[label], b[label]
                    assert x.shape == y.shape
                    err = x.astype("float64") - y.astype("float64")
                    diff[label] = dict(
                        shape=x.shape,
                        values=x.size,
                        equal_bytes=x.tobytes() == y.tobytes(),
                        max_abs=float(abs(err).max()),
                        rmse=float(np.sqrt(np.mean(err**2))),
                        v1_sha256=sha(x.tobytes()),
                        v2_sha256=sha(y.tobytes()),
                    )
            arm = {v: json.loads((folder / f"{v}-{n}.json").read_bytes()) for v in order}
            results.append(dict(n=n, order=order, arms=arm, diff=diff))
            print(n, [arm[v]["times"][0]["wall_s"] for v in ("v1", "v2")], flush=True)
    out = dict(
        schema="m3s2-native-batch-cold-first/1",
        evidence="nine independent fresh-process pairs; three consecutive calls in each; no match; CPU affinity is not a 1.5-core container certification",
        panel=panel,
        results=results,
        v1=release["parent"],
        v2=release["candidate"],
        release_sha256=sha((HERE.parent / "release.json").read_bytes()),
        program_sha256=sha(Path(__file__).read_bytes()),
        all_outputs_bitwise_equal=all(
            h["equal_bytes"] for rec in results for h in rec["diff"].values()
        ),
    )
    dest = HERE / "results.json"
    tmp = dest.with_suffix(".tmp")
    tmp.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    tmp.replace(dest)
    print(sha(dest.read_bytes()), out["all_outputs_bitwise_equal"], flush=True)


if __name__ == "__main__":
    main()

"""Build exit shards through the REAL exit_gen.record() (production path):
rows include unsearched rows (q all-NaN), searched rows with partially
evaluated q (NaN at unevaluated moves), a searched row whose q is entirely
NaN, and fully evaluated rows -- exactly the shapes record() emits.
Assembly of the shard npz mirrors exit_gen.play()'s tail verbatim."""
import os
import sys
import types

import numpy as np

RUN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run")
sys.path.insert(0, RUN)
import cnet2_feat as F  # noqa: E402
from cnet2_build import FIELDS  # noqa: E402
from exit_gen import EXTRA, record  # noqa: E402
from fastsim import FastCarbon  # noqa: E402

NAN = float("nan")
# per game: list of rec specs; None = unsearched step
GAMES = [
    (990001, [None,
              {"slot": 2, "q": [12.0, NAN, -30.0, NAN, 45.0]},
              {"slot": 0, "q": [1.0, 2.0, 3.0, 4.0, 5.0]},
              {"slot": 5, "q": [NAN, NAN, NAN, NAN, NAN]},
              None,
              {"slot": 11, "q": [NAN, NAN, 7.5, NAN, NAN]}]),
    (990010, [None,
              {"slot": 4, "q": [NAN, 3.0, NAN, -8.0, NAN]},
              None,
              {"slot": 1, "q": [0.5, -0.5, 0.25, -0.25, 0.0]}]),
]


def build_game(seed, specs):
    sim = FastCarbon({}).reset(seed=seed)
    obs = sim.obs(0)
    f = F.Feat(obs["player"])
    x = f.observe(obs)
    y0 = np.zeros(F.OWN, np.int64)
    rows = {k: [] for k in list(FIELDS) + list(EXTRA)}
    simw = types.SimpleNamespace(carbon=sim.carbon)
    for spec in specs:
        rec = None
        if spec is not None:
            rec = {"slot": spec["slot"], "q": np.asarray(spec["q"], np.float64)}
        ag = types.SimpleNamespace(state={
            "f": f, "x": x, "y_net": y0.copy(), "acts_out": f.to_actions(y0),
            "rec": rec, "sampled": False})
        record(ag, 0, simw, rows)
    n = len(rows["y"])
    rows["value"] = [np.tanh(120.0 / 3000.0)] * n
    rows["gid"] = [0] * n
    rows.pop("step", None)
    rows["step"] = list(range(n))
    arr = {k: np.asarray(v).astype((FIELDS.get(k) or EXTRA[k])[0]) for k, v in rows.items()}
    return arr


def main(outdir):
    os.makedirs(outdir, exist_ok=True)
    for seed, specs in GAMES:
        arr = build_game(seed, specs)
        fn = os.path.join(outdir, "%d_fxopp_s0.npz" % seed)
        np.savez(fn, **arr)
        q = arr["q"]
        print("shard", fn, "rows", len(arr["y"]),
              "searched", int(arr["searched"].sum()),
              "q_NaN", int(np.isnan(q).sum()), "of", q.size,
              "pi_slot", arr["pi_slot"].tolist())
    print("shards written under", outdir)


if __name__ == "__main__":
    main(sys.argv[1])

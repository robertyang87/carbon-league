"""diag_return -- is 'return to base when loaded' present in the data
and in the net? For self collectors with cargo>=200 and base distance
>=3, classify the true label and the net prediction as toward-base /
stay / other, on train+val of the fixed dataset."""

import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

from bc_features import DIST  # noqa: E402
from train_bc import build_dataset  # noqa: E402
import shadow_alpha as S  # noqa: E402

MOVE_OFF = {1: (0, 1), 2: (1, 0), 3: (0, -1), 4: (-1, 0)}


def xy(i):
    return i % 15, 14 - i // 15


def idx(x, y):
    return (14 - (y % 15)) * 15 + (x % 15)


def move_target(pos, a):
    if a == 0:
        return pos
    dx, dy = MOVE_OFF[int(a)]
    x, y = xy(pos)
    return idx(x + dx, y + dy)


def main():
    _z = np.load(os.path.join(HERE, "bc_data", "claude_family_fixed.npz"),
                 allow_pickle=True)
    d = {k: _z[k] for k in _z.files}
    Xw, yw, tw, gw, Xb, yb, gb = build_dataset(d)
    # per-row context in build_dataset's emission order
    rows = []
    for i in range(len(d["step"])):
        p = int(d["player"][i])
        own_bases = [int(b) for b in d["bases"][i][p] if int(b) >= 0]
        for j in range(10):
            if d["workers_mask"][i][p][j]:
                pos = int(d["workers"][i][p][j][0])
                cargo = float(d["workers"][i][p][j][1])
                db = min(int(DIST[pos, b]) for b in own_bases)
                rows.append((pos, cargo, db, own_bases,
                             int(d["workers"][i][p][j][2])))
    pred = S._forward("w_", Xw).argmax(1)

    def classify(a, pos, own_bases, db):
        if a == 0:
            return "stay"
        tpos = move_target(pos, a)
        nd = min(int(DIST[tpos, b]) for b in own_bases)
        return "toward" if nd < db else "other"

    for name, m_cargo, m_dist in (("cargo>=200, dist>=3", 200, 3),
                                  ("cargo>=100, dist>=3", 100, 3),
                                  ("cargo>=200, dist>=1", 200, 1)):
        sel = [k for k, r in enumerate(rows)
               if r[4] == 0 and r[1] >= m_cargo and r[2] >= m_dist]
        from collections import Counter
        ct = Counter(classify(yw[k], rows[k][0], rows[k][3], rows[k][2])
                     for k in sel)
        cp = Counter(classify(pred[k], rows[k][0], rows[k][3], rows[k][2])
                     for k in sel)
        n = len(sel)
        print(f"{name}: n={n}")
        for c in ("toward", "stay", "other"):
            print(f"  {c:7s} true {ct[c] / n:.3f}   pred {cp[c] / n:.3f}")


if __name__ == "__main__":
    main()

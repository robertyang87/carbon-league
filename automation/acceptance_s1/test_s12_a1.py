"""S1-2 A1 parity: numpy Policy.act(want_u=True) u == torch forward_heads utility.

Positions come from m3s2 replays (real games, DGX corpus). Torch batch is built
exactly like code/cnet2_parity.py (uf/mf/legal/y from the recorded next action).
Run: <venv python> test_s12_a1.py  (CPU only)
"""
import gzip
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("BC_DEVICE", "cpu")

import torch  # noqa: E402

import cnet2_feat as F  # noqa: E402
import cnet2_np as NP  # noqa: E402
from cnet2_model import CNet2  # noqa: E402


def to_dev_cpu(b):
    """Minimal CPU copy of cnet2_train.to_dev(b, aug=False) for the parity batch."""
    t = {}
    for k, v in b.items():
        if v.dtype == np.float16:
            v = v.astype(np.float32)
        t[k] = torch.from_numpy(np.ascontiguousarray(v))
    t["maps"] = t["maps"].float() / 255.0
    t["carbon"] = t["carbon"].float() / 255.0
    for k in ("tok_pos", "tok_kind", "hist_cell", "hist_move", "y", "teacher"):
        t[k] = t[k].long()
    return t

REPLAYS = sorted(os.path.join(HERE, "replays", f) for f in os.listdir(os.path.join(HERE, "replays"))
                 if f.endswith(".json.gz"))


def value_to_u_old(v):
    """Pre-A1 formula (deleted from stage0_teacher): tanh value -> (U, M)."""
    m = 3000.0 * math.atanh(max(-0.999, min(0.999, float(v))))
    w = 1.0 if m > 0 else (0.0 if m < 0 else 0.5)
    return 0.5 * w + 0.5 / (1.0 + math.exp(-m / 1000.0))


def positions(n_target=24):
    """(x, aux, feat, y_next) tuples from replay observations, spread over games/steps."""
    out = []
    per_game = max(1, n_target // max(1, len(REPLAYS)))
    for rp in REPLAYS:
        d = json.load(gzip.open(rp))
        st = d["steps"]
        for p in (0, 1):
            f = F.Feat(p)
            for t in range(20, 290, 37):
                o0 = st[t][0]["observation"]
                obs = {"step": o0["step"], "players": o0["players"],
                       "carbon": st[t][p]["observation"]["carbon"], "player": p}
                x = f.observe(obs)
                y = f.labels(st[t + 1][p].get("action") or {})
                out.append((x, f.aux, f, y))
                if len(out) >= n_target:
                    return out
    return out


def main():
    w = dict(np.load(os.path.join(HERE, "heads_v2.npz")))
    meta = json.loads(str(w["meta"]))
    net = CNet2(**meta["cfg"])
    net.load_state_dict({k: torch.from_numpy(v) for k, v in w.items() if k != "meta"})
    net.eval()
    pol = NP.Policy(w)
    assert "utility.weight" in w and "utility.bias" in w, "heads v2 lacks utility keys"

    pos = positions(24)
    print(f"positions: {len(pos)}")
    max_du = 0.0
    max_old = 0.0
    us = []
    for x, aux, f, y in pos:
        _, _, v, u_np = pol.act(x, aux, teacher=1, want_u=True)
        assert 0.0 <= u_np <= 1.0, f"u out of [0,1]: {u_np}"
        us.append(u_np)
        # torch batch, cnet2_parity construction
        pf = F.Prefix(x, aux)
        uf = np.zeros((F.OWN, F.UF), np.float32)
        mf = np.zeros((F.OWN, 5, F.MF), np.float32)
        lg = np.zeros((F.OWN, 5), bool)
        for s in range(F.OWN):
            if x["tok_mask"][s]:
                uf[s], mf[s], lg[s] = pf.feats(s)
                if not lg[s, y[s]]:
                    y[s] = 0
                pf.advance(s, int(y[s]))
        b = {k: x[k][None] for k in ("maps", "scal", "tok_pos", "tok_kind", "tok_mask",
                                     "tok_cargo", "hist_cell", "hist_cargo", "hist_move")}
        b.update(uf=uf[None].astype(np.float16), mf=mf[None].astype(np.float16), legal=lg[None],
                 y=y[None], carbon=np.zeros((1, F.N), np.uint8), value=np.zeros(1, np.float16),
                 teacher=np.array([1]))
        for k in ("scal", "tok_cargo", "hist_cargo"):
            b[k] = b[k].astype(np.float16)
        tb = to_dev_cpu(b)
        with torch.no_grad():
            _, _, _, u_t, _, _ = net.forward_heads(tb)
        u_t = float(u_t[0])
        max_du = max(max_du, abs(u_np - u_t))
        max_old = max(max_old, abs(u_np - value_to_u_old(v)))
    print(f"max |u_numpy - u_torch| = {max_du:.3e}  (tol 1e-5)")
    print(f"max |u_new - u_old(value_to_u)| = {max_old:.4f}  (must be >> 0: new head, not old value)")
    print(f"u range [{min(us):.4f}, {max(us):.4f}]  n={len(us)}")
    assert max_du <= 1e-5, "A1 parity FAILED"
    assert max_old > 0.01, "A1 FAILED: new u == old value_to_u (old head in disguise)"

    # missing-key loud failure: old weights without utility keys
    w_old = {k: v for k, v in w.items() if not k.startswith("utility.")}
    pol_old = NP.Policy(w_old)
    x0, aux0, _, _ = pos[0]
    try:
        pol_old.act(x0, aux0, teacher=1, want_u=True)
        raise SystemExit("A1 FAILED: old weights did not raise on want_u=True")
    except KeyError as e:
        print(f"old weights want_u=True -> KeyError {e} (loud, named key) OK")
    # default triple unchanged
    r3 = pol.act(x0, aux0, teacher=1)
    assert len(r3) == 3, "default act() must stay a 3-tuple"
    print("A1 PASS")


if __name__ == "__main__":
    main()

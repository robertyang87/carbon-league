"""nn_train_value2 -- value net v2 on data/value2 shards (gen_value2.py); MPS if available.

python3 nn_train_value2.py SHARD_DIR OUT.npz [epochs]
env: VN_W (48), VN_NB (3), VN_HID (128), VN_LR (2e-3), VN_POL ("0" = only v9-perspective states; "all"),
     VN_MAXSH (max shards), VN_DEV (mps/cpu)
Target: final margin / 5000 from the recording side's view. Game-level split: gid % 10 == 0 -> val.
Net = nn_train_value.VNet architecture (ValueNet in nn_search2 serves it).
Reports R2 / RMSE (points) by phase and a material-only linear baseline per phase.
"""
import glob
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nnfeat as F  # noqa: E402

W = int(os.environ.get("VN_W", "48"))
NB = int(os.environ.get("VN_NB", "3"))
HID = int(os.environ.get("VN_HID", "128"))
PHASES = ((0, 50), (50, 100), (100, 150), (150, 200), (200, 250), (250, 300))


class VNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(F.C_PLANES, W, 3, padding=1, padding_mode="circular")
        self.blocks = nn.ModuleList([nn.ModuleList([
            nn.Conv2d(W, W, 3, padding=1, padding_mode="circular"),
            nn.Conv2d(W, W, 3, padding=1, padding_mode="circular")]) for _ in range(NB)])
        self.fc = nn.Linear(W + F.N_SCAL, HID)
        self.out = nn.Linear(HID, 1)

    def forward(self, x, s):
        h = Fn.relu(self.stem(x))
        for c1, c2 in self.blocks:
            h = Fn.relu(h + c2(Fn.relu(c1(h))))
        g = h.mean(dim=(2, 3))
        return self.out(Fn.relu(self.fc(torch.cat([g, s], 1)))).squeeze(1)


def load(shard_dir, pol_filter, maxsh):
    files = sorted(glob.glob(os.path.join(shard_dir, "shard_*.npz")))[:maxsh]
    cols = {k: [] for k in ("G", "Sc", "y", "step", "gid", "mat", "pol")}
    for f in files:
        d = np.load(f)
        m = np.ones(len(d["y"]), bool) if pol_filter == "all" else d["pol"] == int(pol_filter)
        for k in cols:
            cols[k].append(d[k][m])
    return {k: np.concatenate(v) for k, v in cols.items()}, len(files)


def phase_report(pred, y, step, mat):
    out = {}
    for lo, hi in PHASES:
        m = (step >= lo) & (step < hi)
        if m.sum() < 10:
            continue
        e = (pred[m] - y[m]) * 5000
        out[f"{lo}-{hi}"] = {"n": int(m.sum()), "r2": round(float(1 - (e ** 2).mean() / ((y[m] * 5000).var())), 3),
                             "rmse": round(float(np.sqrt((e ** 2).mean()))), "mae": round(float(np.abs(e).mean()))}
    e = (pred - y) * 5000
    out["all"] = {"r2": round(float(1 - (e ** 2).mean() / (y * 5000).var()), 3),
                  "rmse": round(float(np.sqrt((e ** 2).mean()))), "mae": round(float(np.abs(e).mean()))}
    return out


def material_baseline(tr, va, D):
    """per-phase least squares y ~ a + b*cashdiff + c*cargodiff (fit on train, scored on val)."""
    y, step, mat = D["y"], D["step"], D["mat"]
    pred = np.zeros(len(va))
    for lo, hi in PHASES:
        mt = (step[tr] >= lo) & (step[tr] < hi)
        mv = (step[va] >= lo) & (step[va] < hi)
        X = lambda ii: np.stack([np.ones(len(ii)), (mat[ii, 0] - mat[ii, 1]) / 5000, (mat[ii, 2] - mat[ii, 3]) / 5000], 1)
        coef, *_ = np.linalg.lstsq(X(tr[mt]), y[tr[mt]], rcond=None)
        pred[mv] = X(va[mv]) @ coef
    return phase_report(pred, y[va], step[va], mat[va])


def main(shard_dir, out, epochs=8):
    dev = os.environ.get("VN_DEV", "mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(0); np.random.seed(0)
    D, nsh = load(shard_dir, os.environ.get("VN_POL", "0"), int(os.environ.get("VN_MAXSH", "100000")))
    val = D["gid"] % 10 == 0
    tr_idx, va_idx = np.nonzero(~val)[0], np.nonzero(val)[0]
    print(json.dumps({"shards": nsh, "games": int(len(np.unique(D["gid"]))), "train": len(tr_idx), "val": len(va_idx),
                      "W": W, "NB": NB, "HID": HID, "dev": dev}), flush=True)
    print(json.dumps({"material_baseline_val": material_baseline(tr_idx, va_idx, D)}), flush=True)
    RESID = os.environ.get("VN_RESID", "1") == "1"
    m0 = ((D["mat"][:, 0] - D["mat"][:, 1]) + 0.9 * (D["mat"][:, 2] - D["mat"][:, 3])) / 5000.0
    base = m0.astype(np.float32) if RESID else np.zeros(len(m0), np.float32)
    D["y_fit"] = (D["y"] - base).astype(np.float32)
    Gt = torch.from_numpy(D["G"])           # float16, CPU
    St = torch.from_numpy(D["Sc"])
    yt = torch.from_numpy(D["y_fit"])
    net = VNet().to(dev)
    lr = float(os.environ.get("VN_LR", "2e-3"))
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    bs = 1024
    nb_ep = len(tr_idx) // bs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * nb_ep)
    best = 1e9

    def predict(idx):
        net.eval(); pr = []
        with torch.no_grad():
            for b in range(0, len(idx), 8192):
                ii = torch.from_numpy(idx[b:b + 8192])
                pr.append(net(Gt[ii].to(dev).float(), St[ii].to(dev).float()).cpu().numpy())
        net.train()
        return np.concatenate(pr)

    for ep in range(epochs):
        perm = np.random.permutation(tr_idx); t0 = time.time(); tl = 0.0
        for b in range(nb_ep):
            ii = torch.from_numpy(np.sort(perm[b * bs:(b + 1) * bs]))
            xb = Gt[ii].to(dev).float()
            k = np.random.randint(8)
            if k & 1:
                xb = xb.flip(2)
            if k & 2:
                xb = xb.flip(3)
            if k & 4:
                xb = xb.transpose(2, 3)
            loss = Fn.mse_loss(net(xb, St[ii].to(dev).float()), yt[ii].to(dev))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tl += loss.item()
        pr = predict(va_idx) + base[va_idx]
        rep = phase_report(pr, D["y"][va_idx], D["step"][va_idx], D["mat"][va_idx])
        mse = float(((pr - D["y"][va_idx]) ** 2).mean())
        print(json.dumps({"ep": ep, "train_mse": round(tl / max(1, nb_ep), 4), "val_mse": round(mse, 4), "val": rep,
                          "s": round(time.time() - t0)}), flush=True)
        if mse < best:
            best = mse
            sd = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in net.state_dict().items()}
            np.savez_compressed(out, **sd, meta=np.array(json.dumps({"W": W, "NB": NB, "HID": HID, "epoch": ep, "resid": RESID, "cargo_w": 0.9,
                                                                     "val": rep, "shards": nsh})))
    print("best", best)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 8)

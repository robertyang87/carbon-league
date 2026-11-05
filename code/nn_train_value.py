"""nn_train_value -- small state-value CNN: (global planes, scalars) -> final margin / 5000.

python3 nn_train_value.py DATA.npz OUT.npz [epochs]
Net: stem conv C->W, NB residual blocks (circular padding), global mean + scalars -> HID -> 1.
Dihedral augmentation on the planes (scalars are symmetry-invariant). Exports float32 weights.
"""
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

W = int(os.environ.get("VN_W", "32"))
NB = int(os.environ.get("VN_NB", "2"))
HID = int(os.environ.get("VN_HID", "64"))


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


def augment(x):
    k = np.random.randint(8)
    if k & 1:
        x = x.flip(2)
    if k & 2:
        x = x.flip(3)
    if k & 4:
        x = x.transpose(2, 3)
    return x


def evaluate(net, G, S, y, step, idx):
    net.eval()
    pr = []
    with torch.no_grad():
        for b in range(0, len(idx), 4096):
            ii = idx[b:b + 4096]
            pr.append(net(torch.from_numpy(G[ii].astype(np.float32)), torch.from_numpy(S[ii].astype(np.float32))).numpy())
    pr = np.concatenate(pr); yt = y[idx]; st = step[idx]
    out = {"mse": float(((pr - yt) ** 2).mean()), "r2": float(1 - ((pr - yt) ** 2).mean() / yt.var())}
    for lo, hi in ((0, 50), (50, 100), (100, 200), (200, 300)):
        m = (st >= lo) & (st < hi) & (yt != 0)
        out[f"sign_{lo}"] = round(float((np.sign(pr[m]) == np.sign(yt[m])).mean()), 3)
    net.train()
    return out


def export(net, path, meta):
    sd = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in net.state_dict().items()}
    np.savez_compressed(path, **sd, meta=np.array(json.dumps(meta)))


def main(data, out, epochs=6):
    torch.set_num_threads(int(os.environ.get("VN_THREADS", "4")))
    torch.manual_seed(0); np.random.seed(0)
    D = np.load(data)
    G, S, y, step, val = D["G"], D["Sc"], D["y"], D["step"], D["val"]
    tr_idx, va_idx = np.nonzero(~val)[0], np.nonzero(val)[0]
    print(json.dumps({"train": len(tr_idx), "val": len(va_idx), "W": W, "NB": NB}), flush=True)
    net = VNet()
    lr = float(os.environ.get("VN_LR", "2e-3"))
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    bs = 512
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * (len(tr_idx) // bs))
    best = 1e9
    for ep in range(epochs):
        perm = np.random.permutation(tr_idx); t0 = time.time(); tl = 0.0
        for b in range(len(perm) // bs):
            ii = np.sort(perm[b * bs:(b + 1) * bs])
            xb = augment(torch.from_numpy(G[ii].astype(np.float32)))
            loss = Fn.mse_loss(net(xb, torch.from_numpy(S[ii].astype(np.float32))), torch.from_numpy(y[ii]))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tl += loss.item()
        ev = evaluate(net, G, S, y, step, va_idx)
        print(json.dumps({"ep": ep, "train_mse": round(tl / max(1, len(perm) // bs), 4), "val": ev,
                          "s": round(time.time() - t0)}), flush=True)
        if ev["mse"] < best:
            best = ev["mse"]
            export(net, out, {"W": W, "NB": NB, "HID": HID, "epoch": ep, "val": ev})
    print("best", best)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 6)

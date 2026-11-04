"""nn_train_bc -- train the egocentric CNN policy by behaviour cloning.

Model (mirrored exactly by nn_infer.py for NumPy serving):
  stem conv3x3 (C_PLANES -> W) + ReLU, NB residual blocks
  [conv3x3 W->W, ReLU, conv3x3 W->W, +skip, ReLU], circular padding.
  head input = 3x3 centre patch (9W) ++ global mean (W) ++ scalars (N_SCAL)
  -> fc HID + ReLU -> 8 logits (0-4 worker moves, 5-7 base actions; masked
  by kind) and -> 1 value (tanh, aux, target = teacher-side final margin/5000).
Augmentation: random dihedral symmetry of the torus (flip rows / flip cols /
transpose) with the matching action relabelling.

Usage: python3 nn_train_bc.py DATA.npz OUT.npz [epochs]
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

W = int(os.environ.get("BC_W", "48"))
NB = int(os.environ.get("BC_NB", "3"))
HID = int(os.environ.get("BC_HID", "256"))


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(F.C_PLANES, W, 3, padding=1, padding_mode="circular")
        self.blocks = nn.ModuleList()
        for _ in range(NB):
            self.blocks.append(nn.ModuleList([
                nn.Conv2d(W, W, 3, padding=1, padding_mode="circular"),
                nn.Conv2d(W, W, 3, padding=1, padding_mode="circular")]))
        self.fc = nn.Linear(9 * W + W + F.N_SCAL, HID)
        self.pi = nn.Linear(HID, 8)
        self.v = nn.Linear(HID, 1)

    def forward(self, x, s):
        h = Fn.relu(self.stem(x))
        for c1, c2 in self.blocks:
            h = Fn.relu(h + c2(Fn.relu(c1(h))))
        patch = h[:, :, 6:9, 6:9].reshape(h.shape[0], -1)
        g = h.mean(dim=(2, 3))
        z = Fn.relu(self.fc(torch.cat([patch, g, s], dim=1)))
        return self.pi(z), torch.tanh(self.v(z)).squeeze(1)


# action relabelling under symmetries. moves: 0 NONE 1 UP 2 RIGHT 3 DOWN 4 LEFT
FLIP_ROWS = torch.tensor([0, 3, 2, 1, 4, 5, 6, 7])   # vertical flip swaps UP/DOWN
FLIP_COLS = torch.tensor([0, 1, 4, 3, 2, 5, 6, 7])   # horizontal flip swaps LEFT/RIGHT
TRANSPOSE = torch.tensor([0, 4, 3, 2, 1, 5, 6, 7])   # (r,c)->(c,r): UP<->LEFT, DOWN<->RIGHT


def augment(x, y):
    # centre is (7,7); flips about the centre keep it fixed
    if torch.rand(1).item() < 0.5:
        x = torch.flip(x, dims=[2]); y = FLIP_ROWS[y]
    if torch.rand(1).item() < 0.5:
        x = torch.flip(x, dims=[3]); y = FLIP_COLS[y]
    if torch.rand(1).item() < 0.5:
        x = x.transpose(2, 3); y = TRANSPOSE[y]
    return x, y


class MultiX:
    """Index a list of uint8 tensors as one array without concatenating."""

    def __init__(self, parts, offs):
        self.parts, self.offs = parts, offs

    def __getitem__(self, ii):
        ii = ii.numpy() if hasattr(ii, "numpy") else np.asarray(ii)
        out = torch.empty((len(ii),) + tuple(self.parts[0].shape[1:]), dtype=torch.uint8)
        f = np.searchsorted(self.offs, ii, side="right") - 1
        for j, part in enumerate(self.parts):
            m = np.nonzero(f == j)[0]
            if len(m):
                out[torch.from_numpy(m)] = part[torch.from_numpy(ii[m] - self.offs[j])]
        return out


def masked_logits(logits, kind):
    mask = torch.full_like(logits, -1e9)
    worker = kind < 2
    mask[worker, 0:5] = 0.0
    mask[~worker, 5:8] = 0.0
    return logits + mask


def export(net, path, meta):
    sd = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in net.state_dict().items()}
    np.savez_compressed(path, **sd, meta=np.array(json.dumps(meta)))


def main(data, out, epochs=4):
    torch.set_num_threads(int(os.environ.get("BC_THREADS", "4")))
    torch.manual_seed(0)
    # DATA may be a comma-separated list of npz files; X stays split per file (no 2x RAM concat)
    Xl, Scl, yl, kl, isv = [], [], [], [], []
    for path in data.split(","):
        D = np.load(path)
        games = json.load(open(path.replace(".npz", "_games.json")))["games"]
        val_g = np.array([g["val"] for g in games])
        Xl.append(torch.from_numpy(D["X"])); Scl.append(D["Sc"].astype(np.float32))
        yl.append(D["y"].astype(np.int64)); kl.append(D["kind"].astype(np.int64)); isv.append(val_g[D["gid"]])
    sizes = [len(t) for t in yl]
    offs = np.cumsum([0] + sizes)
    X = MultiX(Xl, offs)
    Sc = torch.from_numpy(np.concatenate(Scl)); y = torch.from_numpy(np.concatenate(yl))
    kind = torch.from_numpy(np.concatenate(kl)); is_val = np.concatenate(isv)
    vt = None
    tr_idx = np.nonzero(~is_val)[0]; va_idx = np.nonzero(is_val)[0]
    print(json.dumps({"train": len(tr_idx), "val": len(va_idx), "W": W, "NB": NB}), flush=True)
    net = Net()
    init = os.environ.get("BC_INIT")
    if init:
        w = np.load(init)
        net.load_state_dict({k: torch.from_numpy(w[k]) for k in w.files if k != "meta"})
    max_lr = float(os.environ.get("BC_LR", "2e-3"))
    opt = torch.optim.AdamW(net.parameters(), lr=max_lr, weight_decay=1e-4)
    bs = 512
    steps_total = epochs * (len(tr_idx) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=max_lr, total_steps=steps_total)
    best = -1
    # 断点续跑（容器会被回收重启）：每 BC_CKPT_EVERY 批存完整状态到 OUT.ckpt.pt，启动时若存在则续跑
    ck_path = out + ".ckpt.pt"
    ck_every = int(os.environ.get("BC_CKPT_EVERY", "500"))
    ep0, b0, perm0 = 0, 0, None
    if os.path.exists(ck_path):
        ck = torch.load(ck_path, weights_only=False)
        net.load_state_dict(ck["net"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        ep0, b0, perm0, best = ck["ep"], ck["b"], ck["perm"], ck["best"]
        np.random.set_state(ck["np_rng"]); torch.set_rng_state(ck["torch_rng"])
        print(json.dumps({"resume": {"ep": ep0, "b": b0}}), flush=True)
    for ep in range(ep0, epochs):
        net.train()
        if ep == ep0 and perm0 is not None:
            perm = perm0
        else:
            perm = np.random.permutation(tr_idx)
            b0 = 0
        t0 = time.time(); tl = 0; tn = 0
        for b in range(b0, len(perm) // bs):
            if b > b0 and b % ck_every == 0:
                tmp = ck_path + ".tmp"
                torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                            "ep": ep, "b": b, "perm": perm, "best": best,
                            "np_rng": np.random.get_state(), "torch_rng": torch.get_rng_state()}, tmp)
                os.replace(tmp, ck_path)
            ii = torch.from_numpy(perm[b * bs:(b + 1) * bs])
            xb = X[ii].float() / 127.0
            yb = y[ii]
            xb, yb = augment(xb, yb)
            logits, v = net(xb, Sc[ii])
            loss = Fn.cross_entropy(masked_logits(logits, kind[ii]), yb)
            if vt is not None:
                loss = loss + 0.25 * Fn.mse_loss(v, vt[ii])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tl += loss.item() * len(ii); tn += len(ii)
            if b % 200 == 0:
                print(json.dumps({"ep": ep, "b": b, "loss": round(tl / tn, 4), "s": round(time.time() - t0)}), flush=True)
        acc = evaluate(net, X, Sc, y, kind, va_idx)
        print(json.dumps({"ep": ep, "train_loss": round(tl / tn, 4), "val": acc, "s": round(time.time() - t0)}), flush=True)
        if acc["all"] > best:
            best = acc["all"]
            export(net, out, {"W": W, "NB": NB, "HID": HID, "epoch": ep, "val": acc, "data": os.path.basename(data)})
        if ep + 1 < epochs:   # 轮末也存一次（下一轮从头）
            torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "ep": ep + 1, "b": 0, "perm": None, "best": best,
                        "np_rng": np.random.get_state(), "torch_rng": torch.get_rng_state()}, ck_path)
    if os.path.exists(ck_path):
        os.remove(ck_path)
    print("best", best)


@torch.no_grad()
def evaluate(net, X, Sc, y, kind, idx):
    net.eval()
    ok = {0: [0, 0], 1: [0, 0], 2: [0, 0]}
    for b in range(0, len(idx), 2048):
        ii = torch.from_numpy(idx[b:b + 2048])
        logits, _ = net(X[ii].float() / 127.0, Sc[ii])
        pred = masked_logits(logits, kind[ii]).argmax(1)
        for k in (0, 1, 2):
            m = kind[ii] == k
            ok[k][0] += (pred[m] == y[ii][m]).sum().item(); ok[k][1] += m.sum().item()
    res = {["collector", "planter", "base"][k]: round(ok[k][0] / max(1, ok[k][1]), 4) for k in ok}
    res["all"] = round(sum(v[0] for v in ok.values()) / max(1, sum(v[1] for v in ok.values())), 4)
    return res


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 4)

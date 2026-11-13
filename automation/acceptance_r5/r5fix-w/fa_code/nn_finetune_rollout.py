"""nn_finetune_rollout -- distil counterfactual rollout labels into the policy, anchored by BC data.

python3 nn_finetune_rollout.py INIT.npz BC.npz ROLLOUT_DIR OUT.npz [epochs=1]
  BC.npz       nn_build_bc output (anchor; keeps the policy's style)
  ROLLOUT_DIR  rollout_label.py shards (X, Sc, kind, logits, scores)
Improvement samples: decisions where best branch beats the policy's argmax branch by >= RO_GAIN points;
target = best move. Each step trains on a BC batch + an improvement batch (cycled), loss = CE_bc + RO_W * CE_ro.
Env: BC_W/BC_NB as nn_train_bc, RO_GAIN (100), RO_W (1.0), BC_LR (1e-4).
"""
import glob
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as Fn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nn_train_bc as TB  # noqa: E402


def load_rollout(d, gain_min):
    Xs, Ss, Ks, Ys = [], [], [], []
    n_all = 0
    for f in sorted(glob.glob(os.path.join(d, "shard_*.npz"))):
        D = np.load(f)
        V, L = D["scores"], D["logits"]
        pa = L[:, :5].argmax(1)
        gain = V.max(1) - V[np.arange(len(V)), pa]
        m = gain >= gain_min
        n_all += len(V)
        Xs.append(D["X"][m]); Ss.append(D["Sc"][m]); Ks.append(D["kind"][m]); Ys.append(V[m].argmax(1))
    X = np.concatenate(Xs).astype(np.float32); S = np.concatenate(Ss).astype(np.float32)
    return X, S, np.concatenate(Ks).astype(np.int64), np.concatenate(Ys).astype(np.int64), n_all


def main(init, bc, rdir, out, epochs=1):
    torch.set_num_threads(int(os.environ.get("BC_THREADS", "4")))
    torch.manual_seed(0); np.random.seed(0)
    gain_min = float(os.environ.get("RO_GAIN", "100"))
    w_ro = float(os.environ.get("RO_W", "1.0"))
    Xr, Sr, Kr, Yr, n_all = load_rollout(rdir, gain_min)
    D = np.load(bc)
    Xb, Sb, yb, kb = D["X"], D["Sc"], D["y"].astype(np.int64), D["kind"].astype(np.int64)
    print(json.dumps({"rollout_decisions": n_all, "improvement_samples": len(Yr), "bc": len(yb)}), flush=True)
    net = TB.Net()
    w = np.load(init)
    net.load_state_dict({k: torch.from_numpy(w[k]) for k in w.files if k != "meta"})
    opt = torch.optim.AdamW(net.parameters(), lr=float(os.environ.get("BC_LR", "1e-4")), weight_decay=1e-4)
    bs, bs_r = 512, 64
    steps = epochs * (len(yb) // bs)
    perm = np.concatenate([np.random.permutation(len(yb)) for _ in range(epochs)])
    t0 = time.time()
    for b in range(steps):
        ii = torch.from_numpy(np.sort(perm[b * bs:(b + 1) * bs]))
        xb = torch.from_numpy(Xb[ii.numpy()]).float() / 127.0
        y_ = torch.from_numpy(yb[ii.numpy()])
        xb, y_ = TB.augment(xb, y_)
        lg, _ = net(xb, torch.from_numpy(Sb[ii.numpy()].astype(np.float32)))
        loss = Fn.cross_entropy(TB.masked_logits(lg, torch.from_numpy(kb[ii.numpy()])), y_)
        if len(Yr):
            jj = np.random.randint(0, len(Yr), bs_r)
            xr, yr = TB.augment(torch.from_numpy(Xr[jj]), torch.from_numpy(Yr[jj]))
            lr_, _ = net(xr, torch.from_numpy(Sr[jj]))
            loss = loss + w_ro * Fn.cross_entropy(TB.masked_logits(lr_, torch.from_numpy(Kr[jj])), yr)
        opt.zero_grad(); loss.backward(); opt.step()
        if b % 200 == 0:
            print(json.dumps({"b": b, "of": steps, "loss": round(loss.item(), 4), "s": round(time.time() - t0)}), flush=True)
    TB.export(net, out, {"init": os.path.basename(init), "rollout": rdir, "gain_min": gain_min, "w_ro": w_ro})
    print("done", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]) if len(sys.argv) > 5 else 1)

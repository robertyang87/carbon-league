"""pred_mlp -- small-MLP对照 for the opponent predictor.

Same public information as the bucket model, but continuous resolution:
[wtype, cargo, d_base, d_ebase, d_tree, d_foe, step, on_base, on_ebase,
on_tree]. 2x128 MLP, torch CPU with 2 threads. Reports val top-1
(all/collector/planter) and single-worker forward latency.
"""

import json
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(2)
HERE = os.path.dirname(os.path.abspath(__file__))


def feats(d):
    X = np.stack([d["wtype"].astype(np.float64),
                  np.minimum(d["cargo"], 1000.0) / 500.0,
                  d["d_base"] / 14.0, d["d_ebase"] / 14.0,
                  d["d_tree"] / 14.0, d["d_foe"] / 14.0,
                  d["step"] / 300.0, d["on_base"], d["on_ebase"],
                  d["on_tree"]], axis=1).astype(np.float32)
    return X


def run_family(name, path, epochs, max_train=60000, seed=7):
    d = np.load(path)
    X = feats(d)
    y = d["label"].astype(np.int64)
    wtype = d["wtype"]
    tr = d["split"] == 0
    va = d["split"] == 1
    rng = np.random.default_rng(seed)
    tri = np.where(tr)[0]
    if len(tri) > max_train:
        tri = rng.choice(tri, size=max_train, replace=False)
    Xtr = torch.tensor(X[tri]); ytr = torch.tensor(y[tri])
    Xva = torch.tensor(X[va]); yva = y[va]
    net = torch.nn.Sequential(
        torch.nn.Linear(10, 128), torch.nn.ReLU(),
        torch.nn.Linear(128, 128), torch.nn.ReLU(),
        torch.nn.Linear(128, 5))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lossf = torch.nn.CrossEntropyLoss()
    n = len(tri)
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, 4096):
            b = perm[i:i + 4096]
            opt.zero_grad()
            out = net(Xtr[b])
            loss = lossf(out, ytr[b])
            loss.backward()
            opt.step()
            tot += float(loss) * len(b)
        print(f"  {name} ep{ep} loss {tot / n:.4f}", flush=True)
    net.eval()
    with torch.no_grad():
        pred = net(Xva).argmax(axis=1).numpy()
    res = {"family": name, "n_train_used": int(len(tri))}
    for tn, t in (("all", None), ("collector", 0), ("planter", 1)):
        m = np.ones(len(pred), bool) if t is None else (wtype[va] == t)
        res[f"mlp_{tn}"] = float(np.mean(pred[m] == yva[m]))
    # latency: single-worker forward
    x1 = Xva[:1]
    with torch.no_grad():
        for _ in range(50):
            net(x1)
        t0 = time.time()
        for _ in range(500):
            net(x1)
        res["mlp_us_per_worker_b1"] = (time.time() - t0) / 500 * 1e6
        x10 = Xva[:10]
        t0 = time.time()
        for _ in range(200):
            net(x10)
        res["mlp_us_per_worker_b10"] = (time.time() - t0) / 200 / 10 * 1e6
    return res


def main():
    out = []
    for name, path, ep in (("alpha", "pred_data_alpha.npz", 4),
                           ("deepseek", "pred_data_deepseek.npz", 4),
                           ("v53", "pred_data_v53.npz", 4)):
        p = os.path.join(HERE, path)
        if not os.path.exists(p):
            print("skip", name)
            continue
        r = run_family(name, p, ep)
        out.append(r)
        print(json.dumps(r, indent=1))
    json.dump(out, open(os.path.join(HERE, "pred_mlp_results.json"), "w"),
              indent=1)


if __name__ == "__main__":
    main()

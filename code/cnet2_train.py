"""cnet2_train -- behaviour cloning of cnet2 on a cnet2_build dataset (MPS).

Usage: python cnet2_train.py DATA_DIR OUT.npz EPOCHS [cfg-json]
env: BC_DEVICE (mps), BC_BS (512), BC_LR (1e-3), BC_INIT (npz to start from),
     BC_MAXSTEPS (cap optimiser steps per epoch, for smoke tests)
Augmentation: the 4 symmetries of the canonical frame that keep the base sets.
"""

import json
import os
import queue
import sys
import threading
import time

import numpy as np
import torch
import torch.nn.functional as Fn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cnet2_feat as F  # noqa: E402
from cnet2_build import FIELDS  # noqa: E402
from cnet2_model import CNet2  # noqa: E402

DEV = torch.device(os.environ.get("BC_DEVICE", "mps"))
TW = {0: 1.5, 1: 1.5, 2: 1.5}       # Master, 1C-Zero v9, v8-preview weighted up
LOAD = [k for k in FIELDS if k not in ("gid", "step")]


def load(dirn):
    mm = None if os.environ.get("BC_RAM") == "1" else "r"     # BC_RAM=1: read the whole dataset into memory
    D = {k: np.load(os.path.join(dirn, k + ".npy"), mmap_mode=mm) for k in FIELDS}
    if os.path.exists(os.path.join(dirn, "w.npy")) and os.environ.get("BC_USE_W", "1") == "1":
        D["w"] = np.load(os.path.join(dirn, "w.npy"), mmap_mode=mm)      # per-sample weights (dataset v4+)
    if os.path.exists(os.path.join(dirn, "uw.npy")) and os.environ.get("BC_USE_UW", "1") == "1":
        D["uw"] = np.load(os.path.join(dirn, "uw.npy"), mmap_mode=mm)    # per-unit weights (dataset v6+)
    meta = json.load(open(os.path.join(dirn, "meta.json")))
    val_g = np.array([g["val"] for g in meta["games"]])
    is_val = val_g[np.asarray(D["gid"])]
    return D, is_val, meta


def fetch(D, ii):
    ii = np.sort(ii)
    return {k: np.asarray(D[k][ii]) for k in LOAD + [x for x in ("w", "uw") if x in D]}


SYM_CELL = torch.from_numpy(F.SYM_CELL)
SYM_MOVE = torch.from_numpy(F.SYM_MOVE)


GPU_AUG = os.environ.get("BC_GPU_AUG") == "1"   # move the raw batch to the device first, augment there (CPU-bound main thread)


def to_dev(b, aug):
    t = {}
    B = len(b["y"])
    for k, v in b.items():
        if v.dtype == np.float16:
            v = v.astype(np.float32)
        t[k] = torch.from_numpy(np.ascontiguousarray(v))
        if GPU_AUG:
            t[k] = t[k].to(DEV, non_blocking=True)
    SC, SM = (SYM_CELL.to(DEV), SYM_MOVE.to(DEV)) if GPU_AUG else (SYM_CELL, SYM_MOVE)
    dv = DEV if GPU_AUG else None
    t["maps"] = t["maps"].float() / 255.0
    t["carbon"] = t["carbon"].float() / 255.0
    for k in ("tok_pos", "tok_kind", "hist_cell", "hist_move", "y", "teacher"):
        t[k] = t[k].long()
    if aug:
        g = torch.randint(0, 4, (B,), device=dv)
        cm = SC[g]                                         # [B,N]
        mm = SM[g]                                         # [B,5]
        t["maps"] = torch.gather(t["maps"], 2, cm[:, None, :].expand(-1, F.P, -1))
        t["carbon"] = torch.gather(t["carbon"], 1, cm)
        t["tok_pos"] = torch.gather(cm, 1, t["tok_pos"])
        hc = t["hist_cell"]
        hcm = torch.gather(cm, 1, hc.clamp(min=0).reshape(B, -1)).reshape(hc.shape)
        t["hist_cell"] = torch.where(hc >= 0, hcm, hc)
        mm6 = torch.cat([mm, torch.full((B, 1), 5, dtype=torch.long, device=dv)], 1)
        t["hist_move"] = torch.gather(mm6, 1, t["hist_move"].reshape(B, -1)).reshape(t["hist_move"].shape)
        mf = t["mf"]
        w = torch.gather(mf[:, :10], 2, mm[:, None, :, None].expand(-1, 10, -1, F.MF))
        t["mf"] = torch.cat([w, mf[:, 10:]], 1)
        y = t["y"]
        yw = torch.where(y[:, :10] >= 0, torch.gather(mm, 1, y[:, :10].clamp(min=0)), y[:, :10])
        t["y"] = torch.cat([yw, y[:, 10:]], 1)
    t["maps"] = t["maps"].reshape(B, F.P, F.S, F.S)
    return {k: v.to(DEV, non_blocking=True) for k, v in t.items()}


def loader(D, idx, bs, aug, shuffle=True, nthreads=None):
    nthreads = nthreads or int(os.environ.get("BC_THREADS", "3"))   # data-loading threads (CPU side)
    q = queue.Queue(maxsize=8)
    order = np.random.permutation(idx) if shuffle else idx
    chunks = [order[i:i + bs] for i in range(0, len(order), bs)]
    lock = threading.Lock()
    it = iter(range(len(chunks)))
    res = {}

    def work():
        while True:
            with lock:
                try:
                    j = next(it)
                except StopIteration:
                    return
            res_b = fetch(D, chunks[j])
            q.put((j, res_b))
    ths = [threading.Thread(target=work, daemon=True) for _ in range(nthreads)]
    for th in ths:
        th.start()
    nxt = 0
    while nxt < len(chunks):
        while nxt not in res:
            j, b = q.get()
            res[j] = b
        yield to_dev(res.pop(nxt), aug)
        nxt += 1


AMP = os.environ.get("BC_AMP", "bf16")


def losses(net, b):
    if AMP == "bf16" and DEV.type == "mps":
        with torch.autocast("mps", dtype=torch.bfloat16):
            lg, v, aux = net(b)
        lg, v, aux = lg.float(), v.float(), aux.float()
    else:
        lg, v, aux = net(b)
    y = b["y"]
    valid = y >= 0
    ce = Fn.cross_entropy(lg.reshape(-1, 5), y.clamp(min=0).reshape(-1), reduction="none").reshape(y.shape)
    tw = torch.ones_like(b["value"])
    for k, w in TW.items():
        tw = torch.where(b["teacher"] == k, torch.full_like(tw, w), tw)
    if "w" in b:
        tw = tw * b["w"]
    wv = valid.float() * tw[:, None]
    if "uw" in b:
        wv = wv * b["uw"]
    lpi = (ce * wv).sum() / wv.sum().clamp(min=1)
    lv = Fn.mse_loss(v, b["value"])
    la = Fn.mse_loss(aux, b["carbon"]) * 10.0
    return lpi, lv, la, lg


@torch.no_grad()
def evaluate(net, D, idx, bs=1024):
    net.eval()
    ok = np.zeros(3); n = np.zeros(3); lsum = 0.0; ln = 0
    for b in loader(D, idx, bs, aug=False, shuffle=False):
        lpi, lv, la, lg = losses(net, b)
        pred = lg.argmax(-1)
        y = b["y"]
        kind = b["tok_kind"][:, :F.OWN]
        for k in range(3):
            m = (y >= 0) & (kind == k)
            ok[k] += ((pred == y) & m).sum().item(); n[k] += m.sum().item()
        lsum += lpi.item() * len(y); ln += len(y)
    net.train()
    r = {nm: round(ok[k] / max(n[k], 1), 4) for k, nm in enumerate(("collector", "planter", "base"))}
    r["all"] = round(ok.sum() / max(n.sum(), 1), 4)
    r["workers"] = round(ok[:2].sum() / max(n[:2].sum(), 1), 4)
    r["loss"] = round(lsum / max(ln, 1), 4)
    return r


def export(net, path, meta):
    sd = {k: v.detach().cpu().float().numpy() for k, v in net.state_dict().items()}
    np.savez(path, **sd, meta=np.array(json.dumps(meta)))


def main(data, out, epochs, cfg):
    torch.manual_seed(0); np.random.seed(0)
    D, is_val, meta = load(data)
    tr = np.nonzero(~is_val)[0]; va = np.nonzero(is_val)[0]
    sub = int(os.environ.get("BC_SUBSET", "0"))          # sanity runs: random subset of train / val
    if sub:
        rs = np.random.RandomState(1)
        tr = np.sort(rs.choice(tr, min(sub, len(tr)), replace=False))
        va = np.sort(rs.choice(va, min(max(sub // 4, 256), len(va)), replace=False))
    net = CNet2(**cfg).to(DEV)
    init = os.environ.get("BC_INIT")
    if init:
        w = np.load(init)
        net.load_state_dict({k: torch.from_numpy(w[k]) for k in w.files if k != "meta"})
        for r_ in [int(x) for x in os.environ.get("BC_REINIT_TEACH", "").split(",") if x]:
            with torch.no_grad():   # fresh style row for a re-assigned slot (others keep their warm start)
                net.teach.weight[r_].normal_(0.0, float(net.teach.weight.std()))
            print(json.dumps({"reinit_teacher_row": r_}), flush=True)
    npar = sum(p.numel() for p in net.parameters())
    bs = int(os.environ.get("BC_BS", "512"))
    lr = float(os.environ.get("BC_LR", "1e-3"))
    maxsteps = int(os.environ.get("BC_MAXSTEPS", "0"))
    spe = len(tr) // bs if not maxsteps else min(maxsteps, len(tr) // bs)
    print(json.dumps({"train": len(tr), "val": len(va), "params": npar, "cfg": net.cfg, "bs": bs,
                      "steps_per_epoch": spe}), flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * spe, pct_start=float(os.environ.get("BC_PCT", "0.05")))
    best = -1
    log = open(out.replace(".npz", "_log.jsonl"), "a")
    for ep in range(epochs):
        t0 = time.time(); acc = np.zeros(3); n = 0
        for i, b in enumerate(loader(D, tr, bs, aug=True)):
            if i >= spe:
                break
            lpi, lv, la, _ = losses(net, b)
            loss = lpi + 0.2 * lv + la
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), float(os.environ.get("BC_CLIP", "2.0")))
            opt.step(); sched.step()
            if i % 50 == 0:
                acc += [lpi.item(), lv.item(), la.item()]; n += 1
            if i % 500 == 0:
                print(json.dumps({"ep": ep, "i": i, "pi": round(acc[0] / n, 4), "v": round(acc[1] / n, 4),
                                  "aux": round(acc[2] / n, 4), "s": round(time.time() - t0)}), flush=True)
        r = evaluate(net, D, va)
        rec = {"ep": ep, "val": r, "train_pi": round(acc[0] / max(n, 1), 4), "s": round(time.time() - t0)}
        print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
        export(net, out.replace(".npz", "_last.npz"), {"cfg": net.cfg, "epoch": ep, "val": r, "params": npar})
        if os.environ.get("BC_SAVE_EPOCHS") == "1":       # keep every epoch for the online probes to choose from
            export(net, out.replace(".npz", f"_ep{ep}.npz"), {"cfg": net.cfg, "epoch": ep, "val": r, "params": npar})
        if r["all"] > best:
            best = r["all"]
            export(net, out, {"cfg": net.cfg, "epoch": ep, "val": r, "params": npar, "data": data})
    print("best", best)


if __name__ == "__main__":
    cfg = json.loads(sys.argv[4]) if len(sys.argv) > 4 else {}
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), cfg)

"""Stage 0 full head training: train ONLY utility/wdl/margin heads on a
stock dataset (backbone frozen), per-epoch val metrics + U calibration,
save merged weights (init arrays + trained heads) as a new npz.

Run from code/:
  BC_DEVICE=cuda BC_RAM=1 BC_BS=256 python3 stage0_train_heads.py \
    <ds_dir> <init.npz> <out.npz> [epochs=20]
Labels/loss identical to stage0_smoke_heads.head_losses (atanh-derived).
"""
import json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("BC_DEVICE", "cuda")
os.environ.setdefault("BC_RAM", "1")
import torch
import torch.nn.functional as Fn
import cnet2_train as CT
from cnet2_model import CNet2
from stage0_smoke_heads import head_losses

def targets(b):
    mg = 3000.0 * torch.atanh(b["value"].clamp(-0.999, 0.999))
    w = torch.where(mg > 0, torch.ones_like(mg), torch.where(mg < 0, torch.zeros_like(mg), torch.full_like(mg, 0.5)))
    u_tgt = 0.5 * w + 0.5 * torch.sigmoid(mg / 1000.0)
    cls = torch.where(mg > 0, torch.zeros_like(mg, dtype=torch.long),
                      torch.where(mg < 0, torch.full_like(mg, 2, dtype=torch.long), torch.ones_like(mg, dtype=torch.long)))
    return mg, u_tgt, cls

def evaluate(net, D, va, bs):
    net.eval()
    us, uts, preds, clss, mks, mgs = [], [], [], [], [], []
    with torch.no_grad():
        for b in CT.loader(D, va, bs, aug=False, shuffle=False):
            _, _, _, u, wdl, mk = net.forward_heads(b)
            mg, u_tgt, cls = targets(b)
            us.append(u.cpu().numpy()); uts.append(u_tgt.cpu().numpy())
            preds.append(wdl.argmax(1).cpu().numpy()); clss.append(cls.cpu().numpy())
            mks.append(mk.cpu().numpy()); mgs.append((mg / 1000.0).cpu().numpy())
    u = np.concatenate(us); ut = np.concatenate(uts)
    pr = np.concatenate(preds); cl = np.concatenate(clss)
    mk = np.concatenate(mks); mg = np.concatenate(mgs)
    bins = np.clip((ut * 10).astype(int), 0, 9)
    calib_bin = 0.0
    for i in range(10):
        m = bins == i
        if m.sum() > 0:
            calib_bin += m.mean() * abs(u[m].mean() - ut[m].mean())
    return {"val_l_u": round(float(np.mean((u - ut) ** 2)), 5),
            "val_wdl_acc": round(float((pr == cl).mean()), 4),
            "val_margin_mae": round(float(np.abs(mk - mg).mean() * 1000), 1),
            "val_u_calib_mean": round(float(u.mean() - ut.mean()), 5),
            "val_u_calib_binned": round(float(calib_bin), 5),
            "val_u_range": [round(float(u.min()), 3), round(float(u.max()), 3)],
            "val_n": int(len(u))}

def main():
    data, init, out = sys.argv[1], sys.argv[2], sys.argv[3]
    epochs = int(sys.argv[4]) if len(sys.argv) > 4 else 20
    bs = int(os.environ.get("BC_BS", "256"))
    D, is_val, meta = CT.load(data)
    tr = np.nonzero(~is_val)[0]; va = np.nonzero(is_val)[0]
    w = np.load(init)
    cfg = json.loads(str(w["meta"]))["cfg"]
    net = CNet2(**cfg).to(CT.DEV)
    sd = {k: torch.from_numpy(w[k]) for k in w.files if k != "meta"}
    missing = net.load_state_dict(sd, strict=False)
    assert sorted(missing.missing_keys) == ["margin.bias", "margin.weight", "utility.bias", "utility.weight", "wdl.bias", "wdl.weight"], missing.missing_keys
    for n, p in net.named_parameters():
        p.requires_grad_(n.split(".")[0] in ("utility", "wdl", "margin"))
    steps_per_epoch = int(np.ceil(len(tr) / bs))
    total = steps_per_epoch * epochs
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=total, pct_start=0.05)
    print(json.dumps({"train": len(tr), "val": len(va), "epochs": epochs, "steps_per_epoch": steps_per_epoch, "cfg": cfg}), flush=True)
    t0 = time.time()
    for ep in range(1, epochs + 1):
        net.train()
        hs = []
        for b in CT.loader(D, tr, bs, aug=False, shuffle=True):
            loss, l_u, l_wdl, l_m, acc, mae, ur = head_losses(net, b)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            hs.append((l_u, l_wdl, l_m))
        h = np.array(hs)
        ev = evaluate(net, D, va, bs)
        print("EPOCH", json.dumps({"ep": ep, "train_l_u": round(float(h[:, 0].mean()), 5),
              "train_l_wdl": round(float(h[:, 1].mean()), 4), "train_l_m": round(float(h[:, 2].mean()), 4),
              **ev, "s": round(time.time() - t0)}), flush=True)
    sd_out = {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}
    sd_out["meta"] = w["meta"]
    np.savez(out, **sd_out)
    print("TRAIN-DONE", json.dumps({"out": out, "s": round(time.time() - t0)}), flush=True)

if __name__ == "__main__":
    main()

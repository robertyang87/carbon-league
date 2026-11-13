"""Stage 0 full head training: train ONLY utility/wdl/margin heads on a
stock dataset (backbone frozen), per-epoch val metrics + U calibration,
save merged weights (init arrays + trained heads) as a new npz.

Run from code/:
  BC_DEVICE=cuda BC_RAM=1 BC_BS=256 python3 stage0_train_heads.py \
    <ds_dir> <init.npz> <out.npz> [epochs=20]
Labels/loss identical to stage0_smoke_heads.head_losses (atanh-derived);
margin head loss is MSE (expectation head, F10).
Calibration (F12): the headline U-calibration bins by PREDICTED u and checks
E[U | predicted u] ~= predicted u; the old target-binned statistic is kept
only as a renamed diagnostic (val_u_target_binned_resid).
Hard gates (F10): init missing any non-head key, any non-finite train/val
metric, or any non-finite parameter before saving aborts non-zero and
saves nothing (no TRAIN-DONE).
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
from cnet2_model import CNet2, migrate_state_dict
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
    # Diagnostic ONLY (renamed, F12): residual stratified by TARGET u. This is
    # NOT calibration -- it bins on the observed value, not the prediction.
    tbins = np.clip((ut * 10).astype(int), 0, 9)
    target_bin_resid = 0.0
    for i in range(10):
        m = tbins == i
        if m.sum() > 0:
            target_bin_resid += m.mean() * abs(u[m].mean() - ut[m].mean())
    # Calibration (headline, F12): bin by PREDICTED u; per bin compare the
    # observed mean E[U | predicted u] against the predicted mean, with the
    # bin count and the standard error of the observed mean as uncertainty.
    pbins = np.clip((u * 10).astype(int), 0, 9)
    calib_pred = 0.0
    pred_bins = []
    for i in range(10):
        m = pbins == i
        n_i = int(m.sum())
        if n_i > 0:
            mp = float(u[m].mean()); mo = float(ut[m].mean())
            se = float(ut[m].std(ddof=1) / np.sqrt(n_i)) if n_i > 1 else 0.0
            calib_pred += (n_i / len(u)) * abs(mo - mp)
            pred_bins.append([n_i, round(mp, 4), round(mo, 4), round(se, 4)])
    return {"val_l_u": round(float(np.mean((u - ut) ** 2)), 5),
            "val_wdl_acc": round(float((pr == cl).mean()), 4),
            "val_margin_mae": round(float(np.abs(mk - mg).mean() * 1000), 1),
            "val_u_calib_mean": round(float(u.mean() - ut.mean()), 5),
            "val_u_target_binned_resid": round(float(target_bin_resid), 5),
            "val_u_calib_pred_binned": round(float(calib_pred), 5),
            "val_u_calib_pred_bins": pred_bins,
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
    try:   # F01 migration path: only the 6 new-head keys may be absent (a bad
        # trunk key such as stem.weight is still a hard error); heads already
        # present (this script's own output as init) load as-is.
        mig = migrate_state_dict(net, sd, source="stage0_train_heads init")
    except RuntimeError as e:
        raise SystemExit(str(e))
    print(json.dumps({"init_migrated_missing": mig["migrated_missing"]}), flush=True)
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
        # F10 hard gate: non-finite train/val metrics abort before any print/save.
        nums = [float(h[:, 0].mean()), float(h[:, 1].mean()), float(h[:, 2].mean()),
                ev["val_l_u"], ev["val_wdl_acc"], ev["val_margin_mae"], ev["val_u_calib_mean"],
                ev["val_u_target_binned_resid"], ev["val_u_calib_pred_binned"],
                ev["val_u_range"][0], ev["val_u_range"][1]]
        if not (np.isfinite(h).all() and np.isfinite(nums).all()):
            raise SystemExit(f"F10 gate: non-finite train/val metric at epoch {ep}; nothing saved")
        print("EPOCH", json.dumps({"ep": ep, "train_l_u": round(float(h[:, 0].mean()), 5),
              "train_l_wdl": round(float(h[:, 1].mean()), 4), "train_l_m": round(float(h[:, 2].mean()), 4),
              **ev, "s": round(time.time() - t0)}), flush=True)
    # F10 hard gate: never publish non-finite parameters.
    if not all(bool(torch.isfinite(p).all()) for p in net.parameters()):
        raise SystemExit("F10 gate: non-finite parameter before save; nothing saved")
    sd_out = {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}
    sd_out["meta"] = w["meta"]
    np.savez(out, **sd_out)
    print("TRAIN-DONE", json.dumps({"out": out, "s": round(time.time() - t0)}), flush=True)

if __name__ == "__main__":
    main()

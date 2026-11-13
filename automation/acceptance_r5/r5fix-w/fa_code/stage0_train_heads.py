"""Stage 0 full head training: train ONLY utility/wdl/margin heads on a
stock dataset (backbone frozen), per-epoch val metrics + U calibration,
save merged weights (init arrays + trained heads) as a new npz.

Run from code/:
  BC_DEVICE=cuda BC_RAM=1 BC_BS=256 python3 stage0_train_heads.py \
    <ds_dir> <init.npz> <out.npz> [epochs=20]
Labels: EXACT labels first (owner review pt.7 follow-up, 2026-11-13).
When the dataset carries margin_raw/u_raw/label_status columns
(cnet2_build exact-label contract), targets come from those columns --
per row, never silently mixed: rows with finite raw values are exact,
rows without fall back to the atanh-derived approximation and are
counted as approx in the log and in the product manifest
(<out>.manifest.json: approx_labels=true whenever any approx row is
present, label_source recorded). Without the columns the whole dataset
is approx (source "tanh-inverse(value)") and the manifest says so.
Value-loss masking: rows with label_status == LABEL_UNUSABLE (an
unusable game result -- double execution fault / missing data) are
excluded from the U and margin losses AND from the WDL cross-entropy:
their result class does not exist, so no head target is defined for
them. (The BC policy loss lives in cnet2_train, does not consume these
columns, and is unaffected by this mask.) Approx rows DO enter the
value losses -- they are counted separately, not excluded.
Loss composition otherwise identical to stage0_smoke_heads.head_losses;
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

# Label-status codes -- MUST match cnet2_build (LABEL_* there).
LABEL_EXACT, LABEL_APPROX, LABEL_UNUSABLE = 0, 1, 2


def targets(b):
    """Per-row head targets. Exact raw columns win row by row; rows
    whose raw value is NaN (approx rows) fall back to the atanh-derived
    target from the legacy value field."""
    mg = 3000.0 * torch.atanh(b["value"].clamp(-0.999, 0.999))
    if "margin_raw" in b:
        mr = b["margin_raw"].float()
        mg = torch.where(torch.isfinite(mr), mr, mg)
    w = torch.where(mg > 0, torch.ones_like(mg), torch.where(mg < 0, torch.zeros_like(mg), torch.full_like(mg, 0.5)))
    u_tgt = 0.5 * w + 0.5 * torch.sigmoid(mg / 1000.0)
    if "u_raw" in b:
        ur = b["u_raw"].float()
        u_tgt = torch.where(torch.isfinite(ur), ur, u_tgt)
    cls = torch.where(mg > 0, torch.zeros_like(mg, dtype=torch.long),
                      torch.where(mg < 0, torch.full_like(mg, 2, dtype=torch.long), torch.ones_like(mg, dtype=torch.long)))
    return mg, u_tgt, cls


def _usable_mask(b, n):
    """Rows allowed into value-family losses: everything except
    label_status == LABEL_UNUSABLE. Also returns per-status counts."""
    if "label_status" in b:
        st = b["label_status"]
        return st != LABEL_UNUSABLE, {
            "exact": int((st == LABEL_EXACT).sum()),
            "approx": int((st == LABEL_APPROX).sum()),
            "unusable": int((st == LABEL_UNUSABLE).sum())}
    return torch.ones(n, dtype=torch.bool, device=b["value"].device), {
        "exact": 0, "approx": int(n), "unusable": 0}


def masked_head_losses(net, b):
    """head_losses with exact-label targets + the unusable mask.
    Returns (loss, l_u, l_wdl, l_m, acc, mae, u_range, counts)."""
    lg, v, aux, u, wdl, mk = net.forward_heads(b)
    mg, u_tgt, cls = targets(b)
    usable, counts = _usable_mask(b, mg.numel())
    nu = int(usable.sum())

    def _mse(pred, tgt):
        if nu == 0:
            return (pred * 0.0).sum()   # no usable rows: zero, graph intact
        return Fn.mse_loss(pred[usable], tgt[usable])

    l_u = _mse(u, u_tgt)
    l_m = _mse(mk, mg / 1000.0)
    l_wdl = Fn.cross_entropy(wdl[usable], cls[usable]) if nu else (wdl * 0.0).sum()
    with torch.no_grad():
        acc = float((wdl[usable].argmax(1) == cls[usable]).float().mean()) if nu else 0.0
        mae = float((mk[usable] - mg[usable] / 1000.0).abs().mean() * 1000.0) if nu else 0.0
        ur = (float(u.min()), float(u.max()))
    return l_u + 0.25 * l_wdl + 0.25 * l_m, float(l_u), float(l_wdl), float(l_m), acc, mae, ur, counts


def load_label_columns(data, D):
    """Load the builder's exact-label columns (if any) into D and the
    batch loader. Returns the provenance record used in the log and the
    product manifest. A PARTIAL column set is a hard error -- provenance
    is never guessed."""
    total = len(D["y"])
    info = {"total_rows": int(total), "exact_rows": 0, "approx_rows": int(total),
            "unusable_rows": 0, "approx_labels": True,
            "label_source": "tanh-inverse(value)"}
    cols = ("margin_raw", "u_raw", "outcome_type", "label_status")
    have = [k for k in cols if os.path.exists(os.path.join(data, k + ".npy"))]
    if not have:
        return info
    if "label_status" not in have or "margin_raw" not in have:
        raise SystemExit(
            f"label columns incomplete in {data}: found {have}; refusing to guess provenance")
    for k in have:
        arr = np.load(os.path.join(data, k + ".npy"))
        if len(arr) != total:
            raise SystemExit(
                f"label column {k} has {len(arr)} rows, dataset has {total}: misaligned, refusing")
        D[k] = arr
        if k not in CT.LOAD:
            CT.LOAD.append(k)
    st = np.asarray(D["label_status"])
    info.update({"exact_rows": int((st == LABEL_EXACT).sum()),
                 "approx_rows": int((st == LABEL_APPROX).sum()),
                 "unusable_rows": int((st == LABEL_UNUSABLE).sum())})
    info["approx_labels"] = info["approx_rows"] > 0
    info["label_source"] = ("mixed: margin_raw/u_raw exact + tanh-inverse approx"
                            if info["approx_rows"] else "margin_raw/u_raw (exact)")
    return info

def evaluate(net, D, va, bs):
    net.eval()
    us, uts, preds, clss, mks, mgs, sts = [], [], [], [], [], [], []
    with torch.no_grad():
        for b in CT.loader(D, va, bs, aug=False, shuffle=False):
            _, _, _, u, wdl, mk = net.forward_heads(b)
            mg, u_tgt, cls = targets(b)
            us.append(u.cpu().numpy()); uts.append(u_tgt.cpu().numpy())
            preds.append(wdl.argmax(1).cpu().numpy()); clss.append(cls.cpu().numpy())
            mks.append(mk.cpu().numpy()); mgs.append((mg / 1000.0).cpu().numpy())
            if "label_status" in b:
                sts.append(b["label_status"].cpu().numpy())
    u = np.concatenate(us); ut = np.concatenate(uts)
    pr = np.concatenate(preds); cl = np.concatenate(clss)
    mk = np.concatenate(mks); mg = np.concatenate(mgs)
    if sts:
        # Unusable rows have no defined targets: exclude them from every
        # value metric and from calibration (their fallback targets are
        # placeholders, not labels).
        keep = np.concatenate(sts) != LABEL_UNUSABLE
        u, ut, pr, cl, mk, mg = u[keep], ut[keep], pr[keep], cl[keep], mk[keep], mg[keep]
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
    label_info = load_label_columns(data, D)
    print(json.dumps({"labels": label_info}), flush=True)
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
        cnt = {"exact": 0, "approx": 0, "unusable": 0}
        for b in CT.loader(D, tr, bs, aug=False, shuffle=True):
            loss, l_u, l_wdl, l_m, acc, mae, ur, c1 = masked_head_losses(net, b)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            hs.append((l_u, l_wdl, l_m))
            for k in cnt:
                cnt[k] += c1[k]
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
              "train_label_exact": cnt["exact"], "train_label_approx": cnt["approx"],
              "train_label_unusable": cnt["unusable"],
              "train_value_rows": cnt["exact"] + cnt["approx"],
              **ev, "s": round(time.time() - t0)}), flush=True)
    # F10 hard gate: never publish non-finite parameters.
    if not all(bool(torch.isfinite(p).all()) for p in net.parameters()):
        raise SystemExit("F10 gate: non-finite parameter before save; nothing saved")
    sd_out = {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}
    sd_out["meta"] = w["meta"]
    np.savez(out, **sd_out)
    # Product manifest: label provenance is part of the product record --
    # approx_labels=true whenever ANY training row used the tanh-inverse
    # approximation (pure-approx datasets included), never silent.
    manifest = {"out": out, "dataset": data, "init": init, "epochs": epochs,
                "labels": label_info, "s": round(time.time() - t0)}
    man_path = out + ".manifest.json"
    with open(man_path, "w") as fh:
        json.dump(manifest, fh)
    print("TRAIN-DONE", json.dumps({"out": out, "manifest": man_path,
                                    "labels": label_info,
                                    "s": round(time.time() - t0)}), flush=True)

if __name__ == "__main__":
    main()

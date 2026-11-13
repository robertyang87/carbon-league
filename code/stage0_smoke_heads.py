"""Stage 0 smoke (d): train ONLY the new E[U]/WDL/margin heads on ds_m3s2,
backbone frozen, a few hundred steps. Verifies the link (labels -> loss ->
finite decreasing loss, U outputs in [0,1]).

Labels derived from the dataset's stored value field v = tanh(M/3000):
  M = 3000*atanh(clip(v, +-0.999)); W = 1/0/0.5 by sign(M);
  U = 0.5*W + 0.5*sigmoid(M/1000); WDL class = win 0 / draw 1 / loss 2;
  margin target = M/1000.
v1 caveat: atanh saturates for |M| beyond ~11400 (extinction tails); the
stored value field cannot recover those exactly. Future datasets should
store raw margin / U directly (follow-up, not v1).

Loss (heads only): MSE(u, U) + 0.25*CE(wdl, class) + 0.25*MSE(mk, M/1000).
The margin head is an expectation head (E[M]/1000): its loss MUST be MSE --
smooth_l1/Huber has its minimum at a robust location, not the mean (F10,
REVIEW_FULL_20261113); Huber survives only as diagnostics (MAE below),
never as a training loss for an expectation.
Hard gates (F10): init weights missing any non-head key (e.g. stem.weight),
any non-finite train/val loss, or any non-finite parameter aborts with a
non-zero exit -- a finite=false summary is never a passing smoke.
Run from code/ with:
  BC_DEVICE=cuda BC_RAM=1 BC_BS=256 BC_INIT=<m3s2r1 npz> \
    python3 stage0_smoke_heads.py <ds_m3s2 dir> [steps=300]
Prints JSON lines per 50 steps + SMOKE summary. Saves nothing to the repo.
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

os.environ.setdefault("BC_DEVICE", "cuda")
os.environ.setdefault("BC_RAM", "1")

import torch  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402
import cnet2_train as CT  # noqa: E402
from cnet2_model import CNet2, migrate_state_dict  # noqa: E402


def head_losses(net, b):
    lg, v, aux, u, wdl, mk = net.forward_heads(b)
    mg = 3000.0 * torch.atanh(b["value"].clamp(-0.999, 0.999))
    w = torch.where(mg > 0, torch.ones_like(mg),
                    torch.where(mg < 0, torch.zeros_like(mg), torch.full_like(mg, 0.5)))
    u_tgt = 0.5 * w + 0.5 * torch.sigmoid(mg / 1000.0)
    cls = torch.where(mg > 0, torch.zeros_like(mg, dtype=torch.long),
                      torch.where(mg < 0, torch.full_like(mg, 2, dtype=torch.long),
                                  torch.ones_like(mg, dtype=torch.long)))
    l_u = Fn.mse_loss(u, u_tgt)
    l_wdl = Fn.cross_entropy(wdl, cls)
    l_m = Fn.mse_loss(mk, mg / 1000.0)   # expectation head -> MSE (F10); NOT smooth_l1/Huber
    with torch.no_grad():
        acc = (wdl.argmax(1) == cls).float().mean().item()
        mae = (mk - mg / 1000.0).abs().mean().item() * 1000.0
        ur = (float(u.min()), float(u.max()))
    return l_u + 0.25 * l_wdl + 0.25 * l_m, float(l_u), float(l_wdl), float(l_m), acc, mae, ur


def main():
    data = sys.argv[1]
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    bs = int(os.environ.get("BC_BS", "256"))
    init = os.environ["BC_INIT"]
    D, is_val, meta = CT.load(data)
    tr = np.nonzero(~is_val)[0]
    va = np.nonzero(is_val)[0]
    w = np.load(init)
    cfg = json.loads(str(w["meta"]))["cfg"]
    net = CNet2(**cfg).to(CT.DEV)
    sd = {k: torch.from_numpy(w[k]) for k in w.files if k != "meta"}
    try:   # F01/F10 hard gate: only the 6 new-head keys may be absent in old init weights
        mig = migrate_state_dict(net, sd, source="stage0_smoke init")
    except RuntimeError as e:
        raise SystemExit(str(e))
    print(json.dumps({"init_missing": mig["migrated_missing"],
                      "train": len(tr), "val": len(va), "cfg": cfg, "bs": bs}), flush=True)
    for n, p in net.named_parameters():
        p.requires_grad_(n.split(".")[0] in ("utility", "wdl", "margin"))
    ntr = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(json.dumps({"trainable_params": ntr}), flush=True)
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps, pct_start=0.05)
    hist = []
    t0 = time.time()
    done = 0
    epoch = 0
    while done < steps:
        epoch += 1
        for b in CT.loader(D, tr, bs, aug=False, shuffle=True):
            # loader() already applies to_dev (cnet2_train.loader line 123)
            loss, l_u, l_wdl, l_m, acc, mae, ur = head_losses(net, b)
            if not (bool(torch.isfinite(loss)) and np.isfinite([l_u, l_wdl, l_m, acc, mae]).all()):
                print(json.dumps({"SMOKE-FAIL": "non-finite train loss", "step": done + 1}), flush=True)
                sys.exit(1)   # F10 hard gate: NaN/Inf train loss aborts, never a passing summary
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            done += 1
            hist.append((l_u, l_wdl, l_m))
            if done % 50 == 0 or done == 1:
                print(json.dumps({"step": done, "loss": round(float(loss), 4),
                                  "l_u": round(l_u, 4), "l_wdl": round(l_wdl, 4),
                                  "l_m": round(l_m, 4), "wdl_acc": round(acc, 3),
                                  "margin_mae": round(mae, 1),
                                  "u_range": [round(ur[0], 3), round(ur[1], 3)],
                                  "s": round(time.time() - t0)}), flush=True)
            if done >= steps:
                break
    h = np.array(hist)
    first, last = h[:50].mean(0), h[-50:].mean(0)
    # fixed-batch val check
    net.eval()
    with torch.no_grad():
        vb = next(iter(CT.loader(D, va[:1024], bs, aug=False, shuffle=False)))
        # loader() already applies to_dev
        _, vl_u, vl_wdl, vl_m, vacc, vmae, vur = head_losses(net, vb)
    # F10 hard gate: "finite" covers train history AND val metrics AND parameters
    # (previously val NaN could still print finite=true, and any non-finite run exited 0).
    val_finite = bool(np.isfinite([vl_u, vl_wdl, vl_m, vacc, vmae, vur[0], vur[1]]).all())
    par_finite = bool(all(torch.isfinite(p).all() for p in net.parameters()))
    finite = bool(np.isfinite(h).all()) and val_finite and par_finite
    print("SMOKE", json.dumps({
        "steps": done, "epochs": epoch, "s": round(time.time() - t0),
        "train_first50": [round(float(x), 4) for x in first],
        "train_last50": [round(float(x), 4) for x in last],
        "val_l_u": round(vl_u, 4), "val_l_wdl": round(vl_wdl, 4), "val_l_m": round(vl_m, 4),
        "val_wdl_acc": round(vacc, 3), "val_margin_mae": round(vmae, 1),
        "val_u_range": [round(vur[0], 3), round(vur[1], 3)],
        "finite": finite,
        "decreasing": bool(last[0] < first[0])}), flush=True)
    if not finite:
        print(json.dumps({"SMOKE-FAIL": "non-finite train/val/params",
                          "val_finite": val_finite, "param_finite": par_finite}), flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# =====================================================================
# eval_dense -- the dense main reading for the sparse-label replay
# validation (ACCEPT a7da515d SS3 layer 2): on HELD-OUT games, with
# the hidden true q as scoring truth, read the student's CE and
# anchor KL against the full-label target, the value calibration
# (utility MSE / wdl CE / margin MSE), and the student policy
# entropy. Held-out games never enter any training product, so
# their terminal labels never trained the student (design SS4
# anti-self-evaluation rule).
#
# Usage:
#   python3 eval_dense.py --data DS --student S.npz
#       --kl-reference A.npz --games games.json --out reading.json
#       [--device cuda] [--chunk 32]
# games.json: {"holdout_games": [names...]} (k0_thin_joined holdout
# output). The dataset is the SOURCE full-label product (truth
# intact); only the listed games' roots are read.
# =====================================================================
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

STEP_BANDS = [[0, 50], [50, 100], [100, 150], [150, 200],
              [200, 250], [250, 299]]


def main():
    ap = argparse.ArgumentParser(description="dense held-out reading")
    ap.add_argument("--data", required=True)
    ap.add_argument("--student", required=True)
    ap.add_argument("--kl-reference", required=True)
    ap.add_argument("--games", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--chunk", type=int, default=32)
    a = ap.parse_args()

    os.environ["BC_DEVICE"] = a.device
    import torch
    import torch.nn.functional as Fn
    import cnet2_train as T
    import k0_distill_u as D
    from cnet2_model import CNet2, migrate_state_dict

    dev = torch.device(a.device)
    ds = D.load_annotation(a.data)
    meta = json.load(open(os.path.join(ds["gen_dir"], "meta.json")))
    name_by_gid = {int(g["gid"]): g["game"] for g in meta["games"]}
    hold = set(json.load(open(a.games))["holdout_games"])
    game = np.asarray(ds["keys"]["game"])
    step = np.asarray(ds["keys"]["step"])
    rows = np.nonzero(np.array([name_by_gid[int(g)] in hold
                                for g in game]))[0]
    if rows.size == 0:
        raise SystemExit("eval_dense: no held-out roots selected")

    si = D._read_weights_file(a.student, "--student")
    kl = D._read_weights_file(a.kl_reference, "--kl-reference")
    cfg = si["meta_cfg"]
    net = CNet2(**(cfg or {})).to(dev)
    migrate_state_dict(net, {k: torch.from_numpy(
        np.ascontiguousarray(v)) for k, v in si["arrays"].items()},
        allow_missing_prefixes=("tattn.",), source="--student")
    net.eval()
    anchor = CNet2(**(cfg or {})).to(dev)
    migrate_state_dict(anchor, {k: torch.from_numpy(
        np.ascontiguousarray(v)) for k, v in kl["arrays"].items()},
        allow_missing_prefixes=("tattn.",), source="--kl-reference")
    anchor.eval()

    feats, arms_t, n_arms = ds["feats"], ds["arms"], ds["n_arms"]
    target = ds["target"]
    K = ds["info"]["max_candidates"]
    recs = []
    with torch.no_grad():
        for a0 in range(0, len(rows), a.chunk):
            idx = rows[a0:a0 + a.chunk]
            b, rep, _stats = D.expand_candidate_batch(
                feats, arms_t, idx, K)
            b = T.to_dev(b, aug=False)
            tok, ctx, _ = net.encode(b)
            logits_s = net.decode(b, tok, ctx)
            tok_p, ctx_p, _ = anchor.encode(b)
            logits_p = anchor.decode(b, tok_p, ctx_p)
            arms_b = torch.from_numpy(
                arms_t[idx].reshape(-1, arms_t.shape[2])).to(dev)
            joint, kl = D.joint_logp_kl(logits_s, logits_p, arms_b)
            Br = len(idx)
            joint = joint.reshape(Br, K)
            kl = kl.reshape(Br, K)
            tgt = torch.from_numpy(target[idx]).to(dev)
            narr = torch.from_numpy(n_arms[idx]).to(dev)
            ce = -(tgt * joint).sum(dim=1)
            klr = kl.sum(dim=1) / narr.double()
            u_pred, wdl_logits, margin_pred = net.heads(ctx)
            u_r = u_pred.reshape(Br, K)[:, 0]
            wdl_r = wdl_logits.reshape(Br, K, 3)[:, 0]
            m_r = margin_pred.reshape(Br, K)[:, 0]
            tu = torch.from_numpy(ds["terminal_u"][idx]).to(dev)
            tw = torch.from_numpy(ds["terminal_wdl"][idx]).to(dev)
            tm = torch.from_numpy(ds["terminal_margin"][idx]
                                  / 1000.0).to(dev)
            mse_u = (u_r - tu) ** 2
            ce_w = Fn.cross_entropy(wdl_r, tw, reduction="none")
            mse_m = (m_r - tm) ** 2
            # student policy entropy: per candidate, mean over its
            # valid slots of the categorical entropy; per root, the
            # mean over candidates.
            logp = Fn.log_softmax(logits_s.double(), dim=-1)
            ent_slot = -(logp.exp() * logp).sum(dim=-1) \
                * (arms_b >= 0).double()
            n_valid = (arms_b >= 0).double().sum(dim=1)
            ent_cand = ent_slot.sum(dim=1) / n_valid.clamp(min=1)
            ent = ent_cand.reshape(Br, K).sum(dim=1) \
                / narr.double()
            for i, r in enumerate(idx):
                recs.append({
                    "root": int(r), "step": int(step[r]),
                    "ce": float(ce[i]), "kl": float(klr[i]),
                    "mse_u": float(mse_u[i]), "ce_w": float(ce_w[i]),
                    "mse_m": float(mse_m[i]), "entropy": float(ent[i])})
    arr = {k: np.array([x[k] for x in recs]) for k in
           ("ce", "kl", "mse_u", "ce_w", "mse_m", "entropy")}
    steps = np.array([x["step"] for x in recs])

    def agg(mask):
        return {k: float(arr[k][mask].mean()) for k in arr}

    bands = {}
    for bi, (lo, hi) in enumerate(STEP_BANDS):
        m = (steps >= lo) & (steps <= hi if bi == len(STEP_BANDS) - 1
                             else steps < hi)
        if m.any():
            bands[f"{lo}-{hi}"] = dict(agg(m), roots=int(m.sum()))
    out = {"student": os.path.basename(a.student),
           "student_md5": si["md5"],
           "data_generation": ds["info"]["generation"],
           "holdout_games": sorted(hold),
           "roots": int(len(recs)),
           "overall": dict(agg(np.ones(len(recs), bool)),
                           roots=int(len(recs))),
           "by_step_band": bands}
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print(json.dumps({"eval_dense": out["overall"],
                      "roots": out["roots"]}))


if __name__ == "__main__":
    main()

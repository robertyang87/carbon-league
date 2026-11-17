"""distill_export.py -- export a trained student checkpoint to the production
cnet2_np npz layout and cross-check the two implementations (stage 2).

The exported npz carries the CNet2 state_dict key-for-key (the layout
cnet2_np.Policy loads, identical in convention to the teacher npz) plus a
meta JSON: {cfg, epoch, params, identity:{kind: small_teacher_student,
marker, distilled_from:{teacher_code_md5, teacher_weights_md5}}}.

Cross-check (design section 4): N positions through BOTH paths --
  numpy:  cnet2_np.Policy.act(x, aux, teacher=2, want_u=True) (its own
          autoregressive decode);
  torch:  CNet2.encode + decode teacher-forced on the numpy labels, u from
          heads(). The per-slot logits must agree to float-implementation
          tolerance; report max|dlogits| and max|du| as JSON.
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np

STUDENT_MARKER = "SMALL_TEACHER_V1"
TEACHER_CODE_MD5 = "c6ade5ced42e2c54a77c1499163aa151"
TEACHER_WEIGHTS_MD5 = "06a03d9cfeec239af3065a49fefec382"
X_FIELDS = ("maps", "scal", "tok_pos", "tok_kind", "tok_mask", "tok_cargo",
            "hist_cell", "hist_cargo", "hist_move")
AUX_FIELDS = ("est", "en_min", "en_max", "en_pl", "thr", "dob", "own_base",
              "en_base", "own_tree", "en_tree", "toll", "own_adj")


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def synthetic_positions(n, seed=11):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        x = {"maps": rng.integers(0, 256, (33, 225), np.uint8),
             "scal": rng.standard_normal(20).astype(np.float32),
             "tok_pos": rng.integers(0, 225, 24).astype(np.int64),
             "tok_kind": rng.integers(0, 6, 24).astype(np.int64),
             "tok_mask": (rng.random(24) < 0.8),
             "tok_cargo": (rng.random(24) * 100).astype(np.float32),
             "hist_cell": rng.integers(-1, 225, (24, 8)).astype(np.int64),
             "hist_cargo": (rng.random((24, 8)) * 50).astype(np.float32),
             "hist_move": rng.integers(0, 6, (24, 8)).astype(np.int64),
             "cash": float(rng.random() * 500), "slots": float(rng.integers(0, 5)),
             "left": float(rng.integers(50, 300))}
        aux = {k: (rng.random(225) * 10).astype(np.float64) for k in AUX_FIELDS}
        out.append({"x": x, "aux": aux})
    return out


def shard_positions(path, n):
    d = np.load(path, allow_pickle=False)
    m = min(n, int(d["t_logits"].shape[0]))
    out = []
    for i in range(m):
        x = {k: d["x_" + k][i] for k in X_FIELDS}
        x["cash"] = float(d["x_cash"][i]); x["slots"] = float(d["x_slots"][i])
        x["left"] = float(d["x_left"][i])
        out.append({"x": x, "aux": {k: d["aux_" + k][i] for k in AUX_FIELDS}})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="export student checkpoint to cnet2_np npz + cross-check")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--code-dir", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--check-positions", type=int, default=512)
    ap.add_argument("--check-shard", default="", help="optional datagen shard to draw check positions from")
    ap.add_argument("--report", default="", help="optional JSON report path")
    a = ap.parse_args(argv)

    sys.path.insert(0, a.code_dir)
    import torch
    import cnet2_np as NP
    from cnet2_model import CNet2

    ck = torch.load(a.ckpt, map_location="cpu")
    cfg = ck["cfg"]
    model = CNet2(**cfg)
    model.load_state_dict(ck["model"])
    model.eval()

    sd = {k: v.detach().cpu().numpy().astype(np.float32)
          for k, v in model.state_dict().items()}
    n_params = int(sum(int(v.size) for v in sd.values()))
    meta = {"cfg": cfg, "epoch": int(ck.get("epoch", -1)), "params": n_params,
            "identity": {"kind": "small_teacher_student", "marker": STUDENT_MARKER,
                         "distilled_from": {"teacher_code_md5": TEACHER_CODE_MD5,
                                            "teacher_weights_md5": TEACHER_WEIGHTS_MD5}}}
    sd["meta"] = np.str_(json.dumps(meta))
    np.savez(a.out, **sd)
    out_md5 = md5_file(a.out)
    print(f"exported {a.out}: params={n_params} md5={out_md5}")

    # the production loader must accept it unchanged
    w = dict(np.load(a.out, allow_pickle=False))
    pol = NP.Policy(w)
    assert pol.cfg == cfg, (pol.cfg, cfg)

    positions = (shard_positions(a.check_shard, a.check_positions)
                 if a.check_shard else synthetic_positions(a.check_positions))
    max_dlg, max_du, n_done = 0.0, 0.0, 0
    with torch.no_grad():
        for p in positions:
            x, aux = p["x"], p["aux"]
            lab, lg_np, _v, u_np = pol.act(x, aux, teacher=2, want_u=True)
            b = {"maps": torch.as_tensor(x["maps"].astype(np.float32)[None] / 255.0),
                 "tok_kind": torch.as_tensor(x["tok_kind"][None]),
                 "tok_cargo": torch.as_tensor(x["tok_cargo"].astype(np.float32)[None]),
                 "tok_pos": torch.as_tensor(x["tok_pos"][None]),
                 "tok_mask": torch.as_tensor(x["tok_mask"][None]),
                 "hist_cell": torch.as_tensor(x["hist_cell"][None]),
                 "hist_cargo": torch.as_tensor(x["hist_cargo"].astype(np.float32)[None]),
                 "hist_move": torch.as_tensor(x["hist_move"][None]),
                 "scal": torch.as_tensor(x["scal"].astype(np.float32)[None]),
                 "teacher": torch.tensor([2]),
                 "y": torch.as_tensor(np.maximum(lab, 0)[None])}
            import cnet2_feat as F
            pf = F.Prefix(x, aux)
            uf = np.zeros((1, 12, F.UF), np.float32)
            mf = np.zeros((1, 12, 5, F.MF), np.float32)
            legal = np.zeros((1, 12, 5), bool)
            for s in range(12):
                if x["tok_mask"][s]:
                    u_, m_, l_ = pf.feats(s)
                    uf[0, s], mf[0, s], legal[0, s] = u_, m_, l_
                    if lab[s] >= 0:
                        pf.advance(s, int(lab[s]))
            b["uf"] = torch.as_tensor(uf); b["mf"] = torch.as_tensor(mf)
            b["legal"] = torch.as_tensor(legal)
            tok, ctx, _ = model.encode(b)
            lg_t = model.decode(b, tok, ctx)[0].numpy()
            u_t = float(model.heads(ctx)[0][0])
            max_dlg = max(max_dlg, float(np.abs(lg_t - lg_np).max()))
            max_du = max(max_du, abs(u_t - float(u_np)))
            n_done += 1
    report = {"out_npz": a.out, "out_md5": out_md5, "params": n_params,
              "cfg": cfg, "check_positions": n_done,
              "max_dlogits_torch_vs_np": max_dlg, "max_du_torch_vs_np": max_du,
              "marker": STUDENT_MARKER}
    print(json.dumps(report, indent=1))
    if a.report:
        with open(a.report, "w") as fh:
            json.dump(report, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())

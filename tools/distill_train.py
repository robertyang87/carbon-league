"""distill_train.py -- small-teacher distillation training harness (stage 2).

Design: reports/DESIGN_SMALL_TEACHER_20261117.md (cb6efbe7) section 4, signed
off (5fcdaab4). Student = cnet2_model.CNet2(**cfg), default B tier
(C48/NB4/TA2/D96/L2/NH4/TNH4/E24/G4), trained FROM SCRATCH to fit the current
teacher's own per-position outputs on distill_datagen shards:

  L = 1.0 * KL(softmax(t_logits) || softmax(s_logits))   (per valid slot)
    + 0.1 * MSE(s_logits, t_logits)
    + 1.0 * MSE(s_u, t_u)
    + 0.1 * MSE(s_value, t_value)

(weights are implementation-internal per sign-off #4; the registered numbers
are the section-3/section-4 gates, not these.)

Recipe (initial, design section 4): AdamW lr 1e-3, cosine decay, batch 1024
positions, <=40 epochs, validation = ~5% of GAMES held out whole (no
same-game leakage). Checkpoints every <=45 min of wall clock (model +
optimizer + scheduler + counters) with progress.json rewritten at every
checkpoint and epoch end, so progress is externally readable and any
interruption resumes from the newest checkpoint (--resume).

Symmetry augmentation (--sym-augment) is gated by a teacher self-consistency
check that runs first: the teacher's own outputs on sym_view(x, g) must match
the SYM_MOVE-permuted outputs on x (the production act_tta construction:
encoder sees the symmetric frame, Prefix feats stay in the original frame
with mf rows permuted for worker slots). If the check fails, augmentation
stays OFF and the measured discrepancy is recorded in progress.json.

Teacher forcing uses t_labels from the shards (the teacher decode's own
chosen labels); uf/mf/legal per slot are rebuilt with cnet2_feat.Prefix
advanced along those labels, exactly as Policy.act consumes them.
"""

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

CFG_B = dict(C=48, NB=4, D=96, NH=4, L=2, E=24, G=4, TA=2, TNH=4)
X_FIELDS = ("maps", "scal", "tok_pos", "tok_kind", "tok_mask", "tok_cargo",
            "hist_cell", "hist_cargo", "hist_move")
AUX_FIELDS = ("est", "en_min", "en_max", "en_pl", "thr", "dob", "own_base",
              "en_base", "own_tree", "en_tree", "toll", "own_adj")
LOSS_W = {"kl": 1.0, "logits_mse": 0.1, "u_mse": 1.0, "value_mse": 0.1}


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- data

def load_shards(data_dir):
    man = json.load(open(os.path.join(data_dir, "manifest.json")))
    positions = []
    for g in man["games"]:
        d = np.load(os.path.join(data_dir, g["shard"]), allow_pickle=False)
        n = int(d["t_logits"].shape[0])
        for i in range(n):
            positions.append({
                "x": {k: d["x_" + k][i] for k in X_FIELDS},
                "cash": float(d["x_cash"][i]), "slots": float(d["x_slots"][i]),
                "left": float(d["x_left"][i]),
                "aux": {k: d["aux_" + k][i] for k in AUX_FIELDS},
                "t_logits": d["t_logits"][i], "t_labels": d["t_labels"][i],
                "t_u": float(d["t_u"][i]), "t_value": float(d["t_value"][i]),
                "game": str(d["meta_game"][i])})
    return positions, md5_file(os.path.join(data_dir, "manifest.json"))


def synthetic_positions(n, seed=7, games=("synth0", "synth1", "synth2", "synth3")):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        x = {"maps": rng.integers(0, 256, (33, 225), np.uint8),
             "scal": rng.standard_normal(20).astype(np.float32),
             "tok_pos": rng.integers(0, 225, 24).astype(np.int64),
             "tok_kind": rng.integers(0, 6, 24).astype(np.int64),
             "tok_mask": (rng.random(24) < 0.8),
             "tok_cargo": (rng.random(24) * 100).astype(np.float32),
             "hist_cell": rng.integers(-1, 225, (24, 8)).astype(np.int64),
             "hist_cargo": (rng.random((24, 8)) * 50).astype(np.float32),
             "hist_move": rng.integers(0, 6, (24, 8)).astype(np.int64)}
        x["cash"] = float(rng.random() * 500)
        x["slots"] = float(rng.integers(0, 5))
        x["left"] = float(rng.integers(50, 300))
        aux = {k: (rng.random(225) * 10).astype(np.float64) for k in AUX_FIELDS}
        lab = np.where(x["tok_mask"][:12], rng.integers(0, 5, 12), -1).astype(np.int64)
        lab[10:12] = np.where(lab[10:12] >= 0, lab[10:12] % 3, -1)  # building slots: 3 kinds (aemb has 9 rows)
        out.append({"x": {k: x[k] for k in X_FIELDS}, "cash": x["cash"],
                    "slots": x["slots"], "left": x["left"], "aux": aux,
                    "t_logits": (rng.standard_normal((12, 5)) * 2).astype(np.float32),
                    "t_labels": lab,
                    "t_u": float(rng.random()), "t_value": float(rng.random() * 2 - 1),
                    "game": games[i % len(games)]})
    return out


def split_by_game(positions, val_every=20):
    games = sorted({p["game"] for p in positions})
    val_games = set(games[::val_every]) if len(games) > 1 else set()
    tr = [p for p in positions if p["game"] not in val_games]
    va = [p for p in positions if p["game"] in val_games]
    return tr, va, sorted(val_games)


# ---------------------------------------------------------------- batching

def build_batch(F, chunk, device, sym_gs=None):
    """chunk: list of position dicts. sym_gs: optional per-sample symmetry
    frame (0 = identity). Returns (torch batch dict, targets dict)."""
    import torch
    B = len(chunk)
    xs, ys = [], []
    for i, p in enumerate(chunk):
        g = 0 if sym_gs is None else int(sym_gs[i])
        xs.append(p)
        ys.append(g)
    maps = np.stack([(p["x"]["maps"] if ys[i] == 0 else _sym_x(p, ys[i])["maps"])
                     for i, p in enumerate(xs)])
    b = {"maps": torch.as_tensor(maps.astype(np.float32) / 255.0, device=device),
         "tok_kind": torch.as_tensor(np.stack([p["x"]["tok_kind"] for p in xs]), device=device),
         "tok_cargo": torch.as_tensor(np.stack([p["x"]["tok_cargo"] for p in xs]).astype(np.float32), device=device),
         "tok_pos": torch.as_tensor(np.stack([(p["x"]["tok_pos"] if ys[i] == 0 else _sym_x(p, ys[i])["tok_pos"])
                                              for i, p in enumerate(xs)]), device=device),
         "tok_mask": torch.as_tensor(np.stack([p["x"]["tok_mask"] for p in xs]), device=device),
         "hist_cell": torch.as_tensor(np.stack([(p["x"]["hist_cell"] if ys[i] == 0 else _sym_x(p, ys[i])["hist_cell"])
                                                for i, p in enumerate(xs)]), device=device),
         "hist_cargo": torch.as_tensor(np.stack([p["x"]["hist_cargo"] for p in xs]).astype(np.float32), device=device),
         "hist_move": torch.as_tensor(np.stack([(p["x"]["hist_move"] if ys[i] == 0 else _sym_x(p, ys[i])["hist_move"])
                                                for i, p in enumerate(xs)]), device=device),
         "scal": torch.as_tensor(np.stack([p["x"]["scal"] for p in xs]).astype(np.float32), device=device),
         "teacher": torch.full((B,), 2, dtype=torch.long, device=device)}
    # decode inputs from Prefix advanced along the (possibly permuted) labels
    uf = np.zeros((B, 12, F.UF), np.float32)
    mf = np.zeros((B, 12, 5, F.MF), np.float32)
    legal = np.zeros((B, 12, 5), bool)
    labels = np.stack([_perm_labels(p, ys[i]) for i, p in enumerate(xs)])
    for i, p in enumerate(xs):
        pf = F.Prefix(p["x_full"] if "x_full" in p else _x_with_scalars(p), p["aux"])
        for s in range(12):
            if p["x"]["tok_mask"][s]:
                u_, m_, l_ = pf.feats(s)
                if ys[i] and s < 10:
                    m_ = m_[F.SYM_MOVE[ys[i]]]
                uf[i, s], mf[i, s], legal[i, s] = u_, m_, l_
                orig = int(p["t_labels"][s])
                if orig >= 0:
                    # Prefix lives in the ORIGINAL frame (act_tta construction):
                    # it advances with the original-frame label; the permuted
                    # label is only the student's teacher-forcing target.
                    pf.advance(s, orig)
    b["uf"] = torch.as_tensor(uf, device=device)
    b["mf"] = torch.as_tensor(mf, device=device)
    b["legal"] = torch.as_tensor(legal, device=device)
    b["y"] = torch.as_tensor(np.maximum(labels, 0), device=device)
    t_logits = np.stack([_perm_logits(p, ys[i]) for i, p in enumerate(xs)])
    tgt = {"logits": torch.as_tensor(t_logits, device=device),
           "u": torch.as_tensor(np.array([p["t_u"] for p in xs], np.float32), device=device),
           "value": torch.as_tensor(np.array([p["t_value"] for p in xs], np.float32), device=device),
           "mask": torch.as_tensor(np.stack([p["x"]["tok_mask"][:12] for p in xs]), device=device)}
    return b, tgt


def _x_with_scalars(p):
    x = dict(p["x"])
    x["cash"] = p["cash"]; x["slots"] = p["slots"]; x["left"] = p["left"]
    return x


_SYM_CACHE = {}


def _sym_x(p, g):
    key = (id(p), g)
    if key not in _SYM_CACHE:
        import cnet2_np as NP
        _SYM_CACHE[key] = NP.sym_view(_x_with_scalars(p), g)
    return _SYM_CACHE[key]


def _perm_labels(p, g):
    lab = np.array(p["t_labels"], copy=True)
    if g:
        import cnet2_feat as F
        mm = F.SYM_MOVE[g]
        for s in range(10):
            if lab[s] >= 0:
                lab[s] = int(mm[int(lab[s])])
    return lab


def _perm_logits(p, g):
    lg = np.array(p["t_logits"], copy=True)
    if g:
        import cnet2_feat as F
        mm = F.SYM_MOVE[g]
        for s in range(10):
            lg[s] = lg[s][mm]
    return lg


# ---------------------------------------------------------------- symmetry gate

def symmetry_self_check(positions, weights, teacher_style=2, n_check=16):
    """Teacher self-consistency under symmetry frames (production act_tta
    construction). Returns dict with max diffs; augmentation may be enabled
    only if max|dlogits| (after SYM_MOVE permutation) is small."""
    import cnet2_feat as F
    import cnet2_np as NP
    from cnet2_search import CachePolicy
    pol = CachePolicy(weights)
    res = {"n_check": 0, "max_dlogits": 0.0, "max_du": 0.0, "label_agree": 1.0}
    agree, total = 0, 0
    for p in positions[:n_check]:
        x = _x_with_scalars(p)
        lab0, lg0, _v0, u0 = pol.act(x, p["aux"], teacher=teacher_style, want_u=True)
        for g in (1, 2, 3):
            xv = NP.sym_view(x, g)
            labg, lgg, _vg, ug = pol.act(xv, p["aux"], teacher=teacher_style, want_u=True)
            mm = F.SYM_MOVE[g]
            for s in range(12):
                if not x["tok_mask"][s]:
                    continue
                want = lg0[s][mm] if s < 10 else lg0[s]
                res["max_dlogits"] = max(res["max_dlogits"], float(np.abs(lgg[s] - want).max()))
                total += 1
                want_lab = int(mm[int(lab0[s])]) if (s < 10 and lab0[s] >= 0) else int(lab0[s])
                agree += int(int(labg[s]) == want_lab)
            res["max_du"] = max(res["max_du"], abs(float(ug) - float(u0)))
        res["n_check"] += 1
    res["label_agree"] = (agree / total) if total else 1.0
    return res


# ---------------------------------------------------------------- losses

def compute_loss(Fn, model, b, tgt):
    logits, value, _aux, u, _wdl, _mk = model.forward_heads(b)
    mask = tgt["mask"]
    t_prob = Fn.softmax(tgt["logits"], dim=-1)
    s_logp = Fn.log_softmax(logits, dim=-1)
    kl_slot = Fn.kl_div(s_logp, t_prob, reduction="none").sum(-1)   # [B,12]
    kl = (kl_slot * mask).sum() / mask.sum().clamp(min=1)
    lmse = (((logits - tgt["logits"]) ** 2) * mask[..., None]).sum() / (mask.sum().clamp(min=1) * 5)
    umse = ((u - tgt["u"]) ** 2).mean()
    vmse = ((value - tgt["value"]) ** 2).mean()
    total = LOSS_W["kl"] * kl + LOSS_W["logits_mse"] * lmse \
        + LOSS_W["u_mse"] * umse + LOSS_W["value_mse"] * vmse
    return total, {"kl": float(kl), "logits_mse": float(lmse),
                   "u_mse": float(umse), "value_mse": float(vmse),
                   "total": float(total)}


# ---------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description="small-teacher distillation training")
    ap.add_argument("--data-dir", default="", help="distill_datagen output dir (manifest.json + shards)")
    ap.add_argument("--code-dir", required=True, help="pinned code stack dir (imports cnet2_model/feat/np/search)")
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--cfg-json", default="", help="student cfg JSON (default B tier)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=20261117)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ckpt-minutes", type=float, default=45.0)
    ap.add_argument("--max-steps", type=int, default=0, help="0 = full epochs")
    ap.add_argument("--resume", default="", help="checkpoint .pt to resume from")
    ap.add_argument("--sym-augment", action="store_true")
    ap.add_argument("--teacher-weights", default="", help="needed only for the symmetry self-check")
    ap.add_argument("--smoke-synthetic", type=int, default=0,
                    help="N>0: train on N synthetic positions (construction smoke)")
    a = ap.parse_args(argv)

    sys.path.insert(0, a.code_dir)
    import torch
    import torch.nn.functional as Fn
    import cnet2_feat as F
    from cnet2_model import CNet2

    cfg = json.loads(a.cfg_json) if a.cfg_json else dict(CFG_B)
    os.makedirs(a.ckpt_dir, exist_ok=True)
    progress_path = os.path.join(a.ckpt_dir, "progress.json")
    state = {"stage": "init", "cfg": cfg, "epoch": 0, "step": 0,
             "positions_seen": 0, "last_val": None, "last_ckpt": None,
             "wall_s": 0.0, "sym_augment": False, "sym_check": None,
             "data_manifest_md5": None}
    t_start = time.time()

    if a.smoke_synthetic:
        positions = synthetic_positions(a.smoke_synthetic)
        state["data_manifest_md5"] = "synthetic-smoke"
    else:
        positions, state["data_manifest_md5"] = load_shards(a.data_dir)
    train_pos, val_pos, val_games = split_by_game(positions)
    state["n_train"] = len(train_pos)
    state["n_val"] = len(val_pos)
    state["val_games"] = val_games
    print(f"train {len(train_pos)} / val {len(val_pos)} positions")

    if a.sym_augment:
        if not a.teacher_weights:
            print("sym-augment requested but --teacher-weights missing: augmentation OFF")
        else:
            w = dict(np.load(a.teacher_weights))
            chk = symmetry_self_check(train_pos, w)
            state["sym_check"] = chk
            print("symmetry self-check:", chk)
            if chk["max_dlogits"] <= 1e-3 and chk["label_agree"] >= 0.999:
                state["sym_augment"] = True
            else:
                print("symmetry self-check FAILED: augmentation stays OFF")

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    model = CNet2(**cfg).to(a.device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    steps_per_epoch = max(1, (len(train_pos) + a.batch - 1) // a.batch)
    total_steps = steps_per_epoch * a.epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
    start_epoch, start_step = 0, 0
    if a.resume:
        ck = torch.load(a.resume, map_location=a.device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        start_epoch, start_step = ck["epoch"], ck["step"]
        state["step"] = ck["step"]
        state["epoch"] = ck["epoch"]
        state.update({k: ck["progress"][k] for k in
                      ("positions_seen", "last_val") if k in ck.get("progress", {})})
        print(f"resumed from {a.resume}: epoch {start_epoch} step {start_step}")

    def save_ckpt(tag):
        path = os.path.join(a.ckpt_dir, f"ckpt_{tag}.pt")
        state["wall_s"] = round(time.time() - t_start, 1)
        state["last_ckpt"] = path
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                    "scheduler": sched.state_dict(), "epoch": state["epoch"],
                    "step": state["step"], "cfg": cfg, "progress": dict(state)}, path)
        with open(progress_path, "w") as fh:
            json.dump(state, fh, indent=1)
        return path

    def evaluate():
        model.eval()
        tot, cnt = None, 0
        with torch.no_grad():
            for lo in range(0, len(val_pos), a.batch):
                b, tgt = build_batch(F, val_pos[lo:lo + a.batch], a.device)
                _l, parts = compute_loss(Fn, model, b, tgt)
                tot = parts if tot is None else {k: tot[k] + v for k, v in parts.items()}
                cnt += 1
        model.train()
        return {k: v / max(cnt, 1) for k, v in tot.items()} if tot else None

    state["stage"] = "train"
    last_ckpt_t = time.time()
    stop = False
    for ep in range(start_epoch, a.epochs):
        if stop:
            break
        state["epoch"] = ep
        order = rng.permutation(len(train_pos))
        for bi in range(0, len(order), a.batch):
            chunk = [train_pos[i] for i in order[bi:bi + a.batch]]
            sym_gs = None
            if state["sym_augment"]:
                sym_gs = np.where(rng.random(len(chunk)) < 0.5, 0,
                                  rng.integers(1, 4, len(chunk)))
            b, tgt = build_batch(F, chunk, a.device, sym_gs=sym_gs)
            loss, _parts = compute_loss(Fn, model, b, tgt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            state["step"] += 1
            state["positions_seen"] += len(chunk)
            if time.time() - last_ckpt_t >= a.ckpt_minutes * 60:
                save_ckpt(f"ep{ep}_step{state['step']}")
                last_ckpt_t = time.time()
            if a.max_steps and state["step"] >= (start_step + a.max_steps):
                stop = True
                break
        state["last_val"] = evaluate()
        save_ckpt(f"ep{ep}_end")
        last_ckpt_t = time.time()
        print(f"epoch {ep} done: step {state['step']} val {state['last_val']}")
    state["stage"] = "done"
    save_ckpt("final")
    print("training done:", state["step"], "steps")
    return 0


if __name__ == "__main__":
    sys.exit(main())

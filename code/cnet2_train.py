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


SEQGRU = os.environ.get("BC_SEQGRU") == "1"   # one packed GRU call over the 12 slots (cnet2_model.forward_seq)


def _fwd(net):
    return net.forward_seq if SEQGRU else net


def losses(net, b):
    if AMP == "bf16" and (DEV.type == "mps" or (DEV.type == "cuda" and os.environ.get("BC_CUDA_BF16") == "1")):
        with torch.autocast(DEV.type, dtype=torch.bfloat16):   # dg2 was trained like this on the Mac (mps bf16)
            lg, v, aux = _fwd(net)(b)
        lg, v, aux = lg.float(), v.float(), aux.float()
    else:
        lg, v, aux = _fwd(net)(b)
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


LIGHT = ["y", "tok_mask", "mf", "uf", "legal", "teacher", "tok_kind"]


def train_cached(net, D, tr, va, epochs, bs, opt, sched, spe, out, npar, data):
    """Frozen trunk: run the trunk ONCE per row (no augmentation), keep tok[:, :OWN] and the trunk summary (pre)
    on the device in fp16, then train only the decoder + style rows from the cache (5-10x faster than full passes)."""
    allidx = np.sort(np.concatenate([tr, va])); cpos = np.full(len(D["y"]), -1, np.int64); cpos[allidx] = np.arange(len(allidx))
    t0 = time.time(); toks = []; pres = []
    net.eval()
    with torch.no_grad():
        for a0 in range(0, len(allidx), 2048):
            ii = allidx[a0:a0 + 2048]
            b = to_dev({k: v for k, v in fetch(D, ii).items()}, False); b["_want_pre"] = True
            tok, pre, _ = net.encode(b)
            toks.append(tok[:, :F.OWN].half()); pres.append(pre.half())
    TOK = torch.cat(toks); PRE = torch.cat(pres); del toks, pres
    net.train()
    print(json.dumps({"trunk_cache_rows": int(len(allidx)), "GB": round((TOK.numel() + PRE.numel()) * 2 / 1e9, 2), "s": round(time.time() - t0)}), flush=True)
    extra = [x for x in ("w", "uw") if x in D]
    # decoder inputs also live on the device (sliced there per batch): the CPU-side fancy indexing was the bottleneck
    GD = {}
    for k in LIGHT + extra:
        parts = []
        for a0 in range(0, len(allidx), 65536):
            v = np.asarray(D[k][allidx[a0:a0 + 65536]])
            t_ = torch.from_numpy(np.ascontiguousarray(v))
            if k in ("y", "teacher", "tok_kind"):
                t_ = t_.to(torch.int16) if k != "teacher" else t_.to(torch.int8)
            elif t_.dtype == torch.float32:
                t_ = t_.half()
            parts.append(t_.to(DEV))
        GD[k] = torch.cat(parts)
    print(json.dumps({"light_on_device_GB": round(sum(t.numel() * t.element_size() for t in GD.values()) / 1e9, 2)}), flush=True)

    def light(ii):
        ii = np.sort(ii)
        cp = torch.from_numpy(cpos[ii]).to(DEV)
        b = {}
        for k, t_ in GD.items():
            v = t_[cp]
            if k in ("y", "teacher", "tok_kind"):
                v = v.long()
            elif v.dtype == torch.float16:
                v = v.float()
            b[k] = v
        b["_cp"] = cp
        return b, TOK[cp].float(), PRE[cp].float()

    # BC_SELF_KL=c: soft self-anchor. Before the first step, the logits of the INITIAL net are stored for every row; the
    # loss gets  c * mean over car tokens of KL(initial || current).  Zero gradient at the start, and it pulls the cars
    # back as soon as the shared decoder moves. (A hard-label anchor on the net's own moves does not do this: the policy
    # is soft, so cross-entropy to its own argmax sharpens it and moves the decoder -- measured: cars-only hard-label
    # training alone takes car agreement from 99.5% to 96.4%.)
    self_kl = float(os.environ.get("BC_SELF_KL", "0")); REF = None

    def batch_loss(b, tok, pre):
        lg = net.decode(b, tok, net.ctx_from(pre, b["teacher"]))
        y = b["y"]; valid = y >= 0
        ce = Fn.cross_entropy(lg.reshape(-1, 5), y.clamp(min=0).reshape(-1), reduction="none").reshape(y.shape)
        tw = b["w"] if "w" in b else torch.ones(len(y), device=DEV)
        wv = valid.float() * tw[:, None]
        if "uw" in b:
            wv = wv * b["uw"]
        loss = (ce * wv).sum() / wv.sum().clamp(min=1)
        if REF is not None:
            car = (valid & (b["tok_kind"][:, :F.OWN] == 0))[:, :REF.shape[1]].float()
            lp = Fn.log_softmax(lg[:, :REF.shape[1]].float(), -1); lr_ = Fn.log_softmax(REF[b["_cp"]].float(), -1)
            pr_ = lr_.exp()
            kl = torch.where(pr_ > 0, pr_ * (lr_ - lp), torch.zeros_like(pr_)).sum(-1)     # illegal moves: p = 0, no term
            loss = loss + self_kl * (kl * car).sum() / car.sum().clamp(min=1)
        return loss, lg

    if self_kl > 0:
        net.eval(); refs = []
        with torch.no_grad():
            for a0 in range(0, len(allidx), 4096):
                b_, tok_, pre_ = light(allidx[a0:a0 + 4096])
                refs.append(net.decode(b_, tok_, net.ctx_from(pre_, b_["teacher"]))[:, :F.OWN].half())
        REF = torch.cat(refs); del refs; net.train()
        print(json.dumps({"self_kl": self_kl, "ref_logits_GB": round(REF.numel() * 2 / 1e9, 2)}), flush=True)

    def evaluate_c():
        net.eval(); ok = np.zeros(3); n = np.zeros(3)
        with torch.no_grad():
            for a0 in range(0, len(va), 4096):
                b, tok, pre = light(va[a0:a0 + 4096]); _, lg = batch_loss(b, tok, pre)
                pred = lg.argmax(-1); y = b["y"]; kind = b["tok_kind"][:, :F.OWN]
                for k in range(3):
                    m = (y >= 0) & (kind == k); ok[k] += ((pred == y) & m).sum().item(); n[k] += m.sum().item()
        net.train()
        r = {nm: round(ok[k] / max(n[k], 1), 4) for k, nm in enumerate(("collector", "planter", "base"))}
        r["all"] = round(ok.sum() / max(n.sum(), 1), 4); return r

    best = -1; log = open(out.replace(".npz", "_log.jsonl"), "a")
    for ep in range(epochs):
        t1 = time.time(); acc = 0.0; nb = 0
        perm = np.random.permutation(tr)
        for i in range(spe):
            b, tok, pre = light(perm[i * bs:(i + 1) * bs])
            lpi, _ = batch_loss(b, tok, pre)
            opt.zero_grad(set_to_none=True); lpi.backward()
            torch.nn.utils.clip_grad_norm_([p_ for p_ in net.parameters() if p_.requires_grad], float(os.environ.get("BC_CLIP", "2.0")))
            opt.step(); sched.step()
            if i % 50 == 0:
                acc += lpi.item(); nb += 1
        r = evaluate_c()
        rec = {"ep": ep, "val": r, "train_pi": round(acc / max(nb, 1), 4), "s": round(time.time() - t1), "cached": True}
        print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
        if r["all"] > best:
            best = r["all"]; export(net, out, {"cfg": net.cfg, "epoch": ep, "val": r, "params": npar, "data": data, "cached_trunk": True})
    print("best", best)


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
    cp_ = os.environ.get("BC_COPY_TEACH")     # "5:2": style row 2 starts as a copy of row 5 (AWR: behaviour = slot 5)
    if cp_:
        a_, b_ = (int(x) for x in cp_.split(":"))
        with torch.no_grad():
            net.teach.weight[b_] = net.teach.weight[a_]
        print(json.dumps({"copy_teacher_row": [a_, b_]}), flush=True)
    tp_ = os.environ.get("BC_TRAIN_PREFIX")   # e.g. "teach,din,aemb,gru,whead,bhead,U,V,h0": decoder only, trunk frozen
    if tp_:
        pref = tuple(x.strip() + "." for x in tp_.split(","))
        for n_, p_ in net.named_parameters():
            p_.requires_grad_(n_.startswith(pref))
        tr_rows = os.environ.get("BC_TEACH_ROWS")  # with teach in the prefix list: only these style rows move
        if tr_rows and net.teach.weight.requires_grad:
            keep2 = torch.zeros_like(net.teach.weight); keep2[[int(x) for x in tr_rows.split(",")]] = 1.0
            net.teach.weight.register_hook(lambda g: g * keep2)
        print(json.dumps({"train_prefix": tp_, "trainable": sum(p_.numel() for p_ in net.parameters() if p_.requires_grad)}), flush=True)
    only = os.environ.get("BC_ONLY_TEACH")    # frozen trunk: train ONLY these style rows (e.g. "2"); every other
    if only:                                  # parameter and every other slot stay bit-identical (no weight decay)
        rows_ = [int(x) for x in only.split(",")]
        for n_, p_ in net.named_parameters():
            p_.requires_grad_(n_ == "teach.weight")
        keep_ = torch.zeros_like(net.teach.weight); keep_[rows_] = 1.0
        net.teach.weight.register_hook(lambda g: g * keep_)
        tsel = np.asarray(D["teacher"])
        tr = tr[np.isin(tsel[tr], rows_)]; va = va[np.isin(tsel[va], rows_)]   # those slots' rows only
        print(json.dumps({"only_teach": rows_, "train_rows": len(tr), "val_rows": len(va)}), flush=True)
    npar = sum(p.numel() for p in net.parameters())
    bs = int(os.environ.get("BC_BS", "512"))
    lr = float(os.environ.get("BC_LR", "1e-3"))
    maxsteps = int(os.environ.get("BC_MAXSTEPS", "0"))
    spe = len(tr) // bs if not maxsteps else min(maxsteps, len(tr) // bs)
    print(json.dumps({"train": len(tr), "val": len(va), "params": npar, "cfg": net.cfg, "bs": bs,
                      "steps_per_epoch": spe}), flush=True)
    opt = torch.optim.AdamW([p_ for p_ in net.parameters() if p_.requires_grad], lr=lr,
                            weight_decay=0.0 if (only or os.environ.get("BC_TEACH_ROWS")) else 1e-4)
    # BC_COMPILE=1 (CUDA only): train through torch.compile(fullgraph, static shapes; every training batch is full);
    # evaluation keeps calling the eager `net` (same parameters). Training-equivalent on the GB10: per-layer
    # ||dg||/||g|| 7.7e-4 vs eager in strict fp32 (2026-11-09), far below batch-to-batch gradient variance.
    if DEV.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = os.environ.get("BC_TF32", "1") == "1"
        torch.backends.cudnn.allow_tf32 = os.environ.get("BC_TF32", "1") == "1"
    tnet = net
    if (only or tp_) and os.environ.get("BC_COMPILE") == "1":
        raise SystemExit("BC_ONLY_TEACH with BC_COMPILE=1 is untested (gradient hook under compile); omit BC_COMPILE")
    if only and spe == 0:
        raise SystemExit(f"BC_ONLY_TEACH={only}: zero training rows after filtering")
    if os.environ.get("BC_COMPILE") == "1" and SEQGRU:
        raise SystemExit("BC_COMPILE=1 with BC_SEQGRU=1 would silently train eager (forward_seq is not compiled); pick one")
    if os.environ.get("BC_COMPILE") == "1" and DEV.type == "cuda":
        tnet = torch.compile(net, fullgraph=True, dynamic=False)
        print(json.dumps({"compile": "fullgraph,static"}), flush=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * spe, pct_start=float(os.environ.get("BC_PCT", "0.05")))
    if os.environ.get("BC_CACHE_TRUNK") == "1":
        if not tp_:
            raise SystemExit("BC_CACHE_TRUNK needs BC_TRAIN_PREFIX without trunk layers (the cache is the frozen trunk)")
        return train_cached(net, D, tr, va, epochs, bs, opt, sched, spe, out, npar, data)
    best = -1
    log = open(out.replace(".npz", "_log.jsonl"), "a")
    for ep in range(epochs):
        t0 = time.time(); acc = np.zeros(3); n = 0
        for i, b in enumerate(loader(D, tr, bs, aug=True)):
            if i >= spe:
                break
            lpi, lv, la, _ = losses(tnet, b)
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
    if only and init:          # frozen trunk: prove every parameter except the trained style rows is bit-identical
        w0 = np.load(init); w1 = np.load(out); diff = []
        for k in w0.files:
            if k == "meta":
                continue
            a0, a1 = w0[k], w1[k]
            if k == "teach.weight":
                m_ = np.ones(len(a0), bool); m_[[int(x) for x in only.split(",")]] = False
                re_ = [int(x) for x in os.environ.get("BC_REINIT_TEACH", "").split(",") if x]
                m_[re_] = False
                a0, a1 = a0[m_], a1[m_]
            if not np.array_equal(a0, a1):
                diff.append(k)
        print(json.dumps({"frozen_check": "PASS" if not diff else "FAIL", "changed": diff}), flush=True)


if __name__ == "__main__":
    cfg = json.loads(sys.argv[4]) if len(sys.argv) > 4 else {}
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), cfg)

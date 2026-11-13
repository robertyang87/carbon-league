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
from cnet2_build import FIELDS, load_generation  # noqa: E402
from cnet2_model import CNet2, migrate_state_dict, migrate_optimizer_state, NEW_HEAD_KEYS  # noqa: E402

DEV = torch.device(os.environ.get("BC_DEVICE", "mps"))
TW = {0: 1.5, 1: 1.5, 2: 1.5}       # Master, 1C-Zero v9, v8-preview weighted up
LOAD = [k for k in FIELDS if k not in ("gid", "step")]
AUXK = ("aux_die", "aux_die_m", "aux_dep", "aux_dep_m", "aux_dw", "aux_dw_m")


def _check_finite_weights(D, dirn):
    """F06 hard gate: loss weights must be finite BEFORE any training. A single inf/NaN in w/uw
    silently poisons every weighted batch (batch_loss = NaN); refuse the dataset and name it.
    Also gates the EFFECTIVE product w*uw (owner point 2): w and uw can each be finite float32
    while their product -- the multiplier the loss actually applies -- overflows to inf.
    Returns a summary dict (per-field maxima + product distribution) for the load log."""
    summ = {}
    for k in ("w", "uw"):
        if k not in D:
            continue
        a = np.asarray(D[k])
        if not np.issubdtype(a.dtype, np.floating):
            continue
        flat = a.reshape(-1)
        bad = 0; first = -1; mx = 0.0
        for a0 in range(0, flat.size, 1 << 22):
            blkf = flat[a0:a0 + (1 << 22)].astype(np.float32)
            fin = np.isfinite(blkf)
            if not fin.all():
                bad += int((~fin).sum())
                if first < 0:
                    first = a0 + int(np.argmax(~fin))
            if fin.any():
                mx = max(mx, float(np.abs(blkf[fin]).max()))
        if bad:
            raise SystemExit(f"F06: non-finite loss weights in dataset {dirn!r} field {k!r}: "
                             f"{bad} NaN/Inf entries (first at flat index {first}); refusing to train")
        summ[k + "_max"] = mx
    if "w" in D and "uw" in D:
        w = np.asarray(D["w"]); uw = np.asarray(D["uw"])
        if np.issubdtype(w.dtype, np.floating) and np.issubdtype(uw.dtype, np.floating):
            if w.ndim != 1 or uw.ndim != 2 or uw.shape[0] != w.shape[0]:
                raise SystemExit(f"F06: loss-weight shape mismatch in dataset {dirn!r}: "
                                 f"w{w.shape} vs uw{uw.shape}; refusing to train")
            bad = 0; first = -1; pmax = 0.0; psum = 0.0; pcnt = 0
            for a0 in range(0, w.shape[0], 1 << 15):
                prod = w[a0:a0 + (1 << 15)].astype(np.float32)[:, None] * uw[a0:a0 + (1 << 15)].astype(np.float32)
                fin = np.isfinite(prod)
                if not fin.all():
                    bad += int((~fin).sum())
                    if first < 0:
                        first = a0 * uw.shape[1] + int(np.argmax(~fin))
                if fin.any():
                    pv = np.abs(prod[fin])
                    pmax = max(pmax, float(pv.max())); psum += float(pv.sum(dtype=np.float64)); pcnt += int(fin.sum())
            if bad:
                raise SystemExit(f"F06: non-finite effective loss weights (w*uw) in dataset {dirn!r} field 'w*uw': "
                                 f"{bad} NaN/Inf entries (first at flat index {first}); refusing to train")
            summ["prod_max"] = pmax
            summ["prod_mean"] = psum / max(pcnt, 1)
            summ["prod_n"] = pcnt
    return summ


def old_param_names_checked(old_sd, model, source="checkpoint"):
    """Derive the OLD model's flat parameter order for optimizer migration
    (F01, owner point 3). An old optimizer state_dict keys its moments by
    INTEGER id -- the id is only meaningful against the old model's parameter
    order, so that order must be proven, never guessed:
      * source A is the old checkpoint's OWN key order (torch state_dicts and
        this repo's npz exports both preserve named_parameters order);
      * source B is the order reconstructed from the new model (its
        named_parameters order with the Stage 0 head keys removed -- the
        heads were inserted as one block, nothing else moved).
    The two must agree EXACTLY, the checkpoint must not contain tensors the
    model does not know, the only parameters absent from it may be the new
    heads, and every old tensor's shape must equal the new parameter's shape.
    Any disagreement -> RuntimeError: with same-shape parameters everywhere,
    a wrong order would silently land moments on the wrong weights."""
    new_params = dict(model.named_parameters())
    recon = [n for n in new_params if n not in NEW_HEAD_KEYS]
    sd_keys = list(old_sd.keys())
    unknown = [k for k in sd_keys if k not in new_params]
    if unknown:
        raise RuntimeError(f"{source}: checkpoint holds tensors the model does not know: {unknown[:8]}")
    absent = [n for n in new_params if n not in old_sd]
    if any(n not in NEW_HEAD_KEYS for n in absent):
        raise RuntimeError(f"{source}: checkpoint is missing non-head parameters: "
                           f"{[n for n in absent if n not in NEW_HEAD_KEYS][:8]}")
    order_a = [k for k in sd_keys if k in set(recon)]
    if order_a != recon:
        raise RuntimeError(f"{source}: checkpoint key order disagrees with the reconstructed "
                           f"old parameter order; refusing to guess the optimizer id mapping")
    for n in recon:
        if tuple(old_sd[n].shape) != tuple(new_params[n].shape):
            raise RuntimeError(f"{source}: shape mismatch for {n!r}: checkpoint {tuple(old_sd[n].shape)} "
                               f"vs model {tuple(new_params[n].shape)}")
    return recon


def migrate_optimizer_checked(optimizer, old_opt_sd, old_sd, model, source="checkpoint"):
    """The only sanctioned entry for resuming an OLD optimizer on the new
    (post-heads) model: derives the id->name mapping via
    old_param_names_checked, migrates by name through
    cnet2_model.migrate_optimizer_state, then post-verifies that every
    carried moment's shape equals its parameter's shape. `optimizer` must be
    built over ALL of the model's parameters in named order (the flat-order
    contract); a frozen-prefix optimizer must instead document its own
    provenance. Returns the migration report plus provenance and the set of
    preserved step counts."""
    old_names = old_param_names_checked(old_sd, model, source)
    new_names = [n for n, _ in model.named_parameters()]
    opt_params = [p for g in optimizer.param_groups for p in g["params"]]
    if opt_params != [p for _, p in model.named_parameters()]:
        raise RuntimeError("migrate_optimizer_checked: optimizer must cover all model "
                           "parameters in named order")
    rep = migrate_optimizer_state(optimizer, old_opt_sd, old_names, new_names)
    params = dict(model.named_parameters())
    steps = set()
    for n in rep["carried"]:
        st = optimizer.state[params[n]]
        for mk in ("exp_avg", "exp_avg_sq"):
            if mk in st and tuple(st[mk].shape) != tuple(params[n].shape):
                raise RuntimeError(f"{source}: migrated {mk} for {n!r} has shape "
                                   f"{tuple(st[mk].shape)}, parameter is {tuple(params[n].shape)}")
        if "step" in st:
            steps.add(float(st["step"]))
    rep["provenance"] = "checkpoint key order == reconstructed new-minus-heads order; shapes verified"
    rep["steps_preserved"] = sorted(steps)
    return rep


def _sha256_path(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def _validate_derived_layer(D, dirn, meta):
    """Derived-layer consumer gate (owner pt.4, final acceptance): w/uw in
    the dataset root are post-processing products (set_uw), NOT generation
    fields. Before any of them enters a loss, verify identity against the
    sidecar (derived.json, format "cnet2-derived/1", mirrored into the
    flat meta.json under "derived_layer"): for every weight actually
    consumed -- its field must not be a generation-manifest field, its
    base_generation must equal the generation current.json points at,
    its rows must equal that generation's total, and its dtype/shape/
    sha256 must match the file on disk. A sidecar/meta mirror mismatch
    also refuses. A dataset with a generation pointer but UNDECLARED
    non-generation weights refuses outright (the F05 stale-weight hazard:
    bytes left behind by another build must never be consumed silently).
    Only a pre-generation legacy dataset (no pointer, no sidecar) may
    consume w/uw unverified, and then only with a loud warning.
    Returns a summary dict recorded into meta["derived_layer_check"]."""
    used = [k for k in ("w", "uw") if k in D]
    if not used:
        return None
    side = None
    sp = os.path.join(dirn, "derived.json")
    if os.path.exists(sp):
        try:
            side = json.load(open(sp))
        except Exception as e:
            raise SystemExit(f"derived layer: unreadable sidecar {sp}: {e}; refusing to train")
    mirror = meta.get("derived_layer") if isinstance(meta, dict) else None
    if side is not None and mirror is not None and side != mirror:
        raise SystemExit(f"derived layer: derived.json and meta.json 'derived_layer' mirror "
                         f"disagree in dataset {dirn!r}; refusing to train")
    decl = side if side is not None else mirror
    gen_id = None; gen_fields = set(); gen_total = None
    ptr = os.path.join(dirn, "current.json")
    if os.path.exists(ptr):
        ptrj = json.load(open(ptr))
        gen_id = ptrj["generation"]
        man = json.load(open(os.path.join(dirn, ptrj["dir"], "manifest.json")))
        if man.get("generation") != gen_id:
            raise SystemExit(f"derived layer: generation pointer/manifest mismatch in "
                             f"dataset {dirn!r}; refusing to train")
        gen_fields = set(man.get("fields", {}))
        gen_total = int(man["total"])
    if decl is None:
        if gen_id is None:
            print(json.dumps({"derived_layer_check": "WARNING: derived identity unverified "
                              "(legacy dataset: no generation pointer, no derived.json); "
                              "w/uw consumed on file presence only", "data": dirn}), flush=True)
            return {"mode": "legacy-unverified", "used": used}
        bad = [k for k in used if k not in gen_fields]
        if bad:
            raise SystemExit(f"derived layer: weights {bad} in dataset {dirn!r} are neither "
                             f"fields of generation {gen_id} nor declared in a derived.json "
                             f"sidecar (stale or foreign weights); refusing to train")
        return {"mode": "generation-owned", "generation": gen_id, "used": used}
    if gen_id is None:
        raise SystemExit(f"derived layer: dataset {dirn!r} declares a derived.json sidecar "
                         f"but has no generation pointer; identity cannot be verified; "
                         f"refusing to train")
    entries = decl.get("files", {}) if isinstance(decl, dict) else {}
    checked = []
    for k in used:
        if k in gen_fields:
            if (k + ".npy") in entries:
                raise SystemExit(f"derived layer: {k}.npy in dataset {dirn!r} is BOTH a field "
                                 f"of generation {gen_id} and a declared derived file; "
                                 f"refusing to train")
            continue
        name = k + ".npy"
        e = entries.get(name)
        if e is None:
            raise SystemExit(f"derived layer: {name} in dataset {dirn!r} is consumed but not "
                             f"declared in the derived.json sidecar; refusing to train")
        arr = np.asarray(D[k])
        probs = []
        if e.get("field") != k:
            probs.append(f"sidecar field {e.get('field')!r} != {k!r}")
        if e.get("base_generation") != gen_id:
            probs.append(f"base_generation {e.get('base_generation')!r} != current {gen_id!r} "
                         f"(derived rows belong to another build; regenerate them)")
        if int(e.get("rows", -1)) != gen_total:
            probs.append(f"sidecar rows {e.get('rows')} != generation total {gen_total}")
        if int(arr.shape[0]) != gen_total:
            probs.append(f"file rows {int(arr.shape[0])} != generation total {gen_total}")
        if e.get("dtype") != str(arr.dtype):
            probs.append(f"dtype {e.get('dtype')!r} != file {str(arr.dtype)!r}")
        if [int(x) for x in arr.shape] != [int(x) for x in e.get("shape", [])]:
            probs.append(f"shape {e.get('shape')} != file {list(arr.shape)}")
        if _sha256_path(os.path.join(dirn, name)) != e.get("sha256"):
            probs.append("sha256 mismatch (file bytes changed after registration)")
        if probs:
            raise SystemExit(f"derived layer: {name} in dataset {dirn!r} failed identity "
                             f"checks: " + "; ".join(probs) + "; refusing to train")
        checked.append(name)
    return {"mode": "verified", "generation": gen_id, "files": checked}


def _resolve_load_source(dirn):
    """Residual 3 (F05 torn flat mirror): decide WHERE dataset fields are
    read from. If dirn carries a generation pointer (current.json), the
    fields come from the certified generation directory it points at --
    every manifest field hash-verified via cnet2_build.load_generation --
    and the flat mirror in dirn itself is NEVER read for fields: a
    publish that died mid-refresh leaves new arrays next to an old meta
    in the root, and training on that mix (maps of the new build, labels
    of the old) must be impossible, not silent. A directory that IS a
    generation (manifest.json, no pointer) is verified the same way and
    read directly. Only a pre-generation legacy dataset (neither file)
    falls back to the flat view, and load() then cross-checks field row
    counts (and the builder's flat_total stamp when present) so a torn
    legacy mirror is refused by name instead of read.
    Returns (source_dir, manifest_or_None)."""
    if os.path.exists(os.path.join(dirn, "current.json")):
        # POLICY CHANGE (owner review of 73ebd4af, P1-3): a pointer in
        # force is a certificate, not a hint. An earlier deliberate
        # design fell back to the flat view when the pointed generation
        # was incomplete (to accommodate hand-made manifest fixtures);
        # that fallback is REMOVED. Row-count agreement in the flat
        # mirror cannot substitute for the generation manifest's hash
        # verification: with a pointer present, a missing generation
        # file (FileNotFoundError) or a hash mismatch (ValueError) is a
        # hard refusal, and no flat bytes are read for fields. Fixtures
        # shaped as "pointer present but generation incomplete" must be
        # fixed to a complete sealed generation (or drop the pointer
        # and be legacy), not accommodated here.
        gen_dir, man = load_generation(dirn, verify=True)
        return gen_dir, man
    man_p = os.path.join(dirn, "manifest.json")
    if os.path.exists(man_p):
        man = json.load(open(man_p))
        for _name, info in man.get("fields", {}).items():
            p = os.path.join(dirn, info["file"])
            if not os.path.exists(p) or _sha256_path(p) != info["sha256"]:
                raise SystemExit(
                    f"dataset generation file failed verification: {p}; refusing to train")
        return dirn, man
    return dirn, None


def load(dirn):
    mm = None if os.environ.get("BC_RAM") == "1" else "r"     # BC_RAM=1: read the whole dataset into memory
    src, man = _resolve_load_source(dirn)
    D = {k: np.load(os.path.join(src, k + ".npy"), mmap_mode=mm) for k in FIELDS}
    if man is not None:
        _bad = [k for k in FIELDS if int(np.asarray(D[k]).shape[0]) != int(man["total"])]
        if _bad:
            raise SystemExit(
                f"dataset generation {man.get('generation')} in {dirn!r}: fields {_bad} do not "
                f"have the manifest row count {man['total']}; refusing to train")
        print(json.dumps({"load_source": "generation", "generation": man.get("generation"),
                          "rows": int(man["total"]), "data": dirn}), flush=True)
    if os.path.exists(os.path.join(dirn, "w.npy")) and os.environ.get("BC_USE_W", "1") == "1":
        D["w"] = np.load(os.path.join(dirn, "w.npy"), mmap_mode=mm)      # per-sample weights (dataset v4+)
    if os.path.exists(os.path.join(dirn, "uw.npy")) and os.environ.get("BC_USE_UW", "1") == "1":
        D["uw"] = np.load(os.path.join(dirn, "uw.npy"), mmap_mode=mm)    # per-unit weights (dataset v6+)
    wgate = _check_finite_weights(D, dirn)
    if wgate:
        print(json.dumps({"weight_gate": wgate, "data": dirn}), flush=True)
    if float(os.environ.get("BC_AUX_W", "0")) > 0:                      # auxiliary targets written by aux_labels.py
        for k in AUXK:
            D[k] = np.load(os.path.join(dirn, k + ".npy"), mmap_mode=mm)
    meta = json.load(open(os.path.join(src, "meta.json")))
    meta["dataset_source"] = src
    if man is None:
        # Legacy flat view (no generation pointer): refuse a torn mirror
        # instead of training on it. Fields must all agree on row count,
        # and -- when the builder stamped the flat meta (flat_total) --
        # agree with the stamp.
        _rows = {k: int(np.asarray(D[k]).shape[0]) for k in FIELDS}
        _stamp = meta.get("flat_total")
        if len(set(_rows.values())) != 1 or \
                (_stamp is not None and set(_rows.values()) != {int(_stamp)}):
            raise SystemExit(
                f"legacy flat dataset {dirn!r} has inconsistent field row counts {_rows} "
                f"(flat_total stamp: {_stamp}); this is a torn flat mirror left by an "
                f"interrupted publish -- rebuild the dataset; refusing to train")
        print(json.dumps({"load_source": "WARNING: legacy flat view (no generation pointer); "
                          "fields read from the root after a row-count cross-check",
                          "data": dirn}), flush=True)
    elif "derived_layer" not in meta:
        # The derived-layer mirror lives in the FLAT meta; keep it visible
        # to _validate_derived_layer when fields came from the generation.
        try:
            _root_meta = json.load(open(os.path.join(dirn, "meta.json")))
            if isinstance(_root_meta, dict) and "derived_layer" in _root_meta:
                meta["derived_layer"] = _root_meta["derived_layer"]
        except Exception:
            pass
    if wgate:
        meta["weight_gate"] = wgate
    dchk = _validate_derived_layer(D, dirn, meta)
    if dchk:
        meta["derived_layer_check"] = dchk
        print(json.dumps({"derived_layer_check": dchk, "data": dirn}), flush=True)
    val_g = np.array([g["val"] for g in meta["games"]])
    is_val = val_g[np.asarray(D["gid"])]
    return D, is_val, meta


def fetch(D, ii):
    ii = np.sort(ii)
    return {k: np.asarray(D[k][ii]) for k in LOAD + [x for x in ("w", "uw") + AUXK if x in D]}


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
            elif t_.dtype == torch.float32 and k not in ("w", "uw"):
                t_ = t_.half()
            # F06: loss weights (w/uw) stay float32 -- fp16 overflows real AWR uw (max ~428293.7 -> inf,
            # batch_loss NaN) and underflows small w. Model activations stay half; only the multipliers are f32.
            parts.append(t_.to(DEV))
        GD[k] = torch.cat(parts)
        if k in ("w", "uw") and not bool(torch.isfinite(GD[k]).all()):
            raise SystemExit(f"F06: non-finite loss weights reached the cached trainer (field {k!r}, "
                             f"dataset {data!r}); refusing to train")
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
    exit_w = float(os.environ.get("BC_EXIT_W", "0"))       # needs BC_SELF_KL > 0 (the reference logits)

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
            ref = REF[b["_cp"]].float(); n_ = REF.shape[1]
            lp = Fn.log_softmax(lg[:, :n_].float(), -1); lr_ = Fn.log_softmax(ref, -1)
            pr_ = lr_.exp()
            kl = torch.where(pr_ > 0, pr_ * (lr_ - lp), torch.zeros_like(pr_)).sum(-1)     # illegal moves: p = 0, no term
            if exit_w > 0:
                # search distillation: a COLLECTOR token whose played move differs from the initial net's own move is a
                # search correction -> cross-entropy to the played move, weight exit_w; every other own token
                # (unchanged cars, all planters and bases) is held on the initial net by the KL term.
                # The plain weighted cross-entropy above is not used.
                v_ = valid[:, :n_]; iscar = b["tok_kind"][:, :n_] == 0      # the search only moves collectors: a base /
                chg = v_ & iscar & (y[:, :n_] != ref.argmax(-1))             # planter difference is a rule layer, not a label
                keep = (v_ & ~chg).float(); chg = chg.float()
                loss = exit_w * (ce[:, :n_] * chg).sum() / chg.sum().clamp(min=1) + self_kl * (kl * keep).sum() / keep.sum().clamp(min=1)
            else:
                car = (valid & (b["tok_kind"][:, :F.OWN] == 0))[:, :n_].float()
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
        r["all"] = round(ok.sum() / max(n.sum(), 1), 4)
        if REF is not None and exit_w > 0:
            # of the validation tokens the search changed: how many does the net now play itself (0 before training);
            # of the unchanged ones: how many still agree with the initial net
            c_ok = c_n = k_ok = k_n = 0
            with torch.no_grad():
                for a0 in range(0, len(va), 4096):
                    b, tok, pre = light(va[a0:a0 + 4096]); _, lg = batch_loss(b, tok, pre)
                    ref = REF[b["_cp"]].float(); n_ = REF.shape[1]; y = b["y"][:, :n_]; v_ = y >= 0
                    pred = lg[:, :n_].argmax(-1); r0 = ref.argmax(-1); chg = v_ & (y != r0) & (b["tok_kind"][:, :n_] == 0)
                    c_ok += ((pred == y) & chg).sum().item(); c_n += chg.sum().item()
                    k_ok += ((pred == r0) & v_ & ~chg).sum().item(); k_n += (v_ & ~chg).sum().item()
            net.train()
            r["changed_learned"] = round(c_ok / max(c_n, 1), 4); r["changed_n"] = int(c_n); r["unchanged_kept"] = round(k_ok / max(k_n, 1), 4)
        return r

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
        if r["all"] > best or exit_w > 0:        # distillation: agreement with the labels is not the goal, keep the last epoch
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
        # Stage 0 new heads (utility/wdl/margin) may be absent in old weights, like tattn.*:
        # load through the shared migration path (F01, cnet2_model.migrate_state_dict) --
        # only the new-head keys and new trunk-attention layers (cfg TA, identity at init)
        # may be absent; anything else is a hard error, same as the PPO/resume entry.
        try:
            mig_ = migrate_state_dict(net, {k: torch.from_numpy(w[k]) for k in w.files if k != "meta"},
                                      allow_missing_prefixes=("tattn.",), source="BC_INIT")
        except RuntimeError as e:
            raise SystemExit(str(e))
        if mig_["migrated_missing"]:
            print(json.dumps({"init_new_layers": len(mig_["migrated_missing"])}), flush=True)
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
    # Whole-network search distillation (no trunk cache): BC_EXIT_W / BC_SELF_KL as in train_cached, but the reference
    # is a frozen copy of the initial net run on the same (augmented) batch. Collector tokens whose played move differs
    # from the reference's own move -> cross-entropy to the played move (x BC_EXIT_W); every other own token -> KL to the
    # reference (x BC_SELF_KL); the value head is held on the reference too (the value labels of generated games are
    # not real). Use it when the frozen-trunk decoder cannot learn the corrections (round ex1: 17% learned, flat in
    # data and epochs).
    ex_w = float(os.environ.get("BC_EXIT_W", "0")); ex_kl = float(os.environ.get("BC_SELF_KL", "0")); net0 = None
    # BC_CAR_KL=c (plain imitation, whole net): the COLLECTOR tokens are additionally held on the initial net by
    # c * KL(initial || current); planters and bases learn the teacher as usual. With set_uw.py CAR_W = 0 the cars are
    # not taught at all, with 0 < CAR_W < 1 they are taught weakly against the anchor. Purpose (2026-11-11): Echo1 took
    # Master3s' tree game but lost Grove1's fights for cargo (net effect on the margin vs Master2: +127 instead of +811).
    car_kl = float(os.environ.get("BC_CAR_KL", "0")); netc = None
    if car_kl > 0:
        if ex_w > 0 or ex_kl > 0:
            raise SystemExit("BC_CAR_KL is for plain imitation; do not combine with BC_EXIT_W / BC_SELF_KL")
        import copy
        netc = copy.deepcopy(net).eval()
        for p_ in netc.parameters():
            p_.requires_grad_(False)
        print(json.dumps({"car_kl": car_kl}), flush=True)
    if ex_w > 0 or ex_kl > 0:
        import copy
        net0 = copy.deepcopy(net).eval()
        for p_ in net0.parameters():
            p_.requires_grad_(False)

    # BC_AUX_W=w (with BC_SELF_KL): auxiliary heads on the trunk (stage A of reports/plan_katago.md). Per own unit: destroyed
    # within 10 / 25 / 50 steps, deposits within 25 / 50 steps; per position: change of the worth margin over 10 / 25 / 50
    # steps. The heads live outside the net (saved as *_auxheads.npz): the exported weights and the ladder pack are
    # unchanged, the trunk is what learns. With BC_EXIT_W=0 every own token is held on the reference.
    aux_w = float(os.environ.get("BC_AUX_W", "0")); heads = None; opt_h = None; last_aux = {}
    if aux_w > 0:
        if net0 is None:
            raise SystemExit("BC_AUX_W needs BC_SELF_KL > 0 (whole-network mode with the frozen reference)")
        C_, D_ = net.cfg["C"], net.cfg["D"]
        heads = torch.nn.ModuleDict({"die": torch.nn.Linear(D_, 3), "dep": torch.nn.Linear(D_, 2),
                                     "dw": torch.nn.Sequential(torch.nn.Linear(C_ + D_ + 64, 64), torch.nn.ReLU(), torch.nn.Linear(64, 3))}).to(DEV)
        opt_h = torch.optim.AdamW(heads.parameters(), lr=float(os.environ.get("BC_AUX_LR", "1e-3")))

    def aux_terms(b, tok, pre):
        t10 = tok[:, :10].float()
        dm = b["aux_die_m"].float(); pm = b["aux_dep_m"].float(); wm = b["aux_dw_m"].float()
        l_die = (Fn.binary_cross_entropy_with_logits(heads["die"](t10), b["aux_die"].float(), reduction="none") * dm).sum() / dm.sum().clamp(min=1)
        l_dep = (Fn.smooth_l1_loss(heads["dep"](t10), b["aux_dep"].float(), reduction="none") * pm).sum() / pm.sum().clamp(min=1)
        l_dw = (Fn.smooth_l1_loss(heads["dw"](pre.float()), b["aux_dw"].float(), reduction="none") * wm).sum() / wm.sum().clamp(min=1)
        return l_die, l_dep, l_dw

    def exit_terms(nt, b):
        if heads is not None:
            b2 = dict(b); b2["_want_pre"] = True
            tok, pre, aux = nt.encode(b2); ctx = nt.ctx_from(pre, b["teacher"])
            lg = nt.decode(b, tok, ctx); v = torch.tanh(nt.value(ctx)).squeeze(1)
            last_aux["t"] = aux_terms(b, tok, pre)
        else:
            lg, v, aux = _fwd(nt)(b)
        with torch.no_grad():
            lg0, v0, _ = _fwd(net0)(b)
        lg, lg0 = lg.float(), lg0.float()
        y = b["y"]; v_ = y >= 0; r0 = lg0.argmax(-1)
        chg = v_ & (b["tok_kind"][:, :F.OWN] == 0) & (y != r0)
        if ex_w <= 0:
            chg = torch.zeros_like(chg)                       # no distillation labels: hold every own token
        keep = (v_ & ~chg).float(); chgf = chg.float()
        ce = Fn.cross_entropy(lg.reshape(-1, 5), y.clamp(min=0).reshape(-1), reduction="none").reshape(y.shape)
        lp = Fn.log_softmax(lg, -1); lr_ = Fn.log_softmax(lg0, -1); pr_ = lr_.exp()
        kl = torch.where(pr_ > 0, pr_ * (lr_ - lp), torch.zeros_like(pr_)).sum(-1)
        l_ex = (ce * chgf).sum() / chgf.sum().clamp(min=1); l_kl = (kl * keep).sum() / keep.sum().clamp(min=1)
        return l_ex, l_kl, Fn.mse_loss(v.float(), v0.float()), Fn.mse_loss(aux.float(), b["carbon"]) * 10.0, lg, r0, chg, v_

    @torch.no_grad()
    def evaluate_exit():
        net.eval(); c_ok = c_n = k_ok = k_n = 0; ok = np.zeros(3); nn_ = np.zeros(3); ax = np.zeros(3); an = 0
        for b in loader(D, va, 1024, aug=False, shuffle=False):
            _, _, _, _, lg, r0, chg, v_ = exit_terms(net, b)
            if heads is not None:
                ax += [x.item() for x in last_aux["t"]]; an += 1
            pred = lg.argmax(-1); y = b["y"]; kind = b["tok_kind"][:, :F.OWN]
            c_ok += ((pred == y) & chg).sum().item(); c_n += chg.sum().item()
            k_ok += ((pred == r0) & v_ & ~chg).sum().item(); k_n += (v_ & ~chg).sum().item()
            for k in range(3):
                m = v_ & (kind == k); ok[k] += ((pred == y) & m).sum().item(); nn_[k] += m.sum().item()
        net.train()
        r = {nm: round(ok[k] / max(nn_[k], 1), 4) for k, nm in enumerate(("collector", "planter", "base"))}
        r["all"] = round(ok.sum() / max(nn_.sum(), 1), 4)
        r["changed_learned"] = round(c_ok / max(c_n, 1), 4); r["changed_n"] = int(c_n); r["unchanged_kept"] = round(k_ok / max(k_n, 1), 4)
        if heads is not None and an:
            r["aux_die_bce"], r["aux_dep_l1"], r["aux_dw_l1"] = [round(float(x), 4) for x in ax / an]
        return r

    for ep in range(epochs):
        t0 = time.time(); acc = np.zeros(3); n = 0
        for i, b in enumerate(loader(D, tr, bs, aug=net0 is None)):    # distillation: no symmetry augmentation -- the net is not
            if i >= spe:                                                # exactly equivariant, a mirrored view would turn its
                break                                                   # own moves into "changed" labels
            if net0 is not None:
                l_ex, l_kl, l_v, la, *_ = exit_terms(net if heads is not None else tnet, b)
                lpi, lv = l_ex, l_kl                                  # logged as "pi" (changed tokens) and "v" (KL)
                loss = ex_w * l_ex + ex_kl * l_kl + l_v + la
                if heads is not None:
                    l_die, l_dep, l_dw = last_aux["t"]
                    loss = loss + aux_w * (l_die + 0.3 * l_dep + l_dw)
                    la = l_die + l_dw                                 # logged as "aux"
                    opt_h.zero_grad(set_to_none=True)
            else:
                lpi, lv, la, lg_ = losses(tnet, b)
                loss = lpi + 0.2 * lv + la
                if netc is not None:
                    with torch.no_grad():
                        lg0_ = _fwd(netc)(b)[0].float()
                    car_ = ((b["y"] >= 0) & (b["tok_kind"][:, :F.OWN] == 0)).float()
                    lp_ = Fn.log_softmax(lg_.float(), -1); lr_ = Fn.log_softmax(lg0_, -1); pr_ = lr_.exp()
                    klc = (torch.where(pr_ > 0, pr_ * (lr_ - lp_), torch.zeros_like(pr_)).sum(-1) * car_).sum() / car_.sum().clamp(min=1)
                    loss = loss + car_kl * klc
                    lv = klc                                          # logged in the "v" column
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), float(os.environ.get("BC_CLIP", "2.0")))
            opt.step(); sched.step()
            if heads is not None:
                opt_h.step()
            if i % 50 == 0:
                acc += [lpi.item(), lv.item(), la.item()]; n += 1
            if i % 500 == 0:
                print(json.dumps({"ep": ep, "i": i, "pi": round(acc[0] / n, 4), "v": round(acc[1] / n, 4),
                                  "aux": round(acc[2] / n, 4), "s": round(time.time() - t0)}), flush=True)
        r = evaluate_exit() if net0 is not None else evaluate(net, D, va)
        rec = {"ep": ep, "val": r, "train_pi": round(acc[0] / max(n, 1), 4), "s": round(time.time() - t0)}
        if net0 is not None:
            rec["train_kl"] = round(acc[1] / max(n, 1), 5); best = -1          # distillation: the last epoch is the result
        print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
        export(net, out.replace(".npz", "_last.npz"), {"cfg": net.cfg, "epoch": ep, "val": r, "params": npar})
        if heads is not None:
            np.savez(out.replace(".npz", "_auxheads.npz"), **{k: v.detach().cpu().numpy() for k, v in heads.state_dict().items()})
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

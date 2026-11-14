"""cnet2_train -- behaviour cloning of cnet2 on a cnet2_build dataset (MPS).

Usage: python cnet2_train.py DATA_DIR OUT.npz EPOCHS [cfg-json]
env: BC_DEVICE (mps), BC_BS (512), BC_LR (1e-3), BC_INIT (npz to start from),
     BC_MAXSTEPS (cap optimiser steps per epoch, for smoke tests)
Augmentation: the 4 symmetries of the canonical frame that keep the base sets.
"""

import hashlib
import io
import json
import math
import os
import queue
import shutil
import sys
import threading
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as Fn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cnet2_feat as F  # noqa: E402
from cnet2_build import FIELDS, load_generation, _u_of_margin  # noqa: E402
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


def _scan_finite_array(a):
    """F1 chunked finite scan: None when every entry of the floating
    array `a` is finite, else (bad_count, first_flat_index). Integer and
    bool arrays are always finite. Blocks are widened to float64 so a
    float16/float32 value can neither overflow nor underflow the check."""
    a = np.asarray(a)
    if not np.issubdtype(a.dtype, np.floating):
        return None
    flat = a.reshape(-1)
    bad = 0; first = -1
    for a0 in range(0, flat.size, 1 << 22):
        blk = flat[a0:a0 + (1 << 22)].astype(np.float64)
        fin = np.isfinite(blk)
        if not fin.all():
            bad += int((~fin).sum())
            if first < 0:
                first = a0 + int(np.argmax(~fin))
    return (bad, first) if bad else None


def _check_finite_arrays(mapping, source):
    """F1 hard gate: refuse (SystemExit naming the array) when any
    floating array in `mapping` carries a NaN/Inf. Guards the BC_INIT
    backbone at load (F1a) and every export payload before it is
    written (F1c) -- a poisoned run must fail loudly at the gate, not
    ship weights that only misbehave downstream."""
    for k, a in mapping.items():
        hit = _scan_finite_array(a)
        if hit is not None:
            bad, first = hit
            raise SystemExit(f"F1: non-finite entries in {source} array {k!r}: "
                             f"{bad} NaN/Inf (first at flat index {first}); refusing")


def _fuse_finite(loss, gnorm, where):
    """F1(b) training fuse, checked EVERY step in both training loops:
    a non-finite loss or gradient norm aborts the run (SystemExit ->
    nonzero exit) BEFORE opt.step() and before any export of the epoch,
    so a poisoned step is never stepped on and never saved."""
    lv = float(loss.detach()) if torch.is_tensor(loss) else float(loss)
    gv = float(gnorm.detach()) if torch.is_tensor(gnorm) else float(gnorm)
    if not (math.isfinite(lv) and math.isfinite(gv)):
        raise SystemExit(f"F1: non-finite training signal at {where}: loss={lv!r} "
                         f"grad_norm={gv!r}; aborting without stepping or exporting")


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


_BC_INIT_BLOB = None


def _bc_init_blob():
    """BC_INIT bytes, read ONCE per process: (raw_bytes, sha256), or
    (None, None) when BC_INIT is unset or the file is missing. The env
    record's sha256 and main()'s np.load both consume this single byte
    object, so the recorded init identity is the loaded weights by
    construction (the certify_pack caliber: hash and load the same
    bytes). Before this, the record hashed the path at startup and
    main() re-read the path after the dataset load -- a file swapped
    inside that window was trained from while every record (the stdout
    bc_env line, the exported meta) named the old bytes."""
    global _BC_INIT_BLOB
    if _BC_INIT_BLOB is None:
        init = os.environ.get("BC_INIT")
        if init and os.path.exists(init):
            with open(init, "rb") as fh:
                raw = fh.read()
            _BC_INIT_BLOB = (raw, hashlib.sha256(raw).hexdigest())
        else:
            _BC_INIT_BLOB = (None, None)
    return _BC_INIT_BLOB


_BC_ENV_DEFAULTS = {"BC_AUX_W": "0", "BC_SELF_KL": "0", "BC_EXIT_W": "0",
                    "BC_ONLY_TEACH": None, "BC_TEACH_ROWS": None,
                    "BC_LR": "1e-3", "BC_BS": "512", "BC_MAXSTEPS": "0",
                    "BC_AMP": "bf16", "BC_SEQGRU": "0", "BC_SUBSET": "0"}
_BC_ENV_REC = None


def _bc_env_record():
    """F7: the BC_* switches steer training from the environment and
    used to leave no trace in any artifact -- two runs differing only in
    env were indistinguishable from their products. Record the EFFECTIVE
    value of the behaviour switches (env value, or the code default when
    unset), every other BC_* variable actually present, and for BC_INIT
    the sha256 of the exact bytes main() loads (a bare path string does
    not identify weights; see _bc_init_blob)."""
    rec = {k: os.environ.get(k, d) for k, d in _BC_ENV_DEFAULTS.items()}
    init = os.environ.get("BC_INIT")
    if init:
        rec["BC_INIT"] = {"path": init, "sha256": _bc_init_blob()[1]}
    else:
        rec["BC_INIT"] = None
    for k in sorted(os.environ):
        if k.startswith("BC_") and k not in rec:
            rec[k] = os.environ[k]
    return rec


def _validate_derived_layer(D, dirn, meta, sources=None, gen_ctx=None):
    """Derived-layer consumer gate (owner pt.4, final acceptance): w/uw in
    the dataset root are post-processing products (set_uw), NOT generation
    fields. Before any of them enters a loss, verify identity against the
    sidecar (derived.json, format "cnet2-derived/1", mirrored into the
    flat meta.json under "derived_layer"): for every weight actually
    consumed -- its field must not be a generation-manifest field, its
    base_generation must equal the generation load() pinned at resolve
    time (gen_ctx; the current.json pointer is only checked against
    that pin and a mid-load move refuses -- R3 P1), its rows must equal
    that generation's total, and its dtype/shape/
    sha256 must match the file on disk. A sidecar/meta mirror mismatch
    also refuses. A dataset with a generation pointer but UNDECLARED
    non-generation weights refuses outright (the F05 stale-weight hazard:
    bytes left behind by another build must never be consumed silently).
    Only a pre-generation legacy dataset (no pointer, no sidecar) may
    consume w/uw unverified, and then only with a loud warning.
    P1-A (owner re-review): `sources` records where load() ACTUALLY read
    each weight from -- "generation" (the hash-verified generation
    directory) or "root" (the flat mirror). The generation-owned verdict
    below may only rest on an actual generation read; the previous code
    inferred ownership from the field NAME while the bytes came from the
    unverified root, so a tampered or torn root w was consumed under a
    "generation-owned" label, and a sealed w vanished silently when the
    root copy was missing. `gen_ctx` carries the manifest load() itself
    resolved (for a directory that IS a generation: no pointer exists
    for _validate to re-read).
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
    if gen_ctx is not None:
        # R3 P1 (owner re-review): load() resolved and hash-verified ONE
        # generation before reading any array, and every field/weight in
        # memory was sourced against THAT manifest (gen_ctx). Identity
        # here is judged against the pinned manifest ONLY; the pointer
        # is re-read solely to prove it still names the pinned
        # generation. The previous code let the re-read REPLACE the
        # pinned context whenever a pointer existed, so a publish
        # landing mid-load switched identity to the new generation B:
        # B's derived sidecar then validated cleanly against B while
        # the features in memory were A's -- old features + new weights
        # under a "verified" label. Policy on a moved pointer: REFUSE,
        # naming both generations. An in-load retry was rejected as the
        # fix because it reopens the same race one level up (the retry
        # can itself straddle a second publish); the caller reruns
        # load() so exactly one generation is used end to end.
        gen_id = gen_ctx.get("generation")
        gen_fields = set(gen_ctx.get("fields", ()))
        gen_total = gen_ctx.get("total")
        if os.path.exists(ptr):
            ptrj = json.load(open(ptr))
            if ptrj.get("generation") != gen_id:
                raise SystemExit(
                    f"derived layer: generation pointer in dataset {dirn!r} moved "
                    f"during load: load() pinned and hash-verified {gen_id!r}, "
                    f"the pointer now names {ptrj.get('generation')!r}; the "
                    f"arrays in memory belong to the pinned generation and must "
                    f"never be validated against another generation's identity; "
                    f"refusing to train -- rerun load() end to end")
    elif os.path.exists(ptr):
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
        bad = [k for k in used
               if (sources or {}).get(k) != "generation" or k not in gen_fields]
        if bad:
            raise SystemExit(f"derived layer: weights {bad} in dataset {dirn!r} were not "
                             f"read from the certified generation {gen_id} and are not "
                             f"declared in a derived.json sidecar (stale or foreign "
                             f"weights); refusing to train")
        return {"mode": "generation-owned", "generation": gen_id, "used": used,
                "sources": {k: "generation" for k in used}}
    if gen_id is None:
        raise SystemExit(f"derived layer: dataset {dirn!r} declares a derived.json sidecar "
                         f"but has no generation pointer; identity cannot be verified; "
                         f"refusing to train")
    entries = decl.get("files", {}) if isinstance(decl, dict) else {}
    checked = []
    owned = []
    for k in used:
        src_k = (sources or {}).get(k)
        if src_k == "generation" or (src_k is None and k in gen_fields):
            # Bytes came from the hash-verified generation directory
            # (or, for a caller that does not report sources, the field
            # is generation property by manifest). Ownership is only
            # ever granted on this path -- never for root bytes.
            if k not in gen_fields:
                raise SystemExit(f"derived layer: {k} in dataset {dirn!r} was read from "
                                 f"the generation directory but is not a field of "
                                 f"generation {gen_id}; refusing to train")
            if (k + ".npy") in entries:
                raise SystemExit(f"derived layer: {k}.npy in dataset {dirn!r} is BOTH a field "
                                 f"of generation {gen_id} and a declared derived file; "
                                 f"refusing to train")
            owned.append(k)
            continue
        if k in gen_fields:
            raise SystemExit(f"derived layer: {k}.npy in dataset {dirn!r} is a field of "
                             f"generation {gen_id} but was read from the unverified "
                             f"flat root; refusing to train")
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
    out = {"mode": "verified", "generation": gen_id, "files": checked}
    if owned:
        out["owned"] = owned
    if sources:
        out["sources"] = dict(sources)
    return out


def _resolve_load_source(dirn):
    """Residual 3 (F05 torn flat mirror): decide WHERE dataset fields are
    read from. If dirn carries a generation pointer (current.json), the
    fields come from the certified generation directory it points at --
    every manifest field hash-verified via cnet2_build.load_generation --
    and the flat mirror in dirn itself is NEVER read for fields: a
    publish that died mid-refresh leaves new arrays next to an old meta
    in the root, and training on that mix (maps of the new build, labels
    of the old) must be impossible, not silent. A directory that IS a
    generation (manifest.json, no pointer) is verified the same way --
    field hashes AND, when the manifest carries it, the meta_sha256 seal
    over meta.json -- and read directly. Only a pre-generation legacy
    dataset (neither file)
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
        # Meta seal (Codex batch 1+2 re-review): the field loop above does
        # not cover meta.json, and the split (per-game val flags) lives
        # there -- a tampered meta passed this direct path while the
        # pointer path (load_generation) refused it. Mirror build's
        # load_generation semantics exactly: a manifest carrying
        # meta_sha256 must match sha256(meta.json); generations predating
        # the seal (no meta_sha256) are verified as before. (Build keeps
        # this check inline in load_generation, which requires a pointer
        # and so cannot serve a bare generation directory; the condition
        # and refusal shape are copied, not reinvented.)
        _meta_sha = man.get("meta_sha256")
        if _meta_sha is not None:
            _mp = os.path.join(dirn, "meta.json")
            if not os.path.exists(_mp):
                raise SystemExit(
                    f"dataset generation meta file missing: {_mp}; the manifest "
                    f"carries a meta_sha256 seal; refusing to train")
            if _sha256_path(_mp) != _meta_sha:
                raise SystemExit(
                    f"dataset generation meta failed verification: {_mp} "
                    f"(meta_sha256 mismatch -- meta.json changed after the "
                    f"generation was sealed); refusing to train")
        return dirn, man
    return dirn, None


AUX_SIDECAR = "k0_aux.json"
AUX_FORMAT = "cnet2-k0-aux/1"


def _load_aux_bound(dirn, src, man, meta, mm):
    """F2 (Codex round 3, K0-blocking): BC_AUX_W's six label arrays were
    mmap'd straight from the dataset root by filename -- no identity
    check at all, so a stale bundle written for an older build could be
    paired with newer samples and train silently (and its future
    windows could straddle padding past the real terminal step). The
    only sanctioned source is now the sealed bundle produced by
    k0_aux_labels.py: sidecar k0_aux.json (format "cnet2-k0-aux/1" --
    the same contract k0_train.check_aux_sidecar validates) in the
    dataset root, verified against the generation load() pinned:
      * the dataset must resolve to a certified generation (a legacy
        flat dataset has no generation to bind to -> refuse);
      * sidecar present and readable, format exact;
      * base_generation == the pinned generation (labels written for
        another generation refuse even when the row count matches);
      * sidecar rows == manifest total == rows in memory;
      * the row_key fingerprint recomputed from the pinned generation's
        gid/step files matches (labels bound to this exact row order);
      * lineage == manifest cfg_fp;
      * per-game lineage (game set, file, replay sha256) matches the
        certified meta;
      * all six label files exist, are declared in the sidecar with a
        sha256 equal to the bytes on disk, and carry exactly the
        dataset row count (plus sidecar shape/dtype when declared).
    No sidecar / no generation -> BC_AUX_W > 0 refuses; there is NO
    bare-file fallback. Returns ({AUXK: array}, summary-for-meta)."""
    if man is None:
        raise SystemExit(
            f"BC_AUX_W: dataset {dirn!r} has no certified generation (legacy flat view); "
            f"aux labels cannot be bound to a generation -- refusing to train. "
            f"Seal a generation and label it with k0_aux_labels.py (sidecar {AUX_SIDECAR}, "
            f"format {AUX_FORMAT})")
    gen_id = man.get("generation")
    n_rows = int(man["total"])
    sp = os.path.join(dirn, AUX_SIDECAR)
    if not os.path.exists(sp):
        raise SystemExit(
            f"BC_AUX_W: aux sidecar missing: {sp}; refusing to train. Bare aux_*.npy files "
            f"in the dataset root are not a binding source -- produce the sealed bundle "
            f"with k0_aux_labels.py (format {AUX_FORMAT})")
    try:
        sc = json.load(open(sp))
    except Exception as e:
        raise SystemExit(f"BC_AUX_W: aux sidecar unreadable: {sp}: {e}; refusing to train")
    if sc.get("format") != AUX_FORMAT:
        raise SystemExit(f"BC_AUX_W: aux sidecar format {sc.get('format')!r} != "
                         f"{AUX_FORMAT!r}; refusing to train")
    if sc.get("base_generation") != gen_id:
        raise SystemExit(
            f"BC_AUX_W: aux labels are bound to generation {sc.get('base_generation')!r} "
            f"but the dataset pinned generation {gen_id!r} (old labels on a new "
            f"generation); refusing to train")
    if int(sc.get("rows", -1)) != n_rows:
        raise SystemExit(f"BC_AUX_W: aux sidecar rows {sc.get('rows')} != generation "
                         f"total {n_rows}; refusing to train")
    fields = man.get("fields", {})
    parts = []
    for _nm in ("gid", "step"):
        _info = fields.get(_nm)
        if _info is None:
            raise SystemExit(f"BC_AUX_W: generation manifest lacks field {_nm!r}; cannot "
                             f"verify the aux row-order binding; refusing to train")
        parts.append(_sha256_path(os.path.join(src, _info["file"])))
    _rk = hashlib.sha256("|".join(parts).encode()).hexdigest()
    if (sc.get("row_key") or {}).get("sha256") != _rk:
        raise SystemExit("BC_AUX_W: aux row_key fingerprint mismatch -- the labels were "
                         "computed for a different row order (gid/step) than the pinned "
                         "generation; refusing to train")
    if sc.get("lineage") != man.get("cfg_fp"):
        raise SystemExit(f"BC_AUX_W: aux lineage {sc.get('lineage')!r} != manifest cfg_fp "
                         f"{man.get('cfg_fp')!r}; refusing to train")
    meta_games = {g["gi"]: g for g in meta["games"]}
    lin = {e.get("gi"): e for e in sc.get("games") or []}
    if set(lin) != set(meta_games):
        raise SystemExit(f"BC_AUX_W: aux lineage game set {sorted(lin)} != generation "
                         f"meta games {sorted(meta_games)}; refusing to train")
    for gi, ent in sorted(lin.items()):
        gm = meta_games[gi]
        if ent.get("file") != gm["file"]:
            raise SystemExit(f"BC_AUX_W: game {gi}: lineage file {ent.get('file')!r} != "
                             f"meta file {gm['file']!r}; refusing to train")
        want = ent.get("replay_sha256")
        if want is not None:
            if not os.path.exists(gm["file"]):
                raise SystemExit(f"BC_AUX_W: game {gi}: replay file missing: "
                                 f"{gm['file']}; refusing to train")
            if _sha256_path(gm["file"]) != want:
                raise SystemExit(f"BC_AUX_W: game {gi}: replay hash mismatch (replay "
                                 f"replaced since labelling); refusing to train")
        elif gm["file"].endswith(".json.gz"):
            raise SystemExit(f"BC_AUX_W: game {gi}: sidecar lineage carries no "
                             f"replay_sha256 for {gm['file']!r}; labels are not bound "
                             f"to the replay bytes; refusing to train")
    out = {}
    files_decl = sc.get("files") or {}
    for k in AUXK:
        name = k + ".npy"
        p = os.path.join(dirn, name)
        ent = files_decl.get(name)
        if not os.path.exists(p):
            raise SystemExit(f"BC_AUX_W: aux label file missing: {p}; refusing to train")
        if not isinstance(ent, dict) or ent.get("sha256") is None:
            raise SystemExit(f"BC_AUX_W: aux sidecar declares no sha256 for {name}; "
                             f"refusing to train")
        if _sha256_path(p) != ent["sha256"]:
            raise SystemExit(f"BC_AUX_W: aux label file {name} sha256 mismatch vs sidecar "
                             f"(contents changed since labelling); refusing to train")
        arr = np.load(p, mmap_mode=mm)
        if int(arr.shape[0]) != n_rows:
            raise SystemExit(f"BC_AUX_W: aux label file {name} has {int(arr.shape[0])} "
                             f"rows != dataset rows {n_rows}; refusing to train")
        if ent.get("shape") is not None and                 [int(x) for x in arr.shape] != [int(x) for x in ent["shape"]]:
            raise SystemExit(f"BC_AUX_W: aux label file {name} shape {list(arr.shape)} "
                             f"!= sidecar {ent['shape']}; refusing to train")
        if ent.get("dtype") is not None and str(arr.dtype) != ent["dtype"]:
            raise SystemExit(f"BC_AUX_W: aux label file {name} dtype {arr.dtype} != "
                             f"sidecar {ent['dtype']}; refusing to train")
        out[k] = arr
    summ = {"format": AUX_FORMAT, "generation": gen_id, "rows": n_rows,
            "files": sorted(k + ".npy" for k in AUXK)}
    print(json.dumps({"aux_binding": summ, "data": dirn}), flush=True)
    return out, summ


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
    # F1(a): feature finite gate. The float FEATURE fields are scanned
    # before anything trains on them: one Inf/NaN in scal/mf/uf silently
    # poisons every batch it enters. Label fields are NOT scanned here --
    # the label contract (H) gives NaN a meaning (unusable) in the raw
    # label columns, and the float label "value" keeps its own semantics;
    # the step fuse (F1b) still aborts the run if a poisoned label ever
    # drives a non-finite loss. w/uw keep their dedicated F06 gate below.
    for _k in FIELDS:
        if _k in ("value", "w", "uw"):
            continue
        _hit = _scan_finite_array(D[_k])
        if _hit is not None:
            raise SystemExit(f"F1: non-finite feature entries in dataset {dirn!r} field "
                             f"{_k!r}: {_hit[0]} NaN/Inf (first at flat index {_hit[1]}); "
                             f"refusing to train")
    # Weight fields (owner re-review, P1-A): a field the generation
    # manifest declares is generation property and is read from the
    # certified generation directory ONLY -- _resolve_load_source has
    # already hash-verified those exact bytes via load_generation, and
    # the flat root mirror (torn by an interrupted publish, or tampered)
    # is never a source for them: previously w/uw were loaded from the
    # root unconditionally and merely LABELLED generation-owned when the
    # name appeared in the manifest, so a root w of all-123 trained
    # under a verified label and a sealed w vanished silently when the
    # root copy was deleted. Only a field the manifest does NOT declare
    # may come from the root, and then only as a derived-layer product
    # that _validate_derived_layer verifies against the sidecar below.
    # `wsrc` records the ACTUAL source of every weight consumed.
    gen_fields = set(man.get("fields", {})) if man is not None else set()
    wsrc = {}
    for _k, _env in (("w", "BC_USE_W"), ("uw", "BC_USE_UW")):
        if os.environ.get(_env, "1") != "1":
            continue
        if _k in gen_fields:
            _p = os.path.join(src, _k + ".npy")
            if not os.path.exists(_p):
                raise SystemExit(
                    f"dataset generation {man.get('generation')} in {dirn!r}: manifest "
                    f"declares weight field {_k!r} but {_p} is missing; refusing to train")
            D[_k] = np.load(_p, mmap_mode=mm)
            wsrc[_k] = "generation"
        elif os.path.exists(os.path.join(dirn, _k + ".npy")):
            D[_k] = np.load(os.path.join(dirn, _k + ".npy"), mmap_mode=mm)
            wsrc[_k] = "root"
    wgate = _check_finite_weights(D, dirn)
    if wgate:
        print(json.dumps({"weight_gate": wgate, "data": dirn}), flush=True)
    meta = json.load(open(os.path.join(src, "meta.json")))
    meta["dataset_source"] = src
    if float(os.environ.get("BC_AUX_W", "0")) > 0:                      # auxiliary targets: sealed bound bundle only (F2)
        _aux, _aux_sum = _load_aux_bound(dirn, src, man, meta, mm)
        D.update(_aux)
        meta["aux_binding"] = _aux_sum
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
    _gen_ctx = ({"generation": man.get("generation"),
                 "fields": set(man.get("fields", {})),
                 "total": int(man["total"])} if man is not None else None)
    dchk = _validate_derived_layer(D, dirn, meta, sources=wsrc, gen_ctx=_gen_ctx)
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
    # F6: the daemon workers used to die silently on any fetch error --
    # the consumer then blocked on q.get() forever and the run hung
    # with no diagnosis -- and a consumer that stopped early (the spe
    # break in the training loop) left them parked on a full queue
    # until process exit, accumulating across calls. Workers now
    # record errors into `errors` and set `stop`; the consumer polls
    # with a timeout and raises the worker's error itself; the
    # generator's finally sets `stop` and joins every worker on BOTH
    # the normal and the early-exit path (daemon=True remains only as
    # a backstop, never as the reclaim mechanism).
    stop = threading.Event()
    errors = []

    def work():
        while not stop.is_set():
            with lock:
                try:
                    j = next(it)
                except StopIteration:
                    return
            try:
                res_b = fetch(D, chunks[j])
            except BaseException as e:  # noqa: BLE001 -- re-raised by the consumer below
                errors.append((j, e, traceback.format_exc()))
                stop.set()
                return
            while not stop.is_set():
                try:
                    q.put((j, res_b), timeout=0.25)
                    break
                except queue.Full:
                    continue

    ths = [threading.Thread(target=work, daemon=True, name=f"bc-loader-{t}")
           for t in range(nthreads)]
    for th in ths:
        th.start()
    nxt = 0
    try:
        while nxt < len(chunks):
            while nxt not in res:
                if errors:
                    j, e, tb = errors[0]
                    raise RuntimeError(f"loader worker failed fetching chunk {j}: {e!r}\n{tb}")
                try:
                    j, b = q.get(timeout=0.25)
                except queue.Empty:
                    if not any(th.is_alive() for th in ths):
                        raise RuntimeError(
                            f"loader: all {nthreads} workers exited but chunk {nxt} of "
                            f"{len(chunks)} was never delivered and no error was reported")
                    continue
                res[j] = b
            yield to_dev(res.pop(nxt), aug)
            nxt += 1
    finally:
        stop.set()
        for th in ths:
            th.join(timeout=10)


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
        if not bool(torch.isfinite(lg).all()):
            raise SystemExit("F1: evaluate: non-finite logits from the network; the val "
                             "metrics would be counted from garbage -- refusing to "
                             "report them (check the weights/features feeding this "
                             "evaluation)")
        pred = lg.argmax(-1)
        y = b["y"]
        kind = b["tok_kind"][:, :F.OWN]
        for k in range(3):
            m = (y >= 0) & (kind == k)
            ok[k] += ((pred == y) & m).sum().item(); n[k] += m.sum().item()
        lsum += lpi.item() * len(y); ln += len(y)
    if not math.isfinite(lsum):
        raise SystemExit("F1: evaluate: accumulated val loss is non-finite; refusing "
                         "to report metrics")
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
    sel = _selection_config()
    sel_rows = []
    if sel is not None:
        _selection_open(sel, out)
        print(json.dumps({"selection": sel["public"]}), flush=True)
    for ep in range(epochs):
        t1 = time.time(); acc = 0.0; nb = 0
        perm = np.random.permutation(tr)
        for i in range(spe):
            b, tok, pre = light(perm[i * bs:(i + 1) * bs])
            lpi, _ = batch_loss(b, tok, pre)
            opt.zero_grad(set_to_none=True); lpi.backward()
            _gn = torch.nn.utils.clip_grad_norm_([p_ for p_ in net.parameters() if p_.requires_grad], float(os.environ.get("BC_CLIP", "2.0")))
            _fuse_finite(lpi, _gn, f"train_cached step {i}")
            opt.step(); sched.step()
            if i % 50 == 0:
                acc += lpi.item(); nb += 1
        r = evaluate_c()
        rec = {"ep": ep, "val": r, "train_pi": round(acc / max(nb, 1), 4), "s": round(time.time() - t1), "cached": True}
        print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
        if sel is not None:
            _sel_row = _selection_eval_epoch(sel, net, ep, r["all"], {"cfg": net.cfg, "epoch": ep, "val": r, "params": npar, "data": data, "cached_trunk": True})
            sel_rows.append(_sel_row)
            _srec = {"ep": ep, "selection": {"mean_u": _sel_row.get("mean_u"), "se": _sel_row.get("se"), "ok": _sel_row.get("ok"), "form": _sel_row.get("form"), "wall_s": _sel_row.get("wall_s")}}
            print(json.dumps(_srec), flush=True); log.write(json.dumps(_srec) + "\n"); log.flush()
        elif r["all"] > best or exit_w > 0:        # distillation: agreement with the labels is not the goal, keep the last epoch
            best = r["all"]; export(net, out, {"cfg": net.cfg, "epoch": ep, "val": r, "params": npar, "data": data, "cached_trunk": True})
    if sel is not None:
        best = _selection_finish(sel, sel_rows, net, out, {"params": npar, "data": data, "cached_trunk": True})["agreement"]
    print("best", best)


def export(net, path, meta):
    sd = {k: v.detach().cpu().float().numpy() for k, v in net.state_dict().items()}
    # F1(c): no export may carry a non-finite array -- a poisoned run
    # must fail HERE, loudly, instead of writing weights that only fail
    # (or worse, silently pass) downstream.
    _check_finite_arrays(sd, f"export {path!r}")
    # F7: every product records the effective BC_* environment it was
    # trained under, so runs that differ only in env are separable from
    # their artifacts alone.
    meta = dict(meta)
    meta["bc_env"] = _BC_ENV_REC if _BC_ENV_REC is not None else _bc_env_record()
    np.savez(path, **sd, meta=np.array(json.dumps(meta)))


# ---------------------------------------------------------------------------
# A4 selection (strategy-group ruling, 2026-11-14): BC model selection is no
# longer "best val top-1 teacher agreement". Both branches (main path and
# distillation) rank every epoch's candidate by PAIRED GAMES against a fixed
# examiner -- the Master2 real file, configured via BC_SELECTION_EXAMINER --
# on the registered selection seed block 997000-997199 (selection ONLY, never
# calibration / confirmation / exams), both seats per seed, >= 64 pairs,
# ranked by mean U. U comes from cnet2_build._u_of_margin (the verified fleet
# implementation; no new copy). Uncertainty follows section 20 R2 as
# implemented in rl2_ppo_eval._clustered_se: the SEED is the independent
# unit -- a seed's games (both seats) are averaged into one cluster value
# and the SE / CI run over clusters. Teacher agreement survives only as a
# diagnostic field in the exported meta.
# Pre-registered tie rule: rank 1 vs rank 2 with dU within 1 (paired) SE are
# tied -> take the LATER epoch; still tied -> take the higher agreement.
# Form ruling (strategy group, same date): deployment form (int8 pack under
# the template CONF) is preferred; when per-epoch packing makes it
# infeasible the ranking may run on the naked-proxy form (the trainer's fp32
# export driven through the same cnet2_search under the identical CONF),
# provided (a) provenance records selection_form explicitly and (b) a tie
# inside the band is settled by a deployment-form duel when the pack
# machinery is usable at all. Selection is OFF unless BC_SELECTION_EXAMINER
# is set; with it unset, both branches keep their legacy behaviour exactly.
SELECTION_SEED0 = 997000
SELECTION_SEED_END = 997199
SELECTION_PAIRS_DEFAULT = 64
# Deployment CONF: the 13 keys of code/packs/PACK_CONF_TEMPLATE.txt, verbatim.
SELECTION_CONF = {"SEARCH": False, "ROLL": "self", "GUARD": True,
                  "GUARD_NARROW": True, "MODE": "sweep", "MIX": False,
                  "TEACHER": 2, "TEACHER_LATE": 2, "LATE_STEP": 240,
                  "BUDGET": 0.7, "HARD": 0.85, "ABORT_DL": 0.95,
                  "START_DL": None}
# Table + lower-key rule mirror stage1_teacher_gate._t_crit's stdlib
# fallback verbatim (dense exact 1..30; above, the largest tabulated df <=
# df -- t crit decreases with df, so this never under-covers; no 1.96 tail).
# Kept local rather than imported: the trainer must not depend on the gate
# module's chain. Values can be diffed against the gate mechanically.
_SEL_TCRIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
              7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
              13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101,
              19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064,
              25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
              40: 2.021, 50: 2.009, 60: 2.000, 80: 1.990, 100: 1.984, 120: 1.980}


def _selection_t_crit(df):
    if df < 1:
        return None
    ks = [k for k in _SEL_TCRIT if k <= df]
    return _SEL_TCRIT[max(ks)] if ks else _SEL_TCRIT[1]


def _selection_cluster_stats(vals):
    """Section 20 R2: vals are per-seed cluster values (each seed's games
    averaged). se is None (inestimable) below 2 clusters -- never 0.0."""
    n = len(vals)
    if n == 0:
        return {"mean_u": None, "se": None, "ci95": None, "n_clusters": 0}
    mean = float(np.mean(vals))
    if n < 2:
        return {"mean_u": mean, "se": None, "ci95": None, "n_clusters": n}
    se = float(np.std(vals, ddof=1) / math.sqrt(n))
    t = _selection_t_crit(n - 1)
    return {"mean_u": mean, "se": se,
            "ci95": [mean - t * se, mean + t * se], "n_clusters": n}


def _selection_paired_se(ra, rb):
    """Paired SE of the mean-U difference: per-seed cluster differences on
    the seeds both rows share (identical block by construction)."""
    sa, sb = ra.get("seed_u") or {}, rb.get("seed_u") or {}
    keys = [k for k in sa if k in sb]
    if len(keys) < 2:
        return None
    d = np.array([sa[k] - sb[k] for k in keys], dtype=np.float64)
    return float(np.std(d, ddof=1) / math.sqrt(len(d)))


def _selection_md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _selection_config():
    """Selection config from env; None when selection is off. Every caliber
    violation dies by name BEFORE any training happens."""
    ex = os.environ.get("BC_SELECTION_EXAMINER")
    if not ex:
        return None
    if not os.path.isfile(ex):
        raise SystemExit(f"selection: BC_SELECTION_EXAMINER={ex!r} is not a file -- "
                         "the fixed examiner (Master2 real pack) must exist; refusing to train")
    if os.environ.get("EVAL_STEPS", "299") != "299":
        raise SystemExit("selection: EVAL_STEPS must be unset/299 (selection games are full games)")
    if float(os.environ.get("EVAL_OPP_CASH", "0") or 0) != 0:
        raise SystemExit("selection: EVAL_OPP_CASH must be unset/0 (exam-ruler cash breaks the pairing)")
    for k in ("EVAL_ORACLE_STEPS", "EVAL_REVEAL_STEPS", "EVAL_SNAKE"):
        if os.environ.get(k):
            raise SystemExit(f"selection: {k} must be unset (diagnostic overrides break the pairing)")
    seed0 = int(os.environ.get("BC_SELECTION_SEED0", str(SELECTION_SEED0)))
    pairs = int(os.environ.get("BC_SELECTION_PAIRS", str(SELECTION_PAIRS_DEFAULT)))
    fixture = os.environ.get("BC_SELECTION_FIXTURE") == "1"
    if not (SELECTION_SEED0 <= seed0 <= SELECTION_SEED_END):
        raise SystemExit(f"selection: seed0 {seed0} outside the registered selection block "
                         f"{SELECTION_SEED0}-{SELECTION_SEED_END}")
    if seed0 + pairs - 1 > SELECTION_SEED_END:
        raise SystemExit(f"selection: seed block {seed0}..{seed0 + pairs - 1} overruns "
                         f"the registered block end {SELECTION_SEED_END}")
    if pairs < 2:
        raise SystemExit(f"selection: pairs {pairs} < 2 (clustered SE inestimable)")
    if pairs < SELECTION_PAIRS_DEFAULT and not fixture:
        raise SystemExit(f"selection: pairs {pairs} < {SELECTION_PAIRS_DEFAULT} -- the ruling's minimum; "
                         "only BC_SELECTION_FIXTURE=1 runs may go below it (and are stamped as fixtures)")
    form = os.environ.get("BC_SELECTION_FORM", "auto")
    if form not in ("auto", "deployed", "naked-proxy"):
        raise SystemExit(f"selection: BC_SELECTION_FORM={form!r} not in auto|deployed|naked-proxy")
    sel = {"examiner": os.path.abspath(ex), "examiner_md5": _selection_md5(ex),
           "seed0": seed0, "pairs": pairs, "form": form, "fixture": fixture,
           "workers": int(os.environ.get("BC_SELECTION_WORKERS", "8")), "dir": None}
    sel["public"] = {"method": "examiner_pairs", "examiner": sel["examiner"],
                     "examiner_md5": sel["examiner_md5"], "seed0": seed0, "pairs": pairs,
                     "form_requested": form, "fixture": fixture}
    return sel


def _selection_open(sel, out):
    d = out.replace(".npz", "_sel")
    if os.path.exists(d):
        d = f"{d}_{os.getpid()}"
    os.makedirs(d)
    sel["dir"] = d
    return d


def _selection_build_pack(cand_npz, out_py):
    """Deployment form: quantize the candidate into a cnet2_search pack under
    the template CONF. Any failure (incl. the packer's own SystemExit) is
    reported and returns False -- the caller decides what that means."""
    try:
        packs_dir = os.path.join(HERE, "packs")
        if os.path.isdir(packs_dir) and packs_dir not in sys.path:
            sys.path.insert(0, packs_dir)
        import cnet2_search_pack
        old = os.environ.get("CNET2_NO_ROLL")
        os.environ["CNET2_NO_ROLL"] = "1"     # template CONF is SEARCH=False: no rollout net
        try:
            cnet2_search_pack.main(out_py, repr(SELECTION_CONF), wp=cand_npz)
        finally:
            if old is None:
                os.environ.pop("CNET2_NO_ROLL", None)
            else:
                os.environ["CNET2_NO_ROLL"] = old
        return os.path.isfile(out_py)
    except BaseException:
        print(json.dumps({"selection_pack_error": traceback.format_exc(limit=3)}), flush=True)
        return False


def _selection_policy_file(sel, cand_npz, ep, force_form=None):
    """Resolve the playable policy file for a candidate. Returns
    (path, form) with form in deployed|naked-proxy, or (None, None) for a
    failed FORCED deployment build (duel probe). Explicit deployed + build
    failure dies: no silent degradation to the proxy."""
    want = force_form or sel["form"]
    if want in ("auto", "deployed"):
        if not sel.get("_pack_dead"):
            out_py = os.path.join(sel["dir"], f"ep{ep}_pack.py")
            if _selection_build_pack(cand_npz, out_py):
                return out_py, "deployed"
            sel["_pack_dead"] = True
            print(json.dumps({"selection_pack_fallback": "naked-proxy", "epoch": ep}), flush=True)
        if force_form is not None:
            return None, None
        if want == "deployed":
            raise SystemExit("selection: BC_SELECTION_FORM=deployed but the deployment pack "
                             "build failed -- refusing silent naked-proxy degradation")
    proxy = os.path.join(sel["dir"], f"selproxy_ep{ep}.py")
    src = ("import sys\n"
           f"sys.path.insert(0, {HERE!r})\n"
           "from cnet2_search import make_search_agent\n"
           f"agent = make_search_agent({SELECTION_CONF!r}, weights={cand_npz!r})\n")
    with open(proxy, "w") as f:
        f.write(src)
    return proxy, "naked-proxy"


def _selection_play(task):
    import eval_v53
    return eval_v53.play_game(task)


def _selection_eval(sel, policy_file, form):
    """Paired games vs the examiner over the selection block: every seed at
    both seats. A game counts only with a margin AND zero agent errors on
    both sides (F4/F09 caliber: a margin against or by a silently degraded
    agent is not a measurement); a seed cluster needs both seats valid."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    seeds = list(range(sel["seed0"], sel["seed0"] + sel["pairs"]))
    tasks = [("selection", policy_file, sel["examiner"], s, side)
             for s in seeds for side in (0, 1)]
    t0 = time.time()
    got = {}
    with ProcessPoolExecutor(max_workers=sel["workers"],
                             mp_context=mp.get_context("fork")) as ex_:
        futs = {ex_.submit(_selection_play, t): t for t in tasks}
        for fut, t in futs.items():
            try:
                got[(t[3], t[4])] = fut.result()
            except BaseException:
                got[(t[3], t[4])] = None
                print(json.dumps({"selection_game_error": traceback.format_exc(limit=3)}), flush=True)
    seed_u, errors = {}, 0
    for s in seeds:
        us = []
        for side in (0, 1):
            r = got.get((s, side))
            if (r is None or r.get("margin") is None
                    or (r.get("a_errors") or 0) != 0 or (r.get("b_errors") or 0) != 0):
                errors += 1
                continue
            us.append(_u_of_margin(r["margin"]))
        if len(us) == 2:
            seed_u[s] = float(np.mean(us))
    row = {"form": form, "games": len(tasks), "valid_games": 2 * len(seed_u),
           "errors": errors, "ok": errors == 0 and len(seed_u) == len(seeds),
           "wall_s": round(time.time() - t0, 1), "seed_u": seed_u}
    row.update(_selection_cluster_stats([seed_u[s] for s in seeds if s in seed_u]))
    return row


def _selection_eval_epoch(sel, net, ep, agreement, cand_meta):
    cand = os.path.join(sel["dir"], f"ep{ep}.npz")
    export(net, cand, dict(cand_meta, selection_candidate=True))
    pf, form = _selection_policy_file(sel, cand, ep)
    if sel.get("_form") is None:
        sel["_form"] = form
    elif sel["_form"] != form:
        raise SystemExit(f"selection: form changed mid-run ({sel['_form']} -> {form} at epoch {ep}); "
                         "a ranking may not mix forms")
    row = _selection_eval(sel, pf, form)
    row.update({"epoch": ep, "agreement": agreement, "candidate": cand,
                "candidate_md5": _selection_md5(cand)})
    return row


def _selection_pick(sel, rows):
    """Registered rule: rank by mean U; top-vs-second within 1 paired SE is
    a tie -> deployment-form duel when the ranking ran naked-proxy and the
    machinery is usable, else the later epoch; agreement only as the final
    fallback. Failed epochs never rank."""
    clean = [r for r in rows if r.get("ok") and r.get("mean_u") is not None]
    pick = {"failed_epochs": [r["epoch"] for r in rows if r not in clean]}
    if not clean:
        raise SystemExit("selection: no epoch produced a clean evaluation -- nothing may be "
                         "exported as selected (per-epoch _last/_ep artifacts remain on disk)")
    clean.sort(key=lambda r: (-r["mean_u"], -r["epoch"]))
    top = clean[0]
    pick["winner_epoch"] = top["epoch"]
    if len(clean) == 1:
        pick["rule"] = "sole_candidate"
        return pick
    second = clean[1]
    d = top["mean_u"] - second["mean_u"]
    se_d = _selection_paired_se(top, second)
    pick["tie_band"] = {"top_epoch": top["epoch"], "second_epoch": second["epoch"],
                        "d_u": d, "se_diff": se_d}
    if se_d is not None and d > se_d:
        pick["rule"] = "mean_u"
        return pick
    # Inside the tie band (or the paired SE is inestimable: treat as tied).
    winner, duel = None, None
    if top["form"] == "naked-proxy" and second["form"] == "naked-proxy":
        pf_t = _selection_policy_file(sel, top["candidate"], top["epoch"], force_form="deployed")
        pf_s = _selection_policy_file(sel, second["candidate"], second["epoch"], force_form="deployed")
        if pf_t[0] and pf_s[0]:
            rt = _selection_eval(sel, pf_t[0], "deployed")
            rs = _selection_eval(sel, pf_s[0], "deployed")
            duel = {"machinery_usable": True,
                    "top": {"epoch": top["epoch"], "mean_u": rt.get("mean_u"), "se": rt.get("se"), "ok": rt.get("ok")},
                    "second": {"epoch": second["epoch"], "mean_u": rs.get("mean_u"), "se": rs.get("se"), "ok": rs.get("ok")}}
            if rt.get("ok") and rs.get("ok"):
                dd = rt["mean_u"] - rs["mean_u"]
                sed = _selection_paired_se(rt, rs)
                duel["d_u"] = dd; duel["se_diff"] = sed
                if sed is not None and abs(dd) > sed:
                    winner = top if dd > 0 else second
                    pick["rule"] = "duel_deployed"
            else:
                # V03: the duel RAN but a side's evaluation failed (ok=false:
                # games errored / no valid clusters). That is an executed
                # failure, NOT a statistical tie -- a failed candidate may
                # not win through the later-epoch tie fallback, and the
                # failure must be on the record. (A duel that could not be
                # BUILT at all stays machinery_usable=False below: no
                # deployment comparison exists, nobody's fault.)
                bad = [e for e, r in ((top["epoch"], rt), (second["epoch"], rs))
                       if not (r.get("ok") and r.get("mean_u") is not None)]
                duel["failed_epochs"] = bad
                pick["failed_epochs"] = sorted(set(pick["failed_epochs"]) | set(bad))
                pick["duel"] = duel
                if len(bad) == 2:
                    raise SystemExit(f"selection: the deployment duel failed for BOTH tied "
                                     f"candidates (epochs {bad[0]} and {bad[1]}) -- no healthy "
                                     "deployed comparison exists and a failed candidate may "
                                     "not be selected; refusing to export a winner")
                winner = second if bad[0] == top["epoch"] else top
                pick["rule"] = "duel_opponent_failed"
                pick["winner_epoch"] = winner["epoch"]
                return pick
            pick["duel"] = duel
        else:
            duel = {"machinery_usable": False,
                    "reason": "deployment-form pack build failed for a tied candidate"}
            pick["duel"] = duel
    if winner is None:
        if top["epoch"] != second["epoch"]:
            winner = top if top["epoch"] > second["epoch"] else second
            if duel is not None and duel.get("machinery_usable"):
                pick["rule"] = "duel_tie_later_epoch"
            elif duel is not None:
                pick["rule"] = "tie_later_epoch_no_deployed"
            else:
                pick["rule"] = "tie_later_epoch"
        else:
            winner = top if top["agreement"] >= second["agreement"] else second
            pick["rule"] = "tie_agreement"
    pick["winner_epoch"] = winner["epoch"]
    return pick


def _selection_finish(sel, rows, net, out, meta_extra):
    """Export the picked epoch's candidate to `out` with the full selection
    provenance. The winner's own candidate meta is the base, so the product
    carries exactly what a legacy export of that epoch would have carried,
    plus the selection record."""
    pick = _selection_pick(sel, rows)
    win = next(r for r in rows if r["epoch"] == pick["winner_epoch"])
    z = np.load(win["candidate"])
    sd = {k: torch.from_numpy(np.asarray(z[k])) for k in z.files if k != "meta"}
    net.load_state_dict(sd)
    meta = {}
    if "meta" in z.files:
        meta = json.loads(str(z["meta"]))
    meta.pop("selection_candidate", None)
    meta.update(meta_extra)
    meta.update({
        "selection_method": "examiner_pairs",
        "selection_form": rows[0]["form"],
        "selection_seed_block": [SELECTION_SEED0, SELECTION_SEED_END],
        "selection_seed0": sel["seed0"],
        "selection_pairs": sel["pairs"],
        "selection_examiner": sel["examiner"],
        "selection_examiner_md5": sel["examiner_md5"],
        "selection_epochs": [{k: r.get(k) for k in ("epoch", "mean_u", "se", "ci95", "agreement",
                                                     "ok", "form", "games", "errors", "wall_s",
                                                     "candidate_md5", "seed_u")} for r in rows],
        "selection_pick": pick,
    })
    if sel["fixture"]:
        meta["selection_fixture"] = True
    export(net, out, meta)
    print(json.dumps({"selection_pick": pick}), flush=True)
    return {"agreement": win["agreement"], "epoch": win["epoch"]}


def main(data, out, epochs, cfg):
    torch.manual_seed(0); np.random.seed(0)
    global _BC_ENV_REC
    _BC_ENV_REC = _bc_env_record()          # F7: effective env, printed and exported
    print(json.dumps({"bc_env": _BC_ENV_REC}), flush=True)
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
        _init_raw = _bc_init_blob()[0]
        w = np.load(io.BytesIO(_init_raw)) if _init_raw is not None else np.load(init)
        # F1(a): the init backbone passes a full-tensor finite gate
        # BEFORE migration -- one NaN array in the npz would otherwise
        # be migrated in silently and poison every step (and every
        # export) downstream.
        _init_sd = {k: w[k] for k in w.files if k != "meta"}
        _check_finite_arrays(_init_sd, f"BC_INIT {init!r}")
        # Stage 0 new heads (utility/wdl/margin) may be absent in old weights, like tattn.*:
        # load through the shared migration path (F01, cnet2_model.migrate_state_dict) --
        # only the new-head keys and new trunk-attention layers (cfg TA, identity at init)
        # may be absent; anything else is a hard error, same as the PPO/resume entry.
        try:
            mig_ = migrate_state_dict(net, {k: torch.from_numpy(a) for k, a in _init_sd.items()},
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
    sel = _selection_config()
    sel_rows = []
    if sel is not None:
        _selection_open(sel, out)
        print(json.dumps({"selection": sel["public"]}), flush=True)
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
            if not bool(torch.isfinite(lg).all()):
                raise SystemExit("F1: evaluate_exit: non-finite logits from the network; "
                                 "refusing to report metrics counted from garbage")
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
            _gn = torch.nn.utils.clip_grad_norm_(net.parameters(), float(os.environ.get("BC_CLIP", "2.0")))
            _fuse_finite(loss, _gn, f"main loop ep {ep} step {i}")
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
            _hsd = {k: v.detach().cpu().numpy() for k, v in heads.state_dict().items()}
            _check_finite_arrays(_hsd, f"export {out.replace('.npz', '_auxheads.npz')!r}")
            np.savez(out.replace(".npz", "_auxheads.npz"), **_hsd)
        if os.environ.get("BC_SAVE_EPOCHS") == "1":       # keep every epoch for the online probes to choose from
            export(net, out.replace(".npz", f"_ep{ep}.npz"), {"cfg": net.cfg, "epoch": ep, "val": r, "params": npar})
        if sel is not None:
            _sel_row = _selection_eval_epoch(sel, net, ep, r["all"], {"cfg": net.cfg, "epoch": ep, "val": r, "params": npar})
            sel_rows.append(_sel_row)
            _srec = {"ep": ep, "selection": {"mean_u": _sel_row.get("mean_u"), "se": _sel_row.get("se"), "ok": _sel_row.get("ok"), "form": _sel_row.get("form"), "wall_s": _sel_row.get("wall_s")}}
            print(json.dumps(_srec), flush=True); log.write(json.dumps(_srec) + "\n"); log.flush()
            if heads is not None:
                np.savez(os.path.join(sel["dir"], f"ep{ep}_auxheads.npz"), **{k: v.detach().cpu().numpy() for k, v in heads.state_dict().items()})
        elif r["all"] > best:
            best = r["all"]
            export(net, out, {"cfg": net.cfg, "epoch": ep, "val": r, "params": npar, "data": data})
    if sel is not None:
        _fin = _selection_finish(sel, sel_rows, net, out, {"params": npar, "data": data})
        best = _fin["agreement"]
        if heads is not None:
            _hw = os.path.join(sel["dir"], f"ep{_fin['epoch']}_auxheads.npz")
            if os.path.exists(_hw):
                shutil.copyfile(_hw, out.replace(".npz", "_auxheads.npz"))
    print("best", best)
    if only and init:          # frozen trunk: prove every parameter except the trained style rows is bit-identical
        _init_raw = _bc_init_blob()[0]
        w0 = np.load(io.BytesIO(_init_raw)) if _init_raw is not None else np.load(init)
        w1 = np.load(out); diff = []
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

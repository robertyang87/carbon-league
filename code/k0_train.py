"""k0_train -- K0 training entry (gate 4 of the K0 line): multi-head training
on the UNCHANGED CNet2 backbone, with an explicit gradient-routing protocol.

Heads
  anchor : the backbone's own policy decoder, teacher-forced joint CE to the
           recorded behaviour actions (the behaviour anchor; same CE form as
           cnet2_train.losses, without the teacher/weight multipliers --
           K0's protocol keeps the anchor term plain so A0/A1/B0 differ ONLY
           in the aux routing, never in the anchor).
  U      : Linear(D, 1) on the context vector, sigmoid -> predicted terminal
           U. Defined HERE, outside CNet2 (the BC_AUX_W aux-heads pattern:
           heads live with the entry, the exported backbone is untouched).
           Supervised by the generation's certified u_raw column -- the REAL
           terminal U of each row's game from that row's side, computed by
           the builder with U = 0.5*W + 0.5*sigmoid(margin/1000), W in
           {1, 0.5, 0} for margin > / = / < 0 (cnet2_build._u_of_margin;
           verbatim stage0_teacher._terminal_u / eval_v53 L493-494).
  death  : Linear(D, 3) on the per-slot token representations tok[:, :10]
           -> logits for "this unit is destroyed within 10 / 25 / 50 steps"
           (aux_labels KD horizons). Supervised by the K0 aux labels
           (aux_die / aux_die_m), which are loaded ONLY after the sidecar
           binding validation below passes.

Loss (every term normalised, denominators logged per update):
  L = L_anchor + L_U + lambda_death * L_death        (lambda_death = 0.05)
  L_anchor : sum of joint CE over valid (root, slot) actions / n_valid_slots
  L_U      : MSE(sigmoid(u_logit), u_raw) over roots with a usable finite
             u_raw / n_u_roots
  L_death  : BCEWithLogits over mask-valid (root, slot, horizon) entries /
             n_death_entries

Gradient routing (--aux-route):
  detach   (A0): the death head reads tok.detach() -- aux gradients reach
           the death head only, NEVER the backbone. The backbone is trained
           by anchor + U gradients alone.
  backbone (A1): the death head reads tok -- aux gradients flow into the
           shared backbone.
  B0 reference = --aux-route detach --lambda-death 0 (death term weighted
  exactly 0; logged but inert), or --freeze-backbone (no backbone parameter
  moves at all; only the K0 heads train). All three are this one entry.

A0/A1 protocol (built in, not convention):
  --seed drives everything: torch + numpy seeding, head initialisation, and
  the batch partition (one np.random.RandomState(seed) stream consumed in
  epoch order; the route flag never touches the RNG, so A0 and A1 see
  identical batches in identical order). Same --init weights, same
  optimizer (AdamW) and --lr, same --max-updates cap. The first log line is
  the effective config as one JSON object: init file sha256, data
  fingerprint (generation id + cfg_fp + row-order hash), seed, route,
  lambda, freeze flag, games used / present, root count.

Aux sidecar binding (gate 3 contract, consumer side). --aux-dir must hold
aux_die.npy, aux_die_m.npy and the producer's sidecar k0_aux.json, format
"cnet2-k0-aux/1" -- the single contract k0_aux_labels writes and this
entry validates (same file name, field names and hash compositions):
  {"format": "cnet2-k0-aux/1",
   "base_generation": <must equal the dataset's pinned generation id>,
   "lineage": <must equal the manifest cfg_fp -- the build lineage>,
   "rows": <must equal the manifest row total>,
   "row_key": {"fields": ["gid", "step"],
               "sha256": sha256 over the pinned generation's per-field
               file hashes (gid, step -- the manifest-named files,
               sha256 hex digests joined by "|") -- binds labels to the
               exact row order they were computed for},
   "files": {"aux_die.npy": {"sha256": ..., "shape": ..., "dtype": ...},
             "aux_die_m.npy": {...}} -- per-file entries as the derived
             layer writes them; the two files this entry consumes are
             hash-, shape- and dtype-checked,
   "games": [{gi, key, file, replay_sha256, ...}] -- per-game lineage;
             the game set and file paths must match the certified meta,
             and every replay the labels could come from (meta file
             ending .json.gz, present on disk) must carry a replay_sha256
             equal to the sha256 of the replay FILE BYTES -- the labels
             are bound to the exact replay contents, not to a meta
             description of them}
Any mismatch refuses the start, naming the failed item. No sidecar, no
training: there is no unbound-labels fallback in this entry.

Numeric protection & recovery (spec S19 standard, in code):
  P1 every weight/label tensor is isfinite-checked AT LOAD: init arrays,
     resume tensors (model AND optimizer state), u_raw, aux labels. A
     non-finite entry refuses the start and names the tensor. Sole documented exception: u_raw NaN is
     the builder's own "unusable game" mask -- a NaN is accepted ONLY
     where label_status == LABEL_UNUSABLE, is counted in the header, and
     such roots are excluded from every loss; Inf is never accepted.
  P2 in-training circuit breaker: a non-finite loss term or a non-finite
     gradient exits the process (exit code 3) WITHOUT saving anything.
  P3 checkpoint saving is tmp-file + os.replace (atomic), then a
     <ckpt>.complete.json marker (sha256 of the checkpoint) is written the
     same way, and only AFTER both does the "saved" record enter the
     train log -- a completion line in the log therefore always denotes a
     complete, hash-verified checkpoint on disk. Before anything is
     written, the whole payload (parameters, buffers AND optimizer
     state) must pass the finite check; a non-finite entry aborts the
     save through the P2 breaker, leaving the previous checkpoint and
     its marker untouched.
  P4 batch-stream reconstruction on resume: the batch partition stream
     is consumed ONLY by the per-epoch permutation draws, so its
     position after N completed updates is a pure function of N. A
     resumed run fast-forwards the stream past the completed epochs and
     skips the completed batches of the current epoch, consuming exactly
     the batch sequence the uninterrupted run would have consumed
     (same --bs and root set, as the update counter already presumes).
  P5 resume learning rate: the checkpoint's optimizer state is loaded
     for its moments, but its stored lr does not survive a resume --
     --lr is applied to the param groups right after load_state_dict,
     and every lr the run logs (the effective config, the per-step
     lines) is read back from the optimizer itself, so the recorded
     value is the effective value by construction.

Data: --data is a cnet2_build dataset directory with a sealed generation
(legacy flat datasets are refused: K0 needs the generation id for the
sidecar binding and u_raw for the U supervision). --max-games (default
128) caps the games used: the first max-games game ids present in the
dataset, in dataset order (seed-independent, identical for A0/A1); the
actual counts are printed in the effective config.

Usage:
  python3 k0_train.py --data DS --init INIT.npz --aux-dir DS \
      --out RUN/k0.pt --seed 7 --aux-route detach --max-updates 200
  python3 k0_train.py ... --selftest-routing   # A0/A1 gradient measurement
"""

import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

AUX_FORMAT = "cnet2-k0-aux/1"
AUX_SIDECAR = "k0_aux.json"
DEATH_HORIZONS = (10, 25, 50)


# --------------------------------------------------------------------------
# Pure helpers (no torch: unit-testable on any machine)
# --------------------------------------------------------------------------

def terminal_u(margin):
    """Real terminal U of a finished game from one side, given that side's
    final margin: U = 0.5*W + 0.5*sigmoid(margin/1000), W = 1 / 0.5 / 0 for
    margin > / = / < 0. Identical to cnet2_build._u_of_margin and to
    stage0_teacher._terminal_u / eval_v53 (normal-game rule)."""
    m = float(margin)
    w = 1.0 if m > 0 else (0.0 if m < 0 else 0.5)
    return 0.5 * w + 0.5 / (1.0 + math.exp(-m / 1000.0))


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def row_key_sha256(gen_dir, man):
    """Binding hash over the certified generation's row identity sequence,
    composed exactly as the producer (k0_aux_labels._row_key) writes it:
    sha256 of the per-field file sha256 hex digests (gid, step -- the
    manifest-named files) joined by "|". Binds labels to the exact row
    order they were computed for, with per-field domain separation."""
    fields = man.get("fields", {})
    parts = []
    for name in ("gid", "step"):
        info = fields.get(name)
        if info is None:
            raise SystemExit(f"k0 aux: generation manifest lacks field {name!r}; "
                             f"cannot verify the row-order binding; refusing to start")
        parts.append(_sha256_file(os.path.join(gen_dir, info["file"])))
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def check_aux_sidecar(aux_dir, man, meta, gen_dir, n_rows):
    """Validate the K0 aux bundle against the pinned generation, per the
    unified cnet2-k0-aux/1 contract the producer (k0_aux_labels) writes.
    Returns the parsed sidecar on success; raises SystemExit naming the
    failed binding item otherwise. Pure numpy/json: unit-testable
    without torch."""
    sc_path = os.path.join(aux_dir, AUX_SIDECAR)
    if not os.path.exists(sc_path):
        raise SystemExit(f"k0 aux: sidecar missing: {sc_path}; refusing to start "
                         f"(labels must be bound to the generation by {AUX_FORMAT})")
    sc = json.load(open(sc_path))
    if sc.get("format") != AUX_FORMAT:
        raise SystemExit(f"k0 aux: sidecar format {sc.get('format')!r} != {AUX_FORMAT!r}; refusing to start")
    if sc.get("base_generation") != man.get("generation"):
        raise SystemExit(f"k0 aux: sidecar base_generation {sc.get('base_generation')!r} != pinned generation "
                         f"{man.get('generation')!r}; refusing to start (labels bound to another generation)")
    if int(sc.get("rows", -1)) != int(n_rows) or int(sc.get("rows", -1)) != int(man["total"]):
        raise SystemExit(f"k0 aux: sidecar rows {sc.get('rows')!r} != dataset rows {n_rows} "
                         f"(manifest total {man['total']}); refusing to start")
    rk = sc.get("row_key") or {}
    if rk.get("fields") != ["gid", "step"]:
        raise SystemExit(f"k0 aux: sidecar row_key fields {rk.get('fields')!r} != ['gid', 'step']; "
                         f"refusing to start")
    ro = row_key_sha256(gen_dir, man)
    if rk.get("sha256") != ro:
        raise SystemExit(f"k0 aux: row_key sha256 mismatch (sidecar {rk.get('sha256')!r} "
                         f"!= generation {ro!r}); labels are not bound to this row order; refusing to start")
    if sc.get("lineage") != man.get("cfg_fp"):
        raise SystemExit(f"k0 aux: lineage mismatch (sidecar {sc.get('lineage')!r} != manifest cfg_fp "
                         f"{man.get('cfg_fp')!r}); refusing to start")
    # Replay binding: per-game lineage vs the certified meta. The labels
    # are bound to the replay FILE BYTES (the producer's per-game
    # replay_sha256), never to a meta description of the replays.
    meta_games = {g["gi"]: g for g in meta["games"]}
    lin = {e.get("gi"): e for e in sc.get("games") or []}
    if set(lin) != set(meta_games):
        raise SystemExit(f"k0 aux: sidecar lineage game set {sorted(lin)} != generation meta games "
                         f"{sorted(meta_games)}; refusing to start")
    for gi, ent in sorted(lin.items()):
        gm = meta_games[gi]
        if ent.get("file") != gm["file"]:
            raise SystemExit(f"k0 aux: game {gi}: lineage file {ent.get('file')!r} != meta file "
                             f"{gm['file']!r}; refusing to start")
        want = ent.get("replay_sha256")
        if gm["file"].endswith(".json.gz"):
            if not os.path.exists(gm["file"]):
                raise SystemExit(f"k0 aux: game {gi}: replay file missing: {gm['file']}; "
                                 f"labels bound to it cannot be verified; refusing to start")
            if want is None:
                raise SystemExit(f"k0 aux: game {gi}: sidecar lineage carries no replay_sha256 for "
                                 f"{gm['file']!r}; labels are not bound to the replay bytes; refusing to start")
            if _sha256_file(gm["file"]) != want:
                raise SystemExit(f"k0 aux: game {gi}: replay hash mismatch (replay replaced since "
                                 f"labelling); refusing to start")
        elif want is not None:
            if not os.path.exists(gm["file"]):
                raise SystemExit(f"k0 aux: game {gi}: replay file missing: {gm['file']}; refusing to start")
            if _sha256_file(gm["file"]) != want:
                raise SystemExit(f"k0 aux: game {gi}: replay hash mismatch (replay replaced since "
                                 f"labelling); refusing to start")
    for name in ("aux_die.npy", "aux_die_m.npy"):
        p = os.path.join(aux_dir, name)
        if not os.path.exists(p):
            raise SystemExit(f"k0 aux: label file missing: {p}; refusing to start")
        ent = (sc.get("files") or {}).get(name)
        if not isinstance(ent, dict) or ent.get("sha256") is None:
            raise SystemExit(f"k0 aux: sidecar files map has no sha256 entry for {name}; refusing to start")
        if _sha256_file(p) != ent["sha256"]:
            raise SystemExit(f"k0 aux: {name} sha256 mismatch vs sidecar; refusing to start")
    return sc


def require_finite_np(name, arr, allow_nan_mask=None):
    """S19-P1 for numpy arrays: every entry must be finite, except entries
    where allow_nan_mask is True (the builder's documented unusable mask
    for u_raw), where NaN -- never Inf -- is tolerated. Raises SystemExit
    naming the tensor and the first offending flat index."""
    a = np.asarray(arr)
    if not np.issubdtype(a.dtype, np.floating):
        return
    flat = a.reshape(-1).astype(np.float64)
    bad_inf = ~np.isfinite(flat) & ~np.isnan(flat)
    if bad_inf.any():
        i = int(np.argmax(bad_inf))
        raise SystemExit(f"k0 finite gate: tensor {name!r} has non-finite (Inf) entry at flat index {i}; "
                         f"refusing to start")
    nan = np.isnan(flat)
    if nan.any():
        if allow_nan_mask is None:
            i = int(np.argmax(nan))
            raise SystemExit(f"k0 finite gate: tensor {name!r} has NaN entry at flat index {i}; refusing to start")
        m = np.asarray(allow_nan_mask).reshape(-1).astype(bool)
        unexcused = nan & ~m
        if unexcused.any():
            i = int(np.argmax(unexcused))
            raise SystemExit(f"k0 finite gate: tensor {name!r} has NaN entry at flat index {i} that is NOT "
                             f"covered by the unusable-label mask; refusing to start")


# --------------------------------------------------------------------------
# Training entry (torch below this line)
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="K0 multi-head training entry (anchor + U + death, routed aux)")
    ap.add_argument("--data", required=True, help="cnet2_build dataset dir (sealed generation required)")
    ap.add_argument("--init", default=None, help="backbone init: cnet2_train export .npz")
    ap.add_argument("--resume", default=None, help="K0 checkpoint .pt to resume (heads + optimizer + counters)")
    ap.add_argument("--aux-dir", default=None, help="dir holding aux_die.npy/aux_die_m.npy + k0_aux.json "
                                                    "(the k0_aux_labels sidecar; default: the dataset dir)")
    ap.add_argument("--out", required=True, help="checkpoint output path (.pt)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--aux-route", choices=("detach", "backbone"), default="detach")
    ap.add_argument("--lambda-death", type=float, default=0.05)
    ap.add_argument("--freeze-backbone", action="store_true",
                    help="B0 form: no backbone parameter trains; only the K0 heads do")
    ap.add_argument("--max-updates", type=int, default=200)
    ap.add_argument("--max-games", type=int, default=128)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--cfg", default=None, help="CNet2 cfg JSON (default: init meta cfg or model DEFAULT)")
    ap.add_argument("--save-every", type=int, default=0, help="also save every N updates (0 = final only)")
    ap.add_argument("--grad-log-every", type=int, default=1,
                    help="measure per-term backbone grad norms every N updates")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--selftest-routing", action="store_true",
                    help="measure death-loss backbone grad under both routes on the first batch, print JSON, exit")
    args = ap.parse_args()
    if args.init is None and args.resume is None:
        raise SystemExit("k0: --init or --resume is required (A0/A1 protocol: a declared common start)")
    if args.init is not None and args.resume is not None:
        raise SystemExit("k0: --init and --resume are mutually exclusive")

    os.environ["BC_DEVICE"] = args.device          # cnet2_train reads this at import
    import torch
    import torch.nn.functional as Fn
    import cnet2_train as T
    from cnet2_build import LABEL_UNUSABLE
    from cnet2_model import CNet2, migrate_state_dict

    dev = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    def first_nonfinite(obj, path=""):
        """First dotted path whose floating tensor holds a non-finite
        entry, walking dicts/lists/tuples; None when all finite. Used
        for the optimizer state and the publish payload, whose tensors
        the per-parameter checks do not reach."""
        if torch.is_tensor(obj):
            if obj.dtype.is_floating_point and not bool(torch.isfinite(obj).all()):
                return path or "<root>"
            return None
        if isinstance(obj, dict):
            for k, v in obj.items():
                r = first_nonfinite(v, f"{path}.{k}" if path else str(k))
                if r is not None:
                    return r
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                r = first_nonfinite(v, f"{path}[{i}]")
                if r is not None:
                    return r
        return None

    # ---- data: sealed generation only (load() pins + hash-verifies it) ----
    D, is_val, meta = T.load(args.data)
    gen_dir = meta["dataset_source"]
    man_path = os.path.join(gen_dir, "manifest.json")
    if not os.path.exists(man_path):
        raise SystemExit(f"k0: dataset {args.data!r} has no sealed generation manifest at {man_path}; "
                         f"K0 requires a generation-pinned dataset (sidecar binding + u_raw); refusing to start")
    man = json.load(open(man_path))
    n_rows = int(man["total"])
    if int(np.asarray(D["y"]).shape[0]) != n_rows:
        raise SystemExit(f"k0: dataset rows {np.asarray(D['y']).shape[0]} != manifest total {n_rows}; refusing to start")

    # ---- U supervision: the certified u_raw column (real terminal U) ----
    if "u_raw" not in man.get("fields", {}):
        raise SystemExit(f"k0: generation {man.get('generation')} declares no 'u_raw' field; K0's U head is "
                         f"supervised by real terminal U only (label-aware generation required); refusing to start")
    u_raw = np.load(os.path.join(gen_dir, "u_raw.npy")).astype(np.float64)
    label_status = None
    if "label_status" in man.get("fields", {}):
        label_status = np.load(os.path.join(gen_dir, "label_status.npy"))
    unusable = (label_status == LABEL_UNUSABLE) if label_status is not None else None
    require_finite_np("u_raw", u_raw, allow_nan_mask=unusable)
    n_u_nan = int(np.isnan(u_raw).sum())
    if n_u_nan and unusable is None:
        raise SystemExit("k0 finite gate: u_raw has NaN entries but the generation carries no label_status "
                         "column to excuse them as unusable; refusing to start")

    # ---- aux labels: sidecar binding first, arrays second ----
    aux_dir = args.aux_dir or args.data
    sc = check_aux_sidecar(aux_dir, man, meta, gen_dir, n_rows)
    aux_die = np.load(os.path.join(aux_dir, "aux_die.npy"))
    aux_die_m = np.load(os.path.join(aux_dir, "aux_die_m.npy"))
    if aux_die.shape != (n_rows, 10, 3) or aux_die_m.shape != (n_rows, 10, 3):
        raise SystemExit(f"k0 aux: aux_die/aux_die_m shapes {aux_die.shape}/{aux_die_m.shape} != "
                         f"({n_rows}, 10, 3); refusing to start")
    for name, arr in (("aux_die.npy", aux_die), ("aux_die_m.npy", aux_die_m)):
        ent = sc["files"][name]
        if [int(x) for x in arr.shape] != ent.get("shape") or str(arr.dtype) != ent.get("dtype"):
            raise SystemExit(f"k0 aux: {name} shape/dtype {list(arr.shape)}/{arr.dtype} != sidecar entry "
                             f"{ent.get('shape')}/{ent.get('dtype')}; refusing to start")
    if not set(np.unique(aux_die).tolist()) <= {0, 1}:
        raise SystemExit("k0 finite gate: tensor 'aux_die' has values outside {0, 1}; refusing to start")
    require_finite_np("aux_die", aux_die.astype(np.float64))
    aux_die = aux_die.astype(np.float32)
    aux_die_m = aux_die_m.astype(bool)

    # ---- root set: train rows, game cap, unusable exclusion ----
    gid = np.asarray(D["gid"])
    gids_present = np.unique(gid)
    games_used = gids_present[: max(int(args.max_games), 0)]
    roots = np.nonzero(~is_val)[0]
    roots = roots[np.isin(gid[roots], games_used)]
    if unusable is not None:
        roots = roots[~unusable[roots]]
    if len(roots) == 0:
        raise SystemExit("k0: empty root set after game cap / unusable exclusion; refusing to start")

    # ---- model + heads (heads live here, backbone untouched) ----
    cfg = None
    init_sha = None
    start_sd = None
    if args.init:
        init_sha = _sha256_file(args.init)
        w = np.load(args.init)
        for k in w.files:
            if k != "meta":
                require_finite_np(f"init:{k}", w[k])
        if "meta" in w.files:
            try:
                cfg = json.loads(str(w["meta"])).get("cfg")
            except Exception:
                cfg = None
        start_sd = {k: torch.from_numpy(np.ascontiguousarray(w[k])) for k in w.files if k != "meta"}
    ck = None
    if args.resume:
        init_sha = _sha256_file(args.resume)
        marker_path = args.resume + ".complete.json"
        if not os.path.exists(marker_path):
            raise SystemExit(f"k0: resume checkpoint {args.resume!r} has no complete marker {marker_path!r}; "
                             f"an interrupted save is indistinguishable from a complete one; refusing to start")
        marker = json.load(open(marker_path))
        if marker.get("sha256") != init_sha or not marker.get("complete"):
            raise SystemExit(f"k0: complete marker for {args.resume!r} does not match the file sha256 "
                             f"(marker {marker.get('sha256')!r} != actual {init_sha!r}); refusing to start")
        ck = torch.load(args.resume, map_location=dev, weights_only=False)
        for nm, sd in (("backbone", ck["backbone"]), ("u_head", ck["u_head"]), ("death_head", ck["death_head"])):
            for tn, tv in sd.items():
                if torch.is_tensor(tv) and not bool(torch.isfinite(tv).all()):
                    raise SystemExit(f"k0 finite gate: resume tensor {nm}:{tn} is non-finite; refusing to start")
        bad_opt = first_nonfinite(ck["opt"], "opt")
        if bad_opt is not None:
            raise SystemExit(f"k0 finite gate: resume tensor {bad_opt} is non-finite; refusing to start")
        if cfg is None:
            cfg = ck.get("cfg")
    if args.cfg:
        cfg = json.loads(args.cfg)
    net = CNet2(**(cfg or {})).to(dev)
    if start_sd is not None:
        mig = migrate_state_dict(net, start_sd, allow_missing_prefixes=("tattn.",), source="--init")
    else:
        mig = {"migrated_missing": [], "new_head_init": []}
    dim = net.cfg["D"]
    u_head = torch.nn.Linear(dim, 1).to(dev)
    death_head = torch.nn.Linear(dim, 3).to(dev)

    backbone_params = [p for _, p in net.named_parameters()]
    if args.freeze_backbone:
        for p in backbone_params:
            p.requires_grad_(False)
    trainable = [p for p in backbone_params if p.requires_grad] + list(u_head.parameters()) + list(death_head.parameters())
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)

    updates_done = 0
    if ck is not None:
        net.load_state_dict(ck["backbone"])
        u_head.load_state_dict(ck["u_head"])
        death_head.load_state_dict(ck["death_head"])
        opt.load_state_dict(ck["opt"])
        # P5: load_state_dict restores the checkpoint's param_groups, lr
        # included -- an explicit --lr on the resume command line was
        # silently discarded while every log line kept printing it (a run
        # recorded .01 and optimized at the checkpoint's .001). The CLI
        # value is the caller's intent: apply it to every group. The
        # moments still come from the checkpoint; only the rate is the
        # caller's.
        for _g in opt.param_groups:
            _g["lr"] = args.lr
        updates_done = int(ck["updates"])
    # The lr every record below carries is read back from the optimizer
    # itself, so the recorded value is the effective value by
    # construction, on the fresh path (== args.lr) and on resume alike.
    eff_lr = float(opt.param_groups[0]["lr"])

    # ---- effective config: one JSON line, first in the log ----
    eff = {"k0_effective": {
        "start": ("resume" if args.resume else "init"),
        "start_sha256": init_sha,
        "data": args.data,
        "data_fingerprint": {"generation": man.get("generation"), "cfg_fp": man.get("cfg_fp"),
                             "rows": n_rows, "row_key_sha256": row_key_sha256(gen_dir, man)},
        "aux": {"dir": aux_dir, "sidecar_format": sc.get("format"), "lineage": sc.get("lineage")},
        "seed": args.seed, "route": args.aux_route, "lambda_death": args.lambda_death,
        "freeze_backbone": bool(args.freeze_backbone),
        "lr": eff_lr, "bs": args.bs, "max_updates": args.max_updates,
        "games_in_dataset": len(meta["games"]), "games_present": int(len(gids_present)),
        "games_used": int(len(games_used)), "max_games": args.max_games,
        "roots": int(len(roots)), "u_nan_unusable_rows": n_u_nan,
        "init_migrated_missing": mig["migrated_missing"], "device": args.device}}
    log_path = args.out + ".log.jsonl"
    log = open(log_path, "w")

    def emit(obj):
        line = json.dumps(obj)
        print(line, flush=True)
        log.write(line + "\n"); log.flush()

    def die_nonfinite(what, value):
        emit({"circuit_breaker": {"what": what, "value": str(value), "updates_done": updates_done,
                                  "action": "exit(3) without saving"}})
        log.close()
        sys.exit(3)

    emit(eff)

    # ---- batching: seed-derived partition, route-independent ----
    rs = np.random.RandomState(args.seed)
    # P4: reconstruct the stream position on resume (see header). The
    # stream is consumed only by the per-epoch permutation draws below,
    # so completed epochs are replayed as draws and the completed
    # batches of the current epoch are skipped in the loop.
    batches_per_epoch = (len(roots) + args.bs - 1) // args.bs
    epoch = 0
    skip_batches = 0
    if ck is not None and batches_per_epoch > 0:
        epoch, skip_batches = divmod(int(updates_done), batches_per_epoch)
        for _ in range(epoch):
            rs.permutation(roots)
    u_t = torch.from_numpy(u_raw.astype(np.float32))
    die_t = torch.from_numpy(aux_die)
    die_m_t = torch.from_numpy(aux_die_m)

    def make_batch(idx):
        idx = np.sort(idx)
        b = T.to_dev(T.fetch(D, idx), False)
        return b, idx

    def forward_terms(b, idx, route):
        tok, ctx, _ = net.encode(b)
        logits = net.decode(b, tok, ctx)
        y = b["y"]
        valid = y >= 0
        ce = Fn.cross_entropy(logits.reshape(-1, 5), y.clamp(min=0).reshape(-1),
                              reduction="none").reshape(y.shape)
        n_anchor = int(valid.sum())
        l_anchor = (ce * valid.float()).sum() / max(n_anchor, 1)
        u_pred = torch.sigmoid(u_head(ctx)).squeeze(1)
        u_tg = u_t.to(dev)[torch.from_numpy(idx).to(dev)]
        u_ok = torch.isfinite(u_tg)
        n_u = int(u_ok.sum())
        l_u = Fn.mse_loss(u_pred[u_ok], u_tg[u_ok]) if n_u else torch.zeros((), device=dev)
        tok_d = tok[:, :10].detach() if route == "detach" else tok[:, :10]
        d_logits = death_head(tok_d)
        d_tg = die_t.to(dev)[torch.from_numpy(idx).to(dev)]
        d_m = die_m_t.to(dev)[torch.from_numpy(idx).to(dev)]
        n_d = int(d_m.sum())
        l_death = Fn.binary_cross_entropy_with_logits(
            d_logits[d_m], d_tg[d_m]) if n_d else torch.zeros((), device=dev)
        return (l_anchor, l_u, l_death), (n_anchor, n_u, n_d)

    def backbone_grad_norm(term):
        live = [p for p in backbone_params if p.requires_grad]
        if not live or not term.requires_grad:
            return 0.0
        gs = torch.autograd.grad(term, live, retain_graph=True, allow_unused=True)
        sq = 0.0
        for g in gs:
            if g is not None:
                sq += float((g.float() ** 2).sum())
        return math.sqrt(sq)

    # ---- selftest: same batch, same weights, both routes ----
    if args.selftest_routing:
        first = roots[: args.bs]
        b, idx = make_batch(first)
        out = {}
        for route in ("detach", "backbone"):
            (_, _, l_death), _ = forward_terms(b, idx, route)
            out[route] = backbone_grad_norm(l_death)
        ok = out["detach"] == 0.0 and out["backbone"] > 0.0
        emit({"selftest_routing": {"batch_roots": int(len(first)),
                                   "death_backbone_grad": out, "pass": bool(ok)}})
        log.close()
        sys.exit(0 if ok else 1)

    # ---- checkpoint: S19-P3 atomic save + complete marker, log last ----
    def save_checkpoint(updates):
        for nm, p in list(net.named_parameters()) + list(u_head.named_parameters()) + list(death_head.named_parameters()):
            if not bool(torch.isfinite(p.detach()).all()):
                die_nonfinite(f"param:{nm}", "non-finite at save time")
        payload = {"backbone": net.state_dict(), "u_head": u_head.state_dict(),
                   "death_head": death_head.state_dict(), "opt": opt.state_dict(),
                   "updates": int(updates), "cfg": net.cfg,
                   "effective": eff["k0_effective"]}
        for nm in ("backbone", "u_head", "death_head", "opt"):
            bad = first_nonfinite(payload[nm], nm)
            if bad is not None:
                die_nonfinite(f"publish:{bad}", "non-finite at save time")
        tmp = f"{args.out}.tmp-{os.getpid()}"
        torch.save(payload, tmp)
        os.replace(tmp, args.out)
        sha = _sha256_file(args.out)
        marker = {"checkpoint": os.path.basename(args.out), "sha256": sha,
                  "updates": int(updates), "complete": True, "time": time.time()}
        mtmp = f"{args.out}.complete.json.tmp-{os.getpid()}"
        with open(mtmp, "w") as fh:
            json.dump(marker, fh)
        os.replace(mtmp, args.out + ".complete.json")
        emit({"saved": marker})          # completion enters the log only now

    # ---- training loop ----
    net.train()
    t0 = time.time()
    while updates_done < args.max_updates:
        perm = rs.permutation(roots)
        epoch += 1
        for bi, a0 in enumerate(range(0, len(perm), args.bs)):
            if bi < skip_batches:
                continue
            if updates_done >= args.max_updates:
                break
            idx = perm[a0:a0 + args.bs]
            b, idx = make_batch(idx)
            (l_anchor, l_u, l_death), (n_anchor, n_u, n_d) = forward_terms(b, idx, args.aux_route)
            total = l_anchor + l_u + args.lambda_death * l_death
            for nm, tv in (("anchor", l_anchor), ("u", l_u), ("death", l_death), ("total", total)):
                if not bool(torch.isfinite(tv.detach()).all()):
                    die_nonfinite(f"loss:{nm}", float(tv.detach()))
            measure = (updates_done % max(args.grad_log_every, 1) == 0)
            g_bb = {}
            if measure:
                g_bb = {"anchor": backbone_grad_norm(l_anchor), "u": backbone_grad_norm(l_u),
                        "death": backbone_grad_norm(l_death)}
            opt.zero_grad(set_to_none=True)
            total.backward()
            for nm, p in list(net.named_parameters()) + list(u_head.named_parameters()) + list(death_head.named_parameters()):
                if p.grad is not None and not bool(torch.isfinite(p.grad).all()):
                    die_nonfinite(f"grad:{nm}", "non-finite gradient")
            opt.step()
            updates_done += 1
            emit({"update": updates_done, "epoch": epoch,
                  "loss": {"anchor": float(l_anchor.detach()), "u": float(l_u.detach()),
                           "death": float(l_death.detach()), "total": float(total.detach())},
                  "denominators": {"anchor_slots": n_anchor, "u_roots": n_u, "death_entries": n_d},
                  "grad_backbone": g_bb, "lr": eff_lr, "elapsed_s": round(time.time() - t0, 2)})
            if args.save_every and updates_done % args.save_every == 0 and updates_done < args.max_updates:
                save_checkpoint(updates_done)
        skip_batches = 0
    save_checkpoint(updates_done)
    emit({"done": {"updates": updates_done, "epochs": epoch, "elapsed_s": round(time.time() - t0, 2)}})
    log.close()


if __name__ == "__main__":
    main()

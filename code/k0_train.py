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
  P5 resume learning rate: --lr defaults to None, a sentinel rather
     than a rate. A fresh start reads None as 1e-3, as before. On
     resume, the checkpoint's optimizer state is loaded for its
     moments and -- when --lr is not given -- for its rate too: a bare
     resume must not silently reset the rate to the fresh-start
     default (the mirror image of the split this paragraph exists to
     prevent). An explicit --lr is applied to the param groups right
     after load_state_dict. Every lr the run logs (the effective
     config, the per-step lines) is read back from the optimizer
     itself, so the recorded value is the effective value by
     construction.
  P6 init/resume identity is single-source: --init and --resume are
     each read ONCE into memory; the recorded sha256 (and, for resume,
     the complete-marker comparison) is computed from those bytes, and
     np.load / torch.load parse the same byte object via BytesIO. The
     file cannot be swapped between hashing and loading -- the window
     the certify_pack and cnet2_train init paths were closed for.
  P7 resume protocol equivalence: a resume must continue the run the
     checkpoint actually came from. At load, the stored effective
     config's protocol fields -- route, lambda_death, seed, bs,
     max_games, freeze_backbone, and the data fingerprint (generation
     + row_key_sha256, compared when the record carries them), and
     the aux content identity (the sidecar's byte sha256 plus the
     label files' sha256, also compared when carried -- the sidecar's
     internal checks keep a regenerated bundle self-consistent, so
     without this binding the death targets could be swapped under a
     resume and the run would silently change experiment), and the
     game-subset identity (games_used_sha256, recomputed for this
     invocation and compared against the record -- stricter than the
     fingerprint class: a checkpoint carrying no subset record at all
     is refused unless --accept-unrecorded-subset migrates it, the
     acceptance being recorded) -- are
     checked against this invocation; any mismatch refuses the start,
     naming the field and both values. Exempt by design, as the
     legitimate axes of a continuation: --max-updates, --out,
     --save-every, --resume itself, --lr (P5's rule), --init (not
     applicable to a resume). A checkpoint with no effective record
     is refused outright: an unverifiable resume does not start.

Data: --data is a cnet2_build dataset directory with a sealed generation
(legacy flat datasets are refused: K0 needs the generation id for the
sidecar binding and u_raw for the U supervision). --max-games (default
128) caps the games used: at or above the generation's game count every
game is used (games_sampling "all" in the effective config); below it, a
seeded deterministic sample of that many game ids is drawn from --seed
(games_sampling "seeded") -- never the enumeration-order prefix, which
is the build scan order and so a biased subset. The selected set's
sha256 (games_used_sha256) is recorded in the effective config, and
the P7 resume assertion pins the recomputed value itself, so a resume
provably continues on the producing run's game set -- not merely on
the same seed + cap under whatever selection algorithm the current
code happens to run (W01: the algorithm changed once already, at B2).
A checkpoint from before subset recording (the prefix-selection era)
carries no games_used_sha256: resuming it is refused by default --
its training subset is unverifiable -- unless
--accept-unrecorded-subset is given, in which case the acceptance is
recorded in the new run's effective config (subset_migration).

Stage-A screening (--screen): an eval-only mode of this same entry. It
takes an arm checkpoint (--checkpoint, bound by its complete marker) and
the common --init, walks the SAME sealed data path as training
(generation seal, u_raw finite gate, aux sidecar binding -- any failure
refuses the run, naming the item), and evaluates over the WHOLE
generation the --data pointer binds (no game cap, no split filter: the
screen judges the bound dataset as it stands; holdout-ness is the
caller's property, recorded via the data fingerprint). One JSON line is
printed: for the death head, per-horizon (10 / 25 / 50) and pooled
log-loss against the sidecar labels over mask-valid entries -- the
training loss's own alignment -- with the empirical base-rate log-loss
and the AUC alongside; and the per-root KL(pi_arm || pi_init) of the
policy head (teacher-forced per-slot categoricals summed over the
root's valid action slots, the anchor CE's own factorisation), as
mean / p99. No optimizer is built and nothing is written. A horizon
whose valid entries lack positives or negatives reports a null AUC and
names the missing class instead of inventing a number. Inputs are gated finite on
load, and the quantities the metrics are computed from (the death
probabilities and the per-root KL) plus the metrics themselves are
gated again after evaluation: a checkpoint whose finite parameters
overflow in the forward pass refuses the run, naming the metric,
instead of printing NaN as a result; the emitted line is strict
JSON (allow_nan=False).

Usage:
  python3 k0_train.py --data DS --init INIT.npz --aux-dir DS \
      --out RUN/k0.pt --seed 7 --aux-route detach --max-updates 200
  python3 k0_train.py ... --selftest-routing   # A0/A1 gradient measurement
  python3 k0_train.py --screen --checkpoint ARM.pt --init INIT.npz \
      --data HOLDOUT_DS     # Stage-A screen, one JSON line, no writes
"""

import argparse
import contextlib
import hashlib
import io
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


def _read_bytes(path):
    """Read a file's bytes exactly once. Identity-bound loads (screen
    checkpoint / init) hash, marker-check and parse THESE bytes, so the
    recorded identity and the loaded object have a single source -- the
    file cannot be swapped between hashing and loading (the
    certify_pack / cnet2_train single-source caliber)."""
    with open(path, "rb") as fh:
        return fh.read()


def _read_aux_bundle(aux_dir):
    """Read the aux bundle's bytes exactly once per file (the sidecar
    plus the two label files). Every consumer of the bundle in this
    process -- the sidecar checks, the label parse and the recorded
    content identity -- derives from THESE bytes, so the labels a run
    trains on and the identity it records are the same bytes by
    construction (W02: previously the labels were parsed from one read
    and the identity hashed from a later by-path re-read, so a bundle
    swapped in between was trained as A and recorded as B). A file
    that does not exist is simply absent from the dict; the sidecar
    check names it with its usual refusal."""
    raw = {}
    for name in (AUX_SIDECAR, "aux_die.npy", "aux_die_m.npy"):
        p = os.path.join(aux_dir, name)
        if os.path.exists(p):
            raw[name] = _read_bytes(p)
    return raw


def _load_gen_field(gen_dir, man, name, tag, refuse):
    """One manifest-declared generation field (u_raw / label_status),
    single-read (N2): the bytes are read ONCE, the manifest sha256 is
    checked against those exact bytes, and np.load parses the same
    buffer. Previously these two K0-consumed columns were re-opened by
    path AFTER T.load() had pinned and hash-verified the generation,
    so a file swapped inside that window was trained / screened on
    under the sealed generation's identity. `tag` / `refuse` carry the
    caller's refusal wording ("k0" / "refusing to start" on the
    training path, "k0 screen" / "refusing to run" on the screen
    path)."""
    info = (man.get("fields") or {}).get(name) or {}
    p = os.path.join(gen_dir, info.get("file") or (name + ".npy"))
    raw = _read_bytes(p)
    want = info.get("sha256")
    if want is not None and hashlib.sha256(raw).hexdigest() != want:
        raise SystemExit(
            f"{tag}: generation field {name!r} failed verification: {p} (sha256 mismatch -- "
            f"file bytes changed after the generation was sealed); {refuse}")
    return np.load(io.BytesIO(raw))


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


def check_aux_sidecar(aux_dir, man, meta, gen_dir, n_rows, prefetched=None):
    """Validate the K0 aux bundle against the pinned generation, per the
    unified cnet2-k0-aux/1 contract the producer (k0_aux_labels) writes.
    Returns the parsed sidecar on success; raises SystemExit naming the
    failed binding item otherwise. Pure numpy/json: unit-testable
    without torch. When prefetched (name -> bytes, from
    _read_aux_bundle) carries a file, the sidecar parse and the
    label-file hash checks derive from those bytes instead of
    re-reading by path -- the caller's single read is then the only
    read of the bundle (W02)."""
    sc_path = os.path.join(aux_dir, AUX_SIDECAR)
    if prefetched is not None and prefetched.get(AUX_SIDECAR) is not None:
        sc = json.loads(prefetched[AUX_SIDECAR])
    else:
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
        ent = (sc.get("files") or {}).get(name)
        if not isinstance(ent, dict) or ent.get("sha256") is None:
            raise SystemExit(f"k0 aux: sidecar files map has no sha256 entry for {name}; refusing to start")
        if prefetched is not None and prefetched.get(name) is not None:
            actual = hashlib.sha256(prefetched[name]).hexdigest()
        else:
            if not os.path.exists(p):
                raise SystemExit(f"k0 aux: label file missing: {p}; refusing to start")
            actual = _sha256_file(p)
        if actual != ent["sha256"]:
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
    ap.add_argument("--accept-unrecorded-subset", action="store_true",
                    help="resume migration only: accept a checkpoint carrying no games_used_sha256 subset "
                         "record (pre-B2, prefix-selection era), although its training subset cannot be "
                         "verified against this invocation's; the acceptance is recorded in the new "
                         "checkpoint's effective config (subset_migration)")
    ap.add_argument("--aux-dir", default=None, help="dir holding aux_die.npy/aux_die_m.npy + k0_aux.json "
                                                    "(the k0_aux_labels sidecar; default: the dataset dir)")
    ap.add_argument("--out", default=None, help="checkpoint output path (.pt); required unless --screen")
    ap.add_argument("--screen", action="store_true",
                    help="eval-only Stage-A screen of --checkpoint against --init on --data; "
                         "prints one JSON line, writes nothing")
    ap.add_argument("--checkpoint", default=None,
                    help="arm checkpoint .pt to screen (with --screen; bound by its complete marker)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--aux-route", choices=("detach", "backbone"), default="detach")
    ap.add_argument("--lambda-death", type=float, default=0.05)
    ap.add_argument("--freeze-backbone", action="store_true",
                    help="B0 form: no backbone parameter trains; only the K0 heads do")
    ap.add_argument("--max-updates", type=int, default=200)
    ap.add_argument("--max-games", type=int, default=128)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=None,
                    help="learning rate (default: 1e-3 on a fresh start; on resume, the checkpoint's "
                         "rate unless explicitly given)")
    ap.add_argument("--cfg", default=None, help="CNet2 cfg JSON (default: init meta cfg or model DEFAULT)")
    ap.add_argument("--save-every", type=int, default=0, help="also save every N updates (0 = final only)")
    ap.add_argument("--grad-log-every", type=int, default=1,
                    help="measure per-term backbone grad norms every N updates")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--selftest-routing", action="store_true",
                    help="measure death-loss backbone grad under both routes on the first batch, print JSON, exit")
    args = ap.parse_args()
    if args.screen:
        if args.checkpoint is None or args.init is None:
            raise SystemExit("k0 --screen: --checkpoint (the arm) and --init (the common start) are "
                             "both required")
        if args.resume is not None:
            raise SystemExit("k0 --screen: --resume is not a screen input; the arm is given by --checkpoint")
        run_screen(args)
        return
    if args.out is None:
        raise SystemExit("k0: --out is required (training writes a checkpoint; only --screen runs without one)")
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
    u_raw = _load_gen_field(gen_dir, man, "u_raw", "k0", "refusing to start").astype(np.float64)
    label_status = None
    if "label_status" in man.get("fields", {}):
        label_status = _load_gen_field(gen_dir, man, "label_status", "k0", "refusing to start")
    unusable = (label_status == LABEL_UNUSABLE) if label_status is not None else None
    require_finite_np("u_raw", u_raw, allow_nan_mask=unusable)
    n_u_nan = int(np.isnan(u_raw).sum())
    if n_u_nan and unusable is None:
        raise SystemExit("k0 finite gate: u_raw has NaN entries but the generation carries no label_status "
                         "column to excuse them as unusable; refusing to start")

    # ---- aux labels: ONE read of the bundle (W02); the sidecar
    # binding, the label parse and the identity below all derive
    # from those bytes, never from a second by-path read ----
    aux_dir = args.aux_dir or args.data
    aux_raw = _read_aux_bundle(aux_dir)
    sc = check_aux_sidecar(aux_dir, man, meta, gen_dir, n_rows, prefetched=aux_raw)
    aux_die = np.load(io.BytesIO(aux_raw["aux_die.npy"]))
    aux_die_m = np.load(io.BytesIO(aux_raw["aux_die_m.npy"]))
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

    # ---- aux content identity (V07) ----
    # The sidecar checks above bind the label files to the sidecar and
    # the sidecar to the generation, but they do not bind the bundle to
    # the run that produced a checkpoint: a regenerated bundle -- labels
    # edited, the sidecar's own file hashes updated to match -- stays
    # self-consistent, and a resume would silently continue a DIFFERENT
    # experiment (the death targets are the labels). So the bundle's
    # content identity -- the sidecar's bytes and the label files'
    # bytes -- is computed here, recorded in the effective config, and
    # asserted on resume (P7, fingerprint class: compared when the
    # record carries it).
    # W02: the identity is hashed from aux_raw -- the single read the
    # labels above were parsed from -- never re-read by path. A bundle
    # swapped on disk after that read changes neither the labels this
    # run trains on nor the identity it records; the pre-fix code
    # hashed the files again here, after the parse, and could record
    # the swapped bundle's hash against the original bundle's labels.
    aux_identity = {"sidecar_sha256": hashlib.sha256(aux_raw[AUX_SIDECAR]).hexdigest(),
                    "files_sha256": {name: hashlib.sha256(aux_raw[name]).hexdigest()
                                     for name in ("aux_die.npy", "aux_die_m.npy")}}

    # ---- root set: train rows, game cap, unusable exclusion ----
    gid = np.asarray(D["gid"])
    gids_present = np.unique(gid)
    _cap = max(int(args.max_games), 0)
    if _cap >= len(gids_present):
        games_used = gids_present
        games_sampling = "all"
    else:
        # B2: a cap below the generation's game count takes a seeded
        # deterministic sample, never the enumeration-order prefix --
        # np.unique order is the build scan order (earliest corpora /
        # teachers first), so a prefix is a biased subset that merely
        # looks uniform. The sample is a pure function of (seed,
        # max_games, gids_present), drawn on a dedicated stream so it
        # cannot perturb the batching stream built from the same seed
        # below; the P7 resume assertion pins the recomputed subset's
        # sha256 against the checkpoint's record (W01), so a resume
        # provably recomputes the identical subset -- seed + max_games
        # alone would not pin it across a selection-algorithm change.
        # Sorted after the draw so downstream consumers see the same
        # canonical order as the uncapped path.
        _grs = np.random.RandomState((int(args.seed) * 1000003 + 977) % (2 ** 31))
        games_used = np.sort(_grs.choice(gids_present, size=_cap, replace=False))
        games_sampling = "seeded"
    games_used_sha256 = hashlib.sha256(
        np.asarray(games_used, dtype=np.int64).tobytes()).hexdigest()
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
        # P6: a single byte read -- the recorded sha256 and the np.load
        # parse derive from the same bytes.
        init_raw = _read_bytes(args.init)
        init_sha = hashlib.sha256(init_raw).hexdigest()
        w = np.load(io.BytesIO(init_raw))
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
        # P6: a single byte read -- the recorded sha256, the complete
        # marker comparison and the torch.load parse all derive from
        # the same bytes.
        ck_raw = _read_bytes(args.resume)
        init_sha = hashlib.sha256(ck_raw).hexdigest()
        marker_path = args.resume + ".complete.json"
        if not os.path.exists(marker_path):
            raise SystemExit(f"k0: resume checkpoint {args.resume!r} has no complete marker {marker_path!r}; "
                             f"an interrupted save is indistinguishable from a complete one; refusing to start")
        marker = json.load(open(marker_path))
        if marker.get("sha256") != init_sha or not marker.get("complete"):
            raise SystemExit(f"k0: complete marker for {args.resume!r} does not match the file sha256 "
                             f"(marker {marker.get('sha256')!r} != actual {init_sha!r}); refusing to start")
        ck = torch.load(io.BytesIO(ck_raw), map_location=dev, weights_only=False)
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
    opt = torch.optim.AdamW(trainable, lr=args.lr if args.lr is not None else 1e-3, weight_decay=1e-4)

    updates_done = 0
    subset_migration = None
    if ck is not None:
        # P7 first, before any load_state_dict: the assertion must also
        # pre-empt the bare load errors a mismatched continuation would
        # otherwise die with -- a freeze_backbone mismatch, e.g., changes
        # the optimizer's parameter set, and opt.load_state_dict would
        # raise a plain ValueError naming nothing.
        # The resume must continue THIS run -- compare the protocol
        # fields of the checkpoint's stored effective config against
        # this invocation and refuse any mismatch, naming the field and
        # both values. Exempt by design (see header): --max-updates,
        # --out, --save-every, --resume itself, --lr (P5's rule
        # governs it) and --init (not applicable to a resume). The
        # data fingerprint fields and the aux content identity are
        # compared when the record carries them; the game-subset
        # identity (W01) is stricter -- compared when carried, and a
        # missing record refuses the start unless explicitly migrated.
        _eff0 = ck.get("effective")
        if _eff0 is None:
            raise SystemExit("k0: resume checkpoint carries no 'effective' record; the resume protocol "
                             "cannot be verified against the producing run; refusing to start")
        _proto = (("aux_route (--aux-route)", args.aux_route, _eff0.get("route")),
                  ("lambda_death (--lambda-death)", args.lambda_death, _eff0.get("lambda_death")),
                  ("seed (--seed)", args.seed, _eff0.get("seed")),
                  ("bs (--bs)", args.bs, _eff0.get("bs")),
                  ("max_games (--max-games)", args.max_games, _eff0.get("max_games")),
                  ("freeze_backbone (--freeze-backbone)", bool(args.freeze_backbone),
                   _eff0.get("freeze_backbone")))
        for _name, _mine_v, _ck_v in _proto:
            if _ck_v != _mine_v:
                raise SystemExit(f"k0 resume protocol: {_name} differs from the checkpoint's producing run "
                                 f"(checkpoint {_ck_v!r} != this invocation {_mine_v!r}); refusing to start")
        _fp0 = _eff0.get("data_fingerprint") or {}
        if _fp0.get("generation") is not None and _fp0.get("generation") != man.get("generation"):
            raise SystemExit(f"k0 resume protocol: data_fingerprint.generation differs from the checkpoint's "
                             f"producing run (checkpoint {_fp0.get('generation')!r} != this invocation "
                             f"{man.get('generation')!r}); refusing to start")
        if _fp0.get("row_key_sha256") is not None and \
                _fp0.get("row_key_sha256") != row_key_sha256(gen_dir, man):
            raise SystemExit(f"k0 resume protocol: data_fingerprint.row_key_sha256 differs from the "
                             f"checkpoint's producing run (checkpoint {_fp0.get('row_key_sha256')!r} != this "
                             f"invocation {row_key_sha256(gen_dir, man)!r}); refusing to start")
        # V07: the aux bundle's content identity joins the fingerprint
        # class -- a relabelled bundle is a new experiment, not a
        # continuation, however self-consistent its sidecar is.
        _aux0 = _eff0.get("aux") or {}
        if _aux0.get("sidecar_sha256") is not None and \
                _aux0.get("sidecar_sha256") != aux_identity["sidecar_sha256"]:
            raise SystemExit(f"k0 resume protocol: aux.sidecar_sha256 differs from the checkpoint's "
                             f"producing run (checkpoint {_aux0.get('sidecar_sha256')!r} != this "
                             f"invocation {aux_identity['sidecar_sha256']!r}); refusing to start")
        _auxf0 = _aux0.get("files_sha256") or {}
        for _nm in ("aux_die.npy", "aux_die_m.npy"):
            if _auxf0.get(_nm) is not None and _auxf0.get(_nm) != aux_identity["files_sha256"][_nm]:
                raise SystemExit(f"k0 resume protocol: aux.files_sha256[{_nm}] differs from the "
                                 f"checkpoint's producing run (checkpoint {_auxf0.get(_nm)!r} != this "
                                 f"invocation {aux_identity['files_sha256'][_nm]!r}); refusing to start")
        # W01: the game-subset identity. Seed + max_games (asserted
        # above) pin the selection INPUTS, but the subset is a function
        # of the selection ALGORITHM, which changed at B2 (prefix ->
        # seeded): a prefix-era checkpoint resumed under the seeded
        # algorithm with identical seed + cap silently continued on a
        # different game set -- same protocol fields, different data.
        # So the recomputed subset sha is asserted against the record
        # itself. A checkpoint with no record predates subset
        # recording: its subset is unverifiable, and the resume is
        # refused unless the caller explicitly migrates it with
        # --accept-unrecorded-subset; the acceptance is then written
        # into the new run's effective config (subset_migration).
        _sub0 = _eff0.get("games_used_sha256")
        if _sub0 is None:
            if not args.accept_unrecorded_subset:
                raise SystemExit(f"k0 resume protocol: checkpoint carries no games_used_sha256 subset "
                                 f"record (it predates subset recording -- the prefix-selection era "
                                 f"before B2); its training subset cannot be verified against this "
                                 f"invocation's subset {games_used_sha256!r} (sampling {games_sampling!r}); "
                                 f"refusing to start -- pass --accept-unrecorded-subset to migrate it "
                                 f"explicitly (the acceptance is recorded in the new checkpoint)")
            subset_migration = {"accept_unrecorded_subset": True,
                                "checkpoint_games_sampling": _eff0.get("games_sampling"),
                                "invocation_games_sampling": games_sampling,
                                "invocation_games_used_sha256": games_used_sha256}
        elif _sub0 != games_used_sha256:
            raise SystemExit(f"k0 resume protocol: games_used_sha256 (subset identity) differs from the "
                             f"checkpoint's producing run (checkpoint {_sub0!r}, sampling "
                             f"{_eff0.get('games_sampling')!r} != this invocation {games_used_sha256!r}, "
                             f"sampling {games_sampling!r}); the resume would train on a different game "
                             f"set; refusing to start")
        net.load_state_dict(ck["backbone"])
        u_head.load_state_dict(ck["u_head"])
        death_head.load_state_dict(ck["death_head"])
        opt.load_state_dict(ck["opt"])
        # P5: load_state_dict restores the checkpoint's param_groups, lr
        # included. An explicit --lr is the caller's intent: apply it to
        # every group (the pre-fix code silently discarded it while
        # every log line kept printing it -- a run recorded .01 and
        # optimized at the checkpoint's .001). No --lr (the None
        # sentinel) keeps the checkpoint's rate: a bare resume must not
        # silently reset it to the fresh-start default either. The
        # moments come from the checkpoint in both cases.
        if args.lr is not None:
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
        "aux": {"dir": aux_dir, "sidecar_format": sc.get("format"), "lineage": sc.get("lineage"),
                "sidecar_sha256": aux_identity["sidecar_sha256"],
                "files_sha256": aux_identity["files_sha256"]},
        "seed": args.seed, "route": args.aux_route, "lambda_death": args.lambda_death,
        "freeze_backbone": bool(args.freeze_backbone),
        "lr": eff_lr, "bs": args.bs, "max_updates": args.max_updates,
        "games_in_dataset": len(meta["games"]), "games_present": int(len(gids_present)),
        "games_used": int(len(games_used)), "max_games": args.max_games,
        "games_sampling": games_sampling, "games_used_sha256": games_used_sha256,
        "roots": int(len(roots)), "u_nan_unusable_rows": n_u_nan,
        "init_migrated_missing": mig["migrated_missing"], "device": args.device}}
    if subset_migration is not None:
        eff["k0_effective"]["subset_migration"] = subset_migration
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


# --------------------------------------------------------------------------
# Stage-A screening (--screen): eval-only, no optimizer, no writes
# --------------------------------------------------------------------------

def _screen_metrics(p, y):
    """Log-loss of predicted probabilities p against 0/1 labels y, the
    empirical base-rate log-loss (predict the observed positive rate for
    every entry), and the Mann-Whitney AUC with average ranks for ties.
    AUC is None -- and the missing class is named in "note" -- when the
    entries hold no positives or no negatives: the number does not
    exist there and is never invented."""
    n = int(y.shape[0])
    pos = int(y.sum())
    if n == 0:
        return {"log_loss": None, "base_log_loss": None, "auc": None,
                "n": 0, "positives": 0, "note": "no_valid_entries"}
    pc = np.clip(p, 1e-15, 1.0 - 1e-15)
    ll = float(-np.mean(y * np.log(pc) + (1.0 - y) * np.log(1.0 - pc)))
    base = pos / n
    bc = min(max(base, 1e-15), 1.0 - 1e-15)
    bll = float(-(base * math.log(bc) + (1.0 - base) * math.log(1.0 - bc)))
    out = {"log_loss": ll, "base_log_loss": bll, "auc": None, "n": n, "positives": pos}
    if pos == 0:
        out["note"] = "no_positive_entries: AUC undefined"
    elif pos == n:
        out["note"] = "no_negative_entries: AUC undefined"
    else:
        order = np.argsort(p, kind="mergesort")
        sorted_p = p[order]
        ranks_sorted = np.empty(n, dtype=np.float64)
        j = 0
        while j < n:
            k = j
            while k + 1 < n and sorted_p[k + 1] == sorted_p[j]:
                k += 1
            ranks_sorted[j:k + 1] = 0.5 * (j + k) + 1.0
            j = k + 1
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = ranks_sorted
        pos_rank_sum = float(ranks[y > 0.5].sum())
        out["auc"] = float((pos_rank_sum - pos * (pos + 1) / 2.0) / (pos * (n - pos)))
    return out


def run_screen(args):
    """--screen entry: stdout carries the one screen JSON line and
    nothing else -- library diagnostics from the sealed load path
    (cnet2_train.load prints its source line) are redirected to stderr
    for the whole evaluation."""
    with contextlib.redirect_stdout(sys.stderr):
        out = _screen_eval(args)
    # allow_nan=False is the last line of defence: the screen record
    # must be strict JSON, so a non-finite metric that somehow reached
    # here raises instead of printing a NaN literal as a "result".
    print(json.dumps(out, allow_nan=False), flush=True)


def _screen_eval(args):
    """--screen: the Stage-A screen. Data discipline is the training
    path's own (sealed generation load, u_raw finite gate, aux sidecar
    binding -- any failure refuses the run, naming the item); the arm
    checkpoint is bound by its complete marker and loaded from the same
    bytes its md5 is taken from, the init likewise. Evaluates over the
    whole bound generation and returns the screen record (the caller
    prints it as the single stdout JSON line)."""
    os.environ["BC_DEVICE"] = args.device          # cnet2_train reads this at import
    import torch
    import torch.nn.functional as Fn
    import cnet2_train as T
    from cnet2_build import LABEL_UNUSABLE
    from cnet2_model import CNet2, migrate_state_dict

    dev = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- data: the training path's sealed load + gates, verbatim ----
    D, is_val, meta = T.load(args.data)
    gen_dir = meta["dataset_source"]
    man_path = os.path.join(gen_dir, "manifest.json")
    if not os.path.exists(man_path):
        raise SystemExit(f"k0 screen: dataset {args.data!r} has no sealed generation manifest at {man_path}; "
                         f"refusing to run")
    man = json.load(open(man_path))
    n_rows = int(man["total"])
    if int(np.asarray(D["y"]).shape[0]) != n_rows:
        raise SystemExit(f"k0 screen: dataset rows {np.asarray(D['y']).shape[0]} != manifest total {n_rows}; "
                         f"refusing to run")
    if "u_raw" not in man.get("fields", {}):
        raise SystemExit(f"k0 screen: generation {man.get('generation')} declares no 'u_raw' field; "
                         f"not a label-aware sealed generation; refusing to run")
    u_raw = _load_gen_field(gen_dir, man, "u_raw", "k0 screen", "refusing to run").astype(np.float64)
    label_status = None
    if "label_status" in man.get("fields", {}):
        label_status = _load_gen_field(gen_dir, man, "label_status", "k0 screen", "refusing to run")
    unusable = (label_status == LABEL_UNUSABLE) if label_status is not None else None
    require_finite_np("u_raw", u_raw, allow_nan_mask=unusable)
    if int(np.isnan(u_raw).sum()) and unusable is None:
        raise SystemExit("k0 screen finite gate: u_raw has NaN entries but the generation carries no "
                         "label_status column to excuse them as unusable; refusing to run")
    aux_dir = args.aux_dir or args.data
    aux_raw = _read_aux_bundle(aux_dir)
    sc = check_aux_sidecar(aux_dir, man, meta, gen_dir, n_rows, prefetched=aux_raw)
    aux_die = np.load(io.BytesIO(aux_raw["aux_die.npy"]))
    aux_die_m = np.load(io.BytesIO(aux_raw["aux_die_m.npy"]))
    if aux_die.shape != (n_rows, 10, 3) or aux_die_m.shape != (n_rows, 10, 3):
        raise SystemExit(f"k0 screen aux: aux_die/aux_die_m shapes {aux_die.shape}/{aux_die_m.shape} != "
                         f"({n_rows}, 10, 3); refusing to run")
    for name, arr in (("aux_die.npy", aux_die), ("aux_die_m.npy", aux_die_m)):
        ent = sc["files"][name]
        if [int(x) for x in arr.shape] != ent.get("shape") or str(arr.dtype) != ent.get("dtype"):
            raise SystemExit(f"k0 screen aux: {name} shape/dtype {list(arr.shape)}/{arr.dtype} != sidecar "
                             f"entry {ent.get('shape')}/{ent.get('dtype')}; refusing to run")
    if not set(np.unique(aux_die).tolist()) <= {0, 1}:
        raise SystemExit("k0 screen finite gate: tensor 'aux_die' has values outside {0, 1}; refusing to run")
    require_finite_np("aux_die", aux_die.astype(np.float64))
    labels = aux_die.astype(np.float64)
    label_mask = aux_die_m.astype(bool)
    games = int(len(np.unique(np.asarray(D["gid"]))))

    # ---- arm checkpoint: one byte read; md5, marker check and the
    # torch.load parse all derive from it ----
    arm_raw = _read_bytes(args.checkpoint)
    arm_md5 = hashlib.md5(arm_raw).hexdigest()
    arm_sha = hashlib.sha256(arm_raw).hexdigest()
    marker_path = args.checkpoint + ".complete.json"
    if not os.path.exists(marker_path):
        raise SystemExit(f"k0 screen: arm checkpoint {args.checkpoint!r} has no complete marker "
                         f"{marker_path!r}; refusing to run")
    marker = json.load(open(marker_path))
    if marker.get("sha256") != arm_sha or not marker.get("complete"):
        raise SystemExit(f"k0 screen: complete marker for {args.checkpoint!r} does not match the file "
                         f"sha256 (marker {marker.get('sha256')!r} != actual {arm_sha!r}); refusing to run")
    ck = torch.load(io.BytesIO(arm_raw), map_location=dev, weights_only=False)
    for nm, sd in (("backbone", ck["backbone"]), ("u_head", ck["u_head"]), ("death_head", ck["death_head"])):
        for tn, tv in sd.items():
            if torch.is_tensor(tv) and tv.dtype.is_floating_point and not bool(torch.isfinite(tv).all()):
                raise SystemExit(f"k0 screen finite gate: arm tensor {nm}:{tn} is non-finite; refusing to run")

    # ---- init: one byte read; md5 and the arrays derive from it ----
    init_raw = _read_bytes(args.init)
    init_md5 = hashlib.md5(init_raw).hexdigest()
    w = np.load(io.BytesIO(init_raw))
    for k in w.files:
        if k != "meta":
            require_finite_np(f"init:{k}", w[k])
    init_cfg = None
    if "meta" in w.files:
        try:
            init_cfg = json.loads(str(w["meta"])).get("cfg")
        except Exception:
            init_cfg = None
    start_sd = {k: torch.from_numpy(np.ascontiguousarray(w[k])) for k in w.files if k != "meta"}

    net_arm = CNet2(**(ck.get("cfg") or init_cfg or {})).to(dev)
    net_arm.load_state_dict(ck["backbone"])
    death_head = torch.nn.Linear(net_arm.cfg["D"], 3).to(dev)
    death_head.load_state_dict(ck["death_head"])
    net_init = CNet2(**(init_cfg or ck.get("cfg") or {})).to(dev)
    migrate_state_dict(net_init, start_sd, allow_missing_prefixes=("tattn.",), source="--init")
    net_arm.eval()
    death_head.eval()
    net_init.eval()

    # ---- evaluation over the whole bound generation ----
    prob_chunks = []
    kl_roots = []
    with torch.no_grad():
        for a0 in range(0, n_rows, max(int(args.bs), 1)):
            idx = np.arange(a0, min(a0 + int(args.bs), n_rows))
            b = T.to_dev(T.fetch(D, idx), False)
            tok_a, ctx_a, _ = net_arm.encode(b)
            logits_a = net_arm.decode(b, tok_a, ctx_a)
            tok_i, ctx_i, _ = net_init.encode(b)
            logits_i = net_init.decode(b, tok_i, ctx_i)
            d_prob = torch.sigmoid(death_head(tok_a[:, :10])).detach().cpu().numpy().astype(np.float64)
            prob_chunks.append(d_prob)
            logp = Fn.log_softmax(logits_a.double(), dim=-1)
            logq = Fn.log_softmax(logits_i.double(), dim=-1)
            pa = logp.exp()
            kl_slot = (pa * (logp - logq)).sum(dim=-1).cpu().numpy()
            valid = (b["y"] >= 0).detach().cpu().numpy()
            kl_roots.append((kl_slot * valid).sum(axis=1))
    probs = np.concatenate(prob_chunks, axis=0)
    kl_all = np.concatenate(kl_roots, axis=0)

    # ---- output finite gate (V08) ----
    # The gates above verify the INPUTS (arm/init tensors, u_raw, aux
    # labels). They cannot see overflow that only appears when finite
    # but extreme parameters run through encode/decode: the screen
    # used to succeed and print NaN metrics as its result. So what
    # actually feeds the metrics -- the death probabilities and the
    # per-root KL -- and then the metrics themselves are gated here;
    # any non-finite value refuses the run, naming the item. A null
    # AUC on a single-class horizon stays legitimate (it is an absent
    # number, not a broken one) and is never gated.
    n_bad = int((~np.isfinite(probs)).sum())
    if n_bad:
        raise SystemExit(f"k0 screen finite gate: death probabilities hold {n_bad} non-finite "
                         f"entries of {probs.size}; refusing to run")
    n_bad = int((~np.isfinite(kl_all)).sum())
    if n_bad:
        raise SystemExit(f"k0 screen finite gate: per-root KL holds {n_bad} non-finite entries "
                         f"of {kl_all.shape[0]}; refusing to run")

    per_horizon = {}
    for hi, horizon in enumerate(DEATH_HORIZONS):
        m = label_mask[:, :, hi]
        per_horizon[str(horizon)] = _screen_metrics(probs[:, :, hi][m], labels[:, :, hi][m])
    combined = _screen_metrics(probs[label_mask], labels[label_mask])
    for _where, _m in [(f"death.horizons[{h}]", per_horizon[h]) for h in per_horizon] + \
                      [("death.combined", combined)]:
        for _key in ("log_loss", "base_log_loss", "auc"):
            _v = _m.get(_key)
            if _v is not None and not math.isfinite(_v):
                raise SystemExit(f"k0 screen finite gate: metric {_where}.{_key} is non-finite "
                                 f"({_v}); refusing to run")
    kl_mean = float(kl_all.mean())
    kl_p99 = float(np.percentile(kl_all, 99))
    for _key, _v in (("kl.mean", kl_mean), ("kl.p99", kl_p99)):
        if not math.isfinite(_v):
            raise SystemExit(f"k0 screen finite gate: metric {_key} is non-finite ({_v}); "
                             f"refusing to run")
    out = {"screen": {
        "arm": {"file": args.checkpoint, "md5": arm_md5},
        "init": {"file": args.init, "md5": init_md5},
        "data": args.data,
        "data_fingerprint": {"generation": man.get("generation"), "cfg_fp": man.get("cfg_fp"),
                             "rows": n_rows, "row_key_sha256": row_key_sha256(gen_dir, man)},
        "games": games, "roots": int(n_rows),
        "death": {"horizons": per_horizon, "combined": combined},
        "kl": {"mean": kl_mean, "p99": kl_p99,
               "n": int(kl_all.shape[0])}}}
    return out


if __name__ == "__main__":
    main()

"""k0_distill -- K0 round-2 joint-q distillation entry (design draft
K0_ROUND2_DESIGN_DRAFT_20261114 SS6/SS10 item 2, ruling
K0_ROUND2_RULING_20261114): one generation of joint-q distillation from
a frozen parent into a student bare net, with a self-distillation
control arm selected by a single switch.

Recipe (draft SS6, verbatim where it pins the math):
  * Policy loss: full joint-action cross-entropy
        L_CE(root) = - sum_k q_k * sum_s log pi_student(a^k_s | I, a^k_<s)
    over the root's candidate arms k, each candidate replayed under its
    OWN prefix (teacher-forced decode with y = arm k), NEVER
    renormalised inside the candidate set.
  * Target construction (the teacher's own formula, stage0_teacher.
    evaluate_root):  q_k ∝ sqrt(max(P_A,k, 1e-12)) * exp((Q_k - maxQ)/T),
    T = --q-temp (default 0.05, the teacher's value, in U units).
    P_A = the parent policy's candidate-set-normalised joint priors as
    recorded by the annotator. BOTH arms construct their target in this
    entry from (Q, P_A) through this one function; the stored q is a
    cross-check (a root whose stored q disagrees with the construction
    from its own Q / P_A is refused, naming the root).
  * Control arm (--flatten-q): every candidate's Q is replaced by one
    constant (the root's own Q mean) before the SAME construction --
    the exp term cancels and the target degenerates to
    q0 ∝ P_A^0.5, exactly the draft SS5 definition of arm C. S and C
    differ ONLY in this override: same roots, same candidate sets,
    same lr / steps / batch order (the batch stream is a pure function
    of --seed; the switch never touches the RNG).
  * Anchor: KL(pi_student || pi_parent) -- DIRECTION: student first,
    parent second (strategy-group ruling 2026-11-14). This is the same
    direction the Stage-A screen measures (k0_train --screen computes
    KL(pi_arm || pi_init), per-slot categorical KL summed over the
    root's valid slots), so the quantity the anchor constrains is the
    quantity the monitor reports; reverse KL would penalise exactly
    the probability mass the q signal tries to move. Per candidate:
    sum over the candidate's valid slots s of the categorical
    KL(pi_student(.|I, a_<s) || pi_parent(.|I, a_<s)); per root: the
    mean over the root's candidates. The batch anchor is the
    IPW-weighted mean over roots, multiplied by --kl-beta. No
    hard-label CE anywhere.
  * Frozen (draft SS6.3): the CNet2 utility / wdl / margin heads and
    the aux (true-carbon) conv -- parameter prefixes utility., wdl.,
    margin., aux. -- get requires_grad=False, are excluded from the
    optimizer, and are bitwise-verified against the parent before the
    checkpoint is published (a moved frozen parameter refuses the
    publish, naming the parameter). Everything else (trunk, decoder,
    policy heads) trains.
  * Monitoring (draft SS6.5): per update, the per-root KL defined
    above (mean over the root's candidates of the slot-summed
    KL(pi_student||pi_parent)) is reported as mean / p99 over the
    batch's roots. Recorded only; no threshold, no ruling.
  * IPW (draft SS6.6 / SS13-D): a root's loss weight is 1/pi, where pi
    is the sampling probability RECORDED IN THE PRODUCT (SS13-C). The
    weight activates only from those records: when every recorded pi
    is 1 (full annotation in the window -- this round's default), the
    weights are identically 1. There is no other, unregistered way to
    turn weighting on.

Data (consumed format, the consumer-side reading of the SS13-C frozen
schema; the annotator of SS10 item 1 publishes to it):
  <data> is a dataset root consumed ONLY through its current.json
  pointer to a sealed generation (verify-or-refuse, mirroring
  cnet2_exit_train._verify_exit_generation item for item: pointer
  manifest hash, manifest self-seal, meta seal, every file's bytes +
  sha256 + row count). A sealed generation directory passed directly
  is refused, as on the exit path. manifest format is
  "cnet2-k0-annotation/1"; its producer block carries the teacher
  identity (teacher_config_hash, teacher_weights_md5, belief mode).
  The generation holds one .npy per root-feature field (the
  cnet2_build FIELDS names/dtypes minus y/gid/step, plus gid/step),
  meta.json (games), and the two label bundles of SS13-C ("value and
  policy labels in separate files"):
    policy_labels.npz: game, step, seat (the per-root key), arms
      [R, K, OWN] (label arrays, -1 = no action at that slot), n_arms,
      priors [R, K] (P_A), q [R, K] (the annotator's stored target),
      pi0, pi1, pi [R] with pi = pi0 + (1 - pi0) * pi1.
    value_labels.npz: the same key arrays + Q [R, K] (per-candidate
      mean U from the teacher).
  A record missing pi (absent, NaN, or inconsistent with pi0/pi1) is
  unqualified by SS13-C and refuses the load, naming the root. The
  bundle keys must agree with each other and with the feature rows'
  gid/step, row by row. The producer's teacher_weights_md5 must equal
  the md5 of the --parent file bytes: the priors and q in the product
  were computed under those weights, and the student initialises from
  them -- a different parent refuses the start.

Product: <out>.npz, a standard cnet2 export (cnet2_train.export
format: float32 state arrays + meta JSON), loadable by the existing
tools and by root-continuation drivers; meta carries cfg and the full
provenance under "k0_distill". Beside it, <out>.complete.json (the
fleet atomic-save marker) and <out>.log.jsonl whose first line is the
effective config. This entry has no resume: round-2 runs are short,
single-generation, and restart from the parent by design.

Usage:
  python3 k0_distill.py --data ANNOT_DS --parent PARENT.npz \
      --out RUN/student_s_seed7.npz --seed 7 --max-updates 200
  python3 k0_distill.py --data ANNOT_DS --parent PARENT.npz \
      --out RUN/student_c_seed7.npz --seed 7 --max-updates 200 \
      --flatten-q        # control arm C: identical but for the switch
"""

import argparse
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

ANNOTATION_FORMAT = "cnet2-k0-annotation/1"
POLICY_BUNDLE = "policy_labels"
VALUE_BUNDLE = "value_labels"
Q_TEMP_DEFAULT = 0.05          # stage0_teacher.evaluate_root, U units
FROZEN_PREFIXES = ("utility.", "wdl.", "margin.", "aux.")
KL_DEFINITION = ("KL(pi_student || pi_parent): per-slot categorical KL "
                 "summed over each candidate's valid slots, averaged "
                 "over the root's candidates")
# Root-feature fields of the annotation generation: the cnet2_build
# FIELDS model inputs (everything a batch row carries except the played
# label y, which the candidate arms replace) plus the row key gid/step.
FEATURE_FIELDS = ("maps", "scal", "tok_pos", "tok_kind", "tok_mask",
                  "tok_cargo", "hist_cell", "hist_cargo", "hist_move",
                  "uf", "mf", "legal", "carbon", "teacher", "gid", "step")
FLOAT_FEATURE_FIELDS = ("scal", "tok_cargo", "hist_cargo", "uf", "mf")


# --------------------------------------------------------------------------
# Pure helpers (no torch: unit-testable on any machine)
# --------------------------------------------------------------------------

def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def _md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def _canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def _canonical_sha256(obj):
    return hashlib.sha256(_canonical(obj)).hexdigest()


def construct_target(Q, priors, q_temp=Q_TEMP_DEFAULT, flatten=False):
    """The one q-construction path shared by arms S and C (the
    teacher's formula, stage0_teacher.evaluate_root):
        wq_k = sqrt(max(P_A,k, 1e-12)) * exp((Q_k - maxQ) / q_temp),
        q = wq / sum(wq), over the root's REAL candidates only.
    Q, priors: float64 arrays [K] already restricted to real
    candidates (priors > 0, Q finite). flatten=True replaces every Q
    by the root's Q mean -- the single override that defines arm C;
    the exp term then cancels and the result is q0 ∝ P_A^0.5."""
    Q = np.asarray(Q, dtype=np.float64)
    priors = np.asarray(priors, dtype=np.float64)
    if flatten:
        Q = np.full(Q.shape, float(Q.mean()), dtype=np.float64)
    wq = np.exp((Q - Q.max()) / float(q_temp)) * np.sqrt(np.maximum(priors, 1e-12))
    return wq / wq.sum()


def verify_annotation_generation(data_arg):
    """Resolve the dataset root's pointer and verify the sealed
    generation in full, mirroring the exit path's verify-or-refuse
    (cnet2_exit_train._verify_exit_generation): the pointer's hash of
    the manifest bytes, the manifest's self-seal, the pointer/manifest
    generation match, the meta seal, then every declared file
    (existence, byte size, sha256; row count for .npy fields and for
    the label bundles' key arrays). Returns (gen_dir, manifest, meta).
    A sealed generation directory passed directly is refused: the
    verified generation is the one the pointer certifies."""
    if os.path.isfile(data_arg):
        ptr_path = data_arg
        root = os.path.dirname(os.path.abspath(data_arg))
    elif os.path.exists(os.path.join(data_arg, "current.json")):
        ptr_path = os.path.join(data_arg, "current.json")
        root = data_arg
    elif os.path.exists(os.path.join(data_arg, "manifest.json")):
        raise SystemExit(
            f"k0_distill data: sealed annotation generation at {data_arg!r} must be "
            f"consumed via its dataset root / current pointer, not by direct path; "
            f"refusing to start")
    else:
        raise SystemExit(
            f"k0_distill data: {data_arg!r} is not an annotation dataset root "
            f"(no current.json pointer); refusing to start")
    ptr = json.load(open(ptr_path))
    gen_dir = os.path.join(root, ptr["dir"])
    man_path = os.path.join(gen_dir, "manifest.json")
    if not os.path.exists(man_path):
        raise SystemExit(
            f"k0_distill data: sealed generation has no manifest.json: {gen_dir!r}; "
            f"refusing to start")
    man_raw = open(man_path, "rb").read()
    want = ptr.get("manifest_sha256")
    if want is not None and hashlib.sha256(man_raw).hexdigest() != want:
        raise SystemExit(
            f"k0_distill data: manifest hash mismatch in {gen_dir!r}: manifest.json "
            f"bytes do not match the pointer's manifest_sha256; refusing to start")
    man = json.loads(man_raw)
    if man.get("generation") != ptr.get("generation"):
        raise SystemExit(
            f"k0_distill data: pointer/manifest generation mismatch: pointer "
            f"{ptr.get('generation')!r} vs manifest {man.get('generation')!r}; "
            f"refusing to start")
    if man.get("format") != ANNOTATION_FORMAT:
        raise SystemExit(
            f"k0_distill data: generation format {man.get('format')!r} != "
            f"{ANNOTATION_FORMAT!r}; refusing to start")
    core = {k: v for k, v in man.items() if k != "self_sha256"}
    if man.get("self_sha256") != _canonical_sha256(core):
        raise SystemExit(
            f"k0_distill data: manifest self-seal mismatch in {gen_dir!r}: manifest "
            f"content does not match its self_sha256; refusing to start")
    meta_p = os.path.join(gen_dir, "meta.json")
    if not os.path.exists(meta_p) or _sha256_file(meta_p) != man.get("meta_sha256"):
        raise SystemExit(
            f"k0_distill data: meta seal mismatch in {gen_dir!r}: meta.json does not "
            f"match manifest meta_sha256; refusing to start")
    meta = json.load(open(meta_p))
    for k in sorted(man.get("fields", {})):
        info = man["fields"][k]
        p = os.path.join(gen_dir, info["file"])
        if not os.path.exists(p):
            raise SystemExit(
                f"k0_distill data: generation file missing: field {k!r} ({p!r}); "
                f"refusing to start")
        if os.path.getsize(p) != info["bytes"]:
            raise SystemExit(
                f"k0_distill data: generation file size mismatch: field {k!r} has "
                f"{os.path.getsize(p)} bytes, manifest declares {info['bytes']}; "
                f"refusing to start")
        if _sha256_file(p) != info["sha256"]:
            raise SystemExit(
                f"k0_distill data: generation file hash mismatch: field {k!r} "
                f"({info['file']}); refusing to start")
        if info["file"].endswith(".npy"):
            rows = int(np.load(p, mmap_mode="r").shape[0])
        else:
            with np.load(p) as z:
                rows = int(z["game"].shape[0])
        if rows != info["rows"]:
            raise SystemExit(
                f"k0_distill data: generation file row mismatch: field {k!r} has "
                f"{rows} rows, manifest declares {info['rows']}; refusing to start")
    return gen_dir, man, meta


def _die_rec(what, root=None):
    where = f" at root {root}" if root is not None else ""
    raise SystemExit(f"k0_distill data: record{where}: {what}; refusing to start")


def load_annotation(data_arg, q_temp=Q_TEMP_DEFAULT):
    """Verify + load an annotation generation and run the SS13-C
    record gates. Returns a dict:
      feats   {field: array [R, ...]}      root model inputs (+ gid/step)
      arms    int64 [R, K, OWN]            candidate label arrays
      n_arms  int64 [R]
      priors  float64 [R, K]               P_A (0 on padded candidates)
      q_stored float64 [R, K]              annotator's stored target
      target  float64 [R, K]               constructed target (S form)
      Q       float64 [R, K]               per-candidate Q (NaN padded)
      pi0, pi1, pi  float64 [R]
      keys    {game, step, seat}
      info    {generation, roots, teacher_config_hash,
               teacher_weights_md5, belief, ipw_active, ...}
    Every gate refuses by naming the failed item / root."""
    import cnet2_feat as F
    gen_dir, man, meta = verify_annotation_generation(data_arg)
    fields = man.get("fields", {})
    for k in FEATURE_FIELDS + (POLICY_BUNDLE, VALUE_BUNDLE):
        if k not in fields:
            raise SystemExit(
                f"k0_distill data: generation {man.get('generation')} declares no "
                f"field {k!r}; refusing to start")
    feats = {k: np.load(os.path.join(gen_dir, fields[k]["file"]))
             for k in FEATURE_FIELDS}
    R = int(man["total"])
    for k in FEATURE_FIELDS:
        if int(feats[k].shape[0]) != R:
            raise SystemExit(
                f"k0_distill data: feature field {k!r} has {feats[k].shape[0]} rows, "
                f"manifest total is {R}; refusing to start")
    for k in FLOAT_FEATURE_FIELDS:
        a = np.asarray(feats[k])
        if not np.isfinite(a.astype(np.float64)).all():
            bad = int((~np.isfinite(a.astype(np.float64))).sum())
            raise SystemExit(
                f"k0_distill data: feature field {k!r} holds {bad} non-finite "
                f"entries; refusing to start")
    with np.load(os.path.join(gen_dir, fields[POLICY_BUNDLE]["file"])) as z:
        pol = {k: z[k] for k in z.files}
    with np.load(os.path.join(gen_dir, fields[VALUE_BUNDLE]["file"])) as z:
        val = {k: z[k] for k in z.files}
    for name, bundle, need in (
            ("policy", pol, ("game", "step", "seat", "arms", "n_arms", "priors",
                             "q", "pi0", "pi1", "pi")),
            ("value", val, ("game", "step", "seat", "Q"))):
        for k in need:
            if k not in bundle:
                raise SystemExit(
                    f"k0_distill data: {name} label bundle lacks array {k!r}; "
                    f"refusing to start")
        if int(bundle["game"].shape[0]) != R:
            raise SystemExit(
                f"k0_distill data: {name} label bundle has "
                f"{bundle['game'].shape[0]} roots, manifest total is {R}; "
                f"refusing to start")
    # Key agreement: bundles against each other and against the rows.
    for k in ("game", "step", "seat"):
        if not np.array_equal(np.asarray(pol[k]), np.asarray(val[k])):
            _die_rec(f"policy/value key {k!r} disagrees between the label bundles")
    if not np.array_equal(np.asarray(pol["game"]).astype(np.int64),
                          np.asarray(feats["gid"]).astype(np.int64)):
        _die_rec("policy bundle 'game' disagrees with the feature rows' gid")
    if not np.array_equal(np.asarray(pol["step"]).astype(np.int64),
                          np.asarray(feats["step"]).astype(np.int64)):
        _die_rec("policy bundle 'step' disagrees with the feature rows' step")

    arms = np.asarray(pol["arms"]).astype(np.int64)
    n_arms = np.asarray(pol["n_arms"]).astype(np.int64)
    priors = np.asarray(pol["priors"]).astype(np.float64)
    q_stored = np.asarray(pol["q"]).astype(np.float64)
    Q = np.asarray(val["Q"]).astype(np.float64)
    pi0 = np.asarray(pol["pi0"]).astype(np.float64)
    pi1 = np.asarray(pol["pi1"]).astype(np.float64)
    pi = np.asarray(pol["pi"]).astype(np.float64)
    if arms.ndim != 3 or arms.shape[0] != R or arms.shape[2] != F.OWN:
        raise SystemExit(
            f"k0_distill data: arms shape {arms.shape} != (R={R}, K, OWN={F.OWN}); "
            f"refusing to start")
    K = int(arms.shape[1])
    for name, a in (("priors", priors), ("q", q_stored), ("Q", Q)):
        if a.shape != (R, K):
            raise SystemExit(
                f"k0_distill data: {name} shape {a.shape} != ({R}, {K}); "
                f"refusing to start")
    tok_mask = np.asarray(feats["tok_mask"]).astype(bool)
    target = np.zeros((R, K), dtype=np.float64)
    for r in range(R):
        n = int(n_arms[r])
        if n < 1 or n > K:
            _die_rec(f"n_arms {n} outside [1, {K}]", r)
        real = np.arange(K) < n
        # pi qualification (SS13-C): present, finite, in range, and
        # consistent with its own factors -- a missing pi is an
        # unqualified record, refused here by root.
        for nm, v in (("pi0", pi0[r]), ("pi1", pi1[r]), ("pi", pi[r])):
            if not np.isfinite(v):
                _die_rec(f"{nm} is not finite (a record missing pi is "
                         f"unqualified)", r)
        if not (0.0 <= pi0[r] <= 1.0 and 0.0 <= pi1[r] <= 1.0):
            _die_rec(f"pi0/pi1 outside [0, 1]: {pi0[r]}/{pi1[r]}", r)
        if not (0.0 < pi[r] <= 1.0):
            _die_rec(f"pi {pi[r]} outside (0, 1]", r)
        if abs(pi[r] - (pi0[r] + (1.0 - pi0[r]) * pi1[r])) > 1e-6:
            _die_rec(f"pi {pi[r]} != pi0 + (1-pi0)*pi1 = "
                     f"{pi0[r] + (1.0 - pi0[r]) * pi1[r]}", r)
        # Padded candidates must be inert in every array.
        if not (np.all(arms[r][~real] == -1) and np.all(priors[r][~real] == 0.0)
                and np.all(q_stored[r][~real] == 0.0)
                and np.all(np.isnan(Q[r][~real]))):
            _die_rec("padded candidates are not inert (arms -1, priors 0, "
                     "q 0, Q NaN)", r)
        pr, qq, qqv = priors[r][real], q_stored[r][real], Q[r][real]
        if not np.isfinite(qqv).all():
            _die_rec("Q has a non-finite entry on a real candidate", r)
        if not (np.isfinite(pr).all() and np.all(pr > 0)):
            _die_rec("priors must be finite and > 0 on real candidates", r)
        if abs(pr.sum() - 1.0) > 1e-5:
            _die_rec(f"priors sum {pr.sum()} != 1 over the candidate set", r)
        if not (np.isfinite(qq).all() and np.all(qq >= 0)):
            _die_rec("stored q must be finite and >= 0 on real candidates", r)
        if abs(qq.sum() - 1.0) > 1e-5:
            _die_rec(f"stored q sums to {qq.sum()} != 1", r)
        arm_r = arms[r]
        if int(arm_r.min()) < -1 or int(arm_r.max()) > 4:
            _die_rec("arm action outside [-1, 4]", r)
        # An arm assigns an action exactly at the slots whose unit
        # exists (tok_mask), nowhere else -- the same validity the
        # decoder and the teacher's joint_logp assume (a < 0 = skip).
        exists = tok_mask[r][:F.OWN]
        for k in range(n):
            if not np.array_equal(arm_r[k] >= 0, exists):
                _die_rec(f"candidate {k}: arm validity (arm >= 0) disagrees "
                         f"with the root's tok_mask", r)
        # Stored-q cross-check: the target this entry trains on is
        # constructed here from (Q, priors); the annotator's stored q
        # must agree with that construction from its own record.
        target[r][real] = construct_target(qqv, pr, q_temp)
        d = float(np.abs(target[r][real] - qq).max())
        if d > 2e-6:
            _die_rec(f"stored q disagrees with the SS13-C construction from "
                     f"its own Q/priors (max abs diff {d:.3g})", r)
    prod = man.get("producer") or {}
    for k in ("teacher_config_hash", "teacher_weights_md5", "belief"):
        if not prod.get(k):
            raise SystemExit(
                f"k0_distill data: generation producer identity lacks {k!r}; "
                f"refusing to start")
    ipw_active = bool(np.any(pi < 1.0 - 1e-12))
    info = {"generation": man.get("generation"), "roots": R, "candidates": int(n_arms.sum()),
            "max_candidates": K,
            "teacher_config_hash": prod["teacher_config_hash"],
            "teacher_weights_md5": prod["teacher_weights_md5"],
            "belief": prod["belief"],
            "ipw_active": ipw_active,
            "pi_min": float(pi.min()), "pi_mean": float(pi.mean())}
    return {"feats": feats, "arms": arms, "n_arms": n_arms, "priors": priors,
            "q_stored": q_stored, "target": target, "Q": Q,
            "pi0": pi0, "pi1": pi1, "pi": pi,
            "keys": {"game": np.asarray(pol["game"]), "step": np.asarray(pol["step"]),
                     "seat": np.asarray(pol["seat"])},
            "info": info, "gen_dir": gen_dir, "manifest": man}


def root_weights(pi, ipw_active):
    """Per-root loss weights: 1/pi when the product's records carry a
    registered sampling pi < 1 (ipw_active, decided by the loader from
    the records themselves); identically 1 otherwise. There is no
    other activation path -- full annotation (all pi == 1) makes this
    the identity by construction."""
    pi = np.asarray(pi, dtype=np.float64)
    return (1.0 / pi) if ipw_active else np.ones_like(pi)


def targets_with_flatten(ds, flatten, q_temp=Q_TEMP_DEFAULT):
    """The batch-independent target table for one arm of the
    experiment: arm S constructs from the recorded (Q, priors); arm C
    walks the identical construction with every Q replaced by the
    root's Q mean (construct_target's flatten override). Nothing else
    differs -- in particular this function consumes no randomness, so
    the seed-driven batch order is the same function for both arms."""
    if not flatten:
        return ds["target"]
    out = np.zeros_like(ds["target"])
    for r in range(ds["info"]["roots"]):
        n = int(ds["n_arms"][r])
        out[r][:n] = construct_target(ds["Q"][r][:n], ds["priors"][r][:n],
                                      q_temp, flatten=True)
    return out


def joint_logp_kl(logits_s, logits_p, arms):
    """Per-candidate joint logp under the student and per-candidate
    KL(pi_student || pi_parent), teacher-forced on the candidate's own
    prefix -- the loss path's core, shared by training and the
    fixtures. logits_s / logits_p: float tensors [M, OWN, 5] from the
    student / parent decode of the SAME expanded batch (y = arms);
    arms: int64 [M, OWN]. Computed in float64 (the Stage-A screen's
    caliber). Returns (joint_logp [M], kl [M]) where
      joint_logp = sum over valid slots s of log pi_student(a_s|a_<s),
      kl         = sum over valid slots s of the categorical
                   KL(pi_student(.|a_<s) || pi_parent(.|a_<s)),
    a slot being valid iff arms[:, s] >= 0 (the teacher's joint_logp
    skips exactly those slots)."""
    import torch
    import torch.nn.functional as Fn
    logp = Fn.log_softmax(logits_s.double(), dim=-1)
    logq = Fn.log_softmax(logits_p.double(), dim=-1)
    valid = arms >= 0
    picked = torch.gather(logp, 2, arms.clamp(min=0)[..., None])[..., 0]
    joint = (picked * valid.double()).sum(dim=1)
    kl_slot = (logp.exp() * (logp - logq)).sum(dim=-1)
    kl = (kl_slot * valid.double()).sum(dim=1)
    return joint, kl


def frozen_snapshot(net, prefixes=FROZEN_PREFIXES):
    """{name: cpu tensor clone} for every parameter under the frozen
    prefixes -- the publish gate's reference."""
    return {n: p.detach().cpu().clone() for n, p in net.named_parameters()
            if n.startswith(prefixes)}


def check_frozen(net, snap):
    """Bitwise-compare the frozen parameters against the snapshot.
    Returns (ok, moved): moved lists any frozen parameter whose bytes
    changed -- there must be none; the frozen set is out of the
    optimizer and gradient-free by construction, so a move is a code
    defect, not a training outcome."""
    moved = []
    for n, p in net.named_parameters():
        if n in snap and not np.array_equal(p.detach().cpu().numpy(),
                                            snap[n].numpy()):
            moved.append(n)
    return (not moved), moved


# --------------------------------------------------------------------------
# Training entry (torch below this line)
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="K0 round-2 joint-q distillation (arm S; --flatten-q = arm C)")
    ap.add_argument("--data", required=True,
                    help="annotation dataset root (sealed generation via current.json)")
    ap.add_argument("--parent", required=True,
                    help="parent checkpoint .npz (cnet2 export); the student "
                         "initialises from it and the anchor binds to it")
    ap.add_argument("--out", required=True, help="student checkpoint output (.npz)")
    ap.add_argument("--flatten-q", action="store_true",
                    help="arm C: flatten every candidate's Q before the shared "
                         "target construction (target degenerates to q0 ∝ P_A^0.5)")
    ap.add_argument("--kl-beta", type=float, default=1.0,
                    help="anchor coefficient on KL(pi_student||pi_parent)")
    ap.add_argument("--q-temp", type=float, default=Q_TEMP_DEFAULT,
                    help="teacher target temperature in U units (default: the "
                         "stage0_teacher value 0.05)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=64, help="roots per update")
    ap.add_argument("--max-updates", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--save-every", type=int, default=0,
                    help="also save every N updates (0 = final only)")
    ap.add_argument("--cfg", default=None, help="CNet2 cfg JSON (default: parent meta cfg)")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    os.environ["BC_DEVICE"] = args.device          # cnet2_train reads this at import
    import torch
    import cnet2_train as T
    from cnet2_model import CNet2, migrate_state_dict

    dev = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- data: sealed generation, SS13-C gates (pure path above) ----
    ds = load_annotation(args.data, q_temp=args.q_temp)
    info = ds["info"]
    target = targets_with_flatten(ds, args.flatten_q, q_temp=args.q_temp)
    R, K = info["roots"], info["max_candidates"]

    # ---- parent: one byte read; identities and the parse share it ----
    parent_raw = open(args.parent, "rb").read()
    parent_md5 = hashlib.md5(parent_raw).hexdigest()
    parent_sha = hashlib.sha256(parent_raw).hexdigest()
    if info["teacher_weights_md5"] != parent_md5:
        raise SystemExit(
            f"k0_distill: annotation producer teacher_weights_md5 "
            f"{info['teacher_weights_md5']!r} != --parent file md5 {parent_md5!r}; "
            f"the recorded priors/q were computed under different weights than "
            f"the declared parent; refusing to start")
    w = np.load(io.BytesIO(parent_raw))
    for k in w.files:
        if k != "meta":
            a = np.asarray(w[k])
            if np.issubdtype(a.dtype, np.floating) and not np.isfinite(a.astype(np.float64)).all():
                raise SystemExit(
                    f"k0_distill finite gate: parent array {k!r} has non-finite "
                    f"entries; refusing to start")
    cfg = None
    if "meta" in w.files:
        try:
            cfg = json.loads(str(w["meta"])).get("cfg")
        except Exception:
            cfg = None
    if args.cfg:
        cfg = json.loads(args.cfg)
    start_sd = {k: torch.from_numpy(np.ascontiguousarray(w[k]))
                for k in w.files if k != "meta"}

    net = CNet2(**(cfg or {})).to(dev)
    mig = migrate_state_dict(net, start_sd, allow_missing_prefixes=("tattn.",),
                             source="--parent")
    parent = CNet2(**(cfg or {})).to(dev)
    migrate_state_dict(parent, start_sd, allow_missing_prefixes=("tattn.",),
                       source="--parent")
    parent.eval()
    for p in parent.parameters():
        p.requires_grad_(False)

    # ---- freeze list (SS6.3): out of the optimizer, bitwise at the end ----
    init_snap = {n: p.detach().cpu().clone() for n, p in net.named_parameters()}
    frozen_names = []
    for n, p in net.named_parameters():
        if n.startswith(FROZEN_PREFIXES):
            p.requires_grad_(False)
            frozen_names.append(n)
    if not frozen_names:
        raise SystemExit("k0_distill: freeze list matched no parameters "
                         f"({FROZEN_PREFIXES}); refusing to start")
    snap = frozen_snapshot(net)
    trainable = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.wd)

    # IPW: weights exist only as the records carry them (pi == 1
    # everywhere -> inactive, weights identically 1).
    ipw_w = root_weights(ds["pi"], info["ipw_active"])

    code_md5 = _md5_file(os.path.abspath(__file__))
    eff = {"k0_distill_effective": {
        "format": "k0-distill/1",
        "arm": "C" if args.flatten_q else "S",
        "flatten_q": bool(args.flatten_q),
        "parent": {"file": args.parent, "md5": parent_md5, "sha256": parent_sha},
        "data": {"dir": args.data, "generation": info["generation"],
                 "gen_dir": ds["gen_dir"], "roots": R,
                 "candidates": info["candidates"],
                 "teacher_config_hash": info["teacher_config_hash"],
                 "teacher_weights_md5": info["teacher_weights_md5"],
                 "belief": info["belief"]},
        "recipe": {"q_temp": args.q_temp, "kl_beta": args.kl_beta,
                   "kl_definition": KL_DEFINITION,
                   "lr": args.lr, "wd": args.wd, "bs": args.bs,
                   "max_updates": args.max_updates, "seed": args.seed,
                   "frozen_prefixes": list(FROZEN_PREFIXES),
                   "frozen_params": frozen_names},
        "ipw": {"active": info["ipw_active"], "pi_min": info["pi_min"],
                "pi_mean": info["pi_mean"],
                "source": "pi recorded in the product records (SS13-C)"},
        "init_migrated_missing": mig["migrated_missing"],
        "code_md5": code_md5, "device": args.device}}
    log_path = args.out + ".log.jsonl"
    log = open(log_path, "w")

    def emit(obj):
        line = json.dumps(obj)
        print(line, flush=True)
        log.write(line + "\n"); log.flush()

    def die_nonfinite(what, value):
        emit({"circuit_breaker": {"what": what, "value": str(value),
                                  "action": "exit(3) without saving"}})
        log.close()
        sys.exit(3)

    emit(eff)

    feats = ds["feats"]
    arms_t = ds["arms"]
    n_arms = ds["n_arms"]
    target_t = target

    def expanded_batch(root_idx):
        """Root rows repeated once per candidate slot (fixed K, padded
        candidates masked downstream), y = the candidate arms -- each
        candidate is teacher-forced under its OWN prefix."""
        rep = np.repeat(np.asarray(root_idx), K)
        b = {}
        for k in feats:
            if k in ("gid", "step"):
                continue
            b[k] = np.asarray(feats[k])[rep]
        b["y"] = arms_t[np.asarray(root_idx)].reshape(-1, arms_t.shape[2])
        return T.to_dev(b, aug=False), rep

    def forward_terms(root_idx):
        b, _ = expanded_batch(root_idx)
        tok, ctx, _ = net.encode(b)
        logits_s = net.decode(b, tok, ctx)
        with torch.no_grad():
            tok_p, ctx_p, _ = parent.encode(b)
            logits_p = parent.decode(b, tok_p, ctx_p)
        arms_b = torch.from_numpy(
            arms_t[np.asarray(root_idx)].reshape(-1, arms_t.shape[2])).to(dev)
        joint, kl = joint_logp_kl(logits_s, logits_p, arms_b)
        Br = len(root_idx)
        joint = joint.reshape(Br, K)
        kl = kl.reshape(Br, K)
        tgt = torch.from_numpy(target_t[np.asarray(root_idx)]).to(dev)
        narr = torch.from_numpy(n_arms[np.asarray(root_idx)]).to(dev)
        ce_root = -(tgt * joint).sum(dim=1)
        kl_root = kl.sum(dim=1) / narr.double()
        w_t = torch.from_numpy(ipw_w[np.asarray(root_idx)]).to(dev)
        loss_root = ce_root + float(args.kl_beta) * kl_root
        loss = (w_t * loss_root).sum() / w_t.sum()
        return loss, ce_root, kl_root, joint

    def save_checkpoint(updates):
        ok, moved = check_frozen(net, snap)
        if not ok:
            emit({"frozen_gate": {"moved": moved, "updates": updates,
                                  "action": "refusing to publish"}})
            log.close()
            sys.exit(3)
        for nm, p in net.named_parameters():
            if not bool(torch.isfinite(p.detach()).all()):
                die_nonfinite(f"param:{nm}", "non-finite at save time")
        moved_trainable = sorted(
            n for n, p in net.named_parameters()
            if n not in snap and not np.array_equal(
                p.detach().cpu().numpy(), init_snap[n].numpy()))
        prov = dict(eff["k0_distill_effective"])
        prov["updates_done"] = int(updates)
        prov["frozen_check"] = {"frozen_params": frozen_names, "moved": [],
                                "bitwise_identical_to_parent": True}
        prov["trainable_moved"] = moved_trainable
        meta = {"cfg": net.cfg, "k0_distill": prov}
        tmp = f"{args.out}.tmp-{os.getpid()}.npz"   # np.savez appends .npz otherwise
        T.export(net, tmp, meta)
        os.replace(tmp, args.out)
        sha = _sha256_file(args.out)
        marker = {"checkpoint": os.path.basename(args.out), "sha256": sha,
                  "md5": _md5_file(args.out), "updates": int(updates),
                  "complete": True, "time": time.time()}
        mtmp = f"{args.out}.complete.json.tmp-{os.getpid()}"
        with open(mtmp, "w") as fh:
            json.dump(marker, fh)
        os.replace(mtmp, args.out + ".complete.json")
        emit({"saved": marker})

    rs = np.random.RandomState(args.seed)
    roots = np.arange(R)
    updates_done = 0
    t0 = time.time()
    net.train()
    while updates_done < args.max_updates:
        perm = rs.permutation(roots)
        for a0 in range(0, R, args.bs):
            if updates_done >= args.max_updates:
                break
            idx = perm[a0:a0 + args.bs]
            loss, ce_root, kl_root, _ = forward_terms(idx)
            for nm, tv in (("total", loss),):
                if not bool(torch.isfinite(tv.detach()).all()):
                    die_nonfinite(f"loss:{nm}", float(tv.detach()))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            for nm, p in net.named_parameters():
                if p.grad is not None and not bool(torch.isfinite(p.grad).all()):
                    die_nonfinite(f"grad:{nm}", "non-finite gradient")
            opt.step()
            updates_done += 1
            kl_np = kl_root.detach().cpu().numpy().astype(np.float64)
            emit({"update": updates_done,
                  "loss": {"total": float(loss.detach()),
                           "ce": float((ce_root.detach()).mean()),
                           "anchor_kl_weighted": float(args.kl_beta) * float(kl_np.mean())},
                  "kl_monitor": {"definition": "per-root KL(pi_student||pi_parent), "
                                               "mean over candidates of slot-summed KL",
                                 "mean": float(kl_np.mean()),
                                 "p99": float(np.percentile(kl_np, 99))},
                  "denominators": {"roots": int(len(idx)),
                                   "candidates": int(n_arms[idx].sum()),
                                   "batch_sha256": hashlib.sha256(
                                       np.asarray(idx, dtype=np.int64).tobytes()).hexdigest()},
                  "lr": float(opt.param_groups[0]["lr"]),
                  "elapsed_s": round(time.time() - t0, 2)})
            if args.save_every and updates_done % args.save_every == 0 \
                    and updates_done < args.max_updates:
                save_checkpoint(updates_done)
    save_checkpoint(updates_done)
    emit({"done": {"updates": updates_done, "elapsed_s": round(time.time() - t0, 2)}})
    log.close()


if __name__ == "__main__":
    main()

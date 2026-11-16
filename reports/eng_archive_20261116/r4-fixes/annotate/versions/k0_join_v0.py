"""k0_join -- K0 round-2 seam step between the annotator (SS10 item 1)
and the joint-q distillation entry (SS10 item 2): pack a sealed
annotation generation plus the same games' sealed cnet2_build
generation into the consumer format cnet2-k0-annotation/1.

Why this step exists: the two sides were built against the same
SS13-C schema but landed in different concrete shapes --
  * format name: the annotator publishes "k0-annotation-generation/1";
    the distillation loader certifies only "cnet2-k0-annotation/1";
  * root features (maps/scal/tok_*/hist_*/uf/mf/legal/carbon/teacher)
    are not in the annotation generation at all -- they live in the
    cnet2_build generation of the same games;
  * containers differ: the annotator stores one .npy per field, the
    consumer reads per-field .npy feature files PLUS two label
    bundles (policy_labels.npz / value_labels.npz);
  * the arm padding sentinel differs: the annotator pads non-existent
    arms with -2 ("no such arm"), the consumer's record gates require
    padded arms to be exactly -1.

What join does, and nothing else:
  1. Consumes BOTH inputs verify-or-refuse through their current.json
     pointers only (a sealed generation directory passed directly is
     refused on either side, mirroring both producers' rules). The
     annotation side is verified to the annotator's own caliber
     (k0_annotate.verify_generation + its S13-C record gates); the
     build side to cnet2_build.load_generation's caliber (pointer /
     manifest match, per-file sha256, meta seal).
  2. Joins feature rows to annotation roots 1:1 on (game, step, seat):
     games are matched across the two generations' games tables BY
     NAME; a build game's seat for a row block is read off the build
     generation's own side-major row order (cnet2_build emits each
     side's steps ascending, sides in the meta entry's "sides" order),
     segmented at step resets, and every annotation root must find
     exactly one feature row -- a missing feature row, an ambiguous
     seat segmentation, or a seat the build never produced dies by
     name, naming the root. The emitted feature gid is rewritten to
     the annotation gid (the consumer binds bundle game == feature
     gid, game := annotation gid); the full name/gid mapping is
     recorded in the output meta's games table.
  3. Translates the arm sentinel -2 -> -1 on the padding block of
     every root, with per-root assertions: the padding block must be
     uniformly -2 in the source (the annotator's contract), a -2
     inside a real arm is a corrupt record and dies by name, and a
     post-translation sweep of the whole arms array must find no -2
     left. The translated entry count is recorded in the output meta
     and producer block. No other label value is touched: q, priors
     and Q pass through byte-for-byte (Q padding stays NaN, q/priors
     padding stays 0).
  4. Publishes a sealed "cnet2-k0-annotation/1" generation whose
     producer block dual-binds BOTH source generations (generation id
     + manifest sha256 of each) beside the teacher identity the
     consumer requires (teacher_config_hash, teacher_weights_md5,
     belief, taken from the annotation generation's seal-bound
     producer/meta) and this file's own md5. Sealing, content
     addressing, verify-and-reuse of an identical already-published
     generation, and the pointer flipping last follow the exit
     chain's current rules (a published generation is never deleted
     or overwritten).
  5. Before anything is published, the staged product is loaded
     through the REAL consumer (k0_distill.load_annotation, imported
     from this directory): every consumer gate -- format, bundles,
     per-root record gates, stored-q construction cross-check, arm
     validity against the joined rows' tok_mask -- must pass on the
     staged bytes. A refusal aborts the join with nothing published.

Usage:
  python3 k0_join.py --annot ANNOT_DS --build BUILD_DS --out OUT_DS
  python3 k0_join.py --out OUT_DS --verify   (consumer-load check only)
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

ANNOT_FORMAT = "k0-annotation-generation/1"
OUT_FORMAT = "cnet2-k0-annotation/1"
ARM_PAD_SRC = -2          # annotator: "no such arm" padding
ARM_PAD_DST = -1          # consumer: padded arms must be exactly -1

# The consumer's feature contract (k0_distill.FEATURE_FIELDS): the
# cnet2_build FIELDS model inputs minus the played label y, plus the
# row key gid/step. Named here so a build generation missing one dies
# by field name before any row work starts.
FEATURE_FIELDS = ("maps", "scal", "tok_pos", "tok_kind", "tok_mask",
                  "tok_cargo", "hist_cell", "hist_cargo", "hist_move",
                  "uf", "mf", "legal", "carbon", "teacher", "gid", "step")
ANNOT_FIELDS = ("gid", "step", "seat", "arms", "n_arms", "priors", "q",
                "Q", "Q_std", "pi0", "pi1", "pi", "belief", "n_unknown")


def _sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def _sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _md5_file(p):
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def _die(msg):
    raise SystemExit(f"k0_join: {msg}")


# ------------------------- pointer resolution ------------------------------


def _resolve_pointer(arg, what):
    """Pointer-only consumption for both inputs and the output check:
    returns (root, ptr). A sealed generation directory passed directly
    is refused, mirroring both producers' consumption rules."""
    if os.path.isfile(arg):
        ptr = json.load(open(arg))
        return os.path.dirname(os.path.abspath(arg)), ptr
    if os.path.exists(os.path.join(arg, "current.json")):
        return arg, json.load(open(os.path.join(arg, "current.json")))
    if os.path.exists(os.path.join(arg, "manifest.json")):
        _die(f"{what}: sealed generation at {arg!r} must be consumed via "
             f"its dataset root / current pointer, not by direct path")
    _die(f"{what}: no dataset pointer (current.json) at {arg!r}")


# ------------------------- annotation side ---------------------------------


def _validate_annot_records(arrs, window, K, where):
    """The annotator's S13-C record gates (k0_annotate.validate_records
    caliber), applied to the input before any join work: pi present,
    finite, in (0, 1] and exactly pi0 + (1-pi0)*pi1; steps inside the
    registered window; arm counts sane; q normalised over the recorded
    arms with zero mass on padding; Q finite on arms; no duplicate
    (gid, step, seat) root."""
    lo, hi = window
    n = int(arrs["step"].shape[0])
    pi0 = np.asarray(arrs["pi0"], dtype=np.float64)
    pi1 = np.asarray(arrs["pi1"], dtype=np.float64)
    pi = np.asarray(arrs["pi"], dtype=np.float64)
    for name, v in (("pi0", pi0), ("pi1", pi1)):
        bad = np.nonzero(~np.isfinite(v) | (v < 0.0) | (v > 1.0))[0]
        if len(bad):
            i = int(bad[0])
            _die(f"{where}: root #{i} has invalid {name}={v[i]!r} (pass "
                 f"rates must lie in [0, 1])")
    bad = np.nonzero(~np.isfinite(pi) | (pi <= 0.0) | (pi > 1.0))[0]
    if len(bad):
        i = int(bad[0])
        _die(f"{where}: root #{i} has invalid pi={pi[i]!r} -- records "
             f"without a valid pi are unqualified (S13-C)")
    want = pi0 + (1.0 - pi0) * pi1
    bad = np.nonzero(pi != want)[0]
    if len(bad):
        i = int(bad[0])
        _die(f"{where}: root #{i} pi={pi[i]!r} != pi0 + (1-pi0)*pi1 = "
             f"{want[i]!r} (S13-C identity)")
    step = np.asarray(arrs["step"])
    bad = np.nonzero((step < lo) | (step >= hi))[0]
    if len(bad):
        i = int(bad[0])
        _die(f"{where}: root #{i} step {int(step[i])} outside the "
             f"registered window [{lo}, {hi})")
    na = np.asarray(arrs["n_arms"])
    bad = np.nonzero((na < 1) | (na > K))[0]
    if len(bad):
        i = int(bad[0])
        _die(f"{where}: root #{i} has n_arms {int(na[i])} outside [1, {K}]")
    q = np.asarray(arrs["q"], dtype=np.float64)
    Q = np.asarray(arrs["Q"], dtype=np.float64)
    priors = np.asarray(arrs["priors"], dtype=np.float64)
    for i in range(n):
        m = int(na[i])
        if not np.isfinite(Q[i, :m]).all():
            _die(f"{where}: root #{i} has non-finite Q on a recorded arm")
        if (q[i, :m] < 0).any() or (priors[i, :m] < 0).any():
            _die(f"{where}: root #{i} has negative q/priors on a recorded arm")
        s = float(q[i, :m].sum())
        if abs(s - 1.0) > 1e-9:
            _die(f"{where}: root #{i} q sums to {s!r} over its arms, not 1")
        if (q[i, m:] != 0.0).any() or (priors[i, m:] != 0.0).any():
            _die(f"{where}: root #{i} carries q/priors mass on padding arms")
    keys = set()
    for i in range(n):
        key = (int(arrs["gid"][i]), int(step[i]), int(arrs["seat"][i]))
        if key in keys:
            _die(f"{where}: duplicate root (gid, step, seat) = {key}")
        keys.add(key)


def load_annotation_source(arg):
    """Verify-or-refuse the annotation generation (annotator caliber:
    pointer hash, manifest self-seal, meta seal, teacher identity
    cross-check, per-field bytes/sha/rows, S13-C record gates) and load
    its arrays. Returns (summary, meta, arrs) with summary carrying
    the source identity the output producer block dual-binds."""
    root, ptr = _resolve_pointer(arg, "annotation input")
    gen_dir = os.path.join(root, ptr["dir"])
    man_path = os.path.join(gen_dir, "manifest.json")
    if not os.path.exists(man_path):
        _die(f"annotation input: sealed generation has no manifest.json: "
             f"{gen_dir!r}; refusing")
    man_raw = open(man_path, "rb").read()
    want = ptr.get("manifest_sha256")
    if want is not None and _sha256_bytes(man_raw) != want:
        _die(f"annotation input: manifest hash mismatch in {gen_dir!r}: "
             f"manifest.json bytes do not match the pointer")
    man = json.loads(man_raw)
    if man.get("generation") != ptr.get("generation"):
        _die(f"annotation input: pointer/manifest generation mismatch: "
             f"pointer {ptr.get('generation')!r} vs manifest "
             f"{man.get('generation')!r}")
    if man.get("format") != ANNOT_FORMAT:
        _die(f"annotation input: generation format {man.get('format')!r} "
             f"!= {ANNOT_FORMAT!r}; refusing")
    core = {k: v for k, v in man.items() if k != "self_sha256"}
    if man.get("self_sha256") != _sha256_bytes(_canonical(core)):
        _die(f"annotation input: manifest self-seal mismatch in {gen_dir!r}")
    meta_p = os.path.join(gen_dir, "meta.json")
    if not os.path.exists(meta_p) or _sha256_file(meta_p) != man.get("meta_sha256"):
        _die(f"annotation input: meta seal mismatch in {gen_dir!r}")
    meta = json.load(open(meta_p))
    prod = man.get("producer", {})
    if meta["teacher"]["config_sha256"] != prod.get("config_sha256") \
            or meta["teacher"]["weights_md5"] != prod.get("weights_md5"):
        _die(f"annotation input: generation {gen_dir!r}: meta teacher "
             f"identity does not match the manifest producer identity")
    arrs = {}
    for k in sorted(man.get("fields", {})):
        info = man["fields"][k]
        p = os.path.join(gen_dir, info["file"])
        if not os.path.exists(p):
            _die(f"annotation input: generation file missing: field {k!r} ({p!r})")
        if os.path.getsize(p) != info["bytes"]:
            _die(f"annotation input: generation file size mismatch: field {k!r}")
        if _sha256_file(p) != info["sha256"]:
            _die(f"annotation input: generation file hash mismatch: field {k!r}")
        arr = np.load(p)
        if int(arr.shape[0]) != info["rows"]:
            _die(f"annotation input: generation file row mismatch: field {k!r}")
        arrs[k] = arr
    missing = [k for k in ANNOT_FIELDS if k not in arrs]
    if missing:
        _die(f"annotation input: generation is missing fields {missing}")
    _validate_annot_records(arrs, prod["config"]["window"],
                            prod["config"]["K"], f"annotation input {gen_dir!r}")
    summary = {"generation": man.get("generation"),
               "manifest_sha256": _sha256_bytes(man_raw),
               "total": man.get("total"),
               "teacher_config_hash": prod.get("config_sha256"),
               "teacher_weights_md5": prod.get("weights_md5"),
               "belief": meta["teacher"]["belief"]}
    return summary, meta, arrs


# --------------------------- build side ------------------------------------


def load_build_source(arg):
    """Verify-or-refuse the build generation (cnet2_build
    .load_generation caliber: pointer/manifest generation match,
    per-file sha256, meta seal where present) and load the feature
    arrays. Returns (summary, meta, feats) with feats keyed by field."""
    root, ptr = _resolve_pointer(arg, "build input")
    gen_dir = os.path.join(root, ptr["dir"])
    man_path = os.path.join(gen_dir, "manifest.json")
    if not os.path.exists(man_path):
        _die(f"build input: generation manifest missing: {man_path!r}")
    man = json.load(open(man_path))
    if man.get("generation") != ptr.get("generation"):
        _die("build input: pointer/manifest generation mismatch")
    for name, info in man.get("fields", {}).items():
        p = os.path.join(gen_dir, info["file"])
        if not os.path.exists(p):
            _die(f"build input: generation file missing: {p!r}")
        if _sha256_file(p) != info["sha256"]:
            _die(f"build input: generation file hash mismatch: {p!r}")
    meta_sha = man.get("meta_sha256")
    meta_p = os.path.join(gen_dir, "meta.json")
    if meta_sha is not None:
        if not os.path.exists(meta_p):
            _die(f"build input: generation file missing: {meta_p!r}")
        if _sha256_file(meta_p) != meta_sha:
            _die(f"build input: generation meta hash mismatch: {meta_p!r}")
    meta = json.load(open(meta_p))
    missing = [k for k in FEATURE_FIELDS if k not in man.get("fields", {})]
    if missing:
        _die(f"build input: generation {man.get('generation')} declares "
             f"no feature field(s) {missing}; cannot join")
    feats = {k: np.load(os.path.join(gen_dir, man["fields"][k]["file"]))
             for k in FEATURE_FIELDS}
    total = int(man["total"])
    for k in FEATURE_FIELDS:
        if int(feats[k].shape[0]) != total:
            _die(f"build input: feature field {k!r} has "
                 f"{feats[k].shape[0]} rows, manifest total is {total}")
    summary = {"generation": man.get("generation"),
               "manifest_sha256": _sha256_file(man_path),
               "total": total, "cfg_fp": man.get("cfg_fp")}
    return summary, meta, feats


# ------------------------------ the join -----------------------------------


def _game_stem(key):
    """Build games-table key (replay basename) -> bare game stem, the
    name space the annotation games table uses."""
    for suf in (".json.gz", ".gz", ".json"):
        if key.endswith(suf):
            return key[:-len(suf)]
    return key


def _seat_row_map(feats, build_gi, sides, game_name):
    """{seat: {step: row_index}} for one build game, read off the
    generation's side-major row order: rows of a game are stored as
    each side's steps ascending, sides in the meta entry's order
    (cnet2_build's own layout, cf. its weighted-branch comment).
    Segmentation splits where the step sequence resets; a segment
    count different from the recorded sides, or a duplicate step
    inside one side, makes the seat attribution ambiguous and dies by
    name rather than guessing."""
    gid = np.asarray(feats["gid"]).astype(np.int64)
    step = np.asarray(feats["step"]).astype(np.int64)
    rows = np.nonzero(gid == int(build_gi))[0]
    if len(rows) == 0:
        _die(f"join: build game {game_name!r} (gi {build_gi}) has no rows")
    st = step[rows]
    bounds = [0]
    for i in range(1, len(rows)):
        if st[i] <= st[i - 1]:
            bounds.append(i)
    bounds.append(len(rows))
    if len(bounds) - 1 != len(sides):
        _die(f"join: build game {game_name!r}: step sequence segments "
             f"into {len(bounds) - 1} side block(s) but its meta entry "
             f"records sides {sides}; seat attribution is ambiguous, "
             f"refusing to guess")
    out = {}
    for j, seat in enumerate(sides):
        seg = rows[bounds[j]:bounds[j + 1]]
        m = {}
        for ri in seg:
            s = int(step[ri])
            if s in m:
                _die(f"join: build game {game_name!r} seat {seat}: step "
                     f"{s} appears twice in one side block")
            m[s] = int(ri)
        out[int(seat)] = m
    return out


def join(annot_arg, build_arg):
    """Assemble the consumer product in memory. Returns
    (feat_arrays, policy_bundle, value_bundle, meta, producer, stats).
    Every refusal names the root / game / field responsible."""
    a_sum, a_meta, A = load_annotation_source(annot_arg)
    b_sum, b_meta, B = load_build_source(build_arg)

    # Games matched across the two tables BY NAME; the annotation
    # games table is authoritative for the output gid space.
    build_by_stem = {}
    for g in b_meta["games"]:
        stem = _game_stem(g["key"])
        if stem in build_by_stem:
            _die(f"join: build games table maps two entries to game "
                 f"{stem!r}; refusing")
        build_by_stem[stem] = g
    games_out = []
    seat_maps = {}   # annot gid -> (build game entry, {seat: {step: row}})
    for g in a_meta["games"]:
        name = g["game"]
        bg = build_by_stem.get(name)
        if bg is None:
            _die(f"join: annotation game {name!r} (gid {g['gid']}) has "
                 f"no counterpart in the build generation "
                 f"{b_sum['generation']}; refusing")
        seat_maps[int(g["gid"])] = (
            bg, _seat_row_map(B, bg["gi"], bg["sides"], name))
        games_out.append({"game": name, "gid": int(g["gid"]),
                          "build_gi": int(bg["gi"]), "build_key": bg["key"],
                          "n_roots": int(g["n_roots"])})

    R = int(A["step"].shape[0])
    K = int(A["arms"].shape[1])
    OWN = int(A["arms"].shape[2])
    row_idx = np.empty(R, dtype=np.int64)
    for r in range(R):
        gid = int(A["gid"][r]); step = int(A["step"][r]); seat = int(A["seat"][r])
        bg, smap = seat_maps[gid]
        per_seat = smap.get(seat)
        if per_seat is None:
            _die(f"join: root #{r} (game {bg['key']!r}, gid {gid}, step "
                 f"{step}, seat {seat}): the build generation never "
                 f"produced seat {seat} for this game (sides {bg['sides']})")
        ri = per_seat.get(step)
        if ri is None:
            _die(f"join: root #{r} (game {bg['key']!r}, gid {gid}, step "
                 f"{step}, seat {seat}): no feature row in the build "
                 f"generation {b_sum['generation']} for this root")
        row_idx[r] = ri

    # Feature arrays in annotation root order; gid rewritten to the
    # annotation gid (game := gid), step asserted identical.
    feats = {}
    for k in FEATURE_FIELDS:
        feats[k] = np.asarray(B[k])[row_idx]
    if not np.array_equal(feats["step"].astype(np.int64),
                          np.asarray(A["step"]).astype(np.int64)):
        bad = int(np.nonzero(feats["step"].astype(np.int64)
                             != np.asarray(A["step"]).astype(np.int64))[0][0])
        _die(f"join: root #{bad}: joined feature step "
             f"{int(feats['step'][bad])} != annotation step "
             f"{int(A['step'][bad])} (internal join error)")
    feats["gid"] = np.asarray(A["gid"]).astype(feats["gid"].dtype)

    # Sentinel translation -2 -> -1 on padding arms, per-root asserted.
    arms_src = np.asarray(A["arms"]).astype(np.int64)
    n_arms = np.asarray(A["n_arms"]).astype(np.int64)
    arms = arms_src.copy()
    translated = 0
    roots_with_padding = 0
    for r in range(R):
        n = int(n_arms[r])
        if n < K:
            roots_with_padding += 1
            pad = arms_src[r, n:]
            if not np.all(pad == ARM_PAD_SRC):
                _die(f"join: root #{r}: padding arms are not uniformly "
                     f"{ARM_PAD_SRC} (annotator contract); refusing to "
                     f"translate a non-sentinel padding block")
            arms[r, n:] = ARM_PAD_DST
            translated += int(pad.size)
        if np.any(arms_src[r, :n] == ARM_PAD_SRC):
            _die(f"join: root #{r}: sentinel {ARM_PAD_SRC} inside a "
                 f"real arm (candidate "
                 f"{int(np.nonzero(np.any(arms_src[r, :n] == ARM_PAD_SRC, axis=1))[0][0])}); "
                 f"a real arm carrying the padding sentinel is a "
                 f"corrupt record, refusing")
    residual = int((arms == ARM_PAD_SRC).sum())
    if residual:
        _die(f"join: post-translation sweep found {residual} residual "
             f"{ARM_PAD_SRC} entries in arms; refusing to publish")

    keys = {"game": np.asarray(A["gid"]).astype(np.int64),
            "step": np.asarray(A["step"]).astype(np.int64),
            "seat": np.asarray(A["seat"]).astype(np.int64)}
    policy = dict(keys)
    policy.update({
        "arms": arms, "n_arms": n_arms,
        "priors": np.asarray(A["priors"]).astype(np.float64),
        "q": np.asarray(A["q"]).astype(np.float64),
        "pi0": np.asarray(A["pi0"]).astype(np.float64),
        "pi1": np.asarray(A["pi1"]).astype(np.float64),
        "pi": np.asarray(A["pi"]).astype(np.float64)})
    value = dict(keys)
    value["Q"] = np.asarray(A["Q"]).astype(np.float64)

    code_md5 = _md5_file(os.path.abspath(__file__))
    translation = {"field": "arms", "from": ARM_PAD_SRC, "to": ARM_PAD_DST,
                   "entries": int(translated),
                   "roots_with_padding": int(roots_with_padding),
                   "roots": R}
    sources = {
        "annotation": {"generation": a_sum["generation"],
                       "manifest_sha256": a_sum["manifest_sha256"]},
        "build": {"generation": b_sum["generation"],
                  "manifest_sha256": b_sum["manifest_sha256"],
                  "cfg_fp": b_sum["cfg_fp"]}}
    producer = {"teacher_config_hash": a_sum["teacher_config_hash"],
                "teacher_weights_md5": a_sum["teacher_weights_md5"],
                "belief": a_sum["belief"],
                "k0_join_md5": code_md5,
                "code_md5": {"k0_join.py": code_md5},
                "sources": sources,
                "sentinel_translation": translation}
    meta = {"format": OUT_FORMAT, "total": R, "games": games_out,
            "sources": sources,
            "join": {"sentinel_translation": translation,
                     "candidates": int(n_arms.sum()), "max_candidates": K},
            "teacher": a_meta["teacher"],
            "record_schema": {
                "feature_files": "one .npy per k0_distill FEATURE_FIELDS "
                                   "entry, rows in annotation root order; "
                                   "gid rewritten to the annotation gid",
                "policy_labels": "policy_labels.npz: game (=annotation "
                                 "gid), step, seat, arms (-1 = no action "
                                 "at that slot; padding arms all -1), "
                                 "n_arms, priors, q, pi0, pi1, pi",
                "value_labels": "value_labels.npz: game, step, seat, Q",
                "games": "games table maps annotation gid <-> build gi "
                         "by game name"}}
    stats = {"roots": R, "candidates": int(n_arms.sum()),
             "sentinel_translation": translation}
    return feats, policy, value, meta, producer, stats


# ------------------------------ publish ------------------------------------


def _write_product(staging, feats, policy, value, meta):
    """Write the field files + bundles + meta into a staging
    generation dir; returns the fields table for the manifest."""
    finfo = {}
    for k in FEATURE_FIELDS:
        p = os.path.join(staging, k + ".npy")
        np.save(p, np.asarray(feats[k]))
        finfo[k] = {"file": k + ".npy", "bytes": os.path.getsize(p),
                    "rows": int(np.asarray(feats[k]).shape[0]),
                    "sha256": _sha256_file(p)}
    for name, bundle in (("policy_labels", policy), ("value_labels", value)):
        p = os.path.join(staging, name + ".npz")
        with open(p, "wb") as fh:
            np.savez(fh, **bundle)
        finfo[name] = {"file": name + ".npz", "bytes": os.path.getsize(p),
                       "rows": int(np.asarray(bundle["game"]).shape[0]),
                       "sha256": _sha256_file(p)}
    with open(os.path.join(staging, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1, sort_keys=True)
    return finfo


def _consumer_preflight(ds_root):
    """Load the staged product through the real consumer
    (k0_distill.load_annotation): every consumer gate must pass on
    the staged bytes before anything is published."""
    try:
        import k0_distill as D
    except Exception as e:
        _die(f"consumer preflight unavailable: cannot import "
             f"k0_distill from {HERE!r} ({e!r}); refusing to publish "
             f"an uncertified product")
    ds = D.load_annotation(ds_root)
    return {"generation": ds["info"]["generation"], "roots": ds["info"]["roots"],
            "candidates": ds["info"]["candidates"],
            "ipw_active": ds["info"]["ipw_active"]}


def publish(annot_arg, build_arg, out_dir):
    feats, policy, value, meta, producer, stats = join(annot_arg, build_arg)
    total = int(meta["total"])
    os.makedirs(os.path.join(out_dir, "generations"), exist_ok=True)
    work = os.path.join(out_dir, f".join-work-{os.getpid()}")
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(os.path.join(work, "generations"))
    try:
        staging = os.path.join(work, ".staging")
        os.makedirs(staging)
        finfo = _write_product(staging, feats, policy, value, meta)
        core = {"format": OUT_FORMAT, "total": total, "producer": producer,
                "meta_sha256": _sha256_file(os.path.join(staging, "meta.json")),
                "fields": finfo}
        gen_id = "gen-" + _sha256_bytes(_canonical(core))[:16]
        man = dict(core)
        man["generation"] = gen_id
        man["created"] = time.time()
        man["self_sha256"] = _sha256_bytes(_canonical(man))
        with open(os.path.join(staging, "manifest.json"), "w") as fh:
            json.dump(man, fh, indent=1, sort_keys=True)
        gen_tmp = os.path.join(work, "generations", gen_id)
        os.replace(staging, gen_tmp)
        ptr = {"generation": gen_id, "dir": f"generations/{gen_id}",
               "manifest_sha256": _sha256_file(
                   os.path.join(gen_tmp, "manifest.json")),
               "total": total, "created": time.time()}
        with open(os.path.join(work, "current.json"), "w") as fh:
            json.dump(ptr, fh)
        # Consumer preflight on the staged dataset, before the real
        # output root is touched at all.
        pre = _consumer_preflight(work)

        gen_dir = os.path.join(out_dir, "generations", gen_id)
        reused = False
        if os.path.isdir(gen_dir):
            man_p = os.path.join(gen_dir, "manifest.json")
            if not os.path.exists(man_p):
                if os.listdir(gen_dir):
                    _die(f"publish: generation directory {gen_dir!r} "
                         f"exists without manifest.json and is not "
                         f"empty; refusing to overwrite it")
                os.replace(gen_tmp, gen_dir)  # empty leftover dir only
            else:
                old = json.load(open(man_p))
                old_core = {k: v for k, v in old.items() if k != "self_sha256"}
                if old.get("self_sha256") != _sha256_bytes(_canonical(old_core)):
                    _die(f"publish: published generation {gen_id} in "
                         f"{gen_dir!r} fails its manifest self-seal; "
                         f"refusing to overwrite a published generation")
                want_core = dict(core)
                want_core["generation"] = gen_id
                have_core = {k: v for k, v in old.items()
                             if k not in ("self_sha256", "created")}
                if have_core != want_core:
                    _die(f"publish: published generation {gen_id} in "
                         f"{gen_dir!r} differs from the inputs being "
                         f"published; refusing to overwrite it")
                for k, info in old["fields"].items():
                    p = os.path.join(gen_dir, info["file"])
                    if not os.path.exists(p) or os.path.getsize(p) != info["bytes"] \
                            or _sha256_file(p) != info["sha256"]:
                        _die(f"publish: published generation {gen_id} "
                             f"field {k!r} in {gen_dir!r} does not match "
                             f"its manifest; refusing to overwrite it")
                reused = True
        elif os.path.exists(gen_dir):
            _die(f"publish: {gen_dir!r} exists and is not a directory; "
                 f"refusing to overwrite it")
        else:
            os.replace(gen_tmp, gen_dir)
        ptr = {"generation": gen_id, "dir": f"generations/{gen_id}",
               "manifest_sha256": _sha256_file(
                   os.path.join(gen_dir, "manifest.json")),
               "total": total, "created": time.time()}
        ptr_tmp = os.path.join(out_dir, f".current-{os.getpid()}.tmp")
        with open(ptr_tmp, "w") as fh:
            json.dump(ptr, fh)
        os.replace(ptr_tmp, os.path.join(out_dir, "current.json"))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return {"generation": gen_id, "total": total, "reused": reused,
            "preflight": pre, **stats}


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="K0 round-2 join: annotation generation + build "
                    "generation -> cnet2-k0-annotation/1")
    ap.add_argument("--annot", help="annotation dataset root (pointer)")
    ap.add_argument("--build", help="build dataset root (pointer)")
    ap.add_argument("--out", required=True, help="output dataset root")
    ap.add_argument("--verify", action="store_true",
                    help="consumer-load the sealed product at --out "
                         "(k0_distill.load_annotation) and exit")
    a = ap.parse_args(argv)
    if a.verify:
        ds = _consumer_preflight(a.out)
        print(json.dumps({"k0_join_verified": ds}))
        return 0
    if not (a.annot and a.build):
        ap.error("--annot and --build are required unless --verify")
    summary = publish(a.annot, a.build, a.out)
    print(json.dumps({"k0_joined": summary}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

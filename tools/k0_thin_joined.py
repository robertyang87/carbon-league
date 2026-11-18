#!/usr/bin/env python3
# =====================================================================
# k0_thin_joined -- join-layer replay thinning for the sparse-label
# /3 validation (DESIGN_SPARSE_LABEL_20261118 + ACCEPT a7da515d).
#
# Takes a full-label joined product (cnet2-k0-annotation/2, e.g. the
# ring-1 joined_u gen-90faee2ef4221d59) and publishes a NEW
# cnet2-k0-annotation/3 generation in which a registered sampler's
# chosen roots keep their policy labels (has_q=1) and the remaining
# roots carry the root-level lazy form (has_q=0: n_arms=0, arms -1,
# priors 0, q 0, Q NaN, pi placeholder identity 1.0/0.0/1.0 -- the
# placeholder pi is NOT a record and enters no statistic; the /3
# loader takes its semantics from has_q alone). Terminal (value)
# labels are carried for EVERY root unchanged: value supervision is
# full-coverage by design. The hidden true labels are NOT destroyed:
# the source product is never modified, and the mask is a pure
# function of (source generation, sampler registration) recorded in
# the product meta, so the scoring truth stays recoverable from the
# source.
#
# Sampler (registered in REGISTRATION.md before any run):
#   id "q-sampler-replay/1", seed 20261118, universe = the kept
#   games' roots in stable (game, step, seat) order, per-stratum
#   sampling without replacement; per-root p_q = the stratum's exact
#   inclusion probability drawn/stratum_size.
#   arm "full":     has_q == 1 and p_q == 1.0 everywhere (the
#                   regression-gate / baseline form).
#   arm "random":   strata = step band x seat, rate r per stratum.
#   arm "keyboost": key set K = {step >= 200} u {roots of close
#                   games}; close games are defined ONLY from
#                   build-side observables (the per-game terminal
#                   margin magnitude, a game-result quantity known
#                   before any annotation): games whose |margin| is
#                   at or below the 25th percentile over the kept
#                   games. Total budget B = round(N * r); key draw =
#                   min(round(2.5 * r * |K|), floor(0.8 * B)); the
#                   remainder is drawn uniformly from non-key roots.
#
# Publish discipline mirrors k0_join_u.publish: staging dir, field
# hashes, content-derived generation id, manifest self-seal,
# consumer preflight through the REAL /3 loader (the sparse
# k0_distill_u copy in this directory) before the pointer flips,
# never overwrite a published generation (identical core = reuse).
#
# The seal step writes the consumption sidecar for a /3 product.
# The stock tools/consume_preflight.py classifies targets by the
# /2 formats only and refuses /3 by design, so the /3 line seals
# its own sidecar in the same k0-consumption-sidecar/1 format:
# content-addressed, never silently overwritten, binding the
# generation actually loaded, with the tool block naming THIS
# tool and the checks actually performed stated verbatim. The
# k0_distill_u consumption hook (self-seal + binding) is unchanged
# and applies to these sidecars exactly as to preflight ones.
#
# Usage:
#   python3 k0_thin_joined.py holdout --source DS --out holdout.json
#   python3 k0_thin_joined.py thin --source DS --out DSROOT
#       --arm {full,random,keyboost} [--rate R] [--seed S]
#       [--exclude-games holdout.json] [--annotation-root DS]
#   python3 k0_thin_joined.py verify --data DSROOT
# =====================================================================
import argparse
import hashlib
import io
import json
import os
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

OUT_FORMAT = "cnet2-k0-annotation/3"
SRC_FORMAT = "cnet2-k0-annotation/2"
SAMPLER_ID = "q-sampler-replay/1"
STEP_BANDS = [[0, 50], [50, 100], [100, 150], [150, 200],
              [200, 250], [250, 299]]
KEY_STEP_MIN = 200
CLOSE_QUANTILE = 0.25
KEY_BOOST = 2.5
KEY_BUDGET_CAP = 0.8
FEATURE_FIELDS = ("maps", "scal", "tok_pos", "tok_kind", "tok_mask",
                  "tok_cargo", "hist_cell", "hist_cargo", "hist_move",
                  "uf", "mf", "legal", "carbon", "teacher", "gid", "step")


def _die(msg):
    raise SystemExit(f"k0_thin_joined: {msg}")


def _sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def _sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def _md5_file(p):
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def _canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def _resolve_pointer(arg, what):
    if os.path.isfile(arg):
        ptr_path = arg
        root = os.path.dirname(os.path.abspath(arg))
    elif os.path.exists(os.path.join(arg, "current.json")):
        ptr_path = os.path.join(arg, "current.json")
        root = arg
    else:
        _die(f"{what}: {arg!r} is not a dataset root (no current.json)")
    return root, json.load(open(ptr_path))


def load_source(arg):
    """Load a sealed /2 joined product, verifying pointer, manifest
    self-seal, meta seal and every field hash (the consumer's own
    discipline, mirrored). Returns (root, man, meta, arrays) with
    arrays = {field: array} plus 'policy_labels'/'value_labels'
    bundle dicts, and the manifest's own sha256."""
    root, ptr = _resolve_pointer(arg, "source")
    gen_dir = os.path.join(root, ptr["dir"])
    man_raw = open(os.path.join(gen_dir, "manifest.json"), "rb").read()
    man_sha = hashlib.sha256(man_raw).hexdigest()
    if ptr.get("manifest_sha256") is not None \
            and hashlib.sha256(man_raw).hexdigest() != ptr["manifest_sha256"]:
        _die("source manifest bytes do not match the pointer hash")
    man = json.loads(man_raw)
    if man.get("generation") != ptr.get("generation"):
        _die("source pointer/manifest generation mismatch")
    if man.get("format") != SRC_FORMAT:
        _die(f"source format {man.get('format')!r} != {SRC_FORMAT!r}")
    core = {k: v for k, v in man.items() if k != "self_sha256"}
    if man.get("self_sha256") != _sha256_bytes(_canonical(core)):
        _die("source manifest self-seal mismatch")
    meta_raw = open(os.path.join(gen_dir, "meta.json"), "rb").read()
    if hashlib.sha256(meta_raw).hexdigest() != man.get("meta_sha256"):
        _die("source meta seal mismatch")
    meta = json.loads(meta_raw)
    arrays = {}
    for k, info in sorted(man["fields"].items()):
        p = os.path.join(gen_dir, info["file"])
        raw = open(p, "rb").read()
        if len(raw) != info["bytes"] or hashlib.sha256(raw).hexdigest() \
                != info["sha256"]:
            _die(f"source field {k!r} fails its manifest hash/size")
        if info["file"].endswith(".npy"):
            arrays[k] = np.load(io.BytesIO(raw))
        else:
            with np.load(io.BytesIO(raw)) as z:
                arrays[k] = {kk: z[kk] for kk in z.files}
    return root, man, meta, arrays, man_sha


# ------------------------------- holdout ---------------------------------


def compute_holdout(meta, seed, count):
    names = sorted(g["game"] for g in meta["games"])
    rs = np.random.RandomState(seed)
    order = rs.permutation(len(names))
    hold = sorted(names[i] for i in order[:count])
    train = sorted(set(names) - set(hold))
    return {"seed": int(seed), "count": int(count),
            "universe_games": len(names),
            "holdout_games": hold, "train_games": train,
            "method": "RandomState(seed).permutation over the sorted "
                      "game-name list; first `count` names are held out"}


# ------------------------------- sampler ---------------------------------


def run_sampler(keys, margins_by_gid, arm, rate, seed):
    """keys: (game_gid, step, seat) int arrays in stable order over
    the kept roots. margins_by_gid: {gid: |terminal_margin|} (game
    result observable). Returns (has_q uint8, p_q float64, sampler
    record dict)."""
    R = len(keys[0])
    has_q = np.zeros(R, dtype=np.uint8)
    p_q = np.zeros(R, dtype=np.float64)
    rec = {"id": SAMPLER_ID, "arm": arm, "rate": rate, "seed": int(seed),
           "step_bands": STEP_BANDS, "strata": []}
    if arm == "full":
        has_q[:] = 1
        p_q[:] = 1.0
        rec["strata"].append({"stratum": "full", "roots": int(R),
                              "drawn": int(R), "p_q": 1.0})
        rec["full_coverage"] = True
        return has_q, p_q, rec
    rng = np.random.RandomState(seed)
    game, step, seat = keys
    if arm == "random":
        band = _band_arr(step)
        for bi in range(len(STEP_BANDS)):
            for sd in (0, 1):
                pos = np.nonzero((band == bi) & (seat == sd))[0]
                drawn = int(round(len(pos) * rate))
                _draw(rng, pos, drawn, has_q, p_q, rec,
                      f"band{bi}_seat{sd}")
    elif arm == "keyboost":
        # game-level |margin|: one value per game (the margin
        # magnitude is seat-symmetric).
        gids = np.unique(game)
        gvals = np.array([margins_by_gid[int(g)] for g in gids])
        thr = float(np.quantile(gvals, CLOSE_QUANTILE))
        close_gids = {int(g) for g, v in zip(gids, gvals) if v <= thr}
        is_key = (step >= KEY_STEP_MIN) | np.array(
            [int(g) in close_gids for g in game])
        rec["key_definition"] = {
            "step_min": KEY_STEP_MIN,
            "close_games": "|terminal_margin| <= 25th percentile of "
                           "game-level |terminal_margin| over kept "
                           "games (game-result observable, not q/Q)",
            "close_threshold_abs_margin": thr,
            "close_game_count": len(close_gids),
            "key_roots": int(is_key.sum())}
        budget = int(round(R * rate))
        key_pos = np.nonzero(is_key)[0]
        non_pos = np.nonzero(~is_key)[0]
        key_drawn = min(int(round(KEY_BOOST * rate * len(key_pos))),
                        int(KEY_BUDGET_CAP * budget), len(key_pos))
        non_drawn = min(budget - key_drawn, len(non_pos))
        _draw(rng, key_pos, key_drawn, has_q, p_q, rec, "key")
        _draw(rng, non_pos, non_drawn, has_q, p_q, rec, "nonkey")
        rec["budget"] = budget
    else:
        _die(f"unknown sampler arm {arm!r}")
    rec["roots"] = int(R)
    rec["roots_q"] = int(has_q.sum())
    return has_q, p_q, rec


def _band_arr(step):
    out = np.zeros(len(step), dtype=np.int64)
    for bi, (lo, hi) in enumerate(STEP_BANDS):
        if bi == len(STEP_BANDS) - 1:
            out[(step >= lo) & (step <= hi)] = bi
        else:
            out[(step >= lo) & (step < hi)] = bi
    return out


def _draw(rng, pos, drawn, has_q, p_q, rec, name):
    if len(pos) == 0:
        rec["strata"].append({"stratum": name, "roots": 0, "drawn": 0,
                              "p_q": None})
        return
    drawn = max(0, min(int(drawn), len(pos)))
    if drawn:
        pick = rng.choice(len(pos), size=drawn, replace=False)
        has_q[pos[pick]] = 1
    p_q[pos] = drawn / len(pos)
    rec["strata"].append({"stratum": name, "roots": int(len(pos)),
                          "drawn": int(drawn),
                          "p_q": drawn / len(pos)})


# ------------------------------- thinning --------------------------------


def thin(source_arg, out_dir, arm, rate, seed, exclude_games,
         annotation_root):
    src_root, man, meta, arrs, src_man_sha = load_source(source_arg)
    pol = arrs["policy_labels"]
    val = arrs["value_labels"]
    gid = np.asarray(pol["game"]).astype(np.int64)
    step = np.asarray(pol["step"]).astype(np.int64)
    seat = np.asarray(pol["seat"]).astype(np.int64)
    name_by_gid = {int(g["gid"]): g["game"] for g in meta["games"]}
    keep = np.array([name_by_gid[int(g)] not in exclude_games
                     for g in gid])
    kept_idx = np.nonzero(keep)[0]
    # Stable (game, step, seat) order over kept roots: the source is
    # already in annotation root order; sort explicitly so the
    # sampler universe does not depend on source row order.
    order = np.lexsort((seat[kept_idx], step[kept_idx], gid[kept_idx]))
    kept_idx = kept_idx[order]
    keys = (gid[kept_idx], step[kept_idx], seat[kept_idx])
    t_m = np.asarray(val["terminal_margin"]).astype(np.float64)
    margins_by_gid = {}
    for g in np.unique(gid[kept_idx]):
        rows = np.nonzero(gid[kept_idx] == int(g))[0]
        margins_by_gid[int(g)] = float(abs(t_m[kept_idx][rows[0]]))
    has_q, p_q, srec = run_sampler(keys, margins_by_gid, arm, rate, seed)
    R = len(kept_idx)
    is_v = has_q == 0

    feats = {k: np.asarray(arrs[k])[kept_idx] for k in FEATURE_FIELDS}
    policy = {"game": gid[kept_idx], "step": step[kept_idx],
              "seat": seat[kept_idx]}
    for k in ("arms", "n_arms", "priors", "q", "pi0", "pi1", "pi"):
        policy[k] = np.array(pol[k])[kept_idx]
    if is_v.any():
        policy["arms"][is_v] = -1
        policy["n_arms"][is_v] = 0
        policy["priors"][is_v] = 0.0
        policy["q"][is_v] = 0.0
        policy["pi0"][is_v] = 1.0
        policy["pi1"][is_v] = 0.0
        policy["pi"][is_v] = 1.0
    policy["has_q"] = has_q
    policy["p_q"] = p_q
    value = {"game": gid[kept_idx], "step": step[kept_idx],
             "seat": seat[kept_idx]}
    for k in ("Q", "terminal_u", "terminal_wdl", "terminal_margin"):
        value[k] = np.array(val[k])[kept_idx]
    if is_v.any():
        value["Q"][is_v] = np.nan

    kept_games = [g for g in meta["games"]
                  if g["game"] not in exclude_games]
    new_meta = json.loads(json.dumps(meta))
    new_meta["format"] = OUT_FORMAT
    new_meta["total"] = int(R)
    new_meta["games"] = kept_games
    new_meta["q_sampler"] = srec
    new_meta["roots_q"] = int(has_q.sum())
    new_meta["roots_total"] = int(R)
    new_meta["record_schema"] = dict(new_meta.get("record_schema") or {})
    new_meta["record_schema"]["policy_labels"] = (
        "policy_labels.npz: game (=annotation gid), step, seat, arms, "
        "n_arms, priors, q, pi0, pi1, pi, has_q (uint8), p_q "
        "(float64 design inclusion probability of the root's "
        "stratum). has_q=0 roots carry the root-level lazy form "
        "(n_arms=0, arms -1, priors 0, q 0, pi placeholder identity "
        "1.0/0.0/1.0 -- the placeholder pi is not a record and must "
        "enter no weight, ESS, activation or monitoring statistic); "
        "their Q in value_labels is NaN. Terminal fields cover "
        "every root.")
    producer = json.loads(json.dumps(man["producer"]))
    producer["sparse_derivation"] = {
        "tool": "k0_thin_joined.py",
        "tool_md5": _md5_file(os.path.abspath(__file__)),
        "source_generation": man["generation"],
        "source_manifest_sha256": src_man_sha,
        "sampler": {k: srec[k] for k in ("id", "arm", "rate", "seed")},
        "excluded_games": sorted(exclude_games),
        "mask_sha256": _sha256_bytes(has_q.tobytes()
                                     + p_q.tobytes())}
    summary = publish(out_dir, feats, policy, value, new_meta, producer)
    summary["roots_q"] = int(has_q.sum())
    summary["roots_total"] = int(R)
    if annotation_root:
        seal(out_dir, annotation_root)
        summary["sidecar"] = "sealed"
    return summary


def publish(out_dir, feats, policy, value, meta, producer):
    """Mirror of k0_join_u.publish for the /3 product: staging,
    consumer preflight through the sparse /3 loader, no overwrite
    of a published generation."""
    total = int(meta["total"])
    os.makedirs(os.path.join(out_dir, "generations"), exist_ok=True)
    work = os.path.join(out_dir, f".thin-work-{os.getpid()}")
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(os.path.join(work, "generations"))
    try:
        staging = os.path.join(work, ".staging")
        os.makedirs(staging)
        finfo = {}
        for k in FEATURE_FIELDS:
            p = os.path.join(staging, k + ".npy")
            np.save(p, np.asarray(feats[k]))
            finfo[k] = {"file": k + ".npy",
                        "bytes": os.path.getsize(p),
                        "rows": int(np.asarray(feats[k]).shape[0]),
                        "sha256": _sha256_file(p)}
        for name, bundle in (("policy_labels", policy),
                             ("value_labels", value)):
            p = os.path.join(staging, name + ".npz")
            with open(p, "wb") as fh:
                np.savez(fh, **bundle)
            finfo[name] = {"file": name + ".npz",
                           "bytes": os.path.getsize(p),
                           "rows": int(np.asarray(bundle["game"]).shape[0]),
                           "sha256": _sha256_file(p)}
        with open(os.path.join(staging, "meta.json"), "w") as fh:
            json.dump(meta, fh, indent=1, sort_keys=True)
        core = {"format": OUT_FORMAT, "total": total,
                "producer": producer,
                "meta_sha256": _sha256_file(
                    os.path.join(staging, "meta.json")),
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
        # Consumer preflight: the real /3 loader must accept the
        # staged bytes before anything is published.
        import k0_distill_u as D
        ds = D.load_annotation(work)
        pre = {"generation": ds["info"]["generation"],
               "roots": ds["info"]["roots"],
               "roots_q": ds["info"].get("roots_q")}
        gen_dir = os.path.join(out_dir, "generations", gen_id)
        reused = False
        if os.path.isdir(gen_dir):
            old = json.load(open(os.path.join(gen_dir, "manifest.json")))
            old_core = {k: v for k, v in old.items()
                        if k not in ("self_sha256", "created")}
            want_core = dict(core)
            want_core["generation"] = gen_id
            if old_core != want_core:
                _die(f"published generation {gen_id} in {gen_dir!r} "
                     f"differs from this thinning; refusing to "
                     f"overwrite")
            reused = True
        elif os.path.exists(gen_dir):
            _die(f"{gen_dir!r} exists and is not a directory")
        else:
            os.replace(gen_tmp, gen_dir)
        ptr = {"generation": gen_id, "dir": f"generations/{gen_id}",
               "manifest_sha256": _sha256_file(
                   os.path.join(gen_dir, "manifest.json")),
               "total": total, "created": time.time()}
        tmp = os.path.join(out_dir, f".current-{os.getpid()}.tmp")
        with open(tmp, "w") as fh:
            json.dump(ptr, fh)
        os.replace(tmp, os.path.join(out_dir, "current.json"))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return {"generation": gen_id, "total": total, "reused": reused,
            "preflight": pre}


def seal(out_dir, annotation_root):
    """Seal CONSUMPTION.json for a /3 product (see header note):
    verifies the product through the sparse /3 loader and the
    annotation binding through the real join loader, then writes
    the content-addressed sidecar. Never overwrites a different
    sidecar."""
    import k0_distill_u as D
    import k0_join_u as JU
    root, ptr = _resolve_pointer(out_dir, "seal target")
    ds = D.load_annotation(root)
    gen_dir = ds["gen_dir"]
    man = json.load(open(os.path.join(gen_dir, "manifest.json")))
    man_sha = _sha256_file(os.path.join(gen_dir, "manifest.json"))
    prod = man["producer"]
    a_root, a_ptr = _resolve_pointer(annotation_root, "annotation root")
    a_sum, _a_meta, _a = JU.load_annotation_source(annotation_root)
    src = (prod.get("sources") or {}).get("annotation") or {}
    if a_sum["generation"] != src.get("generation") \
            or a_sum["manifest_sha256"] != src.get("manifest_sha256"):
        _die("annotation root does not match the product's recorded "
             "sources.annotation binding; refusing to seal")
    core = {
        "format": "k0-consumption-sidecar/1",
        "target": {"dataset_root": os.path.abspath(root),
                   "generation": man["generation"],
                   "manifest_sha256": man_sha,
                   "total": man["total"],
                   "teacher_config_hash": prod["teacher_config_hash"],
                   "teacher_weights_md5": prod["teacher_weights_md5"],
                   "kind": "joined"},
        "annotation": {"dataset_root": os.path.abspath(a_root),
                       "generation": a_sum["generation"],
                       "manifest_sha256": a_sum["manifest_sha256"],
                       "total": a_sum["total"],
                       "teacher_config_hash":
                           a_sum["teacher_config_hash"],
                       "teacher_weights_md5":
                           a_sum["teacher_weights_md5"]},
        "merged": None,
        "checks": {
            "target_loader_verified": True,
            "loader": "sparse /3 loader (k0_distill_u impl copy, "
                      "k0-distill-u/3+sparse1)",
            "joined_annotation_binding": True,
            "notice": "cnet2-k0-annotation/3 product derived by "
                      "join-layer replay thinning; the stock "
                      "consume_preflight classifies /2 formats "
                      "only, so this sidecar is sealed by "
                      "k0_thin_joined.py with the checks above "
                      "performed through the real loaders"},
        "tool": {"file": "k0_thin_joined.py",
                 "md5": _md5_file(os.path.abspath(__file__)),
                 "k0_distill_u_md5": _md5_file(
                     os.path.join(HERE, "k0_distill_u.py")),
                 "k0_join_u_md5": _md5_file(JU.__file__),
                 "source_generation": prod["sparse_derivation"]
                     ["source_generation"]}}
    sha = _sha256_bytes(_canonical(core))
    sidecar = dict(core)
    sidecar["locations"] = {"target_arg": out_dir,
                            "annotation_root": annotation_root,
                            "source_args": []}
    sidecar["created"] = time.time()
    sidecar["sidecar_sha256"] = sha
    path = os.path.join(root, "CONSUMPTION.json")
    if os.path.exists(path):
        old = json.load(open(path))
        if old.get("sidecar_sha256") == sha:
            return path, sha, True
        _die(f"existing CONSUMPTION.json in {root!r} has a different "
             f"content hash; a consumption record is never silently "
             f"overwritten -- refusing")
    tmp = f"{path}.tmp-{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(sidecar, fh, indent=1, sort_keys=True)
    os.replace(tmp, path)
    return path, sha, False


def main(argv=None):
    ap = argparse.ArgumentParser(description="join-layer replay "
                                             "thinning to /3")
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("holdout")
    h.add_argument("--source", required=True)
    h.add_argument("--seed", type=int, default=20261119)
    h.add_argument("--count", type=int, default=15)
    h.add_argument("--out", required=True)
    t = sub.add_parser("thin")
    t.add_argument("--source", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--arm", choices=("full", "random", "keyboost"),
                   required=True)
    t.add_argument("--rate", type=float, default=None)
    t.add_argument("--seed", type=int, default=20261118)
    t.add_argument("--exclude-games",
                   help="JSON file with a holdout_games list")
    t.add_argument("--annotation-root",
                   help="also seal CONSUMPTION.json against this "
                        "annotation dataset root")
    v = sub.add_parser("verify")
    v.add_argument("--data", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "holdout":
        _r, _m, meta, _a, _s = load_source(a.source)
        rec = compute_holdout(meta, a.seed, a.count)
        with open(a.out, "w") as fh:
            json.dump(rec, fh, indent=1, sort_keys=True)
        print(json.dumps({"holdout_games": rec["holdout_games"]}))
        return 0
    if a.cmd == "thin":
        if a.arm != "full" and a.rate is None:
            ap.error("--rate is required for sampled arms")
        exclude = set()
        if a.exclude_games:
            exclude = set(json.load(open(a.exclude_games))
                          ["holdout_games"])
        s = thin(a.source, a.out, a.arm, a.rate, a.seed, exclude,
                 a.annotation_root)
        print(json.dumps({"k0_thinned": s}))
        return 0
    if a.cmd == "verify":
        import k0_distill_u as D
        ds = D.load_annotation(a.data)
        print(json.dumps({"k0_thin_verified": {
            "generation": ds["info"]["generation"],
            "roots": ds["info"]["roots"],
            "roots_q": ds["info"].get("roots_q"),
            "format": ds["info"].get("format")}}))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())

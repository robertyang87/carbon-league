#!/usr/bin/env python3
# =====================================================================
# merge_annot_generations -- cross-config merge publisher for sealed
# k0 annotation generations (ring-3 step-segment mixing prerequisite).
#
# Spec basis:
#   * reports/FEASIBILITY_CROSS_CONFIG_MIX_20261118.md (bd36c4fe):
#     the current chain cannot mix generations of different
#     config_sha256 (record gate binds one window per generation; the
#     joined producer identity is scalar). The compliant fix is a NEW
#     merge publisher at the annotation-generation layer; pinned /
#     accepted files are untouched and only imported.
#   * reports/RULING_MERGED_GEN_PROVENANCE_SAMPLER_20261118.md
#     (7283a3f3): the merged generation is a NEW registered identity.
#     Its teacher_config_hash is computed from the normalized source
#     list; it must NOT borrow any source generation's config hash.
#     Training provenance citation is two-layer (merged identity +
#     expanded source list). The sampler: unit = root; each source
#     generation is drawn INDEPENDENTLY (20% of its own roots, never
#     pooled); draws are uniform without replacement over the source
#     roots' stable order (game, step, seat); actual drawn count =
#     floor(source roots * rate) per source; the current segment's
#     new data is taken in full, unsampled.
#   * Strategy-group reinforcements (2026-11-18):
#     (1) REGISTRATION FIRST: a merged generation must be registered
#         before it may be consumed. This tool writes an independent
#         registration file REGISTRATION.json into the output dataset
#         root (normalized source list + merged teacher_config_hash +
#         generation id + registration seed + seed derivation rule)
#         and flips current.json ONLY after the registration file has
#         landed. --verify checks that the registration file agrees
#         with the in-generation source list byte-for-byte (canonical
#         bytes). Registration-file existence is the publish gate.
#     (2) SEED SCHEME: ONE registration seed per merged generation
#         (required, no default; recorded in REGISTRATION.json). The
#         per-source sampling seed is DERIVED, never chosen freely:
#             derived_seed = int.from_bytes(
#                 sha256(utf8(registration_seed)
#                        + utf8(source_generation_id))[:8], "big")
#         Sampling for a source uses random.Random(derived_seed)
#         .sample() over that source's root indices in stable
#         (game, step, seat) order, without replacement; the selected
#         index set is then re-sorted into stable order. Each source's
#         derived seed value is recorded in the source list.
#     (3) OVERLAP DEDUP: a root appearing in more than one source
#         under the same (game, step, seat) may be kept once ONLY if
#         every field's row bytes are identical across the copies
#         (the gid field excluded: it is a per-generation plan label,
#         remapped on merge). ANY byte difference dies, naming the
#         root (game, step, seat) and the differing fields. Never
#         silently keep one copy, never average, never splice.
#   * Maintenance batch (strategy-group ruling
#     reports/CLEAR_MERGE_PUBLISHER_CONDITION_20261118.md, 43907b6a
#     SS2.2; future publishes only -- already-published generations
#     and identity hashes are untouched):
#     (a) GAMES PROVENANCE CARRY: the merged games table carries each
#         source game's provenance triple (pass / selected_sha256 /
#         shard_sha256) so per-game traceability is self-contained
#         in the merged generation. A game present in several
#         sources must record the SAME triple in every source; the
#         agreed values are carried verbatim. Any disagreement --
#         differing values, or a field recorded by one source and
#         absent from another -- dies naming the game and the field;
#         no source is ever silently preferred. A field absent from
#         every source for a game is simply not carried.
#     (c) SEED STORAGE FORM: the registration seed's STORED form
#         (/sampling/seed in meta, "seed" in REGISTRATION.json) is
#         an int when the seed string is the canonical decimal
#         representation of that int, else the string itself --
#         matching the source generations' int seed. The derivation
#         byte caliber NEVER depends on the storage form: derivation
#         always feeds utf8 of the canonical seed string (the exact
#         registration seed string), REGISTRATION.json records that
#         string explicitly as "seed_canonical", and str(stored
#         form) == canonical string in every case, so --verify
#         recomputes the same derived seeds for new (int-form) and
#         legacy (str-form, no seed_canonical field) registrations.
#     Format note (ruling 43907b6a SS2.1, in-register semantic
#     note): the merged generation's sampling block carries "seed"
#     and "seed_derivation" INSTEAD OF the annotator's pi0 / pi1 /
#     selected_sha256 -- pi* records the annotator's internal
#     two-stage sampling (the source-generation layer), while the
#     merge draw is a second, separate layer governed by the
#     registered seed derivation; the two layers are never recorded
#     in each other's terms.
#
# What the tool does:
#   1. Loads >= 2 sealed annotation generations through the REAL
#      loader (k0_join_u.load_annotation_source: pointer, manifest
#      self-seal, meta seal, teacher identity cross-check, per-field
#      bytes/sha/rows, S13-C record gates under each source's own
#      window). Any refusal aborts the merge.
#   2. Cross-source pre-checks, each dying by name: format identical;
#      code_md5s set identical; weights_md5 identical; teacher config
#      dicts identical field-by-field EXCEPT "window" (any other
#      differing field -- train_temp/K/M/H/temps/seed/belief/pi/...
#      -- dies naming the field); per-field dtype/trailing shape
#      identical; each source's recorded config_sha256 recomputes
#      from its config dict (annotator caliber: sha256 of canonical
#      config bytes).
#   3. Per-source selection: full sources contribute every root;
#      sample sources contribute floor(roots * rate) roots drawn by
#      the derived-seed sampler above. Same inputs + same seed =>
#      same selection, by construction.
#   4. Merge: games table union by name; gid re-planned in sorted
#      name order and remapped across all field arrays; roots
#      concatenated in (gid, step, seat) order; overlaps deduped
#      under rule (3) above. Merged config = source config with the
#      window replaced by the union [min lo, max hi).
#   5. Identity: teacher_config_hash = sha256 of the canonical bytes
#      of the normalized source list (per source: generation id,
#      manifest_sha256, source config_sha256, window, mode, rate,
#      source root count, actual drawn count, derived sample seed).
#      The value is written identically in THREE places: manifest
#      producer.config_sha256, manifest producer.teacher_config_hash,
#      meta.teacher.config_sha256. meta.teacher.config carries the
#      merged config object. The merged generation never presents
#      the union config as a single run's config: the manifest
#      producer and the meta both carry the full source list under
#      "merged_from".
#   6. Sealing mirrors the annotator's publish discipline: per-field
#      {file, bytes, rows, sha256}, meta_sha256 seal, manifest
#      self_sha256 over canonical bytes, generation id content-
#      addressed ("gen-" + first 16 hex of the canonical core hash),
#      an already-published identical generation is verified and
#      REUSED, never overwritten. Before anything becomes visible,
#      the staged generation is loaded through the real loader plus
#      the record gates under the union window. REGISTRATION.json
#      lands next; current.json flips LAST.
#
# --verify mode re-runs, on a merged dataset root: the real loader,
# the record gates under the union window, an independent recompute
# of teacher_config_hash against the in-generation source list, the
# per-source derived-seed recomputation against the registration
# seed, and the registration-file vs in-generation-list canonical
# byte equality check. All must pass.
#
# Usage:
#   python3 merge_annot_generations.py --out OUT_DS --seed SEED \
#       --full SRC_DS [--full SRC_DS ...] \
#       [--sample SRC_DS[:RATE] ...] [--code-dir DIR]
#   python3 merge_annot_generations.py --out OUT_DS --verify
#
# k0_join_u is imported, never modified. Its location is resolved
# from --code-dir, else $K0_CODE_DIR, else ../../code or ../code
# relative to this file (repo layout: tools/ beside code/).
# =====================================================================
import argparse
import copy
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

REGISTRATION_FORMAT = "k0-annotation-merge-registration/1"
SEED_DERIVATION = (
    "derived_seed = int.from_bytes(sha256(utf8(registration_seed) + "
    "utf8(source_generation_id))[:8], 'big'); sampling = "
    "random.Random(derived_seed).sample over the source's root "
    "indices in stable (game, step, seat) order, without replacement, "
    "draw = floor(source_roots * rate); selected indices re-sorted "
    "into stable order")


def _die(msg):
    raise SystemExit(f"merge_annot: {msg}")


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


def _atomic_write_json(obj, path):
    tmp = f"{path}.tmp-{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1, sort_keys=True)
    os.replace(tmp, path)


def _load_join_u(code_dir):
    cands = []
    if code_dir:
        cands.append(code_dir)
    if os.environ.get("K0_CODE_DIR"):
        cands.append(os.environ["K0_CODE_DIR"])
    cands.append(os.path.join(HERE, "..", "..", "code"))
    cands.append(os.path.join(HERE, "..", "code"))
    for d in cands:
        if d and os.path.isfile(os.path.join(d, "k0_join_u.py")):
            sys.path.insert(0, os.path.abspath(d))
            import k0_join_u
            return k0_join_u
    _die(f"k0_join_u.py not found in any of {[os.path.abspath(c) for c in cands]}; "
         f"pass --code-dir")


JU = None  # set in main()


def derived_seed(registration_seed, generation_id):
    """The one derivation rule (see header / SEED_DERIVATION)."""
    digest = hashlib.sha256(
        registration_seed.encode("utf-8")
        + generation_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


PROVENANCE_FIELDS = ("pass", "selected_sha256", "shard_sha256")


def _seed_forms(registration_seed):
    """(stored_form, canonical_str) for a registration seed
    (maintenance item (c), ruling 43907b6a SS2.2). The canonical
    string -- the exact registration seed string -- is the ONLY byte
    input to seed derivation, for every generation ever published.
    The stored form is an int iff the string is that int's canonical
    decimal representation (str(int(s)) == s), so str(stored_form)
    == canonical_str in every case and derivation bytes can never
    depend on the storage type. Legacy registrations carry no
    seed_canonical field; their stored form is the seed string
    itself, so str(stored) recovers the canonical string there too.
    """
    s = registration_seed if isinstance(registration_seed, str) \
        else str(registration_seed)
    try:
        v = int(s)
    except (TypeError, ValueError):
        return s, s
    if str(v) == s:
        return v, s
    return s, s


# --------------------------- source loading -------------------------------


def load_source(arg, mode, rate, registration_seed):
    """Load one sealed source generation through the real loader and
    run the per-source pre-checks. Returns a source record dict."""
    summary, meta, arrs = JU.load_annotation_source(arg)
    root, ptr = JU._resolve_pointer(arg, "merge source")
    gen_dir = os.path.join(root, ptr["dir"])
    man = json.load(open(os.path.join(gen_dir, "manifest.json")))
    prod = man.get("producer", {})
    gen = man.get("generation")
    where = f"source {gen} ({arg!r})"
    config = prod.get("config")
    if not isinstance(config, dict) or "window" not in config or "K" not in config:
        _die(f"{where}: producer.config missing or lacks window/K")
    # The annotator seals config_sha256 = sha256(canonical(config));
    # a recorded hash that does not recompute is a corrupt identity.
    if prod.get("config_sha256") != _sha256_bytes(_canonical(config)):
        _die(f"{where}: recorded config_sha256 {prod.get('config_sha256')!r} "
             f"does not recompute from producer.config")
    tmeta = meta.get("teacher", {})
    if _canonical(tmeta.get("config")) != _canonical(config):
        _die(f"{where}: meta.teacher.config does not match producer.config")
    codes = prod.get("code_md5") or prod.get("code_md5s")
    if not isinstance(codes, dict) or not codes:
        _die(f"{where}: producer carries no code_md5/code_md5s set")
    window = config["window"]
    if not (isinstance(window, list) and len(window) == 2):
        _die(f"{where}: config window {window!r} is not a [lo, hi) pair")
    # games table: gid -> name, name -> gid; both must be unambiguous.
    gid2name, name2gid, name2gentry = {}, {}, {}
    for g in meta.get("games", []):
        gid, name = int(g["gid"]), str(g["game"])
        if gid in gid2name or name in name2gid:
            _die(f"{where}: games table has duplicate gid/name ({gid}, {name!r})")
        gid2name[gid] = name
        name2gid[name] = gid
        name2gentry[name] = g
    n = int(arrs["step"].shape[0])
    keys = []
    for i in range(n):
        gid = int(arrs["gid"][i])
        if gid not in gid2name:
            _die(f"{where}: root #{i} references gid {gid} absent from "
                 f"the games table")
        keys.append((gid2name[gid], int(arrs["step"][i]), int(arrs["seat"][i])))
    stable = sorted(range(n), key=lambda i: keys[i])
    dseed = derived_seed(registration_seed, gen)
    if mode == "full":
        selected = list(stable)
        eff_rate = 1.0
    else:
        k = int(math.floor(n * rate))
        chosen = set(random.Random(dseed).sample(stable, k))
        selected = [i for i in stable if i in chosen]
        eff_rate = float(rate)
    return {"arg": arg, "generation": gen, "summary": summary, "meta": meta,
            "man": man, "prod": prod, "config": config, "codes": codes,
            "weights_md5": prod.get("weights_md5"), "arrs": arrs,
            "gid2name": gid2name, "name2gentry": name2gentry,
            "keys": keys, "n_roots": n,
            "mode": mode, "rate": eff_rate, "selected": selected,
            "sample_seed": dseed}


def cross_checks(sources):
    """Cross-source identity pre-checks; every failure dies by name."""
    s0 = sources[0]
    for s in sources[1:]:
        if s["man"].get("format") != s0["man"].get("format"):
            _die(f"format mismatch: {s0['generation']} has "
                 f"{s0['man'].get('format')!r}, {s['generation']} has "
                 f"{s['man'].get('format')!r}")
        if _canonical(s["codes"]) != _canonical(s0["codes"]):
            diff = sorted(k for k in set(s["codes"]) | set(s0["codes"])
                          if s["codes"].get(k) != s0["codes"].get(k))
            _die(f"code_md5s differ between {s0['generation']} and "
                 f"{s['generation']} in file(s) {diff}")
        if s["weights_md5"] != s0["weights_md5"]:
            _die(f"weights_md5 differs across sources: {s0['generation']} "
                 f"has {s0['weights_md5']!r}, {s['generation']} has "
                 f"{s['weights_md5']!r}")
        c0 = {k: v for k, v in s0["config"].items() if k != "window"}
        c1 = {k: v for k, v in s["config"].items() if k != "window"}
        if _canonical(c0) != _canonical(c1):
            diff = sorted(k for k in set(c0) | set(c1)
                          if _canonical(c0.get(k)) != _canonical(c1.get(k)))
            _die(f"config field(s) {diff} differ between "
                 f"{s0['generation']} and {s['generation']} -- only "
                 f"'window' may differ across merged sources")
        t0 = {k: v for k, v in s0["meta"]["teacher"].items()
              if k not in ("config", "config_sha256")}
        t1 = {k: v for k, v in s["meta"]["teacher"].items()
              if k not in ("config", "config_sha256")}
        if _canonical(t0) != _canonical(t1):
            diff = sorted(k for k in set(t0) | set(t1)
                          if _canonical(t0.get(k)) != _canonical(t1.get(k)))
            _die(f"meta.teacher field(s) {diff} differ between "
                 f"{s0['generation']} and {s['generation']}")
        for f in JU.ANNOT_FIELDS:
            a, b = s0["arrs"][f], s["arrs"][f]
            if a.dtype != b.dtype or a.shape[1:] != b.shape[1:]:
                _die(f"field {f!r}: dtype/shape differ across sources "
                     f"({a.dtype}{a.shape[1:]} vs {b.dtype}{b.shape[1:]})")


# ------------------------------ merge -------------------------------------


def merge_sources(sources, registration_seed):
    """Assemble merged arrays + games table + source list. Returns
    (arrs, games_meta, entries, merged_config, dedup_count)."""
    los = [int(s["config"]["window"][0]) for s in sources]
    his = [int(s["config"]["window"][1]) for s in sources]
    merged_config = copy.deepcopy(sources[0]["config"])
    merged_config["window"] = [min(los), max(his)]
    names = sorted({name for s in sources for name in
                    s["gid2name"].values()})
    new_gid = {name: i for i, name in enumerate(names)}
    kept = {}   # (new_gid, step, seat) -> (source_idx, row_idx)
    order = []
    dedup = 0
    for si, s in enumerate(sources):
        for i in s["selected"]:
            name, step, seat = s["keys"][i]
            key = (new_gid[name], step, seat)
            if key in kept:
                sj, j = kept[key]
                other = sources[sj]
                differing = [f for f in JU.ANNOT_FIELDS if f != "gid"
                             and np.asarray(other["arrs"][f][j]).tobytes()
                             != np.asarray(s["arrs"][f][i]).tobytes()]
                if differing:
                    _die(f"overlapping root (game {name!r}, step {step}, "
                         f"seat {seat}) differs across sources "
                         f"{other['generation']} and {s['generation']} in "
                         f"field(s) {differing} -- dedup requires "
                         f"byte-identical copies; refusing to pick one, "
                         f"average, or splice")
                dedup += 1
                continue
            kept[key] = (si, i)
            order.append((si, i, key))
    if not order:
        _die("merge produced zero roots from the declared sources; "
             "refusing to seal an empty generation")
    order.sort(key=lambda t: t[2])
    arrs = {}
    for f in JU.ANNOT_FIELDS:
        if f == "gid":
            arrs[f] = np.array([k[0] for _, _, k in order],
                               dtype=sources[0]["arrs"]["gid"].dtype)
        else:
            arrs[f] = np.stack([np.asarray(sources[si]["arrs"][f][i])
                                for si, i, _ in order])
    counts = {}
    for _, _, k in order:
        counts[k[0]] = counts.get(k[0], 0) + 1
    # Games provenance carry (maintenance item (a), ruling 43907b6a
    # SS2.2): each merged games entry carries the source provenance
    # triple (pass / selected_sha256 / shard_sha256). Caliber: the
    # sources recording the game must AGREE on each field -- the
    # agreed value is carried verbatim; a field absent from every
    # recording source is not carried; ANY disagreement (differing
    # values, or recorded by one source and absent from another)
    # dies naming the game and the field. No silent preference.
    absent = object()
    games_meta = []
    for name in names:
        entry = {"game": name, "gid": new_gid[name],
                 "n_roots": counts.get(new_gid[name], 0)}
        holders = [s for s in sources if name in s["name2gentry"]]
        for f in PROVENANCE_FIELDS:
            vals = {s["generation"]: s["name2gentry"][name].get(
                f, absent) for s in holders}
            present = {g: v for g, v in vals.items() if v is not absent}
            if len(set(present.values())) > 1 \
                    or 0 < len(present) < len(vals):
                detail = {g: ("<absent>" if v is absent else v)
                          for g, v in vals.items()}
                _die(f"game {name!r}: provenance field {f!r} "
                     f"disagrees across sources {detail} -- refusing "
                     f"to carry one source's value silently")
            if present:
                entry[f] = next(iter(present.values()))
        games_meta.append(entry)
    entries = []
    for s in sorted(sources, key=lambda s: s["generation"]):
        entries.append({
            "generation": s["generation"],
            "manifest_sha256": s["summary"]["manifest_sha256"],
            "config_sha256": s["prod"]["config_sha256"],
            "window": [int(s["config"]["window"][0]),
                       int(s["config"]["window"][1])],
            "mode": s["mode"],
            "rate": s["rate"],
            "source_roots": s["n_roots"],
            "drawn_roots": len(s["selected"]),
            "sample_seed": s["sample_seed"]})
    JU._validate_annot_records(arrs, merged_config["window"],
                                merged_config["K"],
                                "merge pre-seal self-check")
    return arrs, games_meta, entries, merged_config, dedup


# ------------------------------ sealing -----------------------------------


def _seal_staging(staging_root, arrs, games_meta, entries, merged_config,
                  sources, merged_hash, dedup, registration_seed):
    """Write a full dataset (generation + pointer) under staging_root.
    Returns (gen_id, core, total)."""
    total = int(arrs["step"].shape[0])
    gen_stage = os.path.join(staging_root, "generations", ".pending")
    os.makedirs(gen_stage)
    for f in JU.ANNOT_FIELDS:
        np.save(os.path.join(gen_stage, f + ".npy"), arrs[f])
    t0 = sources[0]["meta"]["teacher"]
    meta = {"format": JU.ANNOT_FORMAT, "total": total,
            "games": games_meta,
            "teacher": {"version": t0.get("version"),
                        "belief": t0.get("belief"),
                        "weights": t0.get("weights"),
                        "weights_md5": sources[0]["weights_md5"],
                        "config": merged_config,
                        "config_sha256": merged_hash},
            "sampling": {"rule": "merged generation (see merged_from): "
                                 "per-source deterministic draw under "
                                 "the registered seed; full sources "
                                 "contribute every root",
                         "seed": _seed_forms(registration_seed)[0],
                         "seed_derivation": SEED_DERIVATION},
            "merged_from": entries,
            "merge": {"tool": "merge_annot_generations.py",
                      "tool_md5": _md5_file(os.path.abspath(__file__)),
                      "dedup_roots": dedup}}
    if "record_schema" in sources[0]["meta"]:
        meta["record_schema"] = sources[0]["meta"]["record_schema"]
    with open(os.path.join(gen_stage, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1, sort_keys=True)
    codes = sources[0]["codes"]
    producer = {"code_md5": codes,
                "merge_annot_generations_md5":
                    _md5_file(os.path.abspath(__file__)),
                "config_sha256": merged_hash,
                "teacher_config_hash": merged_hash,
                "config": merged_config,
                "weights_md5": sources[0]["weights_md5"],
                "merged_from": entries,
                "merged_dedup_roots": dedup}
    for src_key, dst_key in (("k0_annotate_queue.py", "k0_annotate_md5"),
                             ("k0_annotate.py", "k0_annotate_md5"),
                             ("stage0_teacher.py", "stage0_teacher_md5"),
                             ("stage1_teacher_gate.py",
                              "stage1_teacher_gate_md5")):
        if src_key in codes:
            producer[dst_key] = codes[src_key]
    finfo = {}
    for f in JU.ANNOT_FIELDS:
        p = os.path.join(gen_stage, f + ".npy")
        arr = np.load(p, mmap_mode="r")
        finfo[f] = {"file": f + ".npy", "bytes": os.path.getsize(p),
                    "rows": int(arr.shape[0]), "sha256": _sha256_file(p)}
        del arr
    core = {"format": JU.ANNOT_FORMAT, "total": total,
            "producer": producer,
            "meta_sha256": _sha256_file(os.path.join(gen_stage, "meta.json")),
            "fields": finfo}
    gen_id = "gen-" + _sha256_bytes(_canonical(core))[:16]
    gen_dir = os.path.join(staging_root, "generations", gen_id)
    os.replace(gen_stage, gen_dir)
    man = dict(core)
    man["generation"] = gen_id
    man["created"] = time.time()
    man["self_sha256"] = _sha256_bytes(_canonical(man))
    with open(os.path.join(gen_dir, "manifest.json"), "w") as fh:
        json.dump(man, fh, indent=1, sort_keys=True)
    ptr = {"generation": gen_id, "dir": f"generations/{gen_id}",
           "manifest_sha256": _sha256_file(
               os.path.join(gen_dir, "manifest.json")),
           "total": total, "created": time.time()}
    with open(os.path.join(staging_root, "current.json"), "w") as fh:
        json.dump(ptr, fh)
    return gen_id, core, total


def _check_published(gen_dir, core, gen_id):
    """Reuse check for an already-published identical generation,
    mirroring the annotator's publish rule: self-seal, core equality
    (created excluded), meta bytes, per-field bytes. Dies by name on
    any mismatch; a published generation is never overwritten."""
    man_p = os.path.join(gen_dir, "manifest.json")
    if not os.path.exists(man_p):
        _die(f"publish: generation directory {gen_dir!r} exists without "
             f"manifest.json; refusing to overwrite it")
    old = json.load(open(man_p))
    old_core = {k: v for k, v in old.items() if k != "self_sha256"}
    if old.get("self_sha256") != _sha256_bytes(_canonical(old_core)):
        _die(f"publish: published generation {gen_id} in {gen_dir!r} "
             f"fails its manifest self-seal; refusing to overwrite it")
    want_core = dict(core)
    want_core["generation"] = gen_id
    have_core = {k: v for k, v in old.items()
                 if k not in ("self_sha256", "created")}
    if have_core != want_core:
        _die(f"publish: published generation {gen_id} in {gen_dir!r} "
             f"differs from the inputs being published; refusing to "
             f"overwrite it")
    meta_p = os.path.join(gen_dir, "meta.json")
    if not os.path.exists(meta_p) \
            or _sha256_file(meta_p) != old.get("meta_sha256"):
        _die(f"publish: published generation {gen_id} meta.json in "
             f"{gen_dir!r} does not match its manifest seal; refusing "
             f"to reuse it or move the pointer to it")
    for k, info in old["fields"].items():
        p = os.path.join(gen_dir, info["file"])
        if not os.path.exists(p) or os.path.getsize(p) != info["bytes"] \
                or _sha256_file(p) != info["sha256"]:
            _die(f"publish: published generation {gen_id} field {k!r} "
                 f"in {gen_dir!r} does not match its manifest; "
                 f"refusing to overwrite it")


def _land_registration(out_dir, gen_id, merged_hash, entries,
                       registration_seed):
    """Registration gate: REGISTRATION.json must land in the dataset
    root before the pointer flips. An existing registration for the
    same generation must agree exactly; it is never overwritten."""
    stored_seed, canonical_seed = _seed_forms(registration_seed)
    reg = {"format": REGISTRATION_FORMAT, "generation": gen_id,
           "teacher_config_hash": merged_hash, "seed": stored_seed,
           "seed_canonical": canonical_seed,
           "seed_derivation": SEED_DERIVATION, "merged_from": entries,
           "created": time.time()}
    reg_path = os.path.join(out_dir, "REGISTRATION.json")
    if os.path.exists(reg_path):
        old = json.load(open(reg_path))
        # Seed agreement is judged on the canonical string, so a
        # legacy (str-form, no seed_canonical) registration and this
        # publish agree iff they name the same registration seed.
        old_canonical = old.get("seed_canonical")
        if old_canonical is None:
            old_canonical = _seed_forms(old.get("seed"))[1] \
                if old.get("seed") is not None else None
        same = (old.get("generation") == gen_id
                and old.get("teacher_config_hash") == merged_hash
                and old_canonical == canonical_seed
                and _canonical(old.get("merged_from")) == _canonical(entries))
        if not same:
            _die(f"publish: existing REGISTRATION.json in {out_dir!r} "
                 f"does not match this merge (generation/hash/seed/"
                 f"source list); refusing to overwrite a registration")
        return False
    _atomic_write_json(reg, reg_path)
    return True


def publish(out_dir, sources, registration_seed):
    arrs, games_meta, entries, merged_config, dedup = merge_sources(
        sources, registration_seed)
    merged_hash = _sha256_bytes(_canonical(entries))
    os.makedirs(os.path.join(out_dir, "generations"), exist_ok=True)
    staging_root = os.path.join(
        out_dir, f".staging-merge-{os.getpid()}")
    shutil.rmtree(staging_root, ignore_errors=True)
    os.makedirs(staging_root)
    try:
        gen_id, core, total = _seal_staging(
            staging_root, arrs, games_meta, entries, merged_config,
            sources, merged_hash, dedup, registration_seed)
        # Pre-visibility gate: the staged generation must load through
        # the REAL loader + record gates (verify_dataset also re-runs
        # _validate_annot_records and the hash recompute; registration
        # is an out-root artifact and is checked at the real root).
        verify_dataset(staging_root, require_registration=False)
        staged_gen = os.path.join(staging_root, "generations", gen_id)
        final_gen = os.path.join(out_dir, "generations", gen_id)
        reused = False
        if os.path.isdir(final_gen):
            _check_published(final_gen, core, gen_id)
            reused = True
        elif os.path.exists(final_gen):
            _die(f"publish: {final_gen!r} exists and is not a "
                 f"directory; refusing to overwrite it")
        else:
            os.replace(staged_gen, final_gen)
        # Registration lands BEFORE the pointer flips (hard gate).
        _land_registration(out_dir, gen_id, merged_hash, entries,
                           registration_seed)
        ptr = {"generation": gen_id, "dir": f"generations/{gen_id}",
               "manifest_sha256": _sha256_file(
                   os.path.join(final_gen, "manifest.json")),
               "total": total, "created": time.time()}
        ptr_tmp = os.path.join(out_dir, f".current-{os.getpid()}.tmp")
        with open(ptr_tmp, "w") as fh:
            json.dump(ptr, fh)
        os.replace(ptr_tmp, os.path.join(out_dir, "current.json"))
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
    return {"generation": gen_id, "total": total, "reused": reused,
            "teacher_config_hash": merged_hash, "dedup_roots": dedup,
            "games": len(games_meta),
            "sources": [{"generation": e["generation"], "mode": e["mode"],
                         "drawn_roots": e["drawn_roots"],
                         "source_roots": e["source_roots"]}
                        for e in entries]}


# ------------------------------ verify ------------------------------------


def verify_dataset(root, require_registration=True):
    """Full merged-generation verification. Returns the manifest.
    Raises SystemExit (via _die / the loader) on any failure."""
    summary, meta, arrs = JU.load_annotation_source(root)
    ds_root, ptr = JU._resolve_pointer(root, "verify")
    gen_dir = os.path.join(ds_root, ptr["dir"])
    man = json.load(open(os.path.join(gen_dir, "manifest.json")))
    prod = man.get("producer", {})
    entries = prod.get("merged_from")
    if not entries:
        _die(f"verify: generation {man.get('generation')} in "
             f"{gen_dir!r} carries no producer.merged_from source "
             f"list; not a merged generation")
    merged_hash = _sha256_bytes(_canonical(entries))
    for where, val in (("producer.config_sha256",
                        prod.get("config_sha256")),
                       ("producer.teacher_config_hash",
                        prod.get("teacher_config_hash")),
                       ("meta.teacher.config_sha256",
                        meta["teacher"]["config_sha256"])):
        if val != merged_hash:
            _die(f"verify: {where} = {val!r} != recomputed merged "
                 f"teacher_config_hash {merged_hash!r} from the "
                 f"source list")
    if "merged_from" in meta and \
            _canonical(meta["merged_from"]) != _canonical(entries):
        _die("verify: meta.merged_from differs from "
             "producer.merged_from")
    JU._validate_annot_records(arrs, prod["config"]["window"],
                                prod["config"]["K"],
                                f"verify merged {gen_dir!r}")
    if require_registration:
        reg_path = os.path.join(ds_root, "REGISTRATION.json")
        if not os.path.exists(reg_path):
            _die(f"verify: no REGISTRATION.json in dataset root "
                 f"{ds_root!r}; an unregistered merged generation "
                 f"must not be consumed")
        reg = json.load(open(reg_path))
        if reg.get("generation") != man.get("generation"):
            _die(f"verify: REGISTRATION.json generation "
                 f"{reg.get('generation')!r} != merged generation "
                 f"{man.get('generation')!r}")
        if reg.get("teacher_config_hash") != merged_hash:
            _die(f"verify: REGISTRATION.json teacher_config_hash "
                 f"{reg.get('teacher_config_hash')!r} != recomputed "
                 f"{merged_hash!r}")
        if _canonical(reg.get("merged_from")) != _canonical(entries):
            _die("verify: REGISTRATION.json source list is not "
                 "byte-identical (canonical bytes) to the "
                 "in-generation merged_from list")
        seed = reg.get("seed")
        if seed is None or seed == "":
            _die("verify: REGISTRATION.json carries no registration "
                 "seed")
        # Derived seeds recompute from the canonical seed string
        # ONLY (maintenance item (c)): new registrations record it
        # explicitly as seed_canonical and store the seed in
        # normalized (possibly int) form; legacy registrations have
        # no seed_canonical and store the string itself, recovered
        # via str(stored). Both forms must yield the same string.
        canonical = reg.get("seed_canonical")
        if canonical is None:
            canonical = seed if isinstance(seed, str) else str(seed)
        elif not isinstance(canonical, str) \
                or str(seed) != canonical:
            _die(f"verify: REGISTRATION.json seed {seed!r} is "
                 f"inconsistent with its seed_canonical "
                 f"{canonical!r}; refusing to recompute derived "
                 f"seeds from an ambiguous registration")
        for e in entries:
            want = derived_seed(canonical, e["generation"])
            if e.get("sample_seed") != want:
                _die(f"verify: source {e['generation']} recorded "
                     f"sample_seed {e.get('sample_seed')!r} != "
                     f"derived {want!r} under the registered seed")
    return man


# -------------------------------- main ------------------------------------


def _parse_sample(spec):
    if ":" in spec:
        path, tail = spec.rsplit(":", 1)
        try:
            return path, float(tail)
        except ValueError:
            pass
    return spec, 0.2


def main(argv=None):
    global JU
    ap = argparse.ArgumentParser(
        description="Merge sealed k0 annotation generations across "
                    "configs (window-only differences) into one "
                    "registered merged generation.")
    ap.add_argument("--out", required=True,
                    help="output dataset root (merge target / verify "
                         "target)")
    ap.add_argument("--seed",
                    help="registration seed (REQUIRED for merge; one "
                         "seed per merged generation, no default)")
    ap.add_argument("--full", action="append", default=[],
                    metavar="SRC",
                    help="source dataset taken in full (repeatable)")
    ap.add_argument("--sample", action="append", default=[],
                    metavar="SRC[:RATE]",
                    help="source dataset sampled at RATE (default "
                         "0.2) of its own roots (repeatable)")
    ap.add_argument("--verify", action="store_true",
                    help="verify the merged generation at --out and "
                         "exit")
    ap.add_argument("--code-dir",
                    help="directory containing k0_join_u.py")
    a = ap.parse_args(argv)
    JU = _load_join_u(a.code_dir)
    if a.verify:
        man = verify_dataset(a.out, require_registration=True)
        print(f"verify PASS: {a.out} generation "
              f"{man['generation']} total {man['total']} "
              f"teacher_config_hash "
              f"{man['producer']['teacher_config_hash']}")
        return 0
    if not a.seed:
        _die("--seed is required for a merge (one registration seed "
             "per merged generation; no default)")
    specs = [(p, "full", 1.0) for p in a.full]
    for spec in a.sample:
        p, r = _parse_sample(spec)
        if not (0.0 < r <= 1.0):
            _die(f"sample rate {r} for {p!r} outside (0, 1]")
        specs.append((p, "sample", r))
    if len(specs) < 2:
        _die(f"a merge needs >= 2 source generations, got {len(specs)}")
    sources = [load_source(p, mode, rate, a.seed)
               for p, mode, rate in specs]
    cross_checks(sources)
    result = publish(a.out, sources, a.seed)
    print(json.dumps(result, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())

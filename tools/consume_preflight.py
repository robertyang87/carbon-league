#!/usr/bin/env python3
# =====================================================================
# consume_preflight -- mandatory consumption gate for k0 annotation
# generations, closing the consumer side of the merged-generation
# provenance chain (Codex review 3f28c966 section 5: deleting
# REGISTRATION.json made the publisher's own --verify refuse, but
# the real k0_join_u loader still accepted the generation, and
# k0_distill_u seals only generation + hash into the training
# record without expanding the source list -- the publisher was
# therefore NOT a closed loop on the consumption side).
#
# This tool is the pass-through gate at BOTH consumption points:
#   (a) before k0_join_u consumes an annotation generation, and
#   (b) before k0_distill_u consumes a joined product
#       (format cnet2-k0-annotation/2).
# It verifies, refusing by name on any failure:
#   1. The target generation through the REAL loaders:
#      k0_join_u.load_annotation_source for annotation generations;
#      k0_distill_u.verify_annotation_generation (the consumer's own
#      verifier) for joined products, including the joined product's
#      binding to its annotation generation (generation id +
#      manifest sha256, cross-checked in manifest and meta).
#   2. For a MERGED generation (producer carries merged_from):
#      the full publisher verification (imported from
#      merge_annot_generations.verify_dataset: loader, source-list
#      hash recompute against all three identity slots, record
#      gates, REGISTRATION.json presence + generation/hash/list
#      canonical byte equality, per-source derived-seed recompute);
#      PLUS the check the publisher never performs: every source
#      generation in the registration list is REACHABLE and verifies
#      through the real loader, bound to its registration entry by
#      (generation id, manifest sha256) read from the source's own
#      sealed manifest -- never caller-asserted. An unreachable
#      source, a registered source that is not provided, or a
#      provided source that is not registered all refuse by name
#      (substitution guard).
#   3. On success it seals a CONSUMPTION.json sidecar next to the
#      target dataset root: content-addressed (sidecar_sha256 over
#      the canonical content, "created"/"locations" excluded, the
#      manifest caliber), binding the generation that will actually
#      be loaded (id + manifest sha256), the merged identity, the
#      registration file's own sha256, and the full source list with
#      per-source verification results. An existing sidecar with a
#      different content hash is NEVER silently overwritten.
#
# --check-sidecar FILE re-runs the whole verification and compares
# the fresh record against a previously sealed sidecar field by
# field; any divergence refuses by name. This is the mode the
# proposed k0_distill_u hook uses (see DESIGN_CONSUME_PREFLIGHT.md).
#
# Plain (non-merged) generations: with --require-merged they are
# refused by name; in default mode they pass with an explicit
# notice (non-merged generation, no registration chain) and the
# sidecar records merged=null so the training provenance can cite
# the consumption decision either way.
#
# CALIBER NOTE (registration paths): REGISTRATION.json as sealed by
# the accepted publisher carries source IDENTITIES (generation id,
# manifest sha256, config hash, window, counts, seeds), not
# filesystem paths. Source dataset roots are therefore supplied at
# consumption time via --source and are bound to registration
# entries exclusively through loader-read identities. The security
# property is identical (consumption proceeds only on the
# registered sources, each verified through the real loader); the
# publisher's sealed registration format is untouched.
#
# Pinned / accepted files are only imported, never modified:
# k0_join_u (9054aadc), k0_distill_u (06826f59),
# merge_annot_generations (f1f5ab9c).
#
# Usage:
#   python3 consume_preflight.py TARGET [--source DS ...]
#       [--annotation-root DS] [--require-merged]
#       [--sidecar-out DIR] [--code-dir DIR] [--merge-tool PATH]
#   python3 consume_preflight.py TARGET --check-sidecar FILE
#       [--source DS ...] [--annotation-root DS] [--code-dir DIR]
# TARGET is a dataset root (or current.json pointer) of either an
# annotation generation or a joined product. For a joined product,
# --annotation-root (the dataset root holding the annotation
# generation it binds) is required. --source is repeatable.
# =====================================================================
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

SIDECAR_FORMAT = "k0-consumption-sidecar/1"
SIDECAR_NAME = "CONSUMPTION.json"


def _die(msg):
    raise SystemExit(f"consume_preflight: {msg}")


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


JU = None   # k0_join_u
DU = None   # k0_distill_u
MAG = None  # merge_annot_generations


def _load_modules(code_dir, merge_tool):
    global JU, DU, MAG
    cands = []
    if code_dir:
        cands.append(code_dir)
    if os.environ.get("K0_CODE_DIR"):
        cands.append(os.environ["K0_CODE_DIR"])
    cands.append(os.path.join(HERE, "..", "..", "code"))
    cands.append(os.path.join(HERE, "..", "code"))
    cands.append(os.path.join(HERE, "..", "mixcheck"))
    code_at = None
    for d in cands:
        if d and os.path.isfile(os.path.join(d, "k0_join_u.py")) \
                and os.path.isfile(os.path.join(d, "k0_distill_u.py")):
            code_at = os.path.abspath(d)
            break
    if code_at is None:
        _die(f"k0_join_u.py / k0_distill_u.py not found in any of "
             f"{[os.path.abspath(c) for c in cands]}; pass --code-dir")
    sys.path.insert(0, code_at)
    import k0_join_u
    import k0_distill_u
    JU, DU = k0_join_u, k0_distill_u

    mcands = []
    if merge_tool:
        mcands.append(merge_tool)
    if os.environ.get("K0_MERGE_TOOL"):
        mcands.append(os.environ["K0_MERGE_TOOL"])
    mcands.append(os.path.join(HERE, "merge_annot_generations.py"))
    mcands.append(os.path.join(HERE, "..", "mixbuild",
                               "merge_annot_generations.py"))
    mcands.append(os.path.join(HERE, "..", "..", "tools",
                               "merge_annot_generations.py"))
    mpath = None
    for p in mcands:
        if p and os.path.isfile(p):
            mpath = os.path.abspath(p)
            break
    if mpath is None:
        _die(f"merge_annot_generations.py not found in any of "
             f"{[os.path.abspath(p) for p in mcands]}; "
             f"pass --merge-tool")
    spec = importlib.util.spec_from_file_location(
        "merge_annot_generations", mpath)
    MAG = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(MAG)
    MAG.JU = JU
    return code_at, mpath


# ------------------------- target loading ---------------------------------


def _read_manifest(root_arg, what):
    """Resolve a dataset pointer and read its manifest (the loaders
    re-verify the bytes afterwards; this read only classifies)."""
    root, ptr = JU._resolve_pointer(root_arg, what)
    gen_dir = os.path.join(root, ptr["dir"])
    man_p = os.path.join(gen_dir, "manifest.json")
    if not os.path.exists(man_p):
        _die(f"{what}: sealed generation has no manifest.json: "
             f"{gen_dir!r}; refusing")
    raw = open(man_p, "rb").read()
    return root, ptr, gen_dir, json.loads(raw), _sha256_bytes(raw)


def _verify_annotation_target(root_arg):
    """Verify an annotation-format target through the real loader.
    Returns (record, summary, meta, man) where record carries the
    sidecar target/annotation blocks."""
    root, ptr, gen_dir, man, man_sha = _read_manifest(
        root_arg, "preflight target")
    if man.get("format") != JU.ANNOT_FORMAT:
        _die(f"preflight target: generation format "
             f"{man.get('format')!r} is neither {JU.ANNOT_FORMAT!r} "
             f"nor {JU.OUT_FORMAT!r}; refusing")
    summary, meta, _arrs = JU.load_annotation_source(root_arg)
    if summary["generation"] != man.get("generation") \
            or summary["manifest_sha256"] != man_sha:
        _die("preflight target: loader/manifest identity disagreement "
             f"on {root_arg!r}")
    block = {"dataset_root": os.path.abspath(root),
             "generation": summary["generation"],
             "manifest_sha256": summary["manifest_sha256"],
             "total": summary["total"],
             "teacher_config_hash": summary["teacher_config_hash"],
             "teacher_weights_md5": summary["teacher_weights_md5"]}
    return block, summary, meta, man


def _verify_joined_target(root_arg):
    """Verify a joined-product target through the REAL consumer
    verifier (k0_distill_u.verify_annotation_generation) and extract
    its annotation binding. Returns (target_block, bound) where
    bound = {"generation":..., "manifest_sha256":...}."""
    gen_dir, man, meta, _verified = DU.verify_annotation_generation(
        root_arg)
    root, ptr = JU._resolve_pointer(root_arg, "preflight target")
    man_sha = _sha256_file(os.path.join(gen_dir, "manifest.json"))
    if ptr.get("manifest_sha256") is not None \
            and ptr["manifest_sha256"] != man_sha:
        _die("preflight target: joined manifest bytes do not match "
             "the pointer hash")
    prod = man.get("producer", {})
    src_m = (prod.get("sources") or {}).get("annotation")
    src_t = (meta.get("sources") or {}).get("annotation")
    if not src_m or not src_m.get("generation") \
            or not src_m.get("manifest_sha256"):
        _die("preflight target: joined generation carries no "
             "producer.sources.annotation binding; refusing")
    if src_t != src_m:
        _die(f"preflight target: joined meta.sources.annotation "
             f"{src_t!r} disagrees with manifest producer binding "
             f"{src_m!r}; refusing")
    block = {"dataset_root": os.path.abspath(root),
             "generation": man.get("generation"),
             "manifest_sha256": man_sha,
             "total": man.get("total"),
             "teacher_config_hash": prod.get("teacher_config_hash"),
             "teacher_weights_md5": prod.get("teacher_weights_md5")}
    return block, {"generation": src_m["generation"],
                   "manifest_sha256": src_m["manifest_sha256"]}


# ------------------------- merged chain -----------------------------------


def _load_source_bound(arg):
    """Load one provided source generation through the real loader;
    any refusal is re-raised naming the provided path (the source is
    unreachable FOR CONSUMPTION, which is a refusal either way)."""
    try:
        summary, meta, _arrs = JU.load_annotation_source(arg)
    except SystemExit as e:
        _die(f"provided source at {arg!r} cannot be loaded through "
             f"the real loader, so the registered source it should "
             f"bind is unreachable: {e}")
    root, ptr = JU._resolve_pointer(arg, "provided source")
    gen_dir = os.path.join(root, ptr["dir"])
    man = json.load(open(os.path.join(gen_dir, "manifest.json")))
    return {"arg": arg, "root": root, "summary": summary,
            "meta": meta, "man": man}


def _verify_sources(entries, source_args):
    """Bind every registration entry to a distinct provided source,
    verified through the real loader. Returns the per-source
    verification records for the sidecar."""
    loaded = []
    seen = set()
    for arg in source_args:
        rec = _load_source_bound(arg)
        gen = rec["summary"]["generation"]
        if gen in seen:
            _die(f"provided sources include generation {gen} twice "
                 f"(second at {arg!r}); refusing")
        seen.add(gen)
        loaded.append(rec)
    by_identity = {(r["summary"]["generation"],
                    r["summary"]["manifest_sha256"]): r for r in loaded}
    out = []
    for e in entries:
        key = (e["generation"], e["manifest_sha256"])
        rec = by_identity.get(key)
        if rec is None:
            _die(f"registered source generation {e['generation']} "
                 f"(manifest {e['manifest_sha256']}) is not among "
                 f"the provided, loader-verified sources "
                 f"{sorted(seen) or '[]'}; it is unreachable for "
                 f"consumption -- refusing")
        prod = rec["man"].get("producer", {})
        config = prod.get("config") or {}
        if prod.get("config_sha256") != e["config_sha256"]:
            _die(f"source {e['generation']}: manifest config_sha256 "
                 f"{prod.get('config_sha256')!r} != registration "
                 f"entry {e['config_sha256']!r}; refusing")
        if _sha256_bytes(_canonical(config)) != e["config_sha256"]:
            _die(f"source {e['generation']}: recorded config_sha256 "
                 f"does not recompute from its config dict; refusing")
        if [int(config["window"][0]), int(config["window"][1])] \
                != [int(e["window"][0]), int(e["window"][1])]:
            _die(f"source {e['generation']}: config window "
                 f"{config.get('window')!r} != registration entry "
                 f"window {e['window']!r}; refusing")
        if int(rec["summary"]["total"]) != int(e["source_roots"]):
            _die(f"source {e['generation']}: root count "
                 f"{rec['summary']['total']} != registration entry "
                 f"source_roots {e['source_roots']}; refusing")
        is_merged = bool(prod.get("merged_from"))
        if is_merged:
            MAG.verify_dataset(rec["root"], require_registration=True)
        out.append({"generation": e["generation"],
                    "manifest_sha256": e["manifest_sha256"],
                    "config_sha256": e["config_sha256"],
                    "window": e["window"], "mode": e["mode"],
                    "rate": e["rate"],
                    "source_roots": e["source_roots"],
                    "drawn_roots": e["drawn_roots"],
                    "dataset_root": os.path.abspath(rec["root"]),
                    "loader_verified": True,
                    "is_merged": is_merged})
    extra = [g for (g, _s) in by_identity
             if g not in {e["generation"] for e in entries}]
    if extra:
        _die(f"provided source generation(s) {sorted(extra)} are "
             f"not in the registration list; substitution is "
             f"refused")
    return out


def _verify_merged_chain(annot_root, man, source_args):
    """Full merged-generation consumption chain on an annotation
    dataset root whose manifest is `man`. Returns the sidecar
    'merged' block."""
    MAG.verify_dataset(annot_root, require_registration=True)
    prod = man.get("producer", {})
    entries = prod["merged_from"]
    ds_root, _ptr = JU._resolve_pointer(annot_root, "merged target")
    reg_path = os.path.join(ds_root, "REGISTRATION.json")
    reg = json.load(open(reg_path))
    sources = _verify_sources(entries, source_args)
    return {"teacher_config_hash": prod["teacher_config_hash"],
            "registration": {
                "file": "REGISTRATION.json",
                "file_sha256": _sha256_file(reg_path),
                "format": reg.get("format"),
                "seed": reg.get("seed")},
            "merged_from": entries,
            "sources": sources}


# ------------------------- core verification ------------------------------


def run_verification(target_arg, source_args, annotation_root,
                     require_merged):
    """Run the full gate. Returns (core, locations, target_root):
    core is the content-addressed sidecar body (no created /
    locations / sidecar_sha256); locations is informational only."""
    _r, _p, _g, probe_man, _s = _read_manifest(target_arg,
                                               "preflight target")
    fmt = probe_man.get("format")
    checks = {"target_loader_verified": True}
    if fmt == JU.ANNOT_FORMAT:
        tblock, summary, meta, man = _verify_annotation_target(
            target_arg)
        target_block = dict(tblock, kind="annotation")
        annot_block = tblock
        annot_root = target_arg
        annot_man = man
    elif fmt == JU.OUT_FORMAT:
        tblock, bound = _verify_joined_target(target_arg)
        target_block = dict(tblock, kind="joined")
        checks["joined_annotation_binding"] = True
        if not annotation_root:
            _die("preflight target is a joined product; "
                 "--annotation-root (the dataset root holding the "
                 "annotation generation it binds) is required")
        ablock, _s2, _m2, annot_man = _verify_annotation_target(
            annotation_root)
        if ablock["generation"] != bound["generation"] \
                or ablock["manifest_sha256"] != bound["manifest_sha256"]:
            _die(f"joined product binds annotation generation "
                 f"{bound['generation']} (manifest "
                 f"{bound['manifest_sha256']}) but --annotation-root "
                 f"holds generation {ablock['generation']} "
                 f"(manifest {ablock['manifest_sha256']}); "
                 f"binding mismatch -- refusing")
        if tblock["teacher_config_hash"] \
                != ablock["teacher_config_hash"]:
            _die(f"joined product teacher_config_hash "
                 f"{tblock['teacher_config_hash']!r} != annotation "
                 f"generation's {ablock['teacher_config_hash']!r}; "
                 f"refusing")
        annot_block = ablock
        annot_root = annotation_root
    else:
        _die(f"preflight target: generation format {fmt!r} is "
             f"neither {JU.ANNOT_FORMAT!r} nor {JU.OUT_FORMAT!r}; "
             f"refusing")

    prod = annot_man.get("producer", {})
    merged_block = None
    if prod.get("merged_from"):
        merged_block = _verify_merged_chain(annot_root, annot_man,
                                            source_args)
        checks.update({
            "registration_present": True,
            "registration_list_byte_identical": True,
            "merged_hash_recomputed": True,
            "sources_all_loader_verified": True,
            "derived_seeds_recomputed": True})
    else:
        if require_merged:
            _die(f"generation {annot_block['generation']} is not a "
                 f"merged generation (no producer.merged_from "
                 f"source list); --require-merged refuses plain "
                 f"generations")
        checks["registration_present"] = False
        checks["notice"] = (
            "non-merged generation: no registration chain "
            "（非合并代、无注册链）; passed in default mode and "
            "recorded")
        print(f"consume_preflight: NOTE: generation "
              f"{annot_block['generation']} is a non-merged "
              f"generation -- no registration chain "
              f"（非合并代、无注册链）; passing in default mode "
              f"and recording the decision")

    core = {"format": SIDECAR_FORMAT,
            "target": target_block,
            "annotation": annot_block,
            "merged": merged_block,
            "checks": checks,
            "tool": {"file": "consume_preflight.py",
                     "md5": _md5_file(os.path.abspath(__file__)),
                     "merge_annot_generations_md5":
                         _md5_file(MAG.__file__),
                     "k0_join_u_md5": _md5_file(JU.__file__),
                     "k0_distill_u_md5": _md5_file(DU.__file__)}}
    locations = {"target_arg": target_arg,
                 "annotation_root": annot_root,
                 "source_args": list(source_args)}
    return core, locations, target_block["dataset_root"]


def _sidecar_sha(core):
    return _sha256_bytes(_canonical(core))


def emit_sidecar(core, locations, out_dir):
    """Seal CONSUMPTION.json into out_dir. Content-addressed: an
    existing sidecar with the same hash is reused; a different one
    refuses (a consumption record is never silently replaced)."""
    sha = _sidecar_sha(core)
    sidecar = dict(core)
    sidecar["locations"] = locations
    sidecar["created"] = time.time()
    sidecar["sidecar_sha256"] = sha
    path = os.path.join(out_dir, SIDECAR_NAME)
    if os.path.exists(path):
        old = json.load(open(path))
        if old.get("sidecar_sha256") == sha:
            return path, sha, True
        _die(f"an existing {SIDECAR_NAME} in {out_dir!r} has "
             f"sidecar_sha256 {old.get('sidecar_sha256')!r} != this "
             f"run's {sha!r}; a consumption record is never "
             f"silently overwritten -- refusing")
    _atomic_write_json(sidecar, path)
    return path, sha, False


def check_sidecar(core, sidecar_path):
    """Compare a fresh verification record against a sealed sidecar,
    field by field over the content-addressed core."""
    if not os.path.exists(sidecar_path):
        _die(f"sidecar {sidecar_path!r} does not exist; consumption "
             f"is unrecorded -- refusing")
    old = json.load(open(sidecar_path))
    if old.get("format") != SIDECAR_FORMAT:
        _die(f"sidecar {sidecar_path!r} format "
             f"{old.get('format')!r} != {SIDECAR_FORMAT!r}; refusing")
    old_core = {k: v for k, v in old.items()
                if k not in ("sidecar_sha256", "created", "locations")}
    if old.get("sidecar_sha256") != _sha256_bytes(_canonical(old_core)):
        _die(f"sidecar {sidecar_path!r} fails its own "
             f"sidecar_sha256 seal; refusing")
    if _canonical(old_core) == _canonical(core):
        return old["sidecar_sha256"]
    for k in sorted(set(old_core) | set(core)):
        if _canonical(old_core.get(k)) != _canonical(core.get(k)):
            _die(f"sidecar {sidecar_path!r} disagrees with the "
                 f"fresh verification at field {k!r}: sealed "
                 f"{json.dumps(old_core.get(k), sort_keys=True)[:300]}"
                 f" != fresh "
                 f"{json.dumps(core.get(k), sort_keys=True)[:300]}; "
                 f"refusing")
    _die(f"sidecar {sidecar_path!r} disagrees with the fresh "
         f"verification; refusing")


# -------------------------------- main ------------------------------------


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Mandatory consumption preflight for k0 "
                    "annotation generations (merged-generation "
                    "registration chain + sealed CONSUMPTION.json "
                    "sidecar).")
    ap.add_argument("target",
                    help="dataset root (or current.json) of the "
                         "annotation generation or joined product "
                         "about to be consumed")
    ap.add_argument("--source", action="append", default=[],
                    metavar="DS",
                    help="source generation dataset root bound in "
                         "the merged generation's registration "
                         "(repeatable)")
    ap.add_argument("--annotation-root",
                    help="for a joined-product target: the dataset "
                         "root holding the annotation generation it "
                         "binds")
    ap.add_argument("--require-merged", action="store_true",
                    help="refuse plain (non-merged) generations")
    ap.add_argument("--sidecar-out",
                    help="directory for CONSUMPTION.json (default: "
                         "the target dataset root)")
    ap.add_argument("--check-sidecar", metavar="FILE",
                    help="verify-only: re-run the full gate and "
                         "compare against this sealed sidecar")
    ap.add_argument("--code-dir",
                    help="directory containing k0_join_u.py / "
                         "k0_distill_u.py")
    ap.add_argument("--merge-tool",
                    help="path to merge_annot_generations.py")
    a = ap.parse_args(argv)
    _load_modules(a.code_dir, a.merge_tool)
    core, locations, target_root = run_verification(
        a.target, a.source, a.annotation_root, a.require_merged)
    merged = core["merged"]
    if a.check_sidecar:
        sha = check_sidecar(core, a.check_sidecar)
        print(f"preflight PASS (check-sidecar): target generation "
              f"{core['target']['generation']} kind "
              f"{core['target']['kind']} merged "
              f"{merged['teacher_config_hash'] if merged else 'none'} "
              f"sidecar_sha256 {sha}")
        return 0
    out_dir = a.sidecar_out or target_root
    path, sha, reused = emit_sidecar(core, locations, out_dir)
    print(f"preflight PASS: target generation "
          f"{core['target']['generation']} kind "
          f"{core['target']['kind']} merged "
          f"{merged['teacher_config_hash'] if merged else 'none'} "
          f"sidecar {path} sidecar_sha256 {sha} "
          f"{'reused' if reused else 'sealed'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

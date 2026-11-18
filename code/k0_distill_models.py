"""CNet2 checkpoint loading and candidate scoring shared by the distillation trainers."""
import hashlib
import io
import json

import numpy as np


def _read_weights_file(path, role):
    """One byte read of a weights .npz for one identity role; the
    identity (md5/sha256) and the parse share that read. Returns
    {file, md5, sha256, meta, meta_cfg, arrays} (arrays excludes the
    'meta' entry). Refuses by role name: unreadable or unparseable
    file, or a non-finite float array (the /1 finite gate, applied to
    every identity file alike)."""
    try:
        with open(path, "rb") as fh:
            raw = fh.read()
    except OSError as e:
        raise SystemExit(f"k0_distill_u: {role} file {path!r} is "
                         f"unreadable: {e}; refusing to start")
    md5 = hashlib.md5(raw).hexdigest()
    sha = hashlib.sha256(raw).hexdigest()
    try:
        with np.load(io.BytesIO(raw)) as w:
            names = list(w.files)
            arrays = {k: np.asarray(w[k]) for k in names if k != "meta"}
            meta_raw = str(w["meta"]) if "meta" in names else None
    except Exception as e:
        raise SystemExit(f"k0_distill_u: {role} file {path!r} is not a "
                         f"readable .npz export ({e}); refusing to start")
    for k, a in arrays.items():
        if np.iscomplexobj(a):
            raise SystemExit(
                f"k0_distill_u: {role} array {k!r} is complex; "
                "a CNet2 export must carry real weights")
        if np.issubdtype(a.dtype, np.floating) \
                and not np.isfinite(a.astype(np.float64)).all():
            raise SystemExit(
                f"k0_distill_u finite gate: {role} array {k!r} has "
                f"non-finite entries; refusing to start")
    meta = None
    if "meta" in names:
        try:
            meta = json.loads(meta_raw)
        except Exception as e:
            raise SystemExit(
                f"k0_distill_u: {role} file {path!r} has invalid meta JSON "
                f"({e}); refusing to infer a model configuration")
        if not isinstance(meta, dict):
            raise SystemExit(
                f"k0_distill_u: {role} file {path!r} meta must be an object; "
                "refusing to infer a model configuration")
    meta_cfg = meta.get("cfg") if isinstance(meta, dict) else None
    if meta_cfg is not None and not isinstance(meta_cfg, dict):
        raise SystemExit(
            f"k0_distill_u: {role} file {path!r} meta.cfg must be an object; "
            "refusing to infer a model configuration")
    return {"file": path, "md5": md5, "sha256": sha, "meta": meta,
            "meta_cfg": meta_cfg, "arrays": arrays}


def _build_role_model(weights, role, device, cfg_override=None, legacy_cfg=None):
    """Construct and load one CNet2 role, without borrowing another role's
    architecture when this export declares its own. --cfg remains the
    student's explicit override. A legacy anchor with no meta.cfg keeps
    the old student-config convention, recorded as such in provenance;
    malformed metadata/configuration never enters that compatibility path.

    Only the existing migration whitelist (new value heads / tattn) may
    cold-start. Every other missing key, unexpected key, shape mismatch,
    or non-finite loaded parameter is fatal and names the role.
    """
    import torch
    from cnet2_model import CNet2, DEFAULT, migrate_state_dict

    if cfg_override is not None:
        try:
            requested = json.loads(cfg_override) if isinstance(cfg_override, str) \
                else cfg_override
        except (TypeError, ValueError) as e:
            raise SystemExit(f"k0_distill_u: {role} --cfg is not valid JSON: {e}")
        source = "cli_cfg"
    elif weights["meta_cfg"] is not None:
        requested = weights["meta_cfg"]
        source = "meta.cfg"
    elif legacy_cfg is not None:
        requested = legacy_cfg
        source = "legacy_student_cfg_compatibility"
    else:
        requested = {}
        source = "legacy_cnet2_defaults"
    if not isinstance(requested, dict):
        raise SystemExit(f"k0_distill_u: {role} cfg must be a JSON object")
    cfg = dict(DEFAULT, ZI=0, TA=0, TNH=4)
    cfg.update(requested)
    for key in ("C", "D", "NH", "E", "G", "TNH", "NB", "L", "TA"):
        value = cfg[key]
        minimum = 0 if key in ("NB", "L", "TA") else 1
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise SystemExit(f"k0_distill_u: {role} cfg.{key} must be an integer "
                             f">= {minimum}, got {value!r}")
    if not isinstance(cfg["ZI"], (int, bool)) or cfg["ZI"] not in (0, 1):
        raise SystemExit(f"k0_distill_u: {role} cfg.ZI must be 0 or 1")
    if cfg["C"] % cfg["G"] or cfg["D"] % cfg["NH"]:
        raise SystemExit(f"k0_distill_u: {role} cfg requires C divisible by G "
                         "and D divisible by NH")
    if cfg.get("TA", 0) > cfg["NB"] or \
            (cfg.get("TA", 0) and cfg["C"] % cfg.get("TNH", 4)):
        raise SystemExit(f"k0_distill_u: {role} cfg requires TA <= NB and, "
                         "when TA > 0, C divisible by TNH")
    try:
        model = CNet2(**cfg).to(device)
        state = {k: torch.from_numpy(np.ascontiguousarray(a))
                 for k, a in weights["arrays"].items()}
        migration = migrate_state_dict(
            model, state, allow_missing_prefixes=("tattn.",), source=role)
    except (TypeError, ValueError, RuntimeError) as e:
        raise SystemExit(f"k0_distill_u: {role} weights/configuration do not fit: {e}")
    for name, value in model.state_dict().items():
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise SystemExit(f"k0_distill_u finite gate: {role} loaded parameter "
                             f"{name!r} is non-finite; refusing to start")
    cfg_bytes = json.dumps(model.cfg, sort_keys=True, separators=(",", ":")).encode()
    provenance = {
        "class": "CNet2", "cfg": dict(model.cfg),
        "cfg_sha256": hashlib.sha256(cfg_bytes).hexdigest(), "cfg_source": source,
        "declared_meta_cfg": weights["meta_cfg"],
        "migrated_missing": migration["migrated_missing"]}
    return model, migration, provenance


def candidate_outputs(net, anchor, batch, candidate_count):
    """One encoding per root per model; decode each joint action under its own prefix."""
    import torch

    # expand_candidate_batch emits root-major rows. Only the decoder reads
    # candidate-dependent budget, occupancy, legality and previous actions.
    roots = {key: values[::candidate_count] for key, values in batch.items()}

    def decode(model):
        tokens, context, _ = model.encode(roots)
        logits = model.decode(batch, tokens.repeat_interleave(candidate_count, dim=0),
                              context.repeat_interleave(candidate_count, dim=0))
        return logits, context

    logits, context = decode(net)
    with torch.no_grad():
        reference_logits, _ = decode(anchor)
    return logits, reference_logits, context

#!/usr/bin/env python3
# =====================================================================
# test_sparse -- hard correctness gates for the sparse-label /3
# replay implementation (ACCEPT a7da515d SS3 layer 1). Subcommands:
#
#   loader-equiv   sparse loader vs stock loader on the same /2
#                  product: info + every array must agree.
#   loader-neg     tampered /3 products must die by name: v-root
#                  placeholder pi tampered, v-root q nonzero, v-root
#                  n_arms != 0, q-root stored-q tampered, p_q = 0,
#                  has_q = 2, full-coverage fraud in the meta.
#   grad-zero      (torch, L20) v roots contribute EXACTLY zero to
#                  the policy loss and its gradients (perturbation
#                  invariance), DO flow into the value loss, a
#                  mixed batch's policy term equals the q-only
#                  batch's, a zero-q batch skips the policy term,
#                  and the single-row ctx equals the slot-0 ctx.
#   denom-check    a run log's denominators (roots_q, candidates_q,
#                  ess_q, batch_sha256) recomputed independently
#                  from the dataset + the registered seed stream.
#   stamp-check    a caliber-B checkpoint's stamp: ipw.active is
#                  false and the reconciliation block proves the
#                  declared weights were applied.
# =====================================================================
import argparse
import hashlib
import importlib.util
import io
import json
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def sparse_mod():
    return _load_module("k0_distill_u_sparse",
                        os.path.join(HERE, "k0_distill_u.py"))


# --------------------------- loader-equiv --------------------------------


def cmd_loader_equiv(a):
    stock = _load_module("k0_distill_u_stock",
                         os.path.join(a.stock_dir, "k0_distill_u.py"))
    sparse = sparse_mod()
    d1 = stock.load_annotation(a.data)
    d2 = sparse.load_annotation(a.data)
    bad = []
    for k in ("arms", "n_arms", "priors", "q_stored", "target", "Q",
              "pi0", "pi1", "pi", "terminal_u", "terminal_wdl",
              "terminal_margin"):
        if not np.array_equal(np.asarray(d1[k]), np.asarray(d2[k]),
                              equal_nan=True):
            bad.append(k)
    for k in ("generation", "roots", "candidates", "teacher_config_hash",
              "teacher_weights_md5", "ipw_active", "pi_min", "pi_mean",
              "train_temp"):
        if d1["info"][k] != d2["info"][k]:
            bad.append(f"info.{k}")
    print(json.dumps({"loader_equiv": "FAIL" if bad else "PASS",
                      "mismatches": bad}))
    return 1 if bad else 0


# --------------------------- loader-neg ----------------------------------


def _canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def mktampered(src_root, dst_root, mutate_policy=None, mutate_meta=None):
    """Clone a dataset root cheaply: symlink every field file except
    the ones being mutated; adjust the manifest (field hash/size for
    a mutated bundle, meta seal for a mutated meta) and recompute
    the manifest self-seal. The generation id is kept (test sandbox
    only; the consumer verify checks pointer/manifest agreement,
    seals and field hashes, all of which stay honest here)."""
    ptr = json.load(open(os.path.join(src_root, "current.json")))
    gen = ptr["generation"]
    src_gen = os.path.join(src_root, ptr["dir"])
    dst_gen = os.path.join(dst_root, "generations", gen)
    os.makedirs(dst_gen, exist_ok=True)
    man = json.load(open(os.path.join(src_gen, "manifest.json")))
    for k, info in man["fields"].items():
        dst = os.path.join(dst_gen, info["file"])
        if os.path.exists(dst):
            os.remove(dst)
        if k == "policy_labels" and mutate_policy is not None:
            with np.load(os.path.join(src_gen, info["file"])) as z:
                bundle = {kk: z[kk] for kk in z.files}
            bundle = mutate_policy(bundle)
            with open(dst, "wb") as fh:
                np.savez(fh, **bundle)
            raw = open(dst, "rb").read()
            info["bytes"] = len(raw)
            info["sha256"] = hashlib.sha256(raw).hexdigest()
        else:
            os.symlink(os.path.abspath(
                os.path.join(src_gen, info["file"])), dst)
    meta_p = os.path.join(dst_gen, "meta.json")
    if os.path.exists(meta_p):
        os.remove(meta_p)
    if mutate_meta is not None:
        meta = json.load(open(os.path.join(src_gen, "meta.json")))
        meta = mutate_meta(meta)
        with open(meta_p, "w") as fh:
            json.dump(meta, fh, indent=1, sort_keys=True)
        man["meta_sha256"] = hashlib.sha256(
            open(meta_p, "rb").read()).hexdigest()
    else:
        os.symlink(os.path.abspath(
            os.path.join(src_gen, "meta.json")), meta_p)
    man["self_sha256"] = hashlib.sha256(_canonical(
        {k: v for k, v in man.items() if k != "self_sha256"})).hexdigest()
    with open(os.path.join(dst_gen, "manifest.json"), "w") as fh:
        json.dump(man, fh, indent=1, sort_keys=True)
    ptr["manifest_sha256"] = hashlib.sha256(
        open(os.path.join(dst_gen, "manifest.json"), "rb").read()).hexdigest()
    with open(os.path.join(dst_root, "current.json"), "w") as fh:
        json.dump(ptr, fh)
    return dst_root


def _first_v(bundle):
    return int(np.nonzero(bundle["has_q"] == 0)[0][0])


def _first_q(bundle):
    return int(np.nonzero(bundle["has_q"] == 1)[0][0])


def cmd_loader_neg(a):
    sparse = sparse_mod()
    cases = []

    def case(name, fn, expect_substr, meta_fn=None):
        dst = os.path.join(a.work, name)
        shutil.rmtree(dst, ignore_errors=True)
        root = mktampered(a.data, dst, fn, meta_fn)
        try:
            sparse.load_annotation(root)
            cases.append({"case": name, "verdict": "FAIL",
                          "detail": "loaded without refusal"})
        except SystemExit as e:
            ok = expect_substr in str(e)
            cases.append({"case": name,
                          "verdict": "PASS" if ok else "FAIL",
                          "detail": str(e)[:220]})

    def t_pi(b):
        b = dict(b)
        b["pi"] = b["pi"].copy()
        b["pi"][_first_v(b)] = 0.5
        return b

    def t_q(b):
        b = dict(b)
        b["q"] = b["q"].copy()
        b["q"][_first_v(b), 0] = 0.25
        return b

    def t_narms(b):
        b = dict(b)
        b["n_arms"] = b["n_arms"].copy()
        b["n_arms"][_first_v(b)] = 1
        return b

    def t_storedq(b):
        b = dict(b)
        b["q"] = b["q"].copy()
        r = _first_q(b)
        n = int(b["n_arms"][r])
        b["q"][r, :n] = b["q"][r, :n][::-1]
        return b

    def t_pq0(b):
        b = dict(b)
        b["p_q"] = b["p_q"].copy()
        b["p_q"][_first_q(b)] = 0.0
        return b

    def t_hasq2(b):
        b = dict(b)
        b["has_q"] = b["has_q"].copy()
        b["has_q"][_first_q(b)] = 2
        return b

    def m_fraud(meta):
        meta = json.loads(json.dumps(meta))
        meta["q_sampler"]["full_coverage"] = True
        return meta

    case("v_pi_placeholder", t_pi, "placeholder is not the identity")
    case("v_q_nonzero", t_q, "priors/q are not all 0")
    case("v_narms", t_narms, "lazy form requires n_arms=0")
    case("q_storedq", t_storedq, "stored q disagrees")
    case("pq_zero", t_pq0, "p_q")
    case("hasq_two", t_hasq2, "has_q")
    case("full_coverage_fraud", None, "full coverage", m_fraud)
    bad = [c for c in cases if c["verdict"] != "PASS"]
    print(json.dumps({"loader_neg": cases,
                      "verdict": "FAIL" if bad else "PASS"}, indent=1))
    return 1 if bad else 0


# --------------------------- grad-zero -----------------------------------


def cmd_grad_zero(a):
    os.environ["BC_DEVICE"] = a.device
    import torch
    import cnet2_train as T
    from cnet2_model import CNet2, migrate_state_dict
    D = sparse_mod()
    dev = torch.device(a.device)
    ds = D.load_annotation(a.data)
    target = ds["target"]
    w_root, audit = D.sparse_root_weights(ds, "ipw")
    si = D._read_weights_file(a.teacher, "--teacher")
    cfg = si["meta_cfg"]
    sd = {k: torch.from_numpy(np.ascontiguousarray(v))
          for k, v in si["arrays"].items()}
    net = CNet2(**(cfg or {})).to(dev)
    migrate_state_dict(net, sd, allow_missing_prefixes=("tattn.",),
                       source="--teacher")
    net.eval()
    anchor = CNet2(**(cfg or {})).to(dev)
    migrate_state_dict(anchor, sd, allow_missing_prefixes=("tattn.",),
                       source="--teacher")
    anchor.eval()
    for p in anchor.parameters():
        p.requires_grad_(False)
    params = [p for p in net.parameters() if p.requires_grad]
    has_q = ds["has_q"]
    q_rows = np.nonzero(has_q == 1)[0]
    v_rows = np.nonzero(has_q == 0)[0]
    batch = np.concatenate([q_rows[:8], v_rows[:8]])
    stats = {"cash_recovery_roots": set(), "cash_ulp_roots": set()}
    kw = dict(kl_beta=1.0, lambda_u=1.0, lambda_w=0.5, lambda_m=0.5,
              dev=dev, feat_stats=stats)

    def run(rows):
        return D.forward_terms_impl(net, anchor, T, ds, target,
                                    w_root, rows, **kw)

    def pol_grads():
        out = run(batch)
        # allow_unused: the value heads (and any policy-disconnected
        # parameter) legitimately receive no policy gradient; None
        # is read as exactly zero.
        g = torch.autograd.grad(out["loss_policy"], params,
                                allow_unused=True)
        return out, [torch.zeros_like(p) if x is None
                     else x.detach().clone()
                     for x, p in zip(g, params)]

    out0, g0 = pol_grads()
    base_pol = float(out0["loss_policy"])
    base_val = float(out0["loss_value"])
    # Platform nondeterminism floor: the SAME batch, unperturbed,
    # run again. Any v-perturbation gradient difference is only
    # meaningful above this floor (GPU reduction order varies with
    # batch composition even for untouched inputs).
    _outr, gr = pol_grads()
    gdiff_repeat = max(float((x - y).abs().max())
                       for x, y in zip(g0, gr))
    # Perturb v roots: features of v rows replaced by other rows'
    # features, terminal labels flipped. Policy must not move AT
    # ALL; value must move (v roots really flow into it).
    keep = {}
    for k in ("maps", "scal"):
        keep[k] = ds["feats"][k][batch[8:]].copy()
        ds["feats"][k][batch[8:]] = ds["feats"][k][q_rows[100:108]]
    keep["tu"] = ds["terminal_u"][batch[8:]].copy()
    ds["terminal_u"][batch[8:]] = 1.0 - ds["terminal_u"][batch[8:]]
    out1, g1 = pol_grads()
    pert_pol = float(out1["loss_policy"])
    pert_val = float(out1["loss_value"])
    gdiff = max(float((x - y).abs().max()) for x, y in zip(g0, g1))
    for k, v in keep.items():
        if k == "tu":
            ds["terminal_u"][batch[8:]] = v
        else:
            ds["feats"][k][batch[8:]] = v
    # Mixed batch policy == q-only batch policy.
    outq = run(batch[:8])
    qonly_pol = float(outq["loss_policy"])
    # Zero-q batch: policy skipped, loss == value, backward works.
    outz = run(batch[8:])
    ok_z = bool(outz["zero_q"] and outz["loss_policy"] is None
                and abs(float(outz["loss"]) - float(outz["loss_value"]))
                < 1e-12)
    outz["loss"].backward()
    # Single-row ctx == slot-0 ctx for one q root.
    import cnet2_feat as F
    r0 = int(batch[0])
    b, _rep, _st = D.expand_candidate_batch(ds["feats"], ds["arms"],
                                            np.array([r0]),
                                            ds["info"]["max_candidates"])
    b = T.to_dev(b, aug=False)
    with torch.no_grad():
        _t, ctx_exp, _ = net.encode(b)
        brow = {k: np.asarray(ds["feats"][k])[[r0]]
                for k in ds["feats"] if k not in ("gid", "step")}
        brow["y"] = np.zeros((1, F.OWN), dtype=np.int64)
        brow = T.to_dev(brow, aug=False)
        _t2, ctx_row, _ = net.encode(brow)
    ctx_diff = float((ctx_exp[0] - ctx_row[0]).abs().max())
    # Intrinsic row-replication noise: within ONE expanded batch,
    # the K rows of the same root carry identical encode inputs;
    # the spread of their ctx is the platform's own per-row noise.
    K = ds["info"]["max_candidates"]
    ctx_slots = ctx_exp.reshape(K, -1)
    slot_spread = float((ctx_slots - ctx_slots[0]).abs().max())
    # Batch-shape floor: the same root's slot-0 ctx when its
    # expanded batch holds 8 roots instead of 1 -- GEMM reduction
    # order depends on batch shape, so THIS is the floor a
    # single-row-vs-expanded ctx comparison must be read against.
    b8, _r8, _s8 = D.expand_candidate_batch(ds["feats"], ds["arms"],
                                            batch[:8], K)
    b8 = T.to_dev(b8, aug=False)
    with torch.no_grad():
        _t8, ctx8, _ = net.encode(b8)
    shape_floor = float((ctx8[0] - ctx_exp[0]).abs().max())
    res = {
        "policy_loss_base": base_pol,
        "policy_loss_v_perturbed": pert_pol,
        "policy_loss_exactly_invariant": base_pol == pert_pol,
        "policy_grad_max_abs_diff_under_v_perturbation": gdiff,
        "policy_grad_max_abs_diff_same_input_repeat": gdiff_repeat,
        "value_loss_base": base_val,
        "value_loss_v_perturbed": pert_val,
        "value_loss_moves": base_val != pert_val,
        "mixed_vs_qonly_policy_abs_diff": abs(base_pol - qonly_pol),
        "zero_q_batch_skips_policy": ok_z,
        "single_row_ctx_vs_slot0_max_abs_diff": ctx_diff,
        "ctx_intrinsic_slot_spread_same_batch": slot_spread,
        "ctx_batch_shape_floor": shape_floor}
    res["verdict"] = "PASS" if (
        res["policy_loss_exactly_invariant"]
        and gdiff <= max(3.0 * gdiff_repeat, 1e-6)
        and res["value_loss_moves"]
        and res["mixed_vs_qonly_policy_abs_diff"] < 1e-5
        and ok_z
        and ctx_diff <= max(3.0 * shape_floor, 1e-6)) else "FAIL"
    print(json.dumps({"grad_zero": res}, indent=1))
    return 0 if res["verdict"] == "PASS" else 1


# --------------------------- denom-check ---------------------------------


def cmd_denom_check(a):
    sparse = sparse_mod()
    ds = sparse.load_annotation(a.data)
    w_root, _audit = sparse.sparse_root_weights(ds, a.caliber)
    lines = [json.loads(x) for x in open(a.log) if x.strip()]
    eff = lines[0]["k0_distill_u_effective"]
    seed = eff["recipe"]["seed"]
    bs = eff["recipe"]["bs"]
    R = ds["info"]["roots"]
    has_q = ds["has_q"]
    n_arms = ds["n_arms"]
    rs = np.random.RandomState(seed)
    upd = [x for x in lines if "update" in x]
    bad = []
    done = 0
    stream = []
    while done < len(upd):
        perm = rs.permutation(np.arange(R))
        for a0 in range(0, R, bs):
            if done >= len(upd):
                break
            stream.append(perm[a0:a0 + bs])
            done += 1
    for rec, idx in zip(upd, stream):
        den = rec["denominators"]
        qm = has_q[idx] == 1
        want = {"roots_total": int(len(idx)),
                "roots_q": int(qm.sum()),
                "candidates_q": int(n_arms[idx][qm].sum()),
                "batch_sha256": hashlib.sha256(
                    np.asarray(idx, dtype=np.int64).tobytes()).hexdigest()}
        w = w_root[idx][qm]
        want["ess_q"] = float((w.sum() ** 2) / (w ** 2).sum()) if qm.any() else 0.0
        for k, v in want.items():
            got = den.get(k)
            if isinstance(v, float):
                if got is None or abs(got - v) > 1e-6 * max(1.0, abs(v)):
                    bad.append((rec["update"], k, got, v))
            elif got != v:
                bad.append((rec["update"], k, got, v))
    print(json.dumps({"denom_check": "FAIL" if bad else "PASS",
                      "updates_checked": len(upd),
                      "mismatches": bad[:5]}))
    return 1 if bad else 0


# --------------------------- stamp-check ---------------------------------


def cmd_stamp_check(a):
    # cnet2_train.export stores meta as np.array(json.dumps(meta)).
    with np.load(a.ckpt) as z:
        meta = json.loads(z["meta"].item())
    prov = meta["k0_distill_u"]
    ipw = prov["ipw"]
    rec = ipw.get("reconciliation") or {}
    checks = {
        "caliber_declared": ipw.get("caliber") == "declared",
        "ipw_active_false": ipw.get("active") is False,
        "weights_match_declared": rec.get("weights_match_declared")
            is True,
        "activation_suppressed_noted": "suppressed"
            in str(rec.get("legacy_record_driven_activation", "")),
        "sparse_block_present": "sparse" in prov,
        "prov_weighting_matches": (prov.get("sparse") or {})
            .get("weighting", {}).get("caliber") == "declared"}
    bad = [k for k, v in checks.items() if not v]
    print(json.dumps({"stamp_check": checks,
                      "verdict": "FAIL" if bad else "PASS"}, indent=1))
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description="sparse /3 correctness "
                                             "gates")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("loader-equiv")
    p.add_argument("--data", required=True)
    p.add_argument("--stock-dir", required=True)
    p = sub.add_parser("loader-neg")
    p.add_argument("--data", required=True)
    p.add_argument("--work", required=True)
    p = sub.add_parser("grad-zero")
    p.add_argument("--data", required=True)
    p.add_argument("--teacher", required=True)
    p.add_argument("--device", default="cuda")
    p = sub.add_parser("denom-check")
    p.add_argument("--data", required=True)
    p.add_argument("--log", required=True)
    p.add_argument("--caliber", default="ipw")
    p = sub.add_parser("stamp-check")
    p.add_argument("--ckpt", required=True)
    a = ap.parse_args()
    return {"loader-equiv": cmd_loader_equiv,
            "loader-neg": cmd_loader_neg,
            "grad-zero": cmd_grad_zero,
            "denom-check": cmd_denom_check,
            "stamp-check": cmd_stamp_check}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())

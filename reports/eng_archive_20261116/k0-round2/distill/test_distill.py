"""Acceptance tests for k0_distill (SS10 item 2, five items + arm
identity). Run AFTER make_fixture.py:
  ~/.venvs/k0/bin/python test_distill.py <fixture_dir>
Prints one JSON object per test group; exits non-zero on any failure.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys

import numpy as np

FX = os.path.abspath(sys.argv[1])
RUN = os.path.join(FX, "run")
sys.path.insert(0, RUN)
os.environ.setdefault("BC_DEVICE", "cpu")

RESULTS = {}


def report(name, **kw):
    RESULTS[name] = kw
    print(json.dumps({name: kw}), flush=True)


def load_nets():
    import io
    import torch
    import cnet2_train as T  # noqa: F401  (sets DEV via BC_DEVICE)
    from cnet2_model import CNet2, migrate_state_dict
    raw = open(os.path.join(FX, "parent.npz"), "rb").read()
    w = np.load(io.BytesIO(raw))
    cfg = json.loads(str(w["meta"]))["cfg"]
    sd = {k: torch.from_numpy(np.ascontiguousarray(w[k])) for k in w.files if k != "meta"}
    nets = []
    for _ in range(2):
        net = CNet2(**cfg)
        migrate_state_dict(net, sd, allow_missing_prefixes=("tattn.",), source="parent")
        nets.append(net)
    return nets[0], nets[1]


def expanded(ds, root_idx, K):
    import cnet2_train as T
    rep = np.repeat(np.asarray(root_idx), K)
    b = {}
    for k in ds["feats"]:
        if k in ("gid", "step"):
            continue
        b[k] = np.asarray(ds["feats"][k])[rep]
    b["y"] = ds["arms"][np.asarray(root_idx)].reshape(-1, ds["arms"].shape[2])
    return T.to_dev(b, aug=False)


def single_root_batch(ds, r, y_override):
    import cnet2_train as T
    b = {}
    for k in ds["feats"]:
        if k in ("gid", "step"):
            continue
        b[k] = np.asarray(ds["feats"][k])[r:r + 1]
    b["y"] = np.asarray(y_override)[None, :]
    return T.to_dev(b, aug=False)


def replay_joint_logp(net, ds, r, arm):
    """Independent per-slot replay: for each valid slot s, a FRESH
    decode whose y agrees with the arm strictly before s (slots >= s
    blanked) -- logits at slot s are then the model's own conditional
    for that slot given exactly the candidate's prefix. Summed in
    float64."""
    import torch
    import torch.nn.functional as Fn
    total = None
    for s in range(len(arm)):
        if arm[s] < 0:
            continue
        y = np.where(np.arange(len(arm)) < s, arm, -1).astype(np.int64)
        b = single_root_batch(ds, r, y)
        tok, ctx, _ = net.encode(b)
        lg = net.decode(b, tok, ctx)
        lp = Fn.log_softmax(lg[0, s].double(), dim=-1)[int(arm[s])]
        total = lp if total is None else total + lp
    return total


def t1_joint_logp(KD):
    import torch
    ds = KD.load_annotation(os.path.join(FX, "ds_a"))
    student, parent = load_nets()
    student.eval(); parent.eval()
    K = ds["info"]["max_candidates"]
    idx = [0, 1]
    b = expanded(ds, idx, K)
    with torch.no_grad():
        tok, ctx, _ = student.encode(b)
        lg_s = student.decode(b, tok, ctx)
        tokp, ctxp, _ = parent.encode(b)
        lg_p = parent.decode(b, tokp, ctxp)
        arms_b = torch.from_numpy(ds["arms"][idx].reshape(-1, ds["arms"].shape[2]))
        joint_mod, _ = KD.joint_logp_kl(lg_s, lg_p, arms_b)
    diffs = []
    m = 0
    for r in idx:
        for k in range(int(ds["n_arms"][r])):
            with torch.no_grad():
                ref = replay_joint_logp(student, ds, r, ds["arms"][r][k])
            diffs.append(abs(float(joint_mod[m]) - float(ref)))
            m += 1
    report("t1_joint_logp", max_abs_diff=max(diffs), n_candidates=len(diffs),
           tol=1e-5, pass_=bool(max(diffs) <= 1e-5))


def t2_flatten(KD):
    import torch
    pr = np.array([0.5, 0.3, 0.2])
    a = KD.construct_target(np.array([0.4, 0.5, 0.6]), pr, 0.05, flatten=True)
    bq = KD.construct_target(np.array([0.9, 0.1, 0.3]), pr, 0.05, flatten=True)
    q0_ref = np.sqrt(pr) / np.sqrt(pr).sum()
    ident = bool(np.array_equal(a, bq))
    close = float(np.abs(a - q0_ref).max())
    # gradient check: module loss path vs per-slot-replay autograd,
    # target built independently in this test as q0 from priors.
    ds = KD.load_annotation(os.path.join(FX, "ds_a"))
    K = ds["info"]["max_candidates"]
    idx = [0, 1]
    tgt_flat = KD.targets_with_flatten(ds, True)
    grads = []
    for mode in ("module", "replay"):
        student, parent = load_nets()
        parent.eval()
        for p in parent.parameters():
            p.requires_grad_(False)
        bb = expanded(ds, idx, K)
        if mode == "module":
            tok, ctx, _ = student.encode(bb)
            lg_s = student.decode(bb, tok, ctx)
            with torch.no_grad():
                tokp, ctxp, _ = parent.encode(bb)
                lg_p = parent.decode(bb, tokp, ctxp)
            arms_b = torch.from_numpy(ds["arms"][idx].reshape(-1, ds["arms"].shape[2]))
            joint, _kl = KD.joint_logp_kl(lg_s, lg_p, arms_b)
            joint = joint.reshape(len(idx), K)
            tgt = torch.from_numpy(tgt_flat[idx])
        else:
            rows = []
            for r in idx:
                for k in range(K):
                    if k < int(ds["n_arms"][r]):
                        rows.append(replay_joint_logp(student, ds, r, ds["arms"][r][k]))
                    else:
                        rows.append(torch.zeros((), dtype=torch.float64))
            joint = torch.stack(rows).reshape(len(idx), K)
            tgt = torch.from_numpy(np.stack([
                np.concatenate([np.sqrt(ds["priors"][r][:int(ds["n_arms"][r])]) /
                                np.sqrt(ds["priors"][r][:int(ds["n_arms"][r])]).sum(),
                                np.zeros(K - int(ds["n_arms"][r]))])
                for r in idx]))
        ce_root = -(tgt * joint).sum(dim=1)
        loss = ce_root.mean()
        loss.backward()
        grads.append({n: (p.grad.clone() if p.grad is not None else None)
                      for n, p in student.named_parameters()})
    maxd, worst = 0.0, None
    for n in grads[0]:
        g0, g1 = grads[0][n], grads[1][n]
        if g0 is None and g1 is None:
            continue
        if (g0 is None) != (g1 is None):
            maxd, worst = float("inf"), n
            break
        d = float((g0.double() - g1.double()).abs().max())
        if d > maxd:
            maxd, worst = d, n
    report("t2_flatten", target_identical_across_Q=ident,
           target_vs_sqrt_prior_maxdiff=close, grad_max_abs_diff=maxd,
           grad_worst=worst, tol=1e-4,
           pass_=bool(ident and close <= 1e-12 and maxd <= 1e-4))


def t5_ipw(KD):
    import torch
    w = KD.root_weights(np.array([1.0, 0.5]), True)
    w_off = KD.root_weights(np.array([1.0, 0.5]), False)
    ds_b = KD.load_annotation(os.path.join(FX, "ds_b"))
    ds_a = KD.load_annotation(os.path.join(FX, "ds_a"))
    student, parent = load_nets()
    student.eval(); parent.eval()
    K = ds_b["info"]["max_candidates"]
    idx = [0, 1]
    bb = expanded(ds_b, idx, K)
    with torch.no_grad():
        tok, ctx, _ = student.encode(bb)
        lg_s = student.decode(bb, tok, ctx)
        tokp, ctxp, _ = parent.encode(bb)
        lg_p = parent.decode(bb, tokp, ctxp)
        arms_b = torch.from_numpy(ds_b["arms"][idx].reshape(-1, ds_b["arms"].shape[2]))
        joint, _ = KD.joint_logp_kl(lg_s, lg_p, arms_b)
        joint = joint.reshape(2, K)
        tgt = torch.from_numpy(ds_b["target"][idx])
        ce = -(tgt * joint).sum(dim=1).numpy()
    report("t5_ipw", weights_active=w.tolist(), weights_inactive=w_off.tolist(),
           ratio=float(w[1] / w[0]), ce_identical_roots=[float(ce[0]), float(ce[1])],
           ipw_active_b=bool(ds_b["info"]["ipw_active"]),
           ipw_active_a=bool(ds_a["info"]["ipw_active"]),
           pass_=bool(w[1] / w[0] == 2.0 and abs(ce[0] - ce[1]) <= 1e-9
                      and ds_b["info"]["ipw_active"] and not ds_a["info"]["ipw_active"]))


def t4_negatives(KD):
    out = {}
    # tampered feature file (post-seal byte flip)
    dst = os.path.join(FX, "ds_tamper")
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(os.path.join(FX, "ds_a"), dst)
    gen = json.load(open(os.path.join(dst, "current.json")))["generation"]
    p = os.path.join(dst, "generations", gen, "scal.npy")
    raw = bytearray(open(p, "rb").read())
    raw[-1] ^= 0xFF
    open(p, "wb").write(bytes(raw))
    full = {}
    try:
        KD.load_annotation(dst)
        full["tamper"] = "NOT REFUSED"
    except SystemExit as e:
        full["tamper"] = str(e)
    # sealed generation dir passed directly
    try:
        KD.load_annotation(os.path.join(FX, "ds_a", "generations", gen))
        full["direct_gen_dir"] = "NOT REFUSED"
    except SystemExit as e:
        full["direct_gen_dir"] = str(e)
    out = {k: v[:80] for k, v in full.items()}
    report("t4_negatives", **out,
           pass_=bool("hash mismatch" in full.get("tamper", "")
                      and "via its dataset root" in full.get("direct_gen_dir", "")))


def t3_e2e(KD):
    py = sys.executable
    logs = {}
    for arm, extra in (("S", []), ("C", ["--flatten-q"])):
        outp = os.path.join(FX, f"out_{arm}.npz")
        cmd = [py, "k0_distill.py", "--data", os.path.join(FX, "ds_a"),
               "--parent", os.path.join(FX, "parent.npz"), "--out", outp,
               "--seed", "7", "--max-updates", "6", "--bs", "4", "--lr", "3e-3"] + extra
        r = subprocess.run(cmd, cwd=RUN, capture_output=True, text=True)
        if r.returncode != 0:
            report("t3_e2e", arm=arm, rc=r.returncode, stderr=r.stderr[-500:])
            return
        logs[arm] = [json.loads(l) for l in open(outp + ".log.jsonl")]
    seq_s = [l["denominators"]["batch_sha256"] for l in logs["S"] if "update" in l]
    seq_c = [l["denominators"]["batch_sha256"] for l in logs["C"] if "update" in l]
    parent = np.load(os.path.join(FX, "parent.npz"))
    stu = np.load(os.path.join(FX, "out_S.npz"))
    frozen_ok, moved = True, []
    for k in stu.files:
        if k == "meta":
            continue
        if k.startswith(KD.FROZEN_PREFIXES):
            if not np.array_equal(np.asarray(stu[k]), np.asarray(parent[k])):
                frozen_ok = False
        elif not np.array_equal(np.asarray(stu[k]), np.asarray(parent[k])):
            moved.append(k)
    meta = json.loads(str(stu["meta"]))
    prov = meta["k0_distill"]
    marker = json.load(open(os.path.join(FX, "out_S.npz.complete.json")))
    sha = hashlib.sha256(open(os.path.join(FX, "out_S.npz"), "rb").read()).hexdigest()
    kl_lines = [l["kl_monitor"] for l in logs["S"] if "update" in l]
    report("t3_e2e", batch_order_identical=(seq_s == seq_c), frozen_bitwise=frozen_ok,
           moved_count=len(moved), moved_sample=moved[:4],
           arm=prov["arm"], kl_definition=prov["recipe"]["kl_definition"][:44],
           frozen_check=prov["frozen_check"]["bitwise_identical_to_parent"],
           parent_md5_ok=prov["parent"]["md5"] ==
           hashlib.md5(open(os.path.join(FX, "parent.npz"), "rb").read()).hexdigest(),
           marker_sha_ok=marker["sha256"] == sha,
           kl_monitor_keys=sorted(kl_lines[0].keys()) if kl_lines else [],
           pass_=bool(seq_s == seq_c and frozen_ok and len(moved) >= 2
                      and prov["frozen_check"]["bitwise_identical_to_parent"]
                      and marker["sha256"] == sha))
    # parent mismatch refuses before training
    r = subprocess.run([py, "k0_distill.py", "--data", os.path.join(FX, "ds_a"),
                        "--parent", os.path.join(FX, "parent2.npz"),
                        "--out", os.path.join(FX, "out_bad.npz"),
                        "--max-updates", "1"], cwd=RUN, capture_output=True, text=True)
    report("t4_parent_mismatch", rc=r.returncode,
           named="teacher_weights_md5" in (r.stdout + r.stderr),
           pass_=bool(r.returncode != 0 and "teacher_weights_md5" in (r.stdout + r.stderr)))


def main():
    import k0_distill as KD
    t1_joint_logp(KD)
    t2_flatten(KD)
    t5_ipw(KD)
    t4_negatives(KD)
    t3_e2e(KD)
    bad = [k for k, v in RESULTS.items() if not v.get("pass_")]
    print(json.dumps({"ALL_PASS": not bad, "failed": bad}))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()

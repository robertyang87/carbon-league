#!/usr/bin/env python3
"""Run tiny synthetic fixtures through the actual dense/sparse training CLIs.

From the repository root:
  python reports/sapling_1200k/distill_smoke.py --work-dir runs/distill-smoke-1

Requires numpy and torch. The work directory must not exist. The default
initial checkpoint is data/weights/sapling_1200k/initial.npz; all generated
datasets, weights, logs and the receipt stay in the new work directory.
This is compatibility evidence, never production training or a strength test.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
ENV = dict(PYTHONDONTWRITEBYTECODE="1", BC_DEVICE="cpu", BC_GPU_AUG="0",
           OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
os.environ.update(ENV)
sys.dont_write_bytecode = True
sys.path.insert(0, str(REPO / "code"))


def require(ok, detail):
    if not ok:
        raise RuntimeError(detail)


def identity(path):
    path = Path(path)
    raw = path.read_bytes()
    return dict(path=str(path), bytes=len(raw), md5=hashlib.md5(raw).hexdigest(),
                sha256=hashlib.sha256(raw).hexdigest())


def write_json(path, value):
    # Exclusive creation: this script never replaces an existing artifact.
    with Path(path).open("x") as fh:
        json.dump(value, fh, indent=2, allow_nan=False)
        fh.write("\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work-dir", required=True, type=Path,
                    help="new directory for all generated artifacts; must not exist")
    ap.add_argument("--initial", type=Path,
                    default=REPO / "data/weights/sapling_1200k/initial.npz")
    args = ap.parse_args()
    work, initial = args.work_dir.resolve(), args.initial.resolve()
    if not initial.is_file():
        ap.error(f"initial weights do not exist: {initial}")
    if work.exists():
        ap.error(f"work directory already exists; choose a new path: {work}")
    work.mkdir(parents=True, exist_ok=False)

    import numpy as np
    import torch
    import cnet2_train as T
    import k0_distill_u as D
    import k0_distill_u_sparse as S
    from cnet2_model import CNet2
    from test_k0_distill_u_shared_encode import fixture

    torch.set_num_threads(1)
    torch.manual_seed(101)
    anchor = CNet2(C=16, NB=1, D=16, NH=2, L=1, E=8, G=4)
    teacher = work / "synthetic-anchor.npz"
    T.export(anchor, str(teacher), {
        "cfg": anchor.cfg,
        "purpose": "SYNTHETIC COMPATIBILITY FIXTURE; NOT A PRODUCTION TEACHER"})
    f = fixture()
    R, K = f["arms"].shape[:2]

    def make_dataset(name, sparse=False):
        root = work / name
        gen = root / "generation"
        gen.mkdir(parents=True)
        arms, counts = f["arms"].copy(), f["n_arms"].copy()
        Q = np.full((R, K), np.nan, dtype=np.float64)
        priors = np.zeros((R, K), dtype=np.float64)
        q = np.zeros_like(priors)
        for r, n in enumerate(counts):
            Q[r, :n] = np.linspace(.43, .57, n)
            priors[r, :n] = 1 / n
            q[r, :n] = D.construct_target(Q[r, :n], priors[r, :n])
        extra = {}
        if sparse:
            # Same mixed-root construction as the repository regression test
            # test_sparse_value_only_roots_do_not_run_the_policy. Two synthetic
            # Bernoulli inclusion rates deliberately yield unequal IPW weights.
            extra = dict(has_q=np.array([1, 0, 1], dtype=np.uint8),
                         p_q=np.array([.5, .5, .25], dtype=np.float64))
            arms[1], counts[1], priors[1], q[1], Q[1] = -1, 0, 0, 0, np.nan
        for key in D.FEATURE_FIELDS:
            np.save(gen / (key + ".npy"), f["feats"][key])
        keys = dict(game=f["feats"]["gid"], step=f["feats"]["step"],
                    seat=np.array([0, 1, 0], dtype=np.int64))
        np.savez(gen / "policy_labels.npz", **keys, arms=arms, n_arms=counts,
                 priors=priors, q=q, pi0=np.ones(R), pi1=np.zeros(R),
                 pi=np.ones(R), **extra)
        np.savez(gen / "value_labels.npz", **keys, Q=Q,
                 terminal_u=f["terminal_u"], terminal_wdl=f["terminal_wdl"],
                 terminal_margin=f["terminal_margin"])
        cfg = dict(train_temp=1.0, behavior_temp=1.0, fixture="synthetic-only")
        cfg_sha = D._canonical_sha256(cfg)
        meta = {"games": ["synthetic-900", "synthetic-901", "synthetic-902"],
                "teacher": {"config": cfg, "config_sha256": cfg_sha},
                "synthetic_fixture": True,
                "terminal_label_source": "invented margins; no real terminal games"}
        if sparse:
            meta["q_sampler"] = dict(id="synthetic-fixture/1", roots_q=2,
                                      full_coverage=False,
                                      method="constructed Bernoulli outcome; not sampled production data")
        write_json(gen / "meta.json", meta)
        fields = {}
        for key in D.FEATURE_FIELDS + (D.POLICY_BUNDLE, D.VALUE_BUNDLE):
            p = gen / (key + (".npz" if key.endswith("_labels") else ".npy"))
            ident = identity(p)
            fields[key] = dict(file=p.name, bytes=ident["bytes"],
                               sha256=ident["sha256"], rows=R)
        man = dict(format=S.ANNOTATION_FORMAT_V3 if sparse else D.ANNOTATION_FORMAT,
                   total=R, generation="synthetic-sapling120-" + name,
                   fields=fields, meta_sha256=identity(gen / "meta.json")["sha256"],
                   producer=dict(teacher_config_hash=cfg_sha,
                                 teacher_weights_md5=identity(teacher)["md5"],
                                 belief="conditional", config=cfg, train_temp=1.0),
                   synthetic_fixture=True)
        man["self_sha256"] = D._canonical_sha256(man)
        write_json(gen / "manifest.json", man)
        target = dict(generation=man["generation"],
                      manifest_sha256=identity(gen / "manifest.json")["sha256"])
        write_json(root / "current.json", dict(dir=gen.name, **target))
        # Test-only honest self-seal; no production preflight claim.
        sc = dict(format="synthetic-consumption-fixture/1", synthetic_fixture=True,
                  target=target, merged=None, full_production_preflight_executed=False)
        sc["sidecar_sha256"] = D._canonical_sha256(sc)
        write_json(root / "CONSUMPTION.json", sc)
        loader = S if sparse else D
        ds = loader.load_annotation(str(root))
        loader._require_consumption_sidecar(str(root), ds)
        return root, ds, dict(manifest=identity(gen / "manifest.json"),
                             consumption=identity(root / "CONSUMPTION.json"))

    dense_root, dense_ds, dense_seals = make_dataset("dense-v2")
    sparse_root, sparse_ds, sparse_seals = make_dataset("sparse-v3", sparse=True)
    sparse_weights, weight_audit = S.sparse_root_weights(sparse_ds, "ipw")
    require(np.array_equal(sparse_weights, [2., 0., 4.]), "IPW weights differ")
    with np.load(initial) as z:
        initial_arrays = {k: z[k] for k in z.files if k != "meta"}
        initial_meta = json.loads(str(z["meta"].item()))
    student = CNet2(**initial_meta["cfg"])
    param_count = sum(p.numel() for p in student.parameters())
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    report = {
        "status": "RUNNING", "purpose": "SYNTHETIC COMPATIBILITY ONLY; no production training or strength evidence",
        "source_sha": head, "initial": identity(initial), "student_cfg": initial_meta["cfg"],
        "student_parameter_count": param_count, "synthetic_anchor": identity(teacher),
        "synthetic_anchor_cfg": anchor.cfg,
        "fixture": {"roots": R, "seats": [0, 1, 0],
                    "method": "repository shared_encode fixture: real Feat/Prefix tensors and legal actions; invented Q and terminal labels; real construct_target",
                    "terminal_u": f["terminal_u"].tolist(), "terminal_wdl": f["terminal_wdl"].tolist(),
                    "terminal_margin": f["terminal_margin"].tolist(),
                    "dense_v2": dense_seals, "sparse_v3": sparse_seals,
                    "has_q": [1, 0, 1], "p_q": [.5, .5, .25],
                    "policy_ipw_expected_and_actual": sparse_weights.tolist(),
                    "consumption_scope": "synthetic fixture self-seal passes actual trainer hook; production annotator/join/consume_preflight not executed"},
        "production_boundary": {
            "production_data_used": False, "real_terminal_labels_verified": False,
            "playing_strength_evaluated": False,
            "availability": "150/250-game production arrays were not available to the earlier local review; this rerun does not access or re-verify external machines.",
            "references": ["L20: /root/carbon/katago/loop_pilot/joined_u, gen-90faee2ef4221d59",
                           "L20: /root/sparse_impl/data/base_all1_train, gen-8e357f53df68137e",
                           "L20: /root/carbon/katago/loop2"],
            "train_holdout_validation": "not assessed; no production split consumed"},
        "code": {p: identity(REPO / p) for p in (
            "code/k0_distill_u.py", "code/k0_distill_u_sparse.py", "code/k0_distill_models.py",
            "code/test_k0_distill_u_shared_encode.py")}, "smokes": []}
    cases = [("dense_v2", "k0_distill_u.py", dense_root, 3, 2, False),
             ("sparse_v2_compat", "k0_distill_u_sparse.py", dense_root, 3, 2, False),
             ("sparse_v3_ipw_mixed", "k0_distill_u_sparse.py", sparse_root, 3, 1, True),
             ("sparse_v3_zero_q_batch", "k0_distill_u_sparse.py", sparse_root, 1, 2, True)]
    for name, trainer, data, bs, nupdates, is_v3 in cases:
        out = work / (name + "-student.npz")
        cmd = [sys.executable, str(REPO / "code" / trainer), "--data", str(data),
               "--label-teacher", str(teacher), "--student-init", str(initial),
               "--kl-reference", str(teacher), "--out", str(out),
               "--max-updates", str(nupdates), "--bs", str(bs), "--device", "cpu", "--seed", "7"]
        if is_v3:
            cmd += ["--weighting-caliber", "ipw"]
        start = time.monotonic()
        proc = subprocess.run(cmd, cwd=REPO, env=os.environ, capture_output=True, text=True, timeout=120)
        (work / (name + ".stdout")).write_text(proc.stdout)
        (work / (name + ".stderr")).write_text(proc.stderr)
        require(proc.returncode == 0, (name, proc.returncode, proc.stderr, proc.stdout[-1500:]))
        logs = [json.loads(line) for line in proc.stdout.splitlines() if line.startswith("{")]
        updates = [x for x in logs if "update" in x]
        require([x["update"] for x in updates] == list(range(1, nupdates + 1)), "update count")
        require(all(np.isfinite(v) for x in updates for v in x["loss"].values() if v is not None), "finite loss")
        marker = json.loads(Path(str(out) + ".complete.json").read_text())
        require(marker["complete"] and marker["updates"] == nupdates
                and marker["sha256"] == identity(out)["sha256"], "completion identity")
        with np.load(out) as z:
            ometa = json.loads(str(z["meta"].item()))
            arrays = {k: z[k] for k in z.files if k != "meta"}
        moved = [k for k in initial_arrays if not np.array_equal(initial_arrays[k], arrays[k])]
        frozen = [k for k in initial_arrays if k.startswith(D.FROZEN_PREFIXES)]
        require(moved and not set(frozen).intersection(moved), "updates/frozen tensors")
        require(all(np.isfinite(a).all() for a in arrays.values()), "finite weights")
        prov = ometa["k0_distill_u"]
        require(prov["updates_done"] == nupdates, "provenance update count")
        if is_v3:
            require(prov["ipw"]["active"] and prov["ipw"]["caliber"] == "ipw", "IPW inactive")
        if name == "sparse_v3_ipw_mixed":
            den = updates[0]["denominators"]
            require(den["roots_q"] == 2 and den["roots_total"] == 3, "mixed root counts")
            require(abs(den["ess_q"] - 1.8) < 1e-12, "unequal-weight ESS")
        if name == "sparse_v3_zero_q_batch":
            # Seed 7 orders roots [2, 1, 0]; the second update is value-only.
            value_only = updates[1]
            require(value_only.get("policy_skipped_zero_q") is True, "zero-q branch not reached")
            require(value_only["denominators"]["roots_q"] == 0, "zero-q denominator")
            require(value_only["loss"]["policy"] is None and
                    value_only["loss"]["total"] == value_only["loss"]["value"], "value-only loss")
        result = dict(case=name, trainer=trainer,
                      format_exercised=S.ANNOTATION_FORMAT_V3 if is_v3 else D.ANNOTATION_FORMAT,
                      command=" ".join(f"{k}={v}" for k, v in ENV.items()) + " " + shlex.join(cmd),
                      returncode=proc.returncode, elapsed_seconds=time.monotonic()-start,
                      updates=updates, output=identity(out),
                      complete_marker=identity(str(out) + ".complete.json"),
                      parameter_tensors_changed=len(moved), frozen_tensors_unchanged=frozen,
                      all_output_parameters_finite=True,
                      unfreeze_reconciliation=prov["unfreeze_reconciliation"],
                      sparse=prov.get("sparse"), log=identity(str(out) + ".log.jsonl"))
        report["smokes"].append(result)
        print(json.dumps(dict(case=name, updates=nupdates, output_md5=identity(out)["md5"])), flush=True)
    require(identity(initial) == report["initial"], "initial checkpoint changed")
    report.update(status="PASS", reproduction_script=identity(__file__),
                  source_sha_after=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
                  scope_limits=["Synthetic labels do not validate teacher quality, terminal data or playing strength.",
                                "Production provenance chain, GPU throughput, optimizer resume and long-run stability not tested.",
                                "Zero-q batch skips new policy gradients; AdamW momentum from an earlier policy update can still move policy parameters."])
    receipt = work / "distill_smoke.json"
    write_json(receipt, report)
    print(json.dumps(dict(status="PASS", report=str(receipt), parameters=param_count)), flush=True)


if __name__ == "__main__":
    main()

"""Fork of k0-round2/distill/make_fixture.py for the r4 X01/X02
regression runs. Same datasets and sealing, with the minimal changes
the two fixes legitimately require of a fixture (the original
fixture's roots were NOT honest Prefix records, which neither the
pre-fix consumer nor the five tests ever noticed):

  * exactly two worker slots knocked out (3 and 8) in every root, so
    the root free-slot count is 2 under any prefix;
  * the record's cash encodings are honest for cash = 500
    (scal[2]/[12]/[13]/[19] mutually consistent, tree planes empty so
    the tree price is 20), and the stored uf cash/slot columns carry
    the matching root values -- the X01 recompute recovers the root
    cash from these encodings and dies on inconsistent ones;
  * base-slot legality is the Prefix-honest [T, ok, ok, F, F] with
    ok = True (cash 500, 2 free slots: every base action is legal
    under every candidate prefix at these roots);
  * the generation records the teacher train_temp in BOTH places the
    real products carry it (producer.config.train_temp and
    meta.teacher.config.train_temp), bound by one config hash
    (producer.config_sha256 == producer.teacher_config_hash ==
    meta.teacher.config_sha256) -- the X02 consumer reads it from
    meta and cross-checks; ds_tau2 carries train_temp = 2.0.
"""
import hashlib
import json
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "src")
sys.path.insert(0, SRC)
os.environ.setdefault("BC_DEVICE", "cpu")

import cnet2_feat as F  # noqa: E402

CFG = {"C": 32, "NB": 2, "D": 64, "NH": 4, "L": 1, "E": 16, "G": 8}
R_ROOTS, K_CAND = 6, 4
CASH = 500.0


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


def make_parent(path, seed):
    import torch
    import cnet2_train as T
    from cnet2_model import CNet2
    torch.manual_seed(seed)
    net = CNet2(**CFG)
    T.export(net, path, {"cfg": CFG, "fixture": "k0_distill parent"})
    return _md5_file(path)


def make_root(rng, game, step, seat, n_arms):
    ex = np.ones(F.T, dtype=bool)
    ex[3] = False
    ex[8] = False                      # two worker slots missing
    tok_mask = ex
    legal = np.zeros((F.OWN, 5), dtype=bool)
    for s in range(F.OWN):
        if not tok_mask[s]:
            continue
        if s >= 10:
            legal[s] = [True, True, True, False, False]
        else:
            acts = rng.random(5) < 0.6
            if not acts.any():
                acts[rng.integers(0, 5)] = True
            legal[s] = acts
    arms = np.full((K_CAND, F.OWN), -1, dtype=np.int64)
    for k in range(n_arms):
        a = np.full(F.OWN, -1, dtype=np.int64)
        for s in range(F.OWN):
            if tok_mask[s]:
                opts = np.nonzero(legal[s])[0]
                a[s] = int(opts[rng.integers(0, len(opts))])
        arms[k] = a
    Q = np.full(K_CAND, np.nan)
    Q[:n_arms] = rng.uniform(0.35, 0.65, n_arms)
    pr = rng.uniform(0.2, 1.0, n_arms)
    priors = np.zeros(K_CAND)
    priors[:n_arms] = pr / pr.sum()
    scal = (rng.standard_normal(F.NS) * 0.1).astype(np.float16)
    import math
    scal[2] = np.float16(math.log1p(CASH) / 9.0)
    scal[12] = np.float16(1.0)         # cash >= 30
    scal[13] = np.float16(1.0)         # cash >= price (20, no trees)
    scal[19] = np.float16(min(CASH / 2000.0, 3.0))
    maps = rng.integers(0, 4, (F.P, F.N)).astype(np.uint8)
    maps[16] = 0
    maps[17] = 0                       # no trees -> price 20
    uf = (rng.standard_normal((F.OWN, F.UF)) * 0.1).astype(np.float16)
    uf[:, 6] = np.float16(math.log1p(CASH) / 9.0)
    uf[:, 7] = np.float16(0.2)         # 2 free slots at every slot
    feats = {
        "maps": maps,
        "scal": scal,
        "tok_pos": rng.integers(0, F.N, F.T).astype(np.int16),
        "tok_kind": np.array([1] * 10 + [2] * 2 + [3] * 10 + [4] * 2, dtype=np.int8),
        "tok_mask": tok_mask,
        "tok_cargo": rng.uniform(0, 50, F.T).astype(np.float16),
        "hist_cell": np.where(rng.random((F.T, F.H)) < 0.5, -1,
                              rng.integers(0, F.N, (F.T, F.H))).astype(np.int16),
        "hist_cargo": rng.uniform(0, 50, (F.T, F.H)).astype(np.float16),
        "hist_move": rng.integers(0, 6, (F.T, F.H)).astype(np.int8),
        "uf": uf,
        "mf": (rng.standard_normal((F.OWN, 5, F.MF)) * 0.1).astype(np.float16),
        "legal": legal,
        "carbon": rng.integers(0, 256, F.N).astype(np.uint8),
        "teacher": np.int8(1),
        "gid": np.int32(game), "step": np.int16(step),
    }
    return feats, arms, Q, priors, n_arms


def build_dataset(root_dir, parent_md5, pi_mode, tau=1.0, bad_q=False,
                  nan_pi=False, seed=99):
    sys.path.insert(0, HERE)
    import k0_distill as KD
    rng = np.random.default_rng(seed)
    roots = []
    for r in range(R_ROOTS):
        game, step, seat = r // 3, 200 + r, r % 2
        n_arms = 3 if r == R_ROOTS - 1 else K_CAND
        feats, arms, Q, priors, na = make_root(rng, game, step, seat, n_arms)
        roots.append({"game": game, "step": step, "seat": seat, "n_arms": na,
                      "feats": feats, "arms": arms, "Q": Q, "priors": priors})
    if pi_mode == "ipw":
        roots[1]["feats"] = roots[0]["feats"]
        roots[1]["arms"] = roots[0]["arms"]
        roots[1]["Q"] = roots[0]["Q"]
        roots[1]["priors"] = roots[0]["priors"]
        roots[1]["n_arms"] = roots[0]["n_arms"]
        roots[1]["game"] = roots[0]["game"]
        roots[1]["step"] = roots[0]["step"]
        roots[1]["seat"] = 1
        roots[0]["seat"] = 0
    feat_names = list(roots[0]["feats"].keys())
    feats = {k: np.stack([rt["feats"][k] for rt in roots]) for k in feat_names}
    arms = np.stack([rt["arms"] for rt in roots]).astype(np.int16)
    n_arms = np.array([rt["n_arms"] for rt in roots], dtype=np.int16)
    priors = np.stack([rt["priors"] for rt in roots]).astype(np.float32)
    Q = np.stack([rt["Q"] for rt in roots]).astype(np.float32)
    game = np.array([rt["game"] for rt in roots], dtype=np.int64)
    step = np.array([rt["step"] for rt in roots], dtype=np.int32)
    seat = np.array([rt["seat"] for rt in roots], dtype=np.int8)
    q = np.zeros((R_ROOTS, K_CAND), dtype=np.float32)
    for r, rt in enumerate(roots):
        n = rt["n_arms"]
        # Stored q is the RAW SS13-C construction (untempered), as in
        # the real products; tempering is a consumer-side transform.
        q[r, :n] = KD.construct_target(rt["Q"][:n].astype(np.float64),
                                       rt["priors"][:n].astype(np.float64), 0.05)
    if bad_q:
        q[0, 0], q[0, 1] = q[0, 1], q[0, 0]
    pi = np.ones(R_ROOTS); pi0 = np.ones(R_ROOTS); pi1 = np.ones(R_ROOTS)
    if pi_mode == "ipw":
        for r in range(R_ROOTS):
            if r % 2 == 1:
                pi0[r], pi1[r], pi[r] = 0.5, 0.0, 0.5
    if nan_pi:
        pi[2] = np.nan
    os.makedirs(os.path.join(root_dir, "generations"), exist_ok=True)
    staging = os.path.join(root_dir, f".staging-{os.getpid()}")
    os.makedirs(staging)
    files = {}
    for k in feat_names:
        p = os.path.join(staging, k + ".npy")
        np.save(p, feats[k])
        files[k] = {"file": k + ".npy", "bytes": os.path.getsize(p),
                    "rows": R_ROOTS, "sha256": _sha256_file(p)}
    pol = {"game": game, "step": step, "seat": seat, "arms": arms,
           "n_arms": n_arms, "priors": priors, "q": q,
           "pi0": pi0.astype(np.float32), "pi1": pi1.astype(np.float32),
           "pi": pi.astype(np.float32)}
    val = {"game": game, "step": step, "seat": seat, "Q": Q}
    for name, bundle in (("policy_labels", pol), ("value_labels", val)):
        p = os.path.join(staging, name + ".npz")
        np.savez(p, **bundle)
        files[name] = {"file": name + ".npz", "bytes": os.path.getsize(p),
                       "rows": R_ROOTS, "sha256": _sha256_file(p)}
    games = [{"gi": g, "key": f"fx{g}", "file": f"fx{g}.json.gz", "val": False}
             for g in sorted(set(game.tolist()))]
    teacher_cfg = {"K": 16, "M": 8, "H": 2, "q_temp": 0.05,
                   "train_temp": float(tau), "belief": "sample"}
    cfg_hash = hashlib.sha256(_canonical(teacher_cfg)).hexdigest()
    with open(os.path.join(staging, "meta.json"), "w") as fh:
        json.dump({"games": games, "total": R_ROOTS,
                   "teacher": {"config": teacher_cfg,
                               "config_sha256": cfg_hash,
                               "belief": "sample", "version": "fixture"}}, fh)
    producer = {"annotator_md5": _md5_file(os.path.join(HERE, "k0_distill.py")),
                "teacher_config_hash": cfg_hash,
                "config_sha256": cfg_hash,
                "teacher_weights_md5": parent_md5, "belief": "sample",
                "config": dict(teacher_cfg)}
    core = {"format": "cnet2-k0-annotation/1", "total": R_ROOTS,
            "producer": producer,
            "meta_sha256": _sha256_file(os.path.join(staging, "meta.json")),
            "fields": files}
    gen_id = "gen-" + hashlib.sha256(_canonical(core)).hexdigest()[:16]
    man = dict(core); man["generation"] = gen_id; man["created"] = 0.0
    man["self_sha256"] = hashlib.sha256(_canonical(
        {k: v for k, v in man.items() if k != "self_sha256"})).hexdigest()
    with open(os.path.join(staging, "manifest.json"), "w") as fh:
        json.dump(man, fh)
    gen_dir = os.path.join(root_dir, "generations", gen_id)
    os.replace(staging, gen_dir)
    ptr = {"generation": gen_id, "dir": f"generations/{gen_id}",
           "manifest_sha256": _sha256_file(os.path.join(gen_dir, "manifest.json")),
           "total": R_ROOTS, "created": 0.0}
    with open(os.path.join(root_dir, "current.json"), "w") as fh:
        json.dump(ptr, fh)
    return gen_id


def main(outdir):
    shutil.rmtree(outdir, ignore_errors=True)
    os.makedirs(outdir)
    parent_md5 = make_parent(os.path.join(outdir, "parent.npz"), seed=1234)
    make_parent(os.path.join(outdir, "parent2.npz"), seed=777)
    ids = {}
    ids["ds_a"] = build_dataset(os.path.join(outdir, "ds_a"), parent_md5, "full", tau=1.0)
    ids["ds_b"] = build_dataset(os.path.join(outdir, "ds_b"), parent_md5, "ipw", tau=1.0)
    ids["ds_nanpi"] = build_dataset(os.path.join(outdir, "ds_nanpi"), parent_md5,
                                    "full", nan_pi=True, tau=1.0)
    ids["ds_badq"] = build_dataset(os.path.join(outdir, "ds_badq"), parent_md5,
                                   "full", bad_q=True, tau=1.0)
    ids["ds_tau2"] = build_dataset(os.path.join(outdir, "ds_tau2"), parent_md5,
                                   "full", tau=2.0)
    run_dir = os.path.join(outdir, "run")
    os.makedirs(run_dir)
    for m in ("cnet2_feat.py", "cnet2_model.py", "cnet2_build.py", "cnet2_train.py"):
        shutil.copy(os.path.join(SRC, m), os.path.join(run_dir, m))
    shutil.copy(os.path.join(HERE, "k0_distill.py"), os.path.join(run_dir, "k0_distill.py"))
    print(json.dumps({"parent_md5": parent_md5, "generations": ids, "outdir": outdir}))


if __name__ == "__main__":
    main(sys.argv[1])

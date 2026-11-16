"""Build the X01 reproduction datasets: roots whose feature rows are
honest cnet2_build-style rows (Feat.observe + Prefix driven by the
LOGGED action), with candidate arms that diverge from the logged
action in their action effects (base recruits / destinations /
planter stays). Sealed in the consumed cnet2-k0-annotation/1 format.

Datasets:
  ds_x01   root1: cash 45, logged recruits at base slot 10; candidate B
           does not, and recruits at slot 11 instead (legal under its
           OWN prefix, masked under the logged features -> explosion).
           root2: cash 45, logged does not recruit at 10; candidate
           recruits at 10 (the cash-recovery direction).
           root3: cash 500, worker-destination / planter deviations
           (reserved / plants corruption).
  ds_neg1  root2 variant whose extra candidate recruits at 11 after
           recruiting at 10 with cash 45: illegal under its OWN prefix
           (teacher could never certify it) -- post-fix must die.
  ds_neg2  root1 variant whose stored WORKER legal masks the logged
           arm's worker action: joint logp ~ -1e9 through a path the
           feature recompute does not own -- post-fix the loss
           magnitude die must fire (pre-fix it saves as complete).
Also writes ref_feats.npz: per (root, candidate) the Prefix-driven
candidate-consistent (uf, mf, legal) -- the independent reference.
"""
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "src")
sys.path.insert(0, SRC)
sys.path.insert(0, HERE)
os.environ.setdefault("BC_DEVICE", "cpu")

import cnet2_feat as F  # noqa: E402
from k0_distill_base import construct_target  # noqa: E402

CFG = {"C": 32, "NB": 2, "D": 64, "NH": 4, "L": 1, "E": 16, "G": 8}
TEACHER_CFG = {"H": 2, "K": 16, "M": 8, "behavior_temp": 1.0, "belief": "sample",
               "seed_teacher": 7000, "teacher_style": 2, "train_temp": 1.0}


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


def make_parent(path, seed=1234):
    import torch
    import cnet2_train as T
    from cnet2_model import CNet2
    torch.manual_seed(seed)
    net = CNet2(**CFG)
    T.export(net, path, {"cfg": CFG, "fixture": "x01 repro parent"})
    return _md5_file(path)


def make_obs(step, cash):
    own_workers = {
        "w0": [80, 30.0, "COLLECTOR"],    # (5,5)
        "w1": [93, 20.0, "COLLECTOR"],    # (6,3)
        "w2": [137, 10.0, "COLLECTOR"],   # (9,2)
        "w3": [112, 0.0, "PLANTER"],      # (7,7)
        "w4": [53, 0.0, "PLANTER"],       # (3,8)
    }
    own_bases = {"b0": 64, "b1": 160}     # (4,4), (10,10)
    en_workers = {"e0": [100, 15.0, "COLLECTOR"]}
    en_bases = {"x0": 70, "x1": 154}      # (4,10), (10,4)
    players = [[cash, own_bases, own_workers, {}],
               [500.0, en_bases, en_workers, {}]]
    return {"step": step, "players": players,
            "carbon": np.full(F.N, 40.0), "trees": {}}


def drive(x, aux, choose):
    """One Prefix drive: returns (y, uf, mf, legal) with per-slot
    features as the decoder consumes them (feats(s) before advance)."""
    pf = F.Prefix(x, aux)
    uf = np.zeros((F.OWN, F.UF), np.float32)
    mf = np.zeros((F.OWN, 5, F.MF), np.float32)
    legal = np.zeros((F.OWN, 5), bool)
    y = np.full(F.OWN, -1, np.int64)
    for s in range(F.OWN):
        if not x["tok_mask"][s]:
            continue
        uf[s], mf[s], legal[s] = pf.feats(s)
        a = int(choose(s, legal[s]))
        assert legal[s, a], f"drive chose illegal action {a} at slot {s}"
        y[s] = a
        pf.advance(s, a)
    return y, uf, mf, legal


def feats_for_arm(x, aux, arm):
    """Candidate-consistent features: a fresh Prefix drive forced to
    the arm's actions (no legality assert: features are defined for
    any arm; legality is the consumer's check)."""
    pf = F.Prefix(x, aux)
    uf = np.zeros((F.OWN, F.UF), np.float32)
    mf = np.zeros((F.OWN, 5, F.MF), np.float32)
    legal = np.zeros((F.OWN, 5), bool)
    for s in range(F.OWN):
        if not x["tok_mask"][s]:
            continue
        uf[s], mf[s], legal[s] = pf.feats(s)
        pf.advance(s, int(arm[s]))
    return uf, mf, legal


def base_slots(x):
    return [s for s in range(F.OWN) if x["tok_mask"][s] and s >= 10]


def build_root(step, cash, log_choose, arm_specs):
    """arm_specs: list of dicts overriding the logged action per slot.
    Returns (feat_row, arms [K,OWN], ref list per arm)."""
    f = F.Feat(0)
    x = f.observe(make_obs(step, cash))
    aux = f.aux
    y_log, uf, mf, legal = drive(x, aux, log_choose)
    arms, refs = [], []
    for spec in arm_specs:
        arm = y_log.copy()
        for s, a in spec.items():
            arm[s] = a
        arms.append(arm)
        refs.append(feats_for_arm(x, aux, arm))
    row = {
        "maps": x["maps"], "scal": x["scal"].astype(np.float16),
        "tok_pos": x["tok_pos"].astype(np.int16),
        "tok_kind": x["tok_kind"].astype(np.int8), "tok_mask": x["tok_mask"],
        "tok_cargo": x["tok_cargo"].astype(np.float16),
        "hist_cell": x["hist_cell"].astype(np.int16),
        "hist_cargo": x["hist_cargo"].astype(np.float16),
        "hist_move": x["hist_move"].astype(np.int8),
        "uf": uf.astype(np.float16), "mf": mf.astype(np.float16),
        "legal": legal, "carbon": F.q8(np.full(F.N, 40.0) / 100.0),
        "teacher": np.int8(1),
    }
    return row, np.stack(arms), refs, y_log


def seal_dataset(root_dir, parent_md5, roots, train_temp=1.0):
    """roots: list of dicts(row, arms, Q, priors, game, step, seat)."""
    R = len(roots)
    K = max(int(rt["arms"].shape[0]) for rt in roots)
    feat_names = list(roots[0]["row"].keys())
    feats = {k: np.stack([rt["row"][k] for rt in roots]) for k in feat_names}
    feats["gid"] = np.array([rt["game"] for rt in roots], dtype=np.int32)
    feats["step"] = np.array([rt["step"] for rt in roots], dtype=np.int16)
    arms = np.full((R, K, F.OWN), -1, dtype=np.int16)
    n_arms = np.zeros(R, dtype=np.int16)
    priors = np.zeros((R, K), dtype=np.float32)
    Q = np.full((R, K), np.nan, dtype=np.float32)
    q = np.zeros((R, K), dtype=np.float32)
    for r, rt in enumerate(roots):
        n = int(rt["arms"].shape[0])
        arms[r, :n] = rt["arms"]
        n_arms[r] = n
        priors[r, :n] = rt["priors"]
        Q[r, :n] = rt["Q"]
        q[r, :n] = construct_target(np.asarray(rt["Q"], dtype=np.float64),
                                    np.asarray(rt["priors"], dtype=np.float64), 0.05)
    game = np.array([rt["game"] for rt in roots], dtype=np.int64)
    step = np.array([rt["step"] for rt in roots], dtype=np.int32)
    seat = np.array([rt["seat"] for rt in roots], dtype=np.int8)
    os.makedirs(os.path.join(root_dir, "generations"), exist_ok=True)
    staging = os.path.join(root_dir, f".staging-{os.getpid()}")
    os.makedirs(staging)
    files = {}
    for k in feat_names + ["gid", "step"]:
        p = os.path.join(staging, k + ".npy")
        np.save(p, feats[k])
        files[k] = {"file": k + ".npy", "bytes": os.path.getsize(p),
                    "rows": R, "sha256": _sha256_file(p)}
    pol = {"game": game, "step": step, "seat": seat, "arms": arms,
           "n_arms": n_arms, "priors": priors, "q": q,
           "pi0": np.ones(R, np.float32), "pi1": np.ones(R, np.float32),
           "pi": np.ones(R, np.float32)}
    val = {"game": game, "step": step, "seat": seat, "Q": Q}
    for name, bundle in (("policy_labels", pol), ("value_labels", val)):
        p = os.path.join(staging, name + ".npz")
        np.savez(p, **bundle)
        files[name] = {"file": name + ".npz", "bytes": os.path.getsize(p),
                       "rows": R, "sha256": _sha256_file(p)}
    tcfg = dict(TEACHER_CFG); tcfg["train_temp"] = float(train_temp)
    cfg_sha = hashlib.sha256(_canonical(tcfg)).hexdigest()
    games = [{"gi": g, "key": f"fx{g}", "file": f"fx{g}.json.gz", "val": False}
             for g in sorted(set(game.tolist()))]
    meta = {"games": games, "total": R,
            "teacher": {"belief": "sample", "config": tcfg,
                        "config_sha256": cfg_sha}}
    with open(os.path.join(staging, "meta.json"), "w") as fh:
        json.dump(meta, fh)
    producer = {"teacher_config_hash": cfg_sha,
                "teacher_weights_md5": parent_md5, "belief": "sample",
                "config": tcfg}
    core = {"format": "cnet2-k0-annotation/1", "total": R,
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
           "total": R, "created": 0.0}
    with open(os.path.join(root_dir, "current.json"), "w") as fh:
        json.dump(ptr, fh)
    return gen_id


def stay_except(overrides):
    def choose(s, legal):
        return overrides.get(s, 0)
    return choose


def main(outdir):
    os.makedirs(outdir, exist_ok=True)
    parent_md5 = make_parent(os.path.join(outdir, "parent.npz"))
    refs = {}

    # ---- root 1: explosion geometry (cash 45; bases at slots 10, 11)
    row1, arms1, refs1, y1 = build_root(
        210, 45.0, stay_except({10: 1, 11: 0}),
        [{},                      # arm 0 = logged action itself
         {10: 0, 11: 1},          # arm 1: no recruit at 10, recruit at 11
         {0: 2, 3: 0, 10: 1, 11: 0}])   # arm 2: worker deviations, same bases
    b1 = base_slots_from(row1)
    # ---- root 2: recovery direction (logged: no recruit at 10)
    row2, arms2, refs2, y2 = build_root(
        211, 45.0, stay_except({10: 0, 11: 0}),
        [{}, {10: 1, 11: 0}, {10: 1, 11: 0, 1: 3}])
    # ---- root 3: cash 500, destination/planter deviations
    row3, arms3, refs3, y3 = build_root(
        212, 500.0, stay_except({0: 2, 3: 0, 10: 0, 11: 0}),
        [{}, {0: 0, 3: 2, 1: 2}, {0: 4, 4: 0, 10: 2, 11: 1}])
    roots = [
        {"row": row1, "arms": arms1, "Q": [0.55, 0.62, 0.48],
         "priors": [0.5, 0.3, 0.2], "game": 0, "step": 210, "seat": 0},
        {"row": row2, "arms": arms2, "Q": [0.50, 0.58, 0.44],
         "priors": [0.4, 0.35, 0.25], "game": 0, "step": 211, "seat": 0},
        {"row": row3, "arms": arms3, "Q": [0.60, 0.52, 0.47],
         "priors": [0.45, 0.30, 0.25], "game": 1, "step": 212, "seat": 0},
    ]
    gid = seal_dataset(os.path.join(outdir, "ds_x01"), parent_md5, roots)
    for i, rfs in enumerate((refs1, refs2, refs3)):
        for k, (u, m, l) in enumerate(rfs):
            refs[f"r{i}_k{k}_uf"] = u; refs[f"r{i}_k{k}_mf"] = m
            refs[f"r{i}_k{k}_legal"] = l
    refs["y_log"] = np.stack([y1, y2, y3])
    refs["arms"] = np.stack([np.pad(a, ((0, 3 - len(a)), (0, 0)),
                                    constant_values=-1)
                             for a in (arms1, arms2, arms3)])
    np.savez(os.path.join(outdir, "ref_feats.npz"), **refs)

    # ---- ds_neg1: root2 + an arm illegal under its own prefix
    row2b, arms2b, _, _ = build_root(
        211, 45.0, stay_except({10: 0, 11: 0}),
        [{}, {10: 1, 11: 1}])     # recruit at 10 (cash->15), recruit at 11
    seal_dataset(os.path.join(outdir, "ds_neg1"), parent_md5,
                 [{"row": row2b, "arms": arms2b, "Q": [0.50, 0.58],
                   "priors": [0.5, 0.5], "game": 0, "step": 211, "seat": 0}])

    # ---- ds_neg2: root1 with a poisoned WORKER legal row
    row1p = dict(row1)
    lg = row1["legal"].copy()
    s0 = int(np.nonzero(row1["tok_mask"][:10])[0][0])
    lg[s0, int(y1[s0])] = False       # mask the logged arm's worker action
    row1p["legal"] = lg
    seal_dataset(os.path.join(outdir, "ds_neg2"), parent_md5,
                 [{"row": row1p, "arms": arms1[:1], "Q": [0.55],
                   "priors": [1.0], "game": 0, "step": 210, "seat": 0}])
    print(json.dumps({"parent_md5": parent_md5, "ds_x01_gen": gid,
                      "base_slots_root1": b1,
                      "y_log": [y1.tolist(), y2.tolist(), y3.tolist()]}))


def base_slots_from(row):
    tm = row["tok_mask"]
    return [s for s in range(F.OWN) if tm[s] and s >= 10]


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "repro"))

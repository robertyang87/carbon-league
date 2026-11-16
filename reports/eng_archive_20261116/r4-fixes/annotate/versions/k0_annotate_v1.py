"""k0_annotate -- K0 round-2 sparse annotator (design draft
reports/K0_ROUND2_DESIGN_DRAFT_20261114.md S7/S10 item 1, ruling condition I;
record schema frozen at reports/STAGE1_TEACHER_GATES_20261113.md S13-C).

Per root one record: {game, step, seat, arms, Q, q, pi0, pi1,
pi = pi0 + (1 - pi0) * pi1, teacher config hash, weights md5, root belief
condition}. Value labels (Q, Q_std) and policy labels (arms, priors, q)
are stored as separate field files inside one sealed generation. The two
identity fields that are constant across a run (teacher config hash,
weights md5) live in meta.json's "teacher" block and are bound by the
manifest seal; the per-root-varying part of the belief condition (belief
mode code, unknown-cell count) is a per-root field. meta.json's
"record_schema" states exactly how a per-root record is assembled.

Feed and roots: the caller names a replay dir and an explicit game-id
list. Roots are every (step, seat), seat in {0, 1}, with step inside the
registered window [window_lo, window_hi) and below the game's
first-terminal bound (stage1_teacher_gate.first_terminal_step, the same
bound the gate sampler uses). The window is a registration parameter:
the production registration is step 200-299 (gate 0, ruling condition I);
anything outside 0 <= lo < hi <= 299 is refused, and a root outside the
registered window in feed or product dies by name.

Sampling (two passes, so the frozen pi identity holds exactly):
  pass A: each game is drawn with probability pi0 -> all its roots taken;
  pass B: for games not drawn in pass A, each root is drawn independently
          with probability pi1.
A root's marginal inclusion probability is then exactly
pi0 + (1 - pi0) * pi1; the recorded pi0/pi1 are the run's two rates and
pi is recomputed from them per root. Draws come from one dedicated
stream np.random.default_rng(seed) consumed in a fixed documented order
(feed order for pass A; feed order, then (step, seat), for pass B), and
the selected root set's sha256 is recorded (cf. k0_train's B2 subset
accounting). Records without pi are unqualified (S13-C): the pre-seal
self-check and the consumer-side verification both refuse them by name.

Evaluation reuses the gate's code path verbatim:
stage1_teacher_gate.extract_scene + stage0_teacher.evaluate_root with
seed_base = seed_teacher + step, identity-checked by
gate._expect_teacher. This module contains no q formula of its own:
q (proportional to P_A**0.5 * exp((Q - maxQ) / 0.05)) lives in
stage0_teacher.evaluate_root, and priors P_A are recorded alongside so
any tempering (q ** train_temp, R23) and the round-2 C arm's flat-Q
target q0 proportional to P_A**0.5 are derivable from the product.

Sharding and resume (S13-C item 3): one shard per game under
OUT/shards/<game>.npz, with its sidecar meta (per-array sha256, run
identity, per-game selection hash) embedded in the npz as a "_meta"
JSON entry so a shard lands atomically in a single os.replace;
progress in OUT/progress.json. The run identity binds, beside config /
weights / window / sampling / the game-id set, the CONTENT of the
feed: the sha256 of every game's replay file bytes (the plan reads
each replay exactly once, hashes the bytes it enumerates roots from,
and a resume recomputes them before any shard is trusted -- replacing
a replay under the same id dies by name, naming the game), and the
producer code md5 set (code_md5s(), the same set the publish manifest
carries -- resuming under different code dies instead of mixing code
versions into one generation). A rerun under the same run identity
verifies every finished shard (embedded-meta self-consistency,
per-array hashes, plan root keys, config hash, weights md5, selection
hash, and the file sha256 recorded in progress) and skips it; an
orphan shard that fully verifies is adopted, one that does not verify
dies by name -- a shard is never silently recomputed. A rerun under a
different identity dies by name. Progress/shards recorded before the
content+code binding existed are UNBOUND: they die by default, and
are adopted only under the explicit --accept-unbound-identity flag,
which records the adoption in progress.json (the W01 precedent:
absence of a record may be overridden explicitly, contradiction of a
record never may).

Publish: OUT/generations/gen-<id>/ assembled from the shards in feed
order, manifest schema adapted from exit_merge's cnet2-exit-generation/1
(format "k0-annotation-generation/1"): per-field {file, bytes, rows,
sha256}, producer identity (annotator / teacher / gate code md5s plus
the teacher config hash), meta_sha256 seal, manifest self_sha256.
The generation id is content-addressed; an already-published identical
generation is verified and REUSED exactly as-is, never deleted (the
exit chain's V02 rule); current.json flips last with a single
os.replace. Consumption is verify-or-refuse through the pointer only
(mirroring cnet2_exit_train.resolve_exit_arg): a generation directory
passed directly is refused.

Usage (from code/):
  python3 k0_annotate.py --weights HEADS.npz --replays DIR \
      --games games.json --out OUT_DIR [--window-lo 200] [--window-hi 299] \
      [--belief sample] [--teacher-style 2] [--K 16] [--M 8] [--H 2] \
      [--seed-teacher 7000] [--seed 12345] [--pi0 1.0] [--pi1 1.0] \
      [--isolation-file ids.txt]... [--workers 1]
  python3 k0_annotate.py --out OUT_DIR --verify
"""
import argparse
import hashlib
import io
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor

# Single-threaded BLAS per process (same posture as stage1_teacher_gate):
# teacher calls are many small matmuls; parallelism belongs to --workers.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import cnet2_feat as F  # noqa: E402
import stage0_teacher as T  # noqa: E402
import stage1_teacher_gate as G  # noqa: E402

FORMAT = "k0-annotation-generation/1"
OWN = int(F.OWN)
BELIEF_CODE = {name: i for i, name in enumerate(T.BELIEF_MODES)}

# Field layout of a generation (and of every shard). Value labels and
# policy labels are separate files (S13-C).
FIELDS = {
    "gid": np.int32, "step": np.int32, "seat": np.int8,
    "arms": np.int16, "n_arms": np.int16,
    "priors": np.float64, "q": np.float64,
    "Q": np.float64, "Q_std": np.float64,
    "pi0": np.float64, "pi1": np.float64, "pi": np.float64,
    "belief": np.int8, "n_unknown": np.int32,
}
ARM_PAD = -2  # labels live in [-1, 4]; -2 marks "no such arm" padding


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


def _array_sha(arr):
    a = np.ascontiguousarray(arr)
    return _sha256_bytes(a.dtype.str.encode() + json.dumps(list(a.shape)).encode()
                         + a.tobytes())


def _atomic_write_json(obj, path):
    tmp = f"{path}.tmp-{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1, sort_keys=True)
    os.replace(tmp, path)


def _die(msg):
    raise SystemExit(f"k0_annotate: {msg}")


# ----------------------------- plan ---------------------------------------


def _validate_ids(ids, what):
    """Mirror of the gate's S06/S07 rule: bare replay stems, dup-free."""
    for g in ids:
        if not g or any(c in g for c in "./\\ ") or g != g.strip():
            _die(f"{what}: id {g!r} is not a bare game stem")
    dup = sorted({g for g in ids if ids.count(g) > 1})
    if dup:
        _die(f"{what}: duplicate game ids: {dup[:5]}")
    return list(ids)


def teacher_config(a):
    return {"K": a.K, "M": a.M, "H": a.H,
            "teacher_style": a.teacher_style, "belief": a.belief,
            "train_temp": a.train_temp, "behavior_temp": a.behavior_temp,
            "seed_teacher": a.seed_teacher,
            "guard_cfg": T.TEACHER_GUARD_CFG,
            "window": [a.window_lo, a.window_hi],
            "scene_contract": "prefix-feat-guard-snapshot-v1"}


def code_md5s():
    fns = ("stage0_teacher.py", "cnet2_np.py", "cnet2_search.py",
           "cnet2_feat.py", "fastsim.py", "stage1_teacher_gate.py",
           "k0_annotate.py")
    return {fn: _md5_file(os.path.join(HERE, fn)) for fn in fns}


def _enumerate_roots_bytes(raw, lo, hi):
    """Roots of one game from its exact replay bytes (the bytes the
    plan hashed): the identity's replay sha256 therefore covers the
    very content the roots were enumerated from, never a later
    by-path re-read."""
    import gzip
    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as fh:
        d = json.load(fh)
    ft, _cause = G.first_terminal_step(d["steps"])
    n_steps = min(len(d["steps"]) - 1, ft)
    hi_c = min(hi, n_steps)
    roots = []
    for step in range(lo, hi_c):
        for seat in (0, 1):
            roots.append((step, seat))
    return roots, ft


def enumerate_roots(path, lo, hi):
    """All (step, seat) roots of one game inside the window, bounded by
    the game's first-terminal frame exactly as the gate sampler bounds
    its scenes (step t reads frame t+1, so t < min(len-1, ft))."""
    with open(path, "rb") as fh:
        raw = fh.read()
    return _enumerate_roots_bytes(raw, lo, hi)


def build_plan(a):
    if not (0 <= a.window_lo < a.window_hi <= 299):
        _die(f"registered window must satisfy 0 <= window-lo < window-hi <= 299, "
             f"got [{a.window_lo}, {a.window_hi})")
    if not (0.0 <= a.pi0 <= 1.0 and 0.0 <= a.pi1 <= 1.0):
        _die(f"pi0/pi1 must lie in [0, 1], got pi0={a.pi0} pi1={a.pi1}")
    spec = json.load(open(a.games))
    games = spec["games"] if isinstance(spec, dict) else spec
    games = _validate_ids([str(g) for g in games], "games file")
    if not games:
        _die("games file is empty")
    iso = {}
    for f in a.isolation_file or []:
        raw = open(f).read()
        try:
            obj = json.loads(raw)
            ids = obj["games"] if isinstance(obj, dict) else obj
            ids = [str(x) for x in ids]
        except Exception:
            ids = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        for g in _validate_ids(ids, f"isolation file {f}"):
            iso.setdefault(g, f)
    clash = sorted(set(games) & set(iso))
    if clash:
        _die(f"isolation violation: feed games {clash[:5]} appear in isolation "
             f"lists (e.g. {iso[clash[0]]}); refusing to annotate")
    paths = G.list_replays(a.replays, games)
    cfg = teacher_config(a)
    plan = {"games": games, "paths": paths, "config": cfg,
            "config_sha256": _sha256_bytes(_canonical(cfg)),
            "weights": a.weights, "weights_md5": _md5_file(a.weights),
            "pi0": float(a.pi0), "pi1": float(a.pi1), "seed": int(a.seed),
            "code_md5s": code_md5s(),
            "per_game": []}
    rng = np.random.default_rng(int(a.seed))
    pass_a = {}
    for g in games:
        pass_a[g] = bool(rng.random() < plan["pi0"])
    n_sel = 0
    for gi, (g, p) in enumerate(zip(games, paths)):
        # One read of the replay: the recorded content sha256 is taken
        # over the exact bytes the roots are enumerated from, before
        # anything else consumes the game (X03 / P3(a) consumer half).
        with open(p, "rb") as fh:
            raw = fh.read()
        replay_sha = _sha256_bytes(raw)
        roots, ft = _enumerate_roots_bytes(raw, a.window_lo, a.window_hi)
        if pass_a[g]:
            sel, pas = list(roots), "A"
        else:
            sel, pas = [], "B"
            for rt in roots:
                if rng.random() < plan["pi1"]:
                    sel.append(rt)
        sel_keys = [[g, s, seat] for (s, seat) in sel]
        plan["per_game"].append({
            "game": g, "gid": gi, "path": p, "ft": ft, "pass": pas,
            "roots": sel, "replay_sha256": replay_sha,
            "selected_sha256": _sha256_bytes(_canonical(sel_keys))})
        n_sel += len(sel)
    plan["n_roots"] = n_sel
    plan["selected_sha256"] = _sha256_bytes(_canonical(
        [k for pg in plan["per_game"]
         for k in [[pg["game"], s, seat] for (s, seat) in pg["roots"]]]))
    plan["identity"] = {"config_sha256": plan["config_sha256"],
                        "weights_md5": plan["weights_md5"],
                        "window": [a.window_lo, a.window_hi],
                        "pi0": plan["pi0"], "pi1": plan["pi1"],
                        "seed": plan["seed"],
                        "games_sha256": _sha256_bytes(_canonical(games)),
                        "selected_sha256": plan["selected_sha256"],
                        "replay_sha256": {pg["game"]: pg["replay_sha256"]
                                          for pg in plan["per_game"]},
                        "code_md5s": plan["code_md5s"]}
    if n_sel == 0:
        _die("sampling selected zero roots (check pi0/pi1 and the window); "
             "refusing to produce an empty generation")
    return plan


# --------------------------- evaluation -----------------------------------


_W = {}


def _init_worker(weights_path):
    w = dict(np.load(weights_path))
    from cnet2_search import CachePolicy
    _W["pol"] = CachePolicy(w)


def _eval_root(spec, cfg):
    """One root through the gate's exact code path (cf. gate._q_scene,
    which discards the labels; the annotator keeps them)."""
    scene = G.extract_scene(spec)
    pol = _W["pol"]
    r = T.evaluate_root(G._fresh_sim(scene), scene["me"], pol,
                        scene["x_me"], scene["aux_me"],
                        scene["x_opp"], scene["aux_opp"],
                        G._ns(scene["fm"]), G._ns(scene["fo"]),
                        teacher=cfg["teacher_style"],
                        K=cfg["K"], M=cfg["M"], H=cfg["H"],
                        seed_base=cfg["seed_teacher"] + spec["step"],
                        feats=(scene["fm"], scene["fo"]),
                        guard_snaps=G._teacher_guard_snaps(scene, cfg["belief"]),
                        belief=cfg["belief"])
    G._expect_teacher(r, cfg["belief"])
    return r


def _root_row(r, K):
    n = len(r["arms"])
    arms = np.full((K, OWN), ARM_PAD, np.int16)
    priors = np.zeros(K, np.float64)
    q = np.zeros(K, np.float64)
    Q = np.full(K, np.nan, np.float64)
    Qs = np.full(K, np.nan, np.float64)
    for i, arm in enumerate(r["arms"]):
        arms[i] = np.asarray(arm, dtype=np.int16)
    priors[:n] = np.asarray(r["priors"], dtype=np.float64)
    q[:n] = np.asarray(r["q"], dtype=np.float64)
    Q[:n] = np.asarray(r["Q_mean"], dtype=np.float64)
    Qs[:n] = np.asarray(r["Q_std"], dtype=np.float64)
    return {"arms": arms, "n_arms": n, "priors": priors, "q": q,
            "Q": Q, "Q_std": Qs,
            "belief": BELIEF_CODE[r["belief"]],
            "n_unknown": (-1 if r.get("n_unknown_cells") is None
                          else int(r["n_unknown_cells"]))}


def _annotate_game(task):
    pg, cfg, weights_path, shard_dir, identity = task
    if "pol" not in _W:
        _init_worker(weights_path)
    g = pg["game"]
    K = cfg["K"]
    rows = {k: [] for k in FIELDS}
    pi_val = identity["pi0"] + (1.0 - identity["pi0"]) * identity["pi1"]
    for (step, seat) in pg["roots"]:
        spec = {"game": g, "path": pg["path"], "step": step, "me": seat,
                "ft": pg["ft"]}
        try:
            r = _eval_root(spec, cfg)
        except ValueError as e:
            _die(f"root ({g}, step {step}, seat {seat}) refused by the "
                 f"teacher: {e}")
        row = _root_row(r, K)
        rows["gid"].append(pg["gid"]); rows["step"].append(step)
        rows["seat"].append(seat)
        for k in ("arms", "n_arms", "priors", "q", "Q", "Q_std",
                  "belief", "n_unknown"):
            rows[k].append(row[k])
        rows["pi0"].append(identity["pi0"]); rows["pi1"].append(identity["pi1"])
        rows["pi"].append(pi_val)
    arrs = {}
    for k, dt in FIELDS.items():
        v = rows[k]
        if k == "arms":
            arrs[k] = (np.stack(v).astype(dt) if v else
                       np.zeros((0, K, OWN), dt))
        elif k in ("priors", "q", "Q", "Q_std"):
            arrs[k] = (np.stack(v).astype(dt) if v else
                       np.zeros((0, K), dt))
        else:
            arrs[k] = np.asarray(v, dtype=dt)
    meta = {"game": g, "gid": pg["gid"], "pass": pg["pass"],
            "n_roots": len(pg["roots"]),
            "selected_sha256": pg["selected_sha256"],
            "identity": identity,
            "arrays": {k: _array_sha(arrs[k]) for k in FIELDS},
            "shard_file": g + ".npz"}
    npz_path = os.path.join(shard_dir, g + ".npz")
    tmp = npz_path + f".tmp-{os.getpid()}"
    with open(tmp, "wb") as fh:
        np.savez(fh, _meta=np.asarray(json.dumps(meta, sort_keys=True)), **arrs)
    os.replace(tmp, npz_path)
    return meta


# ------------------------- shard verification ------------------------------

# Run-identity keys recorded before the X03 content+code binding
# existed, and the binding keys themselves. A recorded identity that
# carries all legacy keys with equal values but lacks the binding keys
# is UNBOUND (adoptable only under the explicit migration flag); one
# whose replay content sha256 CONTRADICTS the current plan dies by
# name regardless of any flag.
_IDENTITY_LEGACY_KEYS = ("config_sha256", "weights_md5", "window", "pi0",
                         "pi1", "seed", "games_sha256", "selected_sha256")
_IDENTITY_BINDING_KEYS = ("replay_sha256", "code_md5s")


def _identity_replay_changes(old_id, new_id):
    """Games whose recorded replay content sha256 contradicts the
    current plan's recomputation (present in both maps, values differ)."""
    old_map = (old_id or {}).get("replay_sha256") or {}
    new_map = (new_id or {}).get("replay_sha256") or {}
    return {g: (old_map[g], new_map[g]) for g in old_map
            if g in new_map and old_map[g] != new_map[g]}


def _identity_unbound_missing(old_id, new_id):
    """If old_id is a pre-binding identity that agrees with new_id on
    every legacy key, return the binding keys it lacks; else None."""
    if not old_id:
        return None
    if any(k not in old_id for k in _IDENTITY_LEGACY_KEYS):
        return None
    if any(old_id.get(k) != new_id.get(k) for k in _IDENTITY_LEGACY_KEYS):
        return None
    missing = [k for k in _IDENTITY_BINDING_KEYS if k not in old_id]
    return missing or None


def _verify_shard(shard_dir, pg, identity, recorded_sha=None,
                  accept_unbound=False):
    """Full verification of one finished shard against the plan and the
    run identity. Returns the shard meta (with shard_sha256 filled from
    the file bytes), or None if no shard exists; dies by name on any
    mismatch -- a finished shard is never silently recomputed."""
    g = pg["game"]
    npz_path = os.path.join(shard_dir, g + ".npz")
    if not os.path.exists(npz_path):
        return None
    file_sha = _sha256_file(npz_path)
    if recorded_sha is not None and recorded_sha != file_sha:
        _die(f"shard {g}: file sha256 {file_sha} does not match the "
             f"progress record {recorded_sha}; refusing to trust or "
             f"silently recompute it")
    with np.load(npz_path) as D:
        if "_meta" not in D.files:
            _die(f"shard {g}: no embedded sidecar meta; not a finished "
                 f"k0_annotate shard")
        meta = json.loads(str(D["_meta"].item()))
        missing = [k for k in FIELDS if k not in D.files]
        if missing:
            _die(f"shard {g}: missing fields {missing}")
        for k in FIELDS:
            if _array_sha(D[k]) != meta["arrays"].get(k):
                _die(f"shard {g}: array {k!r} hash mismatch with its "
                     f"embedded sidecar")
        if int(D["step"].shape[0]) != meta.get("n_roots"):
            _die(f"shard {g}: root count mismatch with its sidecar")
        keys = [(int(s), int(seat)) for s, seat in
                zip(D["step"].tolist(), D["seat"].tolist())]
        if keys != [tuple(r) for r in pg["roots"]]:
            _die(f"shard {g}: recorded roots do not match the plan's "
                 f"selected roots for this game")
        if not (np.asarray(D["gid"]) == pg["gid"]).all():
            _die(f"shard {g}: gid column does not match the plan")
    if meta.get("identity") != identity:
        rec_id = meta.get("identity") or {}
        changed = _identity_replay_changes(rec_id, identity)
        if g in changed:
            _die(f"shard {g}: replay content for this game changed "
                 f"since the shard was produced (recorded replay "
                 f"sha256 {changed[g][0]}, current {changed[g][1]}); "
                 f"refusing to mix content generations -- remove the "
                 f"output dir to start over")
        missing = _identity_unbound_missing(rec_id, identity)
        if missing is not None:
            if not accept_unbound:
                _die(f"shard {g}: recorded run identity predates the "
                     f"content/code binding (missing {missing}); the "
                     f"shard is UNBOUND -- rerun with "
                     f"--accept-unbound-identity to adopt it "
                     f"explicitly, or remove the output dir to start "
                     f"over")
            meta["identity_migration"] = {
                "missing": missing, "flag": "--accept-unbound-identity"}
        else:
            _die(f"shard {g}: run identity mismatch (config/weights/"
                 f"window/sampling/feed/code differ from the recorded "
                 f"run); refusing to mix identities -- remove the "
                 f"output dir to start over")
    if meta.get("selected_sha256") != pg["selected_sha256"]:
        _die(f"shard {g}: selection hash mismatch with the current plan")
    meta["shard_sha256"] = file_sha
    return meta


# ------------------------- record validation -------------------------------


def validate_records(arrs, window, K, where):
    """The S13-C record gates, applied pre-seal and on consumer verify:
    pi present, finite, in (0, 1] and equal to pi0 + (1-pi0)*pi1; every
    step inside the registered window; arm counts sane; q normalised
    over the recorded arms with zero mass on padding; Q finite on arms."""
    lo, hi = window
    n = int(arrs["step"].shape[0])
    for k in ("pi0", "pi1", "pi"):
        if k not in arrs:
            _die(f"{where}: record field {k!r} missing -- records without "
                 f"pi are unqualified (S13-C)")
    pi0 = np.asarray(arrs["pi0"], dtype=np.float64)
    pi1 = np.asarray(arrs["pi1"], dtype=np.float64)
    pi = np.asarray(arrs["pi"], dtype=np.float64)
    # pi0 / pi1 are the two pass rates: either may be 0 (that pass off)
    # but both are rates in [0, 1]; the recorded pi itself must be a
    # usable inclusion probability in (0, 1] (IPW divides by it).
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


# ------------------------------ publish ------------------------------------


def publish(out_dir, plan, shard_metas):
    shard_dir = os.path.join(out_dir, "shards")
    order = sorted(shard_metas, key=lambda m: m["gid"])
    parts = {k: [] for k in FIELDS}
    games_meta = []
    for m in order:
        with np.load(os.path.join(shard_dir, m["shard_file"])) as D:
            for k in FIELDS:
                parts[k].append(np.asarray(D[k]))
        games_meta.append({"game": m["game"], "gid": m["gid"],
                           "pass": m["pass"], "n_roots": m["n_roots"],
                           "selected_sha256": m["selected_sha256"],
                           "shard_sha256": m["shard_sha256"]})
    arrs = {k: (np.concatenate(parts[k]) if parts[k] else
                np.zeros((0,), FIELDS[k])) for k in FIELDS}
    total = int(arrs["step"].shape[0])
    validate_records(arrs, plan["config"]["window"], plan["config"]["K"],
                     "pre-seal self-check")
    os.makedirs(os.path.join(out_dir, "generations"), exist_ok=True)
    staging = os.path.join(out_dir, f".staging-annotate-{os.getpid()}")

    shutil.rmtree(staging, ignore_errors=True)
    os.makedirs(staging)
    try:
        for k in FIELDS:
            np.save(os.path.join(staging, k + ".npy"), arrs[k])
        meta = {
            "format": FORMAT, "total": total,
            "games": games_meta,
            "teacher": {"version": (f"v{T.TEACHER_VERSION}"
                                    if plan["config"]["belief"] != "oracle"
                                    else "v1+c1"),
                        "belief": plan["config"]["belief"],
                        "weights": plan["weights"],
                        "weights_md5": plan["weights_md5"],
                        "config": plan["config"],
                        "config_sha256": plan["config_sha256"]},
            "sampling": {"pi0": plan["pi0"], "pi1": plan["pi1"],
                         "seed": plan["seed"],
                         "selected_sha256": plan["selected_sha256"],
                         "rule": "pass A: game drawn w.p. pi0 (all roots); "
                                 "pass B: each root of an undrawn game drawn "
                                 "w.p. pi1; pi = pi0 + (1-pi0)*pi1"},
            "record_schema": {
                "per_root_fields": ["game (via gid -> games)", "step", "seat",
                                    "arms", "Q", "q", "pi0", "pi1", "pi",
                                    "teacher config hash (teacher.config_sha256, "
                                    "generation-level, seal-bound)",
                                    "weights md5 (teacher.weights_md5, "
                                    "generation-level, seal-bound)",
                                    "root belief condition (belief code + "
                                    "n_unknown per root)"],
                "value_label_files": ["Q.npy", "Q_std.npy"],
                "policy_label_files": ["arms.npy", "priors.npy", "q.npy"],
                "aux_fields": {"priors": "P_A, the candidate prior q is built "
                                         "from (C arm q0 prop. to P_A**0.5)",
                               "Q_std": "per-arm scenario std of Q",
                               "n_arms": "recorded arm count; arms/q/Q are "
                                         "padded beyond it (arms pad -2, "
                                         "q/priors pad 0, Q pad NaN)"}},
        }
        with open(os.path.join(staging, "meta.json"), "w") as fh:
            json.dump(meta, fh, indent=1, sort_keys=True)
        # The producer identity is the one frozen into the plan (and
        # the run identity) before work started -- publish never reads
        # a fresh code hash (X03: identity is frozen at plan time;
        # reading it here would let a mid-run code change publish
        # under a different identity than the shards were made under).
        codes = plan.get("code_md5s")
        if codes is None:
            _die("publish: plan carries no frozen code_md5s; refusing "
                 "to read code identity at publish time")
        producer = {"k0_annotate_md5": codes["k0_annotate.py"],
                    "stage0_teacher_md5": codes["stage0_teacher.py"],
                    "stage1_teacher_gate_md5": codes["stage1_teacher_gate.py"],
                    "code_md5": codes,
                    "config_sha256": plan["config_sha256"],
                    "config": plan["config"],
                    "weights_md5": plan["weights_md5"]}
        finfo = {}
        for k in FIELDS:
            p = os.path.join(staging, k + ".npy")
            arr = np.load(p, mmap_mode="r")
            finfo[k] = {"file": k + ".npy", "bytes": os.path.getsize(p),
                        "rows": int(arr.shape[0]), "sha256": _sha256_file(p)}
            del arr
        core = {"format": FORMAT, "total": total, "producer": producer,
                "meta_sha256": _sha256_file(os.path.join(staging, "meta.json")),
                "fields": finfo}
        gen_id = "gen-" + _sha256_bytes(_canonical(core))[:16]
        man = dict(core)
        man["generation"] = gen_id
        man["created"] = time.time()
        # Self-seal over the manifest as it stands (core + generation +
        # created, no seal field yet) -- exactly exit_merge's convention,
        # and what verify_generation recomputes.
        man["self_sha256"] = _sha256_bytes(_canonical(man))
        with open(os.path.join(staging, "manifest.json"), "w") as fh:
            json.dump(man, fh, indent=1, sort_keys=True)
        gen_dir = os.path.join(out_dir, "generations", gen_id)
        # Publish rule (exit chain V02): a landed generation is IMMUTABLE.
        # If gen_id is already published, verify the published copy
        # against the core built above and REUSE it exactly as-is;
        # a published directory that does not verify dies by name and is
        # never silently replaced. Only a genuinely new gen_id is
        # renamed into place, and the pointer flips last.
        reused = False
        if os.path.isdir(gen_dir):
            man_p = os.path.join(gen_dir, "manifest.json")
            if not os.path.exists(man_p):
                if os.listdir(gen_dir):
                    _die(f"publish: generation directory {gen_dir!r} exists "
                         f"without manifest.json and is not empty; refusing "
                         f"to overwrite it")
                os.replace(staging, gen_dir)  # empty leftover dir only
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
                        _die(f"publish: published generation {gen_id} field "
                             f"{k!r} in {gen_dir!r} does not match its "
                             f"manifest; refusing to overwrite it")
                reused = True
        elif os.path.exists(gen_dir):
            _die(f"publish: {gen_dir!r} exists and is not a directory; "
                 f"refusing to overwrite it")
        else:
            os.replace(staging, gen_dir)
        ptr = {"generation": gen_id, "dir": f"generations/{gen_id}",
               "manifest_sha256": _sha256_file(
                   os.path.join(gen_dir, "manifest.json")),
               "total": total, "created": time.time()}
        ptr_tmp = os.path.join(out_dir, f".current-{os.getpid()}.tmp")
        with open(ptr_tmp, "w") as fh:
            json.dump(ptr, fh)
        os.replace(ptr_tmp, os.path.join(out_dir, "current.json"))
    finally:

        shutil.rmtree(staging, ignore_errors=True)
    return {"generation": gen_id, "total": total, "reused": reused,
            "games": len(games_meta)}


# ------------------------------ verify -------------------------------------


def verify_generation(gen_dir, ptr):
    """Verify-or-refuse, mirroring cnet2_exit_train._verify_exit_generation
    plus the S13-C record gates. Returns a summary dict."""
    man_path = os.path.join(gen_dir, "manifest.json")
    if not os.path.exists(man_path):
        _die(f"sealed annotation generation has no manifest.json: "
             f"{gen_dir!r}; refusing")
    man_raw = open(man_path, "rb").read()
    want = ptr.get("manifest_sha256")
    if want is not None and _sha256_bytes(man_raw) != want:
        _die(f"annotation manifest hash mismatch in {gen_dir!r}: "
             f"manifest.json bytes do not match the pointer")
    man = json.loads(man_raw)
    if man.get("generation") != ptr.get("generation"):
        _die(f"annotation pointer/manifest generation mismatch: pointer "
             f"{ptr.get('generation')!r} vs manifest {man.get('generation')!r}")
    core = {k: v for k, v in man.items() if k != "self_sha256"}
    if man.get("self_sha256") != _sha256_bytes(_canonical(core)):
        _die(f"annotation manifest self-seal mismatch in {gen_dir!r}")
    meta_p = os.path.join(gen_dir, "meta.json")
    if not os.path.exists(meta_p) or _sha256_file(meta_p) != man.get("meta_sha256"):
        _die(f"annotation generation meta seal mismatch in {gen_dir!r}")
    meta = json.load(open(meta_p))
    prod = man.get("producer", {})
    if meta["teacher"]["config_sha256"] != prod.get("config_sha256") \
            or meta["teacher"]["weights_md5"] != prod.get("weights_md5"):
        _die(f"annotation generation {gen_dir!r}: meta teacher identity "
             f"does not match the manifest producer identity")
    arrs = {}
    for k in sorted(man.get("fields", {})):
        info = man["fields"][k]
        p = os.path.join(gen_dir, info["file"])
        if not os.path.exists(p):
            _die(f"annotation generation file missing: field {k!r} ({p!r})")
        if os.path.getsize(p) != info["bytes"]:
            _die(f"annotation generation file size mismatch: field {k!r}")
        if _sha256_file(p) != info["sha256"]:
            _die(f"annotation generation file hash mismatch: field {k!r}")
        arr = np.load(p)
        if int(arr.shape[0]) != info["rows"]:
            _die(f"annotation generation file row mismatch: field {k!r}")
        arrs[k] = arr
    missing = [k for k in FIELDS if k not in arrs]
    if missing:
        _die(f"annotation generation is missing fields {missing}")
    validate_records(arrs, prod["config"]["window"], prod["config"]["K"],
                     f"verify {gen_dir!r}")
    return {"generation": man.get("generation"), "dir": gen_dir,
            "format": man.get("format"),
            "files_verified": len(man.get("fields", {})),
            "total": man.get("total"),
            "weights_md5": prod.get("weights_md5"),
            "config_sha256": prod.get("config_sha256")}


def resolve_out_arg(arg):
    """Pointer-only consumption, mirroring cnet2_exit_train.resolve_exit_arg:
    a sealed generation directory passed directly is refused."""
    if os.path.isfile(arg):
        ptr = json.load(open(arg))
        root = os.path.dirname(os.path.abspath(arg))
        return verify_generation(os.path.join(root, ptr["dir"]), ptr)
    if os.path.exists(os.path.join(arg, "current.json")):
        ptr = json.load(open(os.path.join(arg, "current.json")))
        return verify_generation(os.path.join(arg, ptr["dir"]), ptr)
    if os.path.exists(os.path.join(arg, "manifest.json")):
        _die(f"sealed annotation generation at {arg!r} must be consumed via "
             f"its dataset root / current pointer, not by direct path")
    _die(f"no annotation generation pointer at {arg!r}")


def load_annotated(arg):
    """Verify-or-refuse load for consumers (the round-2 distillation
    entry): returns (meta, {field: array})."""
    summary = resolve_out_arg(arg)
    gen_dir = summary["dir"]
    meta = json.load(open(os.path.join(gen_dir, "meta.json")))
    arrs = {k: np.load(os.path.join(gen_dir, k + ".npy")) for k in FIELDS}
    return meta, arrs


# -------------------------------- main -------------------------------------


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights")
    ap.add_argument("--replays")
    ap.add_argument("--games", help="JSON file: list of game ids or "
                                    "{\"games\": [...]}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--window-lo", type=int, default=200)
    ap.add_argument("--window-hi", type=int, default=299)
    ap.add_argument("--belief", default="sample", choices=list(T.BELIEF_MODES))
    ap.add_argument("--teacher-style", type=int, default=2)
    ap.add_argument("--K", type=int, default=G.CANON["K"])
    ap.add_argument("--M", type=int, default=G.CANON["M"])
    ap.add_argument("--H", type=int, default=G.CANON["H"])
    ap.add_argument("--seed-teacher", type=int, default=7000)
    ap.add_argument("--train-temp", type=float, default=1.0)
    ap.add_argument("--behavior-temp", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--pi0", type=float, default=1.0)
    ap.add_argument("--pi1", type=float, default=1.0)
    ap.add_argument("--isolation-file", action="append", default=[])
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--accept-unbound-identity", action="store_true",
                    help="adopt progress/shards recorded before the "
                         "content+code identity binding existed (their "
                         "recorded identity agrees on every pre-binding "
                         "key but carries no replay-content or code "
                         "binding); the adoption is recorded in "
                         "progress.json. Contradicted records are never "
                         "adopted.")
    ap.add_argument("--verify", action="store_true",
                    help="verify the sealed generation at --out and exit")
    a = ap.parse_args(argv)
    if a.verify:
        print(json.dumps({"annotation_data_sealed": resolve_out_arg(a.out)}))
        return 0
    if not (a.weights and a.replays and a.games):
        ap.error("--weights, --replays and --games are required unless --verify")
    plan = build_plan(a)
    out_dir = a.out
    shard_dir = os.path.join(out_dir, "shards")
    os.makedirs(shard_dir, exist_ok=True)
    prog_path = os.path.join(out_dir, "progress.json")
    progress = {"identity": plan["identity"], "games": {}}
    if os.path.exists(prog_path):
        old = json.load(open(prog_path))
        old_id = old.get("identity") or {}
        if old_id != plan["identity"]:
            changed = _identity_replay_changes(old_id, plan["identity"])
            if changed:
                g0 = sorted(changed)[0]
                _die(f"existing progress.json records replay content "
                     f"that has since changed for {len(changed)} "
                     f"game(s) (e.g. {g0}: recorded sha256 "
                     f"{changed[g0][0]}, current {changed[g0][1]}); "
                     f"refusing to mix content generations -- remove "
                     f"the output dir to start over")
            missing = _identity_unbound_missing(old_id, plan["identity"])
            if missing is not None and a.accept_unbound_identity:
                old["identity_migration"] = {
                    "missing": missing,
                    "flag": "--accept-unbound-identity",
                    "time": time.time(),
                    "adopted_code_md5s": plan["identity"]["code_md5s"],
                    "adopted_replay_sha256":
                        plan["identity"]["replay_sha256"]}
                old["identity"] = plan["identity"]
                print(json.dumps(
                    {"progress_identity_migration": {
                        "missing": missing,
                        "flag": "--accept-unbound-identity"}}), flush=True)
            elif missing is not None:
                _die("existing progress.json predates the content/code "
                     "identity binding (missing "
                     f"{missing}); the recorded run is UNBOUND -- rerun "
                     "with --accept-unbound-identity to adopt it "
                     "explicitly, or remove the output dir to start over")
            else:
                diff = sorted(k for k in set(old_id) | set(plan["identity"])
                              if old_id.get(k) != plan["identity"].get(k))
                _die("existing progress.json belongs to a different run "
                     f"identity (differing keys: {diff}); refusing to "
                     "mix -- remove the output dir to start over")
        progress = old
    metas = {}
    todo = []
    for pg in plan["per_game"]:
        rec = progress.get("games", {}).get(pg["game"], {})
        m = _verify_shard(shard_dir, pg, plan["identity"],
                          rec.get("shard_sha256"),
                          accept_unbound=a.accept_unbound_identity)
        if m is not None:
            metas[pg["game"]] = m
            rec_out = {"status": "done", "n_roots": m["n_roots"],
                       "shard_sha256": m["shard_sha256"]}
            if m.get("identity_migration"):
                rec_out["identity_migration"] = m["identity_migration"]
            progress["games"][pg["game"]] = rec_out
            print(json.dumps({"shard_skip_verified": pg["game"],
                              "n_roots": m["n_roots"]}), flush=True)
        else:
            todo.append(pg)
    _atomic_write_json(progress, prog_path)

    def _record(m):
        m["shard_sha256"] = _sha256_file(
            os.path.join(shard_dir, m["shard_file"]))
        metas[m["game"]] = m
        progress["games"][m["game"]] = {
            "status": "done", "n_roots": m["n_roots"],
            "shard_sha256": m["shard_sha256"]}
        _atomic_write_json(progress, prog_path)
        print(json.dumps({"shard_done": m["game"],
                          "n_roots": m["n_roots"]}), flush=True)

    tasks = [(pg, plan["config"], a.weights, shard_dir, plan["identity"])
             for pg in todo]
    if tasks:
        if a.workers and a.workers > 1:
            with ProcessPoolExecutor(max_workers=a.workers,
                                     initializer=_init_worker,
                                     initargs=(a.weights,)) as ex:
                for m in ex.map(_annotate_game, tasks):
                    _record(m)
        else:
            _init_worker(a.weights)
            for t in tasks:
                _record(_annotate_game(t))
    summary = publish(out_dir, plan, [metas[g] for g in plan["games"]])
    summary["verified"] = resolve_out_arg(out_dir)
    print(json.dumps({"annotation_published": summary}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

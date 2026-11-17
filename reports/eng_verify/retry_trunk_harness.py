#!/usr/bin/env python3
"""Retry-trunk re-run harness for small-teacher stage 4D (execution proposal v2).

Caliber: strategy-group ruling
reports/RULING_SMALL_TEACHER_STAGE4D_RETRY_TRUNK_20261118.md (commit
6388b27e) + execution proposal v2
reports/eng_verify/STAGE4D_RETRY_TRUNK_PROPOSAL_V2.md (commit 8d5fc52f).

Mechanism (proposal v2 section 1, verbatim): re-run the whole
stage0_teacher.evaluate_root with the student trunk, arms + priors
pinned to the teacher generation's values. Belief scenarios, priors,
opponent root actions are teacher originals; only the trunk is swapped.

  - Arms / priors: read per root from the teacher shards
    (annot_teacher/shards/selfplay-99440{0,1}.npz, generation
    gen-3f2c17f8b7beb5b6); gen_candidates is NOT called in the re-run.
  - Belief scenarios: same code, same seeds as production
    (belief_carbon, rng [seed_base+m, 0xC0]); trunk-independent, so the
    same worlds are reproduced.
  - Opponent root actions: sampled by the TEACHER trunk (temp 1.0, rng
    seed_base+m) + guard, frozen before the rollout; the student rollout
    only USES the frozen values, never resamples.
  - Guard snapshots: same path as production
    (G._teacher_guard_snaps); branch guards restore from the root
    snapshots.
  - ONLY the trunk is swapped, at exactly three places: continuation-step
    greedy decodes, the leaf _leaf_u, and the guard-internal re-decodes.
    Everything else (belief, priors, opponent actions, arm set,
    aggregation formula, _terminal_u, teacher=2) is the teacher original.

Why skipping gen_candidates is safe (read from the pinned sources, not
assumed): gen_candidates mutates the root guard only through
guard.apply, which updates lostw += len(prevw - cur) and prevw = cur.
The guard snapshot restored into evaluate_root already carries prevw =
the root step's worker uids (extract_scene observes the guard at every
step including the root step), and gen_candidates applies the guard to
the SAME root observation, so cur == prevw and lostw grows by exactly
0. guard_base is per-step state, cleared by reset_step() before every
guard use in the rollout, so the snapshot's guard_base is irrelevant.
The control gate (teacher trunk, bit-identical q) arbitrates this
empirically; the harness additionally records the restored me_snap lostw
per root (diagnostics.jsonl) so the at-risk subset (restored lostw == 0)
is measurable.

Counter note: this harness does NOT call gen_candidates, so its act-call
counter is lower than production's 680/root by exactly the
gen_candidates calls (1 greedy + deviations + 16 scoring). For a full
root (no early terminal) the harness counter is 648 = 8 opponent +
512 continuation + 128 leaf. The counter is a diagnostic only; the
control gate checks q / Q_mean bit-equality, not the counter.

Control gate (proposal section 2, hard gate before the full run): the
same harness with the TEACHER trunk on 8 roots (per game steps 200-201
x seats {0,1}) must reproduce the teacher shards' q and Q_mean
BIT-IDENTICALLY, with arms/priors passed through by assert. In control
mode the harness additionally runs gen_candidates with the teacher trunk
on a COPY of the root guard purely as a diagnostic (results discarded):
it must reproduce the pinned arms/priors, which independently confirms
the Phase-1 setup (scene, feats, guard snaps, seeds) is faithful.

This file is NEW. It modifies no pinned file. It imports the pinned
stack only (stage0_teacher c6ade5ce, stage1_teacher_gate e922d9d8,
cnet2_np d8c15ad4, cnet2_search 94f7fb64, cnet2_feat 2a45b08b, fastsim
aefe6c83, eval_v53 45792dfb, k0_annotate 7f567a2e) and reuses their
helpers directly; it contains no q formula of its own.
"""
import argparse
import hashlib
import io
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

# Single-threaded BLAS per process (same posture as k0_annotate):
# teacher calls are many small matmuls; parallelism belongs to --workers.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# The pinned code stack lives in --code-dir (default: the harness dir).
# It must be on sys.path BEFORE the pinned imports below, because this
# file itself may live outside the code dir (repo copy, staging copy).
_CODE_DIR = HERE
for _i, _a in enumerate(sys.argv):
    if _a == "--code-dir" and _i + 1 < len(sys.argv):
        _CODE_DIR = sys.argv[_i + 1]
    elif _a.startswith("--code-dir="):
        _CODE_DIR = _a.split("=", 1)[1]
sys.path.insert(0, _CODE_DIR)
sys.path.insert(0, HERE)

import cnet2_feat as F  # noqa: E402
import stage0_teacher as T  # noqa: E402
import stage1_teacher_gate as G  # noqa: E402
import k0_annotate as K0  # noqa: E402  (pinned 7f567a2e: _root_row, FIELDS, hashes)
from cnet2_search import BankruptcyGuard, CachePolicy, feat_copy  # noqa: E402
from fastsim import FastCarbon  # noqa: E402

HARNESS_VERSION = "retry-trunk-harness/1"
CALIBER = "换 trunk 重跑"


def _die(msg):
    raise SystemExit(f"retry_trunk_harness: {msg}")


def _md5_file(p):
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# retry_root: evaluate_root split in two phases (proposal section 1.6).
# Phase 1 (setup: belief scenarios + opponent root actions) runs with the
# TEACHER policy; Phase 2 (rollout + aggregation) runs with the passed-in
# policy. Arms / priors are pinned inputs, never generated here.
# ---------------------------------------------------------------------------
def retry_root(sim, me, pol, pol_teacher, x_me, aux_me, x_opp, aux_opp,
               feat_me, feat_opp, teacher=2, K=16, M=8, H=2, seed_base=7000,
               guard="on", guard_snaps=None, feats=None, belief="sample",
               arms=None, priors=None, diagnostic=False):
    """evaluate_root with a pinned arm set and a swappable trunk.

    pol: CachePolicy used for the rollout (teacher in control mode,
    student in re-run mode). pol_teacher: CachePolicy used for Phase 1
    (opponent root actions; always the teacher). arms/priors: the
    teacher generation's per-root values (list of int64 [OWN] arms,
    float64 priors). diagnostic (control mode only): run
    gen_candidates with the teacher policy on a copy of the root guard
    and compare against the pinned arms/priors; results discarded.
    Returns (result_dict, diag_dict); result_dict has exactly
    evaluate_root's shape so K0._root_row / G._expect_teacher apply.
    """
    from eval_v53 import CFG
    t0 = time.time()
    if belief != "sample":
        _die("retry_root supports only belief='sample' (production path)")
    if arms is None or priors is None:
        _die("retry_root requires pinned arms/priors")
    if not isinstance(pol, CachePolicy) or not isinstance(pol_teacher, CachePolicy):
        _die("retry_root requires CachePolicy instances (teacher + rollout)")
    arms = [np.asarray(a, dtype=np.int64) for a in arms]
    priors = np.asarray(priors, dtype=np.float64)
    if not (1 <= len(arms) <= K) or len(arms) != len(priors):
        _die(f"pinned arms/priors shape mismatch: {len(arms)} arms, "
             f"{len(priors)} priors, K={K}")
    if abs(float(priors.sum()) - 1.0) > 1e-9:
        _die(f"pinned priors do not sum to 1: {float(priors.sum())}")
    counter = [0]
    ep_steps = int(CFG.get("episodeSteps", 300))
    if T._is_terminal(sim):
        raise ValueError("retry_root called on an already-terminal root "
                         "position (extinct, or step limit reached)")
    if feats is not None:
        checked_feats = ((feats[0], me),)
        for _f, _seat in checked_feats:
            if _f.seat != _seat or int(_f.step) != int(sim.step_count):
                raise ValueError(
                    f"feats must be (seat {me}) Feats that observed "
                    f"root step {sim.step_count}; got seat {_f.seat} at step {_f.step}")
    if guard == "on":
        gcfg = T.TEACHER_GUARD_CFG
    elif isinstance(guard, dict):
        gcfg = dict(T.TEACHER_GUARD_CFG)
        gcfg.update(guard)
    elif guard in (None, False):
        gcfg = None
    else:
        raise ValueError(f"guard must be 'on', None, or a cfg dict, got {guard!r}")
    g_me_root = BankruptcyGuard(gcfg) if gcfg is not None else None
    g_opp_root = BankruptcyGuard(gcfg) if gcfg is not None else None
    if guard_snaps is not None and gcfg is not None:
        _sm, _so = guard_snaps
        if _sm is not None:
            g_me_root.restore(_sm)
        if _so is not None and belief == "oracle":
            g_opp_root.restore(_so)
    gtot = {"guard_moves": 0, "guard_recruit": 0}

    def act_t(px, paux, **kw):
        counter[0] += 1
        return pol_teacher.act(px, paux, teacher=teacher, **kw)

    def act(px, paux, **kw):
        counter[0] += 1
        return pol.act(px, paux, teacher=teacher, **kw)

    # ---- Phase 1a: branch roots (teacher-independent; same as production) ----
    n_unk = None
    _players = sim.obs(me)["players"]
    roots = []
    for m in range(M):
        carbon_m, n_unk = T.belief_carbon(
            feats[0], _players, sim.config,
            rng=np.random.default_rng([int(seed_base) + m, 0xC0]), mode=belief)
        roots.append(T._legal_root(sim, me, carbon_m))

    # ---- Phase 1b: opponent root actions FIRST (TEACHER policy; frozen) ----
    opp_actions = []
    opp_feats = []
    opp_snaps = []
    for m in range(M):
        rng_m = np.random.default_rng(seed_base + m)
        hypothetical = FastCarbon(CFG)
        hypothetical.load_observation(roots[m])
        model_opp = F.Feat(1 - me)
        model_x = model_opp.observe(hypothetical.obs(1 - me))
        model_aux = model_opp.aux
        model_guard = BankruptcyGuard(gcfg) if gcfg is not None else None
        lab, lg, _ = act_t(model_x, model_aux, sample_rng=rng_m, temp=1.0)
        if model_guard is not None:
            model_guard.reset_step()
            lab = model_guard.apply(model_x, model_aux, np.asarray(lab, dtype=np.int64), lg,
                                    pol_teacher, model_opp.own_uid, teacher, gtot)
        opp_actions.append(F.Feat.to_actions(model_opp, np.asarray(lab, dtype=np.int64)))
        opp_feats.append(model_opp)
        opp_snaps.append(model_guard.snapshot() if model_guard is not None else None)

    # ---- Pinned arms/priors (gen_candidates NOT called) ----
    # me_snap is taken at the same point production takes it (after the
    # opponent loop). Skipping gen_candidates leaves it unchanged: the
    # guard snapshot restored into g_me_root already carries prevw = the
    # root step's worker uids, and gen_candidates' applies on the same
    # root observation add exactly 0 to lostw; guard_base is cleared by
    # reset_step() before every guard use in the rollout.
    me_snap = g_me_root.snapshot() if g_me_root is not None else None
    opp_snap = g_opp_root.snapshot() if g_opp_root is not None else None
    diag = {
        "me_snap_lostw": (int(me_snap["lostw"]) if me_snap is not None else None),
        "opp_snaps_lostw": [(int(s["lostw"]) if s is not None else None)
                            for s in opp_snaps],
        "n_unknown_cells": (int(n_unk) if n_unk is not None else None),
        "diagnostic_gen_candidates": None,
    }
    if diagnostic:
        # Control-only diagnostic: reproduce production's gen_candidates
        # with the teacher policy on a COPY of the root guard; compare
        # against the pinned arms/priors; results discarded. A mismatch
        # here means the Phase-1 setup itself is unfaithful.
        d_counter = [0]
        g_copy = g_me_root.copy() if g_me_root is not None else None
        d_arms, d_priors = T.gen_candidates(
            pol_teacher, x_me, aux_me, teacher, K=K, counter=d_counter,
            guard=g_copy, own_uid=feat_me.own_uid)
        arms_match = (len(d_arms) == len(arms) and all(
            np.array_equal(np.asarray(a, dtype=np.int64), b)
            for a, b in zip(d_arms, arms)))
        priors_max_diff = float(np.max(np.abs(
            np.asarray(d_priors, dtype=np.float64) - priors))) if len(d_arms) == len(arms) else float("inf")
        diag["diagnostic_gen_candidates"] = {
            "arms_match": bool(arms_match),
            "priors_max_abs_diff": priors_max_diff,
            "n_act_calls": int(d_counter[0]),
            "n_arms_diagnostic": int(len(d_arms)),
        }

    # ---- Phase 2: rollout with the parameterized policy ----
    Q = np.zeros((len(arms), M))
    for ci, arm in enumerate(arms):
        a_me = F.Feat.to_actions(feat_me, arm)
        for m in range(M):
            clone = FastCarbon(CFG)
            clone.load_observation(roots[m])
            fb = None
            if feats is not None:
                fb = [None, None]
                fb[me] = feat_copy(feats[0])
                fb[1 - me] = feat_copy(opp_feats[m])
            a_opp = opp_actions[m]
            pre = clone.get_state()  # state before the root step (survivor scoring)
            clone.step(*((a_me, a_opp) if me == 0 else (a_opp, a_me)))
            g_me = BankruptcyGuard(gcfg) if gcfg is not None else None
            g_opp = BankruptcyGuard(gcfg) if gcfg is not None else None
            if g_me is not None:
                g_me.restore(me_snap)
            if g_opp is not None:
                g_opp.restore(opp_snaps[m])
            done = T._is_terminal(clone)
            for _ in range(H):
                if done:
                    break
                f0 = fb[0] if fb is not None else F.Feat(0)
                f1 = fb[1] if fb is not None else F.Feat(1)
                x0 = f0.observe(clone.obs(0))
                x1 = f1.observe(clone.obs(1))
                l0, lg0, _ = act(x0, f0.aux)
                l1, lg1, _ = act(x1, f1.aux)
                g_seat = [None, None]
                if g_me is not None:
                    g_seat[me] = g_me
                    g_seat[1 - me] = g_opp
                if g_seat[0] is not None:
                    g_seat[0].reset_step()
                    l0 = g_seat[0].apply(x0, f0.aux, np.asarray(l0, dtype=np.int64), lg0,
                                         pol, f0.own_uid, teacher, gtot)
                if g_seat[1] is not None:
                    g_seat[1].reset_step()
                    l1 = g_seat[1].apply(x1, f1.aux, np.asarray(l1, dtype=np.int64), lg1,
                                         pol, f1.own_uid, teacher, gtot)
                pre = clone.get_state()  # state before this continuation step
                clone.step(*(f0.to_actions(np.asarray(l0, dtype=np.int64)),
                             f1.to_actions(np.asarray(l1, dtype=np.int64))))
                done = T._is_terminal(clone)
            if done:
                Q[ci, m] = T._terminal_u(clone, me, pre, episode_steps=ep_steps)
            else:
                Q[ci, m] = T._leaf_u(clone, me, pol, teacher, counter,
                                     feat=fb[me] if fb is not None else None)
    q_mean = Q.mean(axis=1)
    wq = np.exp((q_mean - q_mean.max()) / 0.05) * np.sqrt(np.maximum(priors, 1e-12))
    q = wq / wq.sum()
    ent = float(-(q * np.log(np.maximum(q, 1e-12))).sum())
    diag["n_act_calls"] = int(counter[0])
    result = {
        "arms": arms, "n_arms": len(arms), "priors": priors, "Q_mean": q_mean,
        "Q_std": Q.std(axis=1),
        "q": q, "entropy": ent, "agree_greedy": bool(int(np.argmax(q)) == 0),
        "greedy_q_rank": int((q > q[0]).sum()),
        "n_act_calls": counter[0],
        "wall_ms": round((time.time() - t0) * 1000, 1),
        "guard": dict(gtot),
        "teacher_version": "v3",
        "opponent_model": "scenario-root-history",
        "belief": belief, "feat_inherit": feats is not None,
        "n_unknown_cells": n_unk,
    }
    return result, diag



# ---------------------------------------------------------------------------
# Driver: identity verification, worker pool, shard writing, generation
# assembly. Mirrors k0_annotate's _annotate_game / publish layout so the
# output is consumable by the same tooling (shard_compare v4 reads the
# per-game shards directly).
# ---------------------------------------------------------------------------

# Expected identity, read from the teacher generation
# (annot_teacher/generations/gen-3f2c17f8b7beb5b6, verified on L20).
TEACHER_GENERATION = "gen-3f2c17f8b7beb5b6"
EXPECTED_CONFIG_SHA256 = "9a2b2440e9267680c9b27a1d51a50db9cacdbd04582ae101ef54abaa86d8d10c"
EXPECTED_GAMES_SHA256 = "c3649caaa222b0f4e128eb65fec1e6926d80c6eea3aca3ab3b7c7b88b5b87fdf"
EXPECTED_REPLAY_SHA256 = {
    "selfplay-994400": "55d713d64c002d6c204988280804da47b96c979464f4ae9740b2676338581436",
    "selfplay-994401": "866a8ce7cc7125af8962e51f6db7b1a41686c3c13dbab35b2cec0c653a66baf5",
}
# NOTE on replay hashes (verified 2026-11-18): the values 6ed2623e... /
# 6607fe5e... cited in proposal v2 section 4 and the 4D stop report as
# replay "sha256" are in fact the files' MD5s. The teacher generation's
# own identity block records the TRUE sha256 values above, and they match
# sha256sum of the replay bytes. The driver therefore verifies BOTH:
# md5 == 6ed2623e.../6607fe5e... (staging record) AND sha256 == the
# identity-block values (generation record). A future registry correction
# should relabel the 6ed2623e... values as md5.
EXPECTED_REPLAY_MD5 = {
    "selfplay-994400": "6ed2623ea5102e477a373ebc22791564",
    "selfplay-994401": "6607fe5e71c69135787c28ffbdb61b12",
}
EXPECTED_WEIGHTS_MD5 = "06a03d9cfeec239af3065a49fefec382"
EXPECTED_STUDENT_NPZ_MD5 = "b22020ff71a17af8f45a3dd2659f24c8"
EXPECTED_CODE_MD5S = {
    "stage0_teacher.py": "c6ade5ced42e2c54a77c1499163aa151",
    "cnet2_np.py": "d8c15ad4fdd8ee079363cf215d52daf3",
    "cnet2_search.py": "94f7fb6438bb7577b141e388f068e122",
    "cnet2_feat.py": "2a45b08b6f97508541e62867e408748e",
    "fastsim.py": "aefe6c83826efae4de83bef8c643c80b",
    "stage1_teacher_gate.py": "e922d9d890c3e1eff6643e947289acb0",
    "k0_annotate.py": "7f567a2e8544b78e51d16aa7660bc47b",
}
EXPECTED_EXTRA_MD5S = {
    "cnet2_model.py": "3bb5857f833a94797ac6cdc2a5a28963",
    "carbon.json": "cd55531b14438156def23274d594956e",
}
EXPECTED_STUDENT_PLUGIN_MD5 = "5c0dc7579126ab7e10e774eef12cea50"
CONTROL_ROOTS = [(200, 0), (200, 1), (201, 0), (201, 1)]  # per game


_W = {}


def _init_worker(teacher_weights_bytes, student_npz_bytes):
    """Load both policies once per worker (mirrors k0_annotate._init_worker)."""
    _W["pol_teacher"] = CachePolicy(dict(np.load(io.BytesIO(teacher_weights_bytes))))
    _W["pol_student"] = CachePolicy(dict(np.load(io.BytesIO(student_npz_bytes))))


def _verify_code_dir(code_dir):
    for fn, want in EXPECTED_CODE_MD5S.items():
        p = os.path.join(code_dir, fn)
        if not os.path.exists(p):
            _die(f"code file missing in code dir: {fn}")
        got = _md5_file(p)
        if got != want:
            _die(f"code identity mismatch: {fn} md5 {got} != {want}")
    p = os.path.join(code_dir, "cnet2_model.py")
    if _md5_file(p) != EXPECTED_EXTRA_MD5S["cnet2_model.py"]:
        _die("cnet2_model.py md5 mismatch")
    p = os.path.join(code_dir, "env_src", "carbon_challenge-master",
                     "zerosum_env", "envs", "carbon", "carbon.json")
    if _md5_file(p) != EXPECTED_EXTRA_MD5S["carbon.json"]:
        _die("carbon.json md5 mismatch")


def _recompute_config_sha256():
    import types
    a = types.SimpleNamespace(K=16, M=8, H=2, teacher_style=2, belief="sample",
                              train_temp=1.0, behavior_temp=1.0,
                              seed_teacher=7000, window_lo=200, window_hi=299)
    cfg = K0.teacher_config(a)
    return K0._sha256_bytes(K0._canonical(cfg))


def _load_teacher_shards(shard_dir):
    """Read the teacher shards; verify identity + arm/prior sanity."""
    import copy
    games = {}
    identity = None
    for g in ("selfplay-994400", "selfplay-994401"):
        p = os.path.join(shard_dir, g + ".npz")
        if not os.path.exists(p):
            _die(f"teacher shard missing: {p}")
        with np.load(p) as d:
            arrs = {k: d[k] for k in d.files}
            meta = json.loads(str(d["_meta"].item()))
        ident = meta["identity"]
        if identity is None:
            identity = ident
        elif ident != identity:
            _die(f"teacher shard {g}: identity differs from sibling shard")
        if ident["config_sha256"] != EXPECTED_CONFIG_SHA256:
            _die(f"teacher shard {g}: config_sha256 mismatch")
        if ident["games_sha256"] != EXPECTED_GAMES_SHA256:
            _die(f"teacher shard {g}: games_sha256 mismatch")
        if ident["replay_sha256"] != EXPECTED_REPLAY_SHA256:
            _die(f"teacher shard {g}: replay_sha256 mismatch")
        if ident["code_md5s"] != EXPECTED_CODE_MD5S:
            _die(f"teacher shard {g}: code_md5s mismatch")
        arms, priors = arrs["arms"], arrs["priors"]
        if arms.shape != (198, 16, 12) or priors.shape != (198, 16):
            _die(f"teacher shard {g}: unexpected arms/priors shape "
                 f"{arms.shape} {priors.shape}")
        if sorted(set(arrs["n_arms"].tolist())) != [16]:
            _die(f"teacher shard {g}: n_arms not all 16")
        ps = priors.sum(axis=1)
        if float(np.max(np.abs(ps - 1.0))) > 1e-9:
            _die(f"teacher shard {g}: priors do not sum to 1")
        games[g] = {"meta": meta, "identity": ident,
                    "step": arrs["step"], "seat": arrs["seat"], "gid": arrs["gid"],
                    "arms": arms, "priors": priors,
                    "q": arrs["q"], "Q": arrs["Q"]}
    return games, identity


def _verify_materials(args):
    """Fail-closed identity verification before any root runs."""
    _verify_code_dir(args.code_dir)
    if _recompute_config_sha256() != EXPECTED_CONFIG_SHA256:
        _die("recomputed config_sha256 != 9a2b2440... (production config)")
    if _md5_file(args.weights) != EXPECTED_WEIGHTS_MD5:
        _die("teacher weights md5 mismatch")
    if _md5_file(args.student_npz) != EXPECTED_STUDENT_NPZ_MD5:
        _die("student npz md5 mismatch")
    plug = os.path.join(args.code_dir, "stage0_teacher_small.py")
    if os.path.exists(plug) and _md5_file(plug) != EXPECTED_STUDENT_PLUGIN_MD5:
        _die("student plug-in md5 mismatch (expected 5c0dc757)")
    for g, want_md5 in EXPECTED_REPLAY_MD5.items():
        p = os.path.join(args.replays, g + ".json.gz")
        if _md5_file(p) != want_md5:
            _die(f"replay {g}: md5 mismatch (staging record)")
        if _sha256_file(p) != EXPECTED_REPLAY_SHA256[g]:
            _die(f"replay {g}: sha256 mismatch (generation identity)")
    games, identity = _load_teacher_shards(args.teacher_shards)
    return games, identity


def _run_root(task):
    """Worker: one root through retry_root. Returns (row, diag)."""
    (game, gid, step, seat, replay_path, arms_row, priors_row,
     mode, teacher_style, seed_teacher) = task
    spec = {"game": game, "path": replay_path, "step": int(step),
            "me": int(seat), "ft": 299}
    scene = G.extract_scene(spec)
    pol = _W["pol_teacher"] if mode == "control" else _W["pol_student"]
    try:
        r, diag = retry_root(
            G._fresh_sim(scene), scene["me"], pol, _W["pol_teacher"],
            scene["x_me"], scene["aux_me"], scene["x_opp"], scene["aux_opp"],
            G._ns(scene["fm"]), G._ns(scene["fo"]),
            teacher=teacher_style, K=16, M=8, H=2,
            seed_base=seed_teacher + int(step),
            feats=(scene["fm"], scene["fo"]),
            guard_snaps=G._teacher_guard_snaps(scene, "sample"),
            belief="sample", arms=[arms_row[i] for i in range(len(arms_row))],
            priors=priors_row, diagnostic=(mode == "control"))
    except ValueError as e:
        _die(f"root ({game}, step {step}, seat {seat}) refused: {e}")
    G._expect_teacher(r, "sample")
    row = K0._root_row(r, 16)
    row_out = {"gid": gid, "step": int(step), "seat": int(seat)}
    for k in ("arms", "n_arms", "priors", "q", "Q", "Q_std", "belief", "n_unknown"):
        row_out[k] = row[k]
    diag_out = {"game": game, "step": int(step), "seat": int(seat)}
    diag_out.update(diag)
    diag_out["wall_ms"] = r["wall_ms"]
    return row_out, diag_out


def _write_shard(out_dir, game, meta_teacher, identity, rows, extra):
    """Write one per-game shard in k0_annotate format."""
    K = 16
    arrs = {}
    for k, dt in K0.FIELDS.items():
        v = rows[k]
        if k == "arms":
            arrs[k] = np.stack(v).astype(dt) if v else np.zeros((0, K, K0.OWN), dt)
        elif k in ("priors", "q", "Q", "Q_std"):
            arrs[k] = np.stack(v).astype(dt) if v else np.zeros((0, K), dt)
        else:
            arrs[k] = np.asarray(v, dtype=dt)
    meta = {"game": game, "gid": meta_teacher["gid"],
            "pass": meta_teacher["pass"], "n_roots": len(rows["step"]),
            "selected_sha256": meta_teacher["selected_sha256"],
            "identity": identity,
            "arrays": {k: K0._array_sha(arrs[k]) for k in K0.FIELDS},
            "shard_file": game + ".npz"}
    meta.update(extra)
    shard_dir = os.path.join(out_dir, "shards")
    os.makedirs(shard_dir, exist_ok=True)
    npz_path = os.path.join(shard_dir, game + ".npz")
    tmp = npz_path + f".tmp-{os.getpid()}"
    with open(tmp, "wb") as fh:
        np.savez(fh, _meta=np.asarray(json.dumps(meta, sort_keys=True)), **arrs)
    os.replace(tmp, npz_path)
    return meta, _sha256_file(npz_path)


def _assemble_generation(out_dir, gen_id, games_meta, identity, extra):
    """Assemble per-field .npy files + meta.json + manifest.json, mirroring
    k0_annotate.publish's generation layout."""
    gen_dir = os.path.join(out_dir, "generations", gen_id)
    os.makedirs(gen_dir, exist_ok=True)
    # Re-read the written shards in feed order and stack.
    stacked = {}
    for g in ("selfplay-994400", "selfplay-994401"):
        with np.load(os.path.join(out_dir, "shards", g + ".npz")) as d:
            for k in K0.FIELDS:
                stacked.setdefault(k, []).append(d[k])
    manifest_fields = {}
    for k, parts in stacked.items():
        arr = np.concatenate(parts, axis=0) if parts[0].ndim > 0 else np.asarray(parts)
        p = os.path.join(gen_dir, k + ".npy")
        np.save(p, arr)
        manifest_fields[k] = {"file": k + ".npy", "bytes": int(os.path.getsize(p)),
                              "rows": int(arr.shape[0]), "sha256": _sha256_file(p)}
    teacher_meta_path = extra["_teacher_meta_path"]
    with open(teacher_meta_path) as fh:
        meta = json.load(fh)
    meta["games"] = [{
        "game": gm["game"], "gid": gm["gid"], "n_roots": gm["n_roots"],
        "pass": gm["pass"], "selected_sha256": gm["selected_sha256"],
        "shard_sha256": games_meta[gm["game"]]["shard_sha256"],
    } for gm in meta["games"]]
    meta["total"] = sum(gm["n_roots"] for gm in meta["games"])
    meta["retry_trunk"] = {
        "caliber": CALIBER, "trunk": extra["trunk"],
        "student_npz_md5": extra["student_npz_md5"],
        "teacher_generation": TEACHER_GENERATION,
        "harness_md5": extra["harness_md5"],
        "harness": os.path.basename(__file__),
        "generation": gen_id,
    }
    meta_path = os.path.join(gen_dir, "meta.json")
    K0._atomic_write_json(meta, meta_path)
    manifest = {
        "created": time.time(), "fields": manifest_fields,
        "format": K0.FORMAT, "generation": gen_id,
        "meta_sha256": _sha256_file(meta_path),
        "producer": {"code_md5": EXPECTED_CODE_MD5S,
                     "harness_md5": extra["harness_md5"],
                     "student_npz_md5": extra["student_npz_md5"],
                     "teacher_generation": TEACHER_GENERATION,
                     "caliber": CALIBER},
        "total": meta["total"],
    }
    manifest["self_sha256"] = K0._sha256_bytes(K0._canonical(
        {k: v for k, v in manifest.items()}))
    K0._atomic_write_json(manifest, os.path.join(gen_dir, "manifest.json"))
    return gen_dir


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("control", "student"), required=True,
                    help="control: teacher trunk, 8 roots, diagnostic on; "
                         "student: student trunk, all 396 roots")
    ap.add_argument("--teacher-shards", required=True,
                    help="dir with the teacher shards (selfplay-99440{0,1}.npz)")
    ap.add_argument("--teacher-generation", required=True,
                    help="teacher generation dir (for meta.json template)")
    ap.add_argument("--replays", required=True)
    ap.add_argument("--weights", required=True, help="teacher heads npz")
    ap.add_argument("--student-npz", required=True)
    ap.add_argument("--code-dir", default=HERE,
                    help="dir with the pinned code stack (default: harness dir)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seat", default="",
                    help="seat label recorded in the run summary (not verified)")
    ap.add_argument("--only", action="append", default=[],
                    help="restrict to specific roots, game:step:seat "
                         "(repeatable; bypasses the mode root-count asserts; "
                         "for spot checks only)")
    args = ap.parse_args(argv)

    t_start = time.time()
    harness_md5 = _md5_file(os.path.abspath(__file__))
    games, identity = _verify_materials(args)
    print(f"[retry] identities verified; harness md5 {harness_md5}", flush=True)

    with open(args.weights, "rb") as fh:
        teacher_weights_bytes = fh.read()
    with open(args.student_npz, "rb") as fh:
        student_npz_bytes = fh.read()

    # Root task list: control = 8 roots (steps 200-201 x seats 0/1 per game);
    # student = every root the teacher shards hold, in shard order.
    only = set()
    for spec in args.only:
        g2, s2, seat2 = spec.split(":")
        only.add((g2, int(s2), int(seat2)))
    tasks = []
    for g in ("selfplay-994400", "selfplay-994401"):
        gd = games[g]
        gid = int(gd["gid"][0])
        for i in range(len(gd["step"])):
            step, seat = int(gd["step"][i]), int(gd["seat"][i])
            if args.mode == "control" and (step, seat) not in CONTROL_ROOTS:
                continue
            if only and (g, step, seat) not in only:
                continue
            tasks.append((g, gid, step, seat,
                          os.path.join(args.replays, g + ".json.gz"),
                          gd["arms"][i], gd["priors"][i],
                          args.mode, 2, 7000))
    if not only:
        if args.mode == "control" and len(tasks) != 8:
            _die(f"control mode expects 8 roots, got {len(tasks)}")
        if args.mode == "student" and len(tasks) != 396:
            _die(f"student mode expects 396 roots, got {len(tasks)}")
    print(f"[retry] mode={args.mode} roots={len(tasks)} workers={args.workers}",
          flush=True)

    rows_by_game = {g: {k: [] for k in K0.FIELDS} for g in games}
    diags = []
    with ProcessPoolExecutor(max_workers=args.workers,
                             initializer=_init_worker,
                             initargs=(teacher_weights_bytes, student_npz_bytes)) as ex:
        for row_out, diag_out in ex.map(_run_root, tasks):
            g = diag_out["game"]
            rb = rows_by_game[g]
            rb["gid"].append(row_out["gid"]); rb["step"].append(row_out["step"])
            rb["seat"].append(row_out["seat"])
            for k in ("arms", "n_arms", "priors", "q", "Q", "Q_std",
                      "belief", "n_unknown"):
                rb[k].append(row_out[k])
            rb["pi0"].append(1.0); rb["pi1"].append(1.0); rb["pi"].append(1.0)
            diags.append(diag_out)
            if len(diags) % 50 == 0 or len(diags) == len(tasks):
                print(f"[retry] {len(diags)}/{len(tasks)} roots "
                      f"({(time.time() - t_start) / 60:.1f} min)", flush=True)

    trunk = "teacher" if args.mode == "control" else "student"
    gen_id_src = {"identity": identity,
                  "student_npz_md5": EXPECTED_STUDENT_NPZ_MD5,
                  "harness_md5": harness_md5, "caliber": CALIBER,
                  "trunk": trunk}
    gen_id = "gen-retry-trunk-" + K0._sha256_bytes(
        K0._canonical(gen_id_src))[:12]
    extra = {"caliber": CALIBER, "trunk": trunk,
             "student_npz_md5": EXPECTED_STUDENT_NPZ_MD5,
             "teacher_generation": TEACHER_GENERATION,
             "harness_md5": harness_md5,
             "_teacher_meta_path": os.path.join(
                 args.teacher_generation, "meta.json")}
    games_meta = {}
    for g in ("selfplay-994400", "selfplay-994401"):
        meta_t = games[g]["meta"]
        meta_w, shard_sha = _write_shard(args.out, g, meta_t, identity,
                                         rows_by_game[g], extra)
        games_meta[g] = {"game": g, "gid": meta_t["gid"],
                         "n_roots": meta_t["n_roots"], "pass": meta_t["pass"],
                         "selected_sha256": meta_t["selected_sha256"],
                         "shard_sha256": shard_sha}
    gen_dir = _assemble_generation(args.out, gen_id, games_meta, identity, extra)
    diag_path = os.path.join(args.out, "diagnostics.jsonl")
    with open(diag_path, "w") as fh:
        for d in sorted(diags, key=lambda x: (x["game"], x["step"], x["seat"])):
            fh.write(json.dumps(d, sort_keys=True) + "\n")
    n_calls = [d["n_act_calls"] for d in diags]
    summary = {
        "mode": args.mode, "trunk": trunk, "seat": args.seat,
        "generation": gen_id, "total_roots": len(diags),
        "n_act_calls": {"min": int(min(n_calls)), "max": int(max(n_calls)),
                        "mean": float(sum(n_calls) / len(n_calls))},
        "me_snap_lostw_zero_roots": int(sum(
            1 for d in diags if d.get("me_snap_lostw") == 0)),
        "wall_min": round((time.time() - t_start) / 60, 2),
        "harness_md5": harness_md5,
        "student_npz_md5": EXPECTED_STUDENT_NPZ_MD5,
        "teacher_generation": TEACHER_GENERATION,
        "identity": {"config_sha256": identity["config_sha256"],
                     "games_sha256": identity["games_sha256"]},
        "out_dir": os.path.abspath(args.out),
        "generation_dir": os.path.abspath(gen_dir),
    }
    K0._atomic_write_json(summary, os.path.join(args.out, "run_summary.json"))
    K0._atomic_write_json(
        {"games": {g: {"n_roots": games_meta[g]["n_roots"],
                       "shard_sha256": games_meta[g]["shard_sha256"],
                       "status": "done"} for g in games_meta},
         "identity": identity, "total": len(diags)},
        os.path.join(args.out, "progress.json"))
    print(json.dumps(summary, indent=1, sort_keys=True))
    return summary


if __name__ == "__main__":
    main()

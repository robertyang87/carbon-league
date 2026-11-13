"""Stage 1 teacher gates: q-benefit gate (S1-2a-q) and paired ΔU gate (S1-2a).

Spec: reports/STAGE1_TEACHER_GATES_20261113.md (owner-corrected final,
2026-11-13). This driver evaluates the ACTUAL q distribution the distillation
will train on -- not agreement / entropy / argmax diagnostics (those are
reported as diagnostics only).

Scene source: real replays (NOT self-play, NOT the training corpus). The
caller supplies a split manifest {"q": [game_id...], "du": [game_id...]}
(game_id = replay filename stem); the two gates' game sets must be disjoint
and, when --ds-games-file is given (the ds_m3s2 game-id list), disjoint from
the training corpus too -- the driver refuses to run otherwise. Without
--ds-games-file the prereg header records the overlap check as NOT RUN.

Modes:
  q   -- per scene, evaluate_root (new U leaf, production guard) and score
         Q(argmax q) - Q(greedy); plus an independent anchor: paired real
         continuations to terminal (guided first move vs greedy first move,
         then both seats guarded-greedy, common seeds) on a seeded subset.
         Prereg verdict (canonical config only): PASS iff mean Q gain
         >= +0.01 U, strict-improvement rate >= 25%, anchor ΔU point
         estimate > 0. FAIL iff gain <= 0 or rate < 10%. Else INDETERMINATE
         (one seeded doubling allowed by the spec, never silent re-runs).
  du  -- paired continuations per root scene: guided arm asks the teacher
         every own step, greedy arm plays guarded greedy; opponent is the
         same bare net, guarded greedy, common seeds per root. Reports
         per-root ΔU, mean, paired-t 95% CI and a bootstrap percentile CI
         (sensitivity only). --pilot (Stage A): no verdict; reports sigma
         and the locked Stage B N = ceil((140*sigma)^2) clamped to
         [150, 600], plus the measured per-call teacher cost.
         Verdict (Stage B, canonical config, N >= --n-required): PASS iff
         ΔU >= +0.02 and the t-CI lower bound > 0.

Guard posture everywhere (candidate generation, continuations, both arms,
opponent): production config GUARD=True, GUARD_NARROW=True, GUARD_CASH=40.0,
LOWW=2 (stage0_teacher.TEACHER_GUARD_CFG). Guard state inside a continuation
starts fresh at the root scene and evolves within that continuation only;
each arm holds its own instances (no cross-arm / cross-seat sharing).

Non-canonical teacher params (K/M/H overridden) or reduced scene counts
force verdict = "DIAGNOSTIC ONLY" in the output header.

Usage (from code/):
  python3 stage1_teacher_gate.py --weights <heads.npz> --replays <dir>
      --split-manifest split.json --mode q|du [--ds-games-file ids.txt]
      [--seed-scene 995300] [--seed-cont-base 996100] [--workers 8]
      [--pilot] [--n-required 150] [--out gate_q.json]
"""
import argparse
import copy
import gzip
import hashlib
import json
import math
import os
import sys
import time
import types
from concurrent.futures import ProcessPoolExecutor

# Single-threaded BLAS per process (same posture as stage0_guided_match):
# teacher calls are many small matmuls; BLAS thread oversubscription
# measured ~12x CPU burn for no wall gain on Studio. Parallelism belongs
# to --workers, not to BLAS.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import cnet2_feat as F  # noqa: E402
import cnet2_np as NP  # noqa: E402
from cnet2_search import BankruptcyGuard, CachePolicy  # noqa: E402
from eval_v53 import CFG  # noqa: E402
from fastsim import FastCarbon  # noqa: E402
import stage0_teacher as T  # noqa: E402

CANON = {"K": 16, "M": 8, "H": 2}
LAYERS_Q = [(40, 120), (120, 200), (200, 240)]
LAYERS_DU = [(0, 40), (40, 120), (120, 200), (200, 240)]
DU_OPENING_SHARE = 0.10  # opening layer share of ΔU-gate scenes (spec (c))
Q_PASS = {"gain": 0.01, "rate": 0.25}
Q_FAIL = {"gain": 0.0, "rate": 0.10}
DU_PASS = 0.02
ANCHOR_N = 30


def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def game_id_of(path):
    name = os.path.basename(path)
    return name[:-len(".json.gz")] if name.endswith(".json.gz") else os.path.splitext(name)[0]


def list_replays(replay_dir, game_ids):
    by_id = {}
    for fn in sorted(os.listdir(replay_dir)):
        if fn.endswith(".json.gz"):
            by_id[game_id_of(fn)] = os.path.join(replay_dir, fn)
    missing = [g for g in game_ids if g not in by_id]
    if missing:
        raise SystemExit(f"split manifest games not in replay dir: {missing[:5]}")
    return [by_id[g] for g in game_ids]


def sample_scene_specs(game_paths, layers, seed, opening_share=None):
    """Deterministic (game, step, me) scene specs: one position per game per
    layer, sampled with the prereg scene RNG."""
    rng = np.random.default_rng(seed)
    specs = []
    for gp in game_paths:
        with gzip.open(gp) as fh:
            d = json.load(fh)
        n_steps = len(d["steps"]) - 1  # step t uses actions of steps[t+1]
        use_layers = layers
        if opening_share is not None:
            # ΔU gate: opening layer only for a seeded share of games.
            if rng.random() >= opening_share:
                use_layers = layers[1:]
        for (lo, hi) in use_layers:
            hi_c = min(hi, n_steps - 1)
            if hi_c <= lo:
                continue
            t = int(rng.integers(lo, hi_c))
            me = int(rng.integers(0, 2))
            specs.append({"game": game_id_of(gp), "path": gp, "step": t, "me": me})
    rng.shuffle(specs)
    return specs


def extract_scene(spec):
    """Replay to spec['step']; return a scene with the sim snapshot and live
    Feat objects for both seats (fresh Feats observing the root obs, the same
    history simplification the teacher itself uses for branch Feats)."""
    with gzip.open(spec["path"]) as fh:
        d = json.load(fh)
    st = d["steps"]
    sim = FastCarbon(CFG).reset(seed=1, full_carbon=st[0][0]["observation"]["full_carbon"])
    for t in range(spec["step"]):
        sim.step(st[t + 1][0].get("action") or {}, st[t + 1][1].get("action") or {})
    me = spec["me"]
    fm = F.Feat(me)
    fo = F.Feat(1 - me)
    x_me = fm.observe(sim.obs(me))
    x_opp = fo.observe(sim.obs(1 - me))
    return {"full": sim.full_observation(0), "step": spec["step"], "me": me,
            "game": spec["game"], "fm": fm, "fo": fo,
            "x_me": x_me, "aux_me": fm.aux, "x_opp": x_opp, "aux_opp": fo.aux}


def _ns(feat):
    return types.SimpleNamespace(seat=feat.seat, own_uid=list(feat.own_uid))


def _guarded_greedy(pol, guard, feat, x, teacher, stats):
    labels, logits, _ = pol.act(x, feat.aux, teacher=teacher)
    guard.reset_step()
    return guard.apply(x, feat.aux, np.asarray(labels, dtype=np.int64), logits,
                       pol, feat.own_uid, teacher, stats)


def continue_game(pol, scene, first_labels, cont_seed, teacher=1, guided_KMH=None,
                  stats=None):
    """Real continuation to terminal from a scene. If guided_KMH is set, the
    `me` seat asks the teacher every step and plays argmax q; otherwise the
    `me` seat plays `first_labels` on step 1 (if given) and guarded greedy
    afterwards. The opponent always plays guarded greedy. Both seats hold
    their own guard instances, fresh at the root. Returns U from me's seat
    (F01/F04 via stage0_teacher._terminal_u) and call timing stats."""
    stats = stats if stats is not None else {}
    me = scene["me"]
    sim = FastCarbon(CFG)
    sim.load_observation(scene["full"])
    feats = {me: copy.deepcopy(scene["fm"]), 1 - me: copy.deepcopy(scene["fo"])}
    guards = {0: BankruptcyGuard(T.TEACHER_GUARD_CFG),
              1: BankruptcyGuard(T.TEACHER_GUARD_CFG)}
    pre = sim.get_state()
    step_no = 0
    while True:
        st = sim.get_state()
        if T._extinct_state(st, 0) or T._extinct_state(st, 1) or sim.step_count >= 299:
            break
        xs = {}
        for seat in (0, 1):
            xs[seat] = feats[seat].observe(sim.obs(seat))
        acts = {}
        for seat in (0, 1):
            if seat == me and guided_KMH is not None:
                other = 1 - me
                r = T.evaluate_root(
                    sim, me, pol, copy.deepcopy(xs[me]), copy.deepcopy(feats[me].aux),
                    copy.deepcopy(xs[other]), copy.deepcopy(feats[other].aux),
                    _ns(feats[me]), _ns(feats[other]), teacher=teacher,
                    K=guided_KMH["K"], M=guided_KMH["M"], H=guided_KMH["H"],
                    seed_base=cont_seed + step_no)
                stats["teacher_calls"] = stats.get("teacher_calls", 0) + 1
                stats["teacher_ms"] = stats.get("teacher_ms", 0.0) + r["wall_ms"]
                acts[seat] = np.asarray(r["arms"][int(np.argmax(r["q"]))], dtype=np.int64)
            elif seat == me and step_no == 0 and first_labels is not None:
                acts[seat] = np.asarray(first_labels, dtype=np.int64)
            else:
                acts[seat] = _guarded_greedy(pol, guards[seat], feats[seat],
                                             xs[seat], teacher, stats)
        pre = sim.get_state()
        sim.step(feats[0].to_actions(np.asarray(acts[0], dtype=np.int64)),
                 feats[1].to_actions(np.asarray(acts[1], dtype=np.int64)))
        step_no += 1
    return T._terminal_u(sim, me, pre, episode_steps=int(CFG.get("episodeSteps", 300)))


def t_ci(vals, alpha=0.05):
    """Paired t interval (two-sided 95%). Falls back to a normal interval for
    large n; n < 2 -> (mean, nan, nan)."""
    v = np.asarray(vals, dtype=np.float64)
    n = len(v)
    mean = float(v.mean()) if n else float("nan")
    if n < 2:
        return mean, float("nan"), float("nan")
    sd = float(v.std(ddof=1))
    # t critical values for common df, else 1.96 (n large in gate practice)
    from statistics import NormalDist
    z = NormalDist().inv_cdf(1 - alpha / 2)
    half = z * sd / math.sqrt(n) if n > 60 else _t_crit(n - 1) * sd / math.sqrt(n)
    return mean, mean - half, mean + half


def _t_crit(df):
    table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
             7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 12: 2.179, 15: 2.131,
             20: 2.086, 25: 2.060, 30: 2.042, 40: 2.021, 50: 2.009, 60: 1.990}
    if df in table:
        return table[df]
    keys = sorted(table)
    for k in keys:
        if df <= k:
            return table[k]
    return 1.96


def bootstrap_ci(vals, seed, n_boot=10000):
    v = np.asarray(vals, dtype=np.float64)
    if len(v) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    means = v[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ----------------------------- workers -------------------------------------

_W = {}


def _init_worker(weights_path):
    w = dict(np.load(weights_path))
    _W["pol"] = CachePolicy(w)


def _q_scene(task):
    spec, cfg = task
    scene = extract_scene(spec)
    pol = _W["pol"]
    r = T.evaluate_root(_fresh_sim(scene), scene["me"], pol,
                        scene["x_me"], scene["aux_me"], scene["x_opp"], scene["aux_opp"],
                        _ns(scene["fm"]), _ns(scene["fo"]), teacher=1,
                        K=cfg["K"], M=cfg["M"], H=cfg["H"],
                        seed_base=cfg["seed_teacher"] + spec["step"])
    q = np.asarray(r["q"])
    Q = np.asarray(r["Q_mean"])
    best = int(np.argmax(q))
    gain = float(Q[best] - Q[0])
    # The gate must not certify the whole q from its argmax alone: also score
    # the distribution itself -- expected Q under q (the soft behaviour the
    # distillation actually trains towards), and under q**tau renormalised
    # when generation-time temperature tau is in play.
    soft = float(q @ Q - Q[0])
    tau = cfg.get("behavior_temp", 1.0)
    if tau != 1.0:
        qt = np.power(np.maximum(q, 1e-12), tau)
        qt = qt / qt.sum()
        soft_t = float(qt @ Q - Q[0])
    else:
        soft_t = soft
    return {"game": spec["game"], "step": spec["step"], "me": spec["me"],
            "gain": gain, "improved": bool(best != 0 and gain > 0),
            "soft_gain": soft, "soft_gain_behavior": soft_t,
            "entropy": float(r["entropy"]), "agree": bool(r["agree_greedy"]),
            "best_arm": [int(v) for v in r["arms"][best]],
            "wall_ms": r["wall_ms"], "guard": r["guard"]}


def _fresh_sim(scene):
    sim = FastCarbon(CFG)
    sim.load_observation(scene["full"])
    return sim


def _anchor_task(task):
    spec, cfg = task
    scene = extract_scene(spec)
    pol = _W["pol"]
    r = T.evaluate_root(_fresh_sim(scene), scene["me"], pol,
                        scene["x_me"], scene["aux_me"], scene["x_opp"], scene["aux_opp"],
                        _ns(scene["fm"]), _ns(scene["fo"]), teacher=1,
                        K=cfg["K"], M=cfg["M"], H=cfg["H"],
                        seed_base=cfg["seed_teacher"] + spec["step"])
    best_labels = r["arms"][int(np.argmax(r["q"]))]
    greedy_labels = r["arms"][0]
    u_q = continue_game(pol, scene, best_labels, cfg["seed_cont"] + spec["step"])
    u_g = continue_game(pol, scene, greedy_labels, cfg["seed_cont"] + spec["step"])
    return {"game": spec["game"], "step": spec["step"], "me": spec["me"],
            "du": u_q - u_g, "u_q": u_q, "u_g": u_g}


def _du_scene(task):
    spec, cfg = task
    scene = extract_scene(spec)
    pol = _W["pol"]
    stats = {}
    u_guided = continue_game(pol, scene, None, cfg["seed_cont"] + spec["step"],
                             guided_KMH=cfg, stats=stats)
    u_greedy = continue_game(pol, scene, None, cfg["seed_cont"] + spec["step"])
    return {"game": spec["game"], "step": spec["step"], "me": spec["me"],
            "du": u_guided - u_greedy, "u_guided": u_guided, "u_greedy": u_greedy,
            "teacher_calls": stats.get("teacher_calls", 0),
            "teacher_ms": stats.get("teacher_ms", 0.0)}


def run_tasks(fn, tasks, workers):
    if workers and workers > 1:
        with ProcessPoolExecutor(max_workers=workers,
                                 initializer=_init_worker,
                                 initargs=(_W["weights_path"],)) as ex:
            return list(ex.map(fn, tasks))
    _init_worker(_W["weights_path"])
    return [fn(t) for t in tasks]


# ----------------------------- main ----------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--replays", required=True)
    ap.add_argument("--split-manifest", required=True)
    ap.add_argument("--mode", required=True, choices=["q", "du"])
    ap.add_argument("--ds-games-file", default=None,
                    help="training-corpus game-id list; overlap check enforced when given")
    ap.add_argument("--seed-scene", type=int, default=995300)
    ap.add_argument("--seed-teacher", type=int, default=7000)
    ap.add_argument("--seed-cont", type=int, default=996100)
    ap.add_argument("--seed-anchor", type=int, default=995310)
    ap.add_argument("--K", type=int, default=CANON["K"])
    ap.add_argument("--M", type=int, default=CANON["M"])
    ap.add_argument("--H", type=int, default=CANON["H"])
    ap.add_argument("--n-anchor", type=int, default=ANCHOR_N)
    ap.add_argument("--behavior-temp", type=float, default=1.0,
                    help="temperature applied to q at generation time, if any; "
                         "the gate scores that behaviour distribution too")
    ap.add_argument("--pilot", action="store_true", help="du mode: Stage A, no verdict")
    ap.add_argument("--n-required", type=int, default=150)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--max-scenes", type=int, default=0, help="diagnostic cap (forces DIAGNOSTIC ONLY)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    _W["weights_path"] = a.weights
    t_start = time.time()

    split = json.load(open(a.split_manifest))
    q_games, du_games = split.get("q", []), split.get("du", [])
    overlap = sorted(set(q_games) & set(du_games))
    if overlap:
        raise SystemExit(f"split manifest overlap between q and du sets: {overlap[:5]}")
    games = q_games if a.mode == "q" else du_games
    if not games:
        raise SystemExit(f"split manifest has no games for mode {a.mode}")
    ds_check = "NOT RUN"
    if a.ds_games_file:
        ds_ids = {ln.strip() for ln in open(a.ds_games_file) if ln.strip()}
        bad = sorted(set(games) & ds_ids)
        if bad:
            raise SystemExit(f"scene games overlap the training corpus: {bad[:5]}")
        ds_check = f"RUN, disjoint (ds n={len(ds_ids)})"
    paths = list_replays(a.replays, games)
    layers = LAYERS_Q if a.mode == "q" else LAYERS_DU
    specs = sample_scene_specs(paths, layers, a.seed_scene,
                               opening_share=DU_OPENING_SHARE if a.mode == "du" else None)
    capped = False
    if a.max_scenes and len(specs) > a.max_scenes:
        specs = specs[:a.max_scenes]
        capped = True
    canonical = (a.K, a.M, a.H) == (CANON["K"], CANON["M"], CANON["H"]) and not capped
    cfg = {"K": a.K, "M": a.M, "H": a.H, "seed_teacher": a.seed_teacher,
           "seed_cont": a.seed_cont, "behavior_temp": a.behavior_temp}
    header = {
        "gate": "q" if a.mode == "q" else "du",
        "weights": a.weights, "weights_md5": md5_of(a.weights),
        "code_md5": {fn: md5_of(os.path.join(HERE, fn)) for fn in
                     ("stage0_teacher.py", "cnet2_np.py", "cnet2_search.py",
                      "stage1_teacher_gate.py")},
        "guard_cfg": T.TEACHER_GUARD_CFG, "K": a.K, "M": a.M, "H": a.H,
        "behavior_temp": a.behavior_temp,
        "seeds": {"scene": a.seed_scene, "teacher": a.seed_teacher,
                  "cont": a.seed_cont, "anchor": a.seed_anchor},
        "games": games, "n_scenes": len(specs), "ds_overlap_check": ds_check,
        "canonical_config": canonical,
    }
    print("PREREG", json.dumps(header), flush=True)
    out = {"prereg": header}

    if a.mode == "q":
        rows = run_tasks(_q_scene, [(s, cfg) for s in specs], a.workers)
        gains = np.array([r["gain"] for r in rows])
        rate = float(np.mean([r["improved"] for r in rows])) if rows else float("nan")
        res = {"n": len(rows), "q_gain_mean": float(gains.mean()),
               "q_soft_gain_mean": float(np.mean([r["soft_gain"] for r in rows])),
               "q_soft_gain_behavior_mean": float(np.mean(
                   [r["soft_gain_behavior"] for r in rows])),
               "behavior_temp": a.behavior_temp,
               "strict_improve_rate": rate,
               "entropy_mean": float(np.mean([r["entropy"] for r in rows])),
               "agree_rate": float(np.mean([r["agree"] for r in rows]))}
        rng = np.random.default_rng(a.seed_anchor)
        idx = rng.choice(len(specs), size=min(a.n_anchor, len(specs)), replace=False)
        anchor_specs = [specs[int(i)] for i in idx]
        arows = run_tasks(_anchor_task, [(s, cfg) for s in anchor_specs], a.workers)
        dus = [r["du"] for r in arows]
        m_, lo_, hi_ = t_ci(dus)
        res["anchor"] = {"n": len(arows), "du_mean": m_, "du_ci95": [lo_, hi_]}
        verdict = "DIAGNOSTIC ONLY"
        if canonical and len(rows) >= 150 and a.n_anchor >= ANCHOR_N:
            if res["q_gain_mean"] <= Q_FAIL["gain"] or rate < Q_FAIL["rate"]:
                verdict = "FAIL"
            elif (res["q_gain_mean"] >= Q_PASS["gain"] and rate >= Q_PASS["rate"]
                    and m_ > 0):
                verdict = "PASS"
            else:
                verdict = "INDETERMINATE"
        res["verdict"] = verdict
        out["q_gate"] = res
        out["rows"] = rows
        out["anchor_rows"] = arows
    else:
        rows = run_tasks(_du_scene, [(s, cfg) for s in specs], a.workers)
        dus = np.array([r["du"] for r in rows])
        m_, lo_, hi_ = t_ci(dus)
        blo, bhi = bootstrap_ci(dus, a.seed_cont)
        calls = sum(r["teacher_calls"] for r in rows)
        ms = sum(r["teacher_ms"] for r in rows)
        res = {"n": len(rows), "du_mean": float(m_), "du_ci95_t": [lo_, hi_],
               "du_ci95_boot": [blo, bhi], "sigma": float(dus.std(ddof=1)) if len(dus) > 1 else None,
               "teacher_ms_per_call": (ms / calls) if calls else None,
               "teacher_calls": calls}
        if a.pilot:
            if res["sigma"]:
                n_req = int(math.ceil((140.0 * res["sigma"]) ** 2))
                res["stageB_N_formula"] = max(150, min(600, n_req))
                res["stageB_N_raw"] = n_req
            res["verdict"] = "PILOT (no verdict)"
        else:
            verdict = "DIAGNOSTIC ONLY"
            if canonical and len(rows) >= a.n_required:
                verdict = "PASS" if (m_ >= DU_PASS and lo_ > 0) else "FAIL"
            res["verdict"] = verdict
        out["du_gate"] = res
        out["rows"] = rows

    out["wall_s"] = round(time.time() - t_start, 1)
    print("GATE-RESULT", json.dumps({k: v for k, v in out.items() if k != "rows"}), flush=True)
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())

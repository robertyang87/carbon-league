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
         the HARD gain Q(argmax q) - Q(greedy) and the SOFT gain
         E_q[Q] - Q(greedy) under the behaviour distribution q**tau
         (tau = --behavior-temp). Independent anchor on a seeded subset:
         EVERY candidate arm is continued to a real terminal (frozen
         continuation, common seeds), giving both the hard anchor
         (argmax arm vs greedy arm) and the soft anchor
         (behaviour-weighted mean U vs greedy arm). Prereg verdict
         (formal runs only, see below): PASS iff hard gain >= +0.01 U,
         strict-improvement rate >= 25%, hard anchor > 0, soft gain > 0
         and soft anchor > 0. FAIL iff hard gain <= 0, rate < 10%,
         soft gain <= 0, or the soft anchor's clustered CI upper bound
         < 0. Else INDETERMINATE (one seeded doubling allowed by the
         spec, never silent re-runs). Q_select (teacher Q, used to pick
         arms) and Q_eval (real terminal U, anchor) are recorded
         separately in the header -- the gate never certifies the soft
         target from its argmax alone (S02).
  du  -- paired continuations per root scene: guided arm asks the teacher
         every own step and plays argmax q (this certifies the HARD
         guided policy, not the soft target), greedy arm plays guarded
         greedy; opponent is the same bare net, guarded greedy, common
         seeds per root. All intervals are CLUSTERED BY SOURCE GAME
         (roots from one game are not independent samples, S03): the
         point estimate and t CI are computed over per-game means, with
         a cluster bootstrap as sensitivity. --pilot (Stage A): no
         verdict; writes a signed pilot artifact locking sigma_game and
         Stage B's required number of GAMES. Formal Stage B must consume
         that artifact (--pilot-artifact), on games disjoint from the
         pilot's, with n_games >= the locked N. Verdict: PASS iff the
         game-level ΔU >= +0.02 and its clustered t-CI lower bound > 0.

Formal-run gating (S06/S07/S08): a verdict other than DIAGNOSTIC ONLY
requires the canonical teacher config, an enforced training-set
isolation check (--ds-games-file, non-empty, stem namespace, content
hash recorded), duplicate-free split manifests, unique roots, and (du)
a valid pilot artifact. Anything missing degrades to DIAGNOSTIC ONLY.
Exit code: 1 iff the verdict is FAIL, else 0; --out is written
atomically. Teacher style defaults to the production value 2
(producer CONF TEACHER=2), recorded in the header. Root Feat objects
are fresh (isolated-root scope): guard history enters continuations
through guard snapshots evolved inside the continuation (S04).

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
    layer, sampled with the prereg scene RNG. Root identity is
    (game, step, me); duplicates are removed and counted.

    opening_share (ΔU gate, S09): the share of opening-layer roots in the
    TOTAL root count. With L non-opening layers per game, a fraction f of
    games adding one opening root gives share f/(L+f); the sampler solves
    f = share*L/(1-share) (share .10, L=3 -> f=1/3 of games)."""
    rng = np.random.default_rng(seed)
    specs = []
    open_games = set()
    if opening_share is not None:
        n_open = int(round(opening_share * (len(layers) - 1) / (1.0 - opening_share)
                           * len(game_paths)))
        order = list(range(len(game_paths)))
        rng.shuffle(order)
        open_games = set(order[:max(0, n_open)])
    for gi, gp in enumerate(game_paths):
        with gzip.open(gp) as fh:
            d = json.load(fh)
        n_steps = len(d["steps"]) - 1  # step t uses actions of steps[t+1]
        use_layers = layers if gi in open_games else layers[1:]
        if opening_share is None:
            use_layers = layers
        for (lo, hi) in use_layers:
            hi_c = min(hi, n_steps - 1)
            if hi_c <= lo:
                continue
            t = int(rng.integers(lo, hi_c))
            me = int(rng.integers(0, 2))
            specs.append({"game": game_id_of(gp), "path": gp, "step": t, "me": me})
    rng.shuffle(specs)
    seen, uniq = set(), []
    for s in specs:
        key = (s["game"], s["step"], s["me"])
        if key not in seen:
            seen.add(key)
            uniq.append(s)
    return uniq


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
                # S04: the teacher sees each seat's guard history AS EVOLVED
                # inside this continuation (snapshots of the live guards),
                # not a fresh guard -- GUARD_NARROW depends on lostw.
                snaps = (guards[me].snapshot(), guards[other].snapshot())
                r = T.evaluate_root(
                    sim, me, pol, copy.deepcopy(xs[me]), copy.deepcopy(feats[me].aux),
                    copy.deepcopy(xs[other]), copy.deepcopy(feats[other].aux),
                    _ns(feats[me]), _ns(feats[other]), teacher=teacher,
                    K=guided_KMH["K"], M=guided_KMH["M"], H=guided_KMH["H"],
                    seed_base=cont_seed + step_no, guard_snaps=snaps)
                stats["teacher_calls"] = stats.get("teacher_calls", 0) + 1
                stats["teacher_ms"] = stats.get("teacher_ms", 0.0) + r["wall_ms"]
                acts[seat] = np.asarray(r["arms"][int(np.argmax(r["q"]))], dtype=np.int64)
                # Advance the live guard's state with the executed arm
                # (state evolution only -- lostw/prevw update at apply()
                # entry; the returned labels are discarded, the arm was
                # already guard-processed inside the teacher).
                guards[me].reset_step()
                guards[me].apply(xs[me], feats[me].aux, acts[seat],
                                 np.zeros((F.OWN, 5), dtype=np.float32),
                                 pol, feats[me].own_uid, teacher, {})
            elif seat == me and step_no == 0 and first_labels is not None:
                acts[seat] = np.asarray(first_labels, dtype=np.int64)
                # Keep the seat's guard history in step with the executed
                # first move (state evolution only, output discarded).
                guards[me].reset_step()
                guards[me].apply(xs[me], feats[me].aux, acts[seat],
                                 np.zeros((F.OWN, 5), dtype=np.float32),
                                 pol, feats[me].own_uid, teacher, {})
            else:
                acts[seat] = _guarded_greedy(pol, guards[seat], feats[seat],
                                             xs[seat], teacher, stats)
        pre = sim.get_state()
        sim.step(feats[0].to_actions(np.asarray(acts[0], dtype=np.int64)),
                 feats[1].to_actions(np.asarray(acts[1], dtype=np.int64)))
        step_no += 1
    return T._terminal_u(sim, me, pre, episode_steps=int(CFG.get("episodeSteps", 300)))


def _t_crit(df, alpha=0.05):
    """Exact t critical value via scipy when available; the stdlib-only
    fallback uses a table and is labelled approximate at the call site."""
    try:
        from scipy import stats as _st
        return float(_st.t.ppf(1 - alpha / 2, df)), True
    except Exception:
        table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
                 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 12: 2.179, 15: 2.131,
                 20: 2.086, 25: 2.060, 30: 2.042, 40: 2.021, 50: 2.009, 60: 1.990}
        if df in table:
            return table[df], False
        for k in sorted(table):
            if df <= k:
                return table[k], False
        return 1.96, False


def t_ci(vals, alpha=0.05):
    """t interval (two-sided 95%) over independent units; n < 2 ->
    (mean, None, None). Returns (mean, lo, hi, exact_flag)."""
    v = np.asarray(vals, dtype=np.float64)
    n = len(v)
    mean = float(v.mean()) if n else None
    if n < 2:
        return mean, None, None, False
    sd = float(v.std(ddof=1))
    crit, exact = _t_crit(n - 1, alpha)
    half = crit * sd / math.sqrt(n)
    return mean, mean - half, mean + half, exact


def bootstrap_ci(vals, seed, n_boot=10000):
    """Row-level percentile bootstrap (sensitivity only -- the clustered
    interval in cluster_stats is the inferential one)."""
    v = np.asarray(vals, dtype=np.float64)
    if len(v) < 2:
        return None, None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    means = v[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def cluster_stats(rows, value_key="du"):
    """S03: roots from one source game are NOT independent. Aggregate to
    per-game means and compute the interval over games. Returns dict with
    root-level and game-level summaries."""
    by_game = {}
    for r in rows:
        by_game.setdefault(r["game"], []).append(float(r[value_key]))
    game_means = np.array([np.mean(v) for v in by_game.values()])
    m, lo, hi, exact = t_ci(game_means)
    root_vals = np.array([float(r[value_key]) for r in rows])
    return {"n_rows": len(rows), "n_games": len(by_game),
            "root_mean": float(root_vals.mean()) if len(root_vals) else None,
            "game_mean": m, "game_ci95_t": [lo, hi], "t_exact": exact,
            "sigma_game": float(game_means.std(ddof=1)) if len(game_means) > 1 else None,
            "sigma_root": float(root_vals.std(ddof=1)) if len(root_vals) > 1 else None}


def cluster_bootstrap_ci(rows, seed, value_key="du", n_boot=10000):
    """Cluster bootstrap: resample GAMES with replacement, keep all their
    roots, average per-game means."""
    by_game = {}
    for r in rows:
        by_game.setdefault(r["game"], []).append(float(r[value_key]))
    games = list(by_game)
    if len(games) < 2:
        return None, None
    gm = np.array([np.mean(by_game[g]) for g in games])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(games), size=(n_boot, len(games)))
    means = gm[idx].mean(axis=1)
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
                        _ns(scene["fm"]), _ns(scene["fo"]),
                        teacher=cfg.get("teacher_style", 2),
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
    """S02: continue EVERY candidate arm to a real terminal under the frozen
    continuation (same seed for all arms of the scene -- continuations are
    deterministic given the first move, so pairing is exact). Reports the
    hard anchor (argmax-q arm vs greedy arm) and the soft anchor
    (behaviour-distribution-weighted U vs greedy arm)."""
    spec, cfg = task
    scene = extract_scene(spec)
    pol = _W["pol"]
    style = cfg.get("teacher_style", 2)
    r = T.evaluate_root(_fresh_sim(scene), scene["me"], pol,
                        scene["x_me"], scene["aux_me"], scene["x_opp"], scene["aux_opp"],
                        _ns(scene["fm"]), _ns(scene["fo"]), teacher=style,
                        K=cfg["K"], M=cfg["M"], H=cfg["H"],
                        seed_base=cfg["seed_teacher"] + spec["step"])
    q = np.asarray(r["q"], dtype=np.float64)
    tau = cfg.get("behavior_temp", 1.0)
    if tau != 1.0:
        bw = np.power(np.maximum(q, 1e-12), tau)
        bw = bw / bw.sum()
    else:
        bw = q
    us = [continue_game(pol, scene, arm, cfg["seed_cont"] + spec["step"],
                        teacher=style) for arm in r["arms"]]
    best = int(np.argmax(q))
    u_soft = float(bw @ np.array(us))
    return {"game": spec["game"], "step": spec["step"], "me": spec["me"],
            "du": us[best] - us[0], "u_q": us[best], "u_g": us[0],
            "du_soft": u_soft - us[0], "u_soft": u_soft,
            "n_arms": len(us)}


def _du_scene(task):
    spec, cfg = task
    scene = extract_scene(spec)
    pol = _W["pol"]
    style = cfg.get("teacher_style", 2)
    stats = {}
    u_guided = continue_game(pol, scene, None, cfg["seed_cont"] + spec["step"],
                             teacher=style, guided_KMH=cfg, stats=stats)
    u_greedy = continue_game(pol, scene, None, cfg["seed_cont"] + spec["step"],
                             teacher=style)
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


def _sanitize(o):
    """NaN/inf -> None so output is strict JSON."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_sanitize(v) for v in o]
    return o


def _atomic_dump(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(_sanitize(obj), fh, indent=1, allow_nan=False)
    os.replace(tmp, path)


def _verify_pilot_artifact(path, header, formal_games):
    """S08: a formal du run must consume the Stage A artifact: identity
    hash, weights/code/config equality, games disjoint from the pilot's,
    and the game count must reach the locked N (checked by the caller via
    the returned locked N in the reason string on success)."""
    if not path:
        return False, 0, "missing --pilot-artifact"
    try:
        art = json.load(open(path))
    except Exception as e:
        return False, 0, f"unreadable artifact: {e}"
    body = {k: v for k, v in art.items() if k != "artifact_id"}
    want = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    if art.get("artifact_id") != want:
        return False, 0, "artifact_id hash mismatch"
    if art.get("weights_md5") != header["weights_md5"]:
        return False, 0, "weights differ from pilot"
    if art.get("code_md5") != header["code_md5"]:
        return False, 0, "code differs from pilot"
    for k in ("K", "M", "H", "teacher_style", "behavior_temp"):
        hv = header["K"] if k == "K" else header["M"] if k == "M" else \
            header["H"] if k == "H" else header[k]
        if art.get(k) != hv:
            return False, 0, f"config field {k} differs from pilot"
    both = sorted(set(art.get("pilot_games", [])) & set(formal_games))
    if both:
        return False, 0, f"formal games overlap pilot games: {both[:5]}"
    locked = int(art.get("locked_N_games", 0))
    return True, locked, f"ok (locked_N_games={locked})"


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
    ap.add_argument("--pilot-artifact", default=None,
                    help="du formal: Stage A artifact JSON written by a --pilot run")
    ap.add_argument("--n-required", type=int, default=150)
    ap.add_argument("--teacher-style", type=int, default=2,
                    help="policy style slot; production producer uses 2")
    ap.add_argument("--train-data-version", default=None,
                    help="identity of the training corpus the ds manifest belongs to")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--max-scenes", type=int, default=0,
                    help="cap on sampled scenes; a run actually capped by it is DIAGNOSTIC ONLY")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    _W["weights_path"] = a.weights
    t_start = time.time()

    def validate_ids(ids, what):
        """S06/S07: ids must be bare replay stems, duplicate-free."""
        for g in ids:
            if not g or any(c in g for c in "./\\ ") or g != g.strip():
                raise SystemExit(f"{what}: id {g!r} is not a bare game stem")
        dup = sorted({g for g in ids if ids.count(g) > 1})
        if dup:
            raise SystemExit(f"{what}: duplicate game ids: {dup[:5]}")
        return list(ids)

    split = json.load(open(a.split_manifest))
    q_games = validate_ids(split.get("q", []), "split manifest q")
    du_games = validate_ids(split.get("du", []), "split manifest du")
    overlap = sorted(set(q_games) & set(du_games))
    if overlap:
        raise SystemExit(f"split manifest overlap between q and du sets: {overlap[:5]}")
    games = q_games if a.mode == "q" else du_games
    if not games:
        raise SystemExit(f"split manifest has no games for mode {a.mode}")
    ds_check = "NOT RUN"
    ds_md5 = None
    if a.ds_games_file:
        raw = open(a.ds_games_file, "rb").read()
        ds_md5 = hashlib.md5(raw).hexdigest()
        ds_ids = validate_ids([ln.strip() for ln in raw.decode().splitlines() if ln.strip()],
                                "ds manifest")
        if not ds_ids:
            # An empty manifest proves nothing: the run completes as a
            # diagnostic, it can never be formal (S06).
            ds_check = "NOT RUN (empty ds manifest)"
        else:
            bad = sorted(set(games) & set(ds_ids))
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
           "seed_cont": a.seed_cont, "behavior_temp": a.behavior_temp,
           "teacher_style": a.teacher_style}
    code_md5 = {fn: md5_of(os.path.join(HERE, fn)) for fn in
                ("stage0_teacher.py", "cnet2_np.py", "cnet2_search.py",
                 "stage1_teacher_gate.py")}
    reasons = []
    if not canonical:
        reasons.append("non-canonical teacher config or capped scenes")
    if not ds_check.startswith("RUN, disjoint"):
        reasons.append("training-set isolation check not run (S06)")
    formal = not reasons
    header = {
        "gate": "q" if a.mode == "q" else "du",
        "run_id": hashlib.sha256(f"{time.time()}-{os.getpid()}".encode()).hexdigest()[:16],
        "weights": a.weights, "weights_md5": md5_of(a.weights),
        "code_md5": code_md5,
        "guard_cfg": T.TEACHER_GUARD_CFG, "K": a.K, "M": a.M, "H": a.H,
        "behavior_temp": a.behavior_temp, "teacher_style": a.teacher_style,
        "q_select_source": "teacher Q (stage0_teacher.evaluate_root)",
        "q_eval_source": "real terminal continuations (anchor / paired arms)",
        "root_feat_history": "fresh Feats at root (isolated-root scope)",
        "seeds": {"scene": a.seed_scene, "teacher": a.seed_teacher,
                  "cont": a.seed_cont, "anchor": a.seed_anchor},
        "games": games, "n_scenes": len(specs), "ds_overlap_check": ds_check,
        "ds_manifest_md5": ds_md5, "train_data_version": a.train_data_version,
        "n_required": a.n_required,
        "canonical_config": canonical, "formal": formal,
        "diagnostic_reasons": reasons,
    }
    print("PREREG", json.dumps(header), flush=True)
    out = {"prereg": header}

    verdict = "DIAGNOSTIC ONLY"
    if a.mode == "q":
        rows = run_tasks(_q_scene, [(s, cfg) for s in specs], a.workers)
        gains = np.array([r["gain"] for r in rows])
        rate = float(np.mean([r["improved"] for r in rows])) if rows else float("nan")
        res = {"n": len(rows), "n_games": len({r["game"] for r in rows}),
               "q_gain_mean": float(gains.mean()),
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
        hard_c = cluster_stats(arows, "du")
        soft_c = cluster_stats(arows, "du_soft")
        res["anchor"] = {"n": len(arows), "hard": hard_c, "soft": soft_c,
                         "soft_ci95_boot_cluster": list(
                             cluster_bootstrap_ci(arows, a.seed_anchor, "du_soft"))}
        soft_gain_b = res["q_soft_gain_behavior_mean"]
        soft_hi = res["anchor"]["soft_ci95_boot_cluster"][1]
        if formal and len(rows) >= 150 and a.n_anchor >= ANCHOR_N:
            if (res["q_gain_mean"] <= Q_FAIL["gain"] or rate < Q_FAIL["rate"]
                    or soft_gain_b <= 0
                    or (soft_hi is not None and soft_hi < 0)):
                verdict = "FAIL"
            elif (res["q_gain_mean"] >= Q_PASS["gain"] and rate >= Q_PASS["rate"]
                    and hard_c["root_mean"] > 0 and soft_gain_b > 0
                    and soft_c["root_mean"] > 0):
                verdict = "PASS"
            else:
                verdict = "INDETERMINATE"
        res["verdict"] = verdict
        out["q_gate"] = res
        out["rows"] = rows
        out["anchor_rows"] = arows
    else:
        rows = run_tasks(_du_scene, [(s, cfg) for s in specs], a.workers)
        cs = cluster_stats(rows, "du")
        blo, bhi = cluster_bootstrap_ci(rows, a.seed_cont, "du")
        calls = sum(r["teacher_calls"] for r in rows)
        ms = sum(r["teacher_ms"] for r in rows)
        res = {"n": cs["n_rows"], "n_games": cs["n_games"],
               "du_mean_roots": cs["root_mean"], "du_mean": cs["game_mean"],
               "du_ci95_t_cluster": cs["game_ci95_t"], "t_exact": cs["t_exact"],
               "du_ci95_boot_cluster": [blo, bhi],
               "sigma_game": cs["sigma_game"], "sigma_root": cs["sigma_root"],
               "teacher_ms_per_call": (ms / calls) if calls else None,
               "teacher_calls": calls}
        if a.pilot:
            if cs["sigma_game"]:
                from statistics import NormalDist
                raw = int(math.ceil((140.0 * cs["sigma_game"]) ** 2))
                locked = max(50, min(200, raw))
                power = float(NormalDist().cdf(
                    0.02 / (cs["sigma_game"] / math.sqrt(locked)) - 1.96))
                art = {"kind": "stage1_du_pilot_artifact",
                       "weights_md5": header["weights_md5"], "code_md5": code_md5,
                       "guard_cfg": T.TEACHER_GUARD_CFG,
                       "K": a.K, "M": a.M, "H": a.H,
                       "teacher_style": a.teacher_style,
                       "behavior_temp": a.behavior_temp,
                       "pilot_games": sorted({r["game"] for r in rows}),
                       "seeds": header["seeds"],
                       "sigma_game": cs["sigma_game"],
                       "sigma_root": cs["sigma_root"],
                       "locked_N_games_raw": raw, "locked_N_games": locked,
                       "power_at_locked_for_0.02": power}
                art["artifact_id"] = hashlib.sha256(json.dumps(
                    art, sort_keys=True).encode()).hexdigest()
                res["pilot_artifact"] = art
                res["stageB_N_games"] = locked
                res["stageB_N_formula"] = locked  # games unit (S03/S08)
                res["stageB_N_raw"] = raw
                if a.out:
                    _atomic_dump(art, a.out + ".pilot.json")
            res["verdict"] = "PILOT (no verdict)"
            verdict = res["verdict"]
        else:
            art_ok, locked_n, art_why = _verify_pilot_artifact(
                a.pilot_artifact, header, games)
            res["pilot_check"] = art_why
            if art_ok and cs["n_games"] < locked_n:
                art_ok = False
                res["pilot_check"] = (f"n_games {cs['n_games']} below locked "
                                      f"N {locked_n} (S08)")
            need_rows = max(a.n_required, 150)
            if formal and art_ok and len(rows) >= need_rows:
                gm, glo = cs["game_mean"], cs["game_ci95_t"][0]
                verdict = "PASS" if (gm is not None and gm >= DU_PASS
                                     and glo is not None and glo > 0) else "FAIL"
            elif formal and not art_ok:
                header["diagnostic_reasons"].append(f"pilot artifact: {art_why}")
            res["verdict"] = verdict
        out["du_gate"] = res
        out["rows"] = rows

    out["wall_s"] = round(time.time() - t_start, 1)
    print("GATE-RESULT", json.dumps(
        _sanitize({k: v for k, v in out.items() if k != "rows"}),
        allow_nan=False), flush=True)
    if a.out:
        _atomic_dump(out, a.out)
    return 1 if verdict == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())

"""K0 round 2 -- root-continuation paired measurement driver (design
draft K0_ROUND2_DESIGN_DRAFT_20261114 sec 4.2 / sec 10 item 3).

The teacher gate's `du` mode pairs a teacher-guided arm against a greedy
arm on the same root scene. This driver is its policy-vs-policy variant:
BOTH arms are bare nets and the teacher is never called. Per root, the
student arm plays the root seat (`me`) with the student net while the
opponent seat plays the PARENT net; the parent arm plays both seats with
the parent net. du = U_student - U_parent per root, aggregated clustered
by game (t interval primary, cluster bootstrap reported alongside) --
exactly the sec 4.2 estimand: the bare-net strength difference inside
the measured window, with the continuation environment (opponent =
parent bare net) held fixed, as in the pilot.

Why not call stage1_teacher_gate.continue_game directly: it binds ONE
policy object to both seats, so it cannot express "opponent = parent"
for the student arm. The continuation loop below (continue_game_2pol)
is a line-mapped adaptation of the gate's bare-net path (gate md5
e922d9d890c3e1eff6643e947289acb0, continue_game lines 304-373 with
guided_KMH=None and first_labels=None, the gate greedy-arm call shape):
same guard construction/restore, same step-0 observation handling, same
observe cadence, same terminal test, same terminal scoring. Its
fidelity is proven by fixture: with both seats on the parent net it
reproduces the gate's continue_game value-for-value.

No-teacher isolation (auditable): this module NEVER imports
stage1_teacher_gate or stage0_teacher. The pure continuation primitives
it needs are extracted from those files' SOURCE by AST allowlist and
compiled in a controlled namespace:

  from stage1_teacher_gate.py: md5_of, game_id_of, _ft_frame_obs,
    _ft_side_extinct, first_terminal_step, extract_scene, _fresh_sim,
    _guarded_greedy, _t_crit, t_ci, bootstrap_ci, cluster_stats,
    cluster_bootstrap_ci, _sanitize, _atomic_dump
  from stage0_teacher.py: _extinct_state, _terminal_u and the literal
    TEACHER_GUARD_CFG -- bound as a shim `T` exposing ONLY those three
    names (no evaluate_root: a guided call is structurally impossible).

An AST audit at import asserts the extracted gate functions reference
no `T` attribute beyond TEACHER_GUARD_CFG. The artifact header records
both source files' md5s, so the extraction basis is pinned per run.
Support modules (cnet2_feat / cnet2_np / cnet2_search / eval_v53 /
fastsim) import no teacher module (verified import graph).

Seeds: per-root continuation seed = seed_cont + step, identical for
both arms -- the gate du caliber (gate default seed_cont 995400). On
the bare-net path the seed is inert numerically (guarded greedy is
deterministic; in the gate it only reaches the teacher's seed_base),
but it is threaded and recorded per row for caliber parity.

Roots: explicit manifest (JSON list of {game, path, step, me}). Hard
validation: step inside [--step-min, --step-max] (default 200..299,
the registered window), step before the game's first-terminal frame
(padding roots are terminal scenes, not measurements -- hard die),
step usable by extract_scene (step < len(steps)-1), game id matches
the replay file name, no duplicate (game, step, me). The root-set
identity (roots_sha256 over {game, step, me, replay_md5}) is recorded
in the header and in every row.
"""

import argparse
import ast
import copy
import gzip
import hashlib
import json
import math
import os
import sys
import types

# Single-threaded BLAS per process (same posture as the gate).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import cnet2_feat as F  # noqa: E402
import cnet2_np as NP  # noqa: E402,F401  (gate import surface parity)
from cnet2_search import BankruptcyGuard, CachePolicy  # noqa: E402
from eval_v53 import CFG  # noqa: E402
from fastsim import FastCarbon  # noqa: E402

GATE_SRC = os.path.join(HERE, "stage1_teacher_gate.py")
TEACHER_SRC = os.path.join(HERE, "stage0_teacher.py")

_GATE_FUNCS = ["md5_of", "game_id_of", "_ft_frame_obs", "_ft_side_extinct",
               "first_terminal_step", "extract_scene", "_fresh_sim",
               "_guarded_greedy", "_t_crit", "t_ci", "bootstrap_ci",
               "cluster_stats", "cluster_bootstrap_ci", "_sanitize",
               "_atomic_dump"]
_TEACHER_FUNCS = ["_extinct_state", "_terminal_u"]
_TEACHER_ATTRS_ALLOWED = {"TEACHER_GUARD_CFG"}


def _file_md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _t_attrs(defs):
    """All attribute names accessed on the name `T` inside AST defs."""
    out = set()
    for d in defs:
        for node in ast.walk(d):
            if (isinstance(node, ast.Attribute)
                    and isinstance(node.value, ast.Name)
                    and node.value.id == "T"):
                out.add(node.attr)
    return out


def _extract_funcs(path, names, extra_globals):
    """Compile ONLY the named top-level functions from `path` in a
    controlled namespace. Fails loudly on a missing name."""
    with open(path) as fh:
        tree = ast.parse(fh.read(), path)
    defs = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            defs[node.name] = node
    missing = [n for n in names if n not in defs]
    if missing:
        raise SystemExit(f"k0_duel: extraction failed: {path} lacks {missing}")
    mod = ast.Module(body=[defs[n] for n in names], type_ignores=[])
    ns = dict(extra_globals)
    exec(compile(mod, path, "exec"), ns)
    return {n: ns[n] for n in names}, [defs[n] for n in names]


def _teacher_guard_cfg_literal(path):
    with open(path) as fh:
        tree = ast.parse(fh.read(), path)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "TEACHER_GUARD_CFG"
                for t in node.targets):
            call = node.value
            if not (isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Name)
                    and call.func.id == "dict" and not call.args):
                raise SystemExit("k0_duel: TEACHER_GUARD_CFG is not a "
                                 "literal dict(...) in stage0_teacher.py")
            return {kw.arg: ast.literal_eval(kw.value)
                    for kw in call.keywords}
    raise SystemExit("k0_duel: TEACHER_GUARD_CFG not found in "
                     "stage0_teacher.py")


# --- teacher shim: ONLY the pure terminal helpers + the guard cfg -------
_TF, _ = _extract_funcs(TEACHER_SRC, _TEACHER_FUNCS,
                        {"np": np, "math": math})
_T = types.SimpleNamespace(
    TEACHER_GUARD_CFG=_teacher_guard_cfg_literal(TEACHER_SRC),
    _extinct_state=_TF["_extinct_state"],
    _terminal_u=_TF["_terminal_u"])

# --- gate primitives, compiled against the shim --------------------------
_G, _GDEFS = _extract_funcs(GATE_SRC, _GATE_FUNCS, {
    "np": np, "math": math, "os": os, "json": json, "gzip": gzip,
    "hashlib": hashlib, "copy": copy, "types": types, "F": F,
    "CFG": CFG, "FastCarbon": FastCarbon,
    "BankruptcyGuard": BankruptcyGuard, "T": _T})
_seen_t = _t_attrs(_GDEFS)
if not _seen_t <= _TEACHER_ATTRS_ALLOWED:
    raise SystemExit(f"k0_duel: extracted gate functions reference "
                     f"unexpected T attributes {sorted(_seen_t)}")

md5_of = _G["md5_of"]
game_id_of = _G["game_id_of"]
first_terminal_step = _G["first_terminal_step"]
extract_scene = _G["extract_scene"]
_fresh_sim = _G["_fresh_sim"]
_guarded_greedy = _G["_guarded_greedy"]
cluster_stats = _G["cluster_stats"]
cluster_bootstrap_ci = _G["cluster_bootstrap_ci"]
_sanitize = _G["_sanitize"]
_atomic_dump = _G["_atomic_dump"]

SOURCE_MD5 = {"stage1_teacher_gate": _file_md5(GATE_SRC),
              "stage0_teacher": _file_md5(TEACHER_SRC)}


def continue_game_2pol(pol_me, pol_opp, scene, cont_seed, teacher_style=2,
                       stats=None):
    """Bare-net continuation of `scene` to terminal; U from the root
    seat. Line-mapped adaptation of the gate's continue_game bare path
    (see module docstring): the only change is per-seat policies --
    pol_me plays the root seat, pol_opp the opponent seat. The guided
    branch does not exist here (the shim carries no evaluate_root) and
    first_labels is always None, as in the gate's greedy-arm call.
    cont_seed is threaded for caliber parity with the gate du call
    shape; on the bare path it is numerically inert, exactly as in the
    gate's greedy arm."""
    stats = stats if stats is not None else {}
    me = scene["me"]
    pols = {me: pol_me, 1 - me: pol_opp}
    sim = _fresh_sim(scene)
    feats = {me: copy.deepcopy(scene["fm"]),
             1 - me: copy.deepcopy(scene["fo"])}
    guards = {0: BankruptcyGuard(_T.TEACHER_GUARD_CFG),
              1: BankruptcyGuard(_T.TEACHER_GUARD_CFG)}
    for seat in (0, 1):
        guards[seat].restore(scene["guard_snaps"][seat])
    pre = sim.get_state()
    step_no = 0
    while True:
        st = sim.get_state()
        if (_T._extinct_state(st, 0) or _T._extinct_state(st, 1)
                or sim.step_count >= 299):
            break
        if step_no == 0:
            xs = {me: copy.deepcopy(scene["x_me"]),
                  1 - me: copy.deepcopy(scene["x_opp"])}
        else:
            xs = {seat: feats[seat].observe(sim.obs(seat))
                  for seat in (0, 1)}
        acts = {}
        for seat in (0, 1):
            acts[seat] = _guarded_greedy(pols[seat], guards[seat],
                                         feats[seat], xs[seat],
                                         teacher_style, stats)
        pre = sim.get_state()
        sim.step(feats[0].to_actions(np.asarray(acts[0], dtype=np.int64)),
                 feats[1].to_actions(np.asarray(acts[1], dtype=np.int64)))
        step_no += 1
    return _T._terminal_u(sim, me, pre,
                          episode_steps=int(CFG.get("episodeSteps", 300)))


def _load_policy(path):
    return CachePolicy(dict(np.load(path)))


def load_roots(manifest_path, step_min, step_max):
    """Load and hard-validate the root manifest. Returns (specs,
    roots_sha256). Every spec gains ft / n_steps / replay_md5."""
    with open(manifest_path) as fh:
        raw = json.load(fh)
    if not isinstance(raw, list) or not raw:
        raise SystemExit("k0_duel: roots manifest must be a non-empty list")
    specs, seen = [], set()
    game_info = {}
    for i, s in enumerate(raw):
        try:
            game, path = str(s["game"]), str(s["path"])
            step, me = int(s["step"]), int(s["me"])
        except Exception:
            raise SystemExit(f"k0_duel: root #{i} lacks game/path/step/me")
        if me not in (0, 1):
            raise SystemExit(f"k0_duel: root #{i} me={me} not in (0, 1)")
        if not (step_min <= step <= step_max):
            raise SystemExit(
                f"k0_duel: root #{i} step {step} outside the registered "
                f"window [{step_min}, {step_max}]")
        if not os.path.isfile(path):
            raise SystemExit(f"k0_duel: root #{i} replay not found: {path}")
        if game_id_of(path) != game:
            raise SystemExit(
                f"k0_duel: root #{i} game {game!r} != replay file id "
                f"{game_id_of(path)!r}")
        key = (game, step, me)
        if key in seen:
            raise SystemExit(f"k0_duel: duplicate root {key}")
        seen.add(key)
        if path not in game_info:
            with gzip.open(path) as fh:
                d = json.load(fh)
            ft, _cause = first_terminal_step(d["steps"])
            game_info[path] = (ft, len(d["steps"]), md5_of(path))
        ft, n_steps, replay_md5 = game_info[path]
        if step >= ft:
            raise SystemExit(
                f"k0_duel: root #{i} step {step} >= first terminal {ft} "
                f"of {game}: terminal scene, not a measurement")
        if step >= n_steps - 1:
            raise SystemExit(
                f"k0_duel: root #{i} step {step} has no successor frame "
                f"in {game} (len {n_steps})")
        specs.append({"game": game, "path": path, "step": step, "me": me,
                      "ft": ft, "replay_md5": replay_md5})
    canon = sorted(({"game": s["game"], "step": s["step"], "me": s["me"],
                     "replay_md5": s["replay_md5"]} for s in specs),
                   key=lambda r: (r["game"], r["step"], r["me"]))
    roots_sha = hashlib.sha256(json.dumps(
        canon, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return specs, roots_sha


_W = {}


def _init_worker(student_path, parent_path):
    _W["pol_student"] = _load_policy(student_path)
    _W["pol_parent"] = _load_policy(parent_path)


def _duel_task(task):
    spec, cfg = task
    scene = extract_scene(spec)
    seed = cfg["seed_cont"] + spec["step"]
    u_parent = continue_game_2pol(_W["pol_parent"], _W["pol_parent"],
                                  scene, seed,
                                  teacher_style=cfg["teacher_style"])
    u_student = continue_game_2pol(_W["pol_student"], _W["pol_parent"],
                                   scene, seed,
                                   teacher_style=cfg["teacher_style"])
    return {"game": spec["game"], "step": spec["step"], "me": spec["me"],
            "seat": spec["me"], "ft": spec["ft"],
            "u_student": u_student, "u_parent": u_parent,
            "du": u_student - u_parent, "seed": seed,
            "student_md5": cfg["student_md5"],
            "parent_md5": cfg["parent_md5"],
            "roots_sha256": cfg["roots_sha256"]}


def run_duel(student_path, parent_path, specs, roots_sha256,
             seed_cont=995400, teacher_style=2, workers=1):
    cfg = {"seed_cont": seed_cont, "teacher_style": teacher_style,
           "student_md5": md5_of(student_path),
           "parent_md5": md5_of(parent_path),
           "roots_sha256": roots_sha256}
    tasks = [(s, cfg) for s in specs]
    if workers and workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers,
                                 initializer=_init_worker,
                                 initargs=(student_path,
                                           parent_path)) as ex:
            rows = list(ex.map(_duel_task, tasks))
    else:
        _init_worker(student_path, parent_path)
        rows = [_duel_task(t) for t in tasks]
    stats = cluster_stats(rows, "du")
    blo, bhi = cluster_bootstrap_ci(rows, seed_cont, "du")
    return {"kind": "k0_duel_result", "version": 1,
            "student": {"path": student_path, "md5": cfg["student_md5"]},
            "parent": {"path": parent_path, "md5": cfg["parent_md5"]},
            "roots_sha256": roots_sha256,
            "seed_cont": seed_cont, "teacher_style": teacher_style,
            "sources": dict(SOURCE_MD5),
            "summary": {"delta_u": stats["game_mean"],
                        "n_roots": stats["n_rows"],
                        "n_games": stats["n_games"],
                        "cluster": stats,
                        "cluster_bootstrap_ci95": [blo, bhi]},
            "rows": rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--student", required=True)
    ap.add_argument("--parent", required=True)
    ap.add_argument("--roots", required=True,
                    help="JSON manifest: [{game, path, step, me}]")
    ap.add_argument("--out", required=True)
    ap.add_argument("--step-min", type=int, default=200)
    ap.add_argument("--step-max", type=int, default=299)
    ap.add_argument("--seed-cont", type=int, default=995400)
    ap.add_argument("--teacher-style", type=int, default=2)
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    specs, roots_sha = load_roots(a.roots, a.step_min, a.step_max)
    result = run_duel(a.student, a.parent, specs, roots_sha,
                      seed_cont=a.seed_cont,
                      teacher_style=a.teacher_style, workers=a.workers)
    result["window"] = {"step_min": a.step_min, "step_max": a.step_max}
    _atomic_dump(_sanitize(result), a.out)
    s = result["summary"]
    print(f"k0_duel: {s['n_roots']} roots / {s['n_games']} games  "
          f"delta_u={s['delta_u']}  "
          f"t_ci95={s['cluster']['game_ci95_t']}  "
          f"boot_ci95={s['cluster_bootstrap_ci95']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

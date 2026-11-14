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

Finite gates (X07): corrupt inputs must die, never measure as if
healthy. Weights pass a full finite check at load (die naming the
file and the tensor); root scene features are finite-checked per
root; every produced U is finite-checked per arm, and every row is
re-checked before aggregation. Before these gates, a checkpoint
with NaN/Inf tensors measured "normally" through the real CLI:
paired against itself its du is exactly 0 and its U is a finite,
plausible number -- indistinguishable from a healthy same-weights
run -- and against a healthy parent it fabricated a clean-looking
nonzero delta_u. The gates check only; they touch no numerics.
The same discipline applies DURING inference: every step's policy
logits and value must be finite or the root's measurement is
rejected on the spot (see _greedy_checked) -- a corruption that
surfaces only in the inference outputs, with labels and even the
final U bit-identical to the healthy run, is otherwise invisible
to every output-side check.

Identity binding (X04): a weights file is read ONCE. run_duel
hashes the captured bytes for the header/rows, and the workers
parse exactly those bytes -- there is no second, by-path read for
a swap to slip between identity and consumption (record A, measure
B is structurally impossible, not merely checked for). Replays
are bound the same way: load_roots reads each replay once (md5,
first-terminal and length all derive from that single read) and
scene extraction consumes those captured bytes, materialized
under the original basename in a per-process scratch dir --
extract_scene reads only spec["path"], so the scratch path IS
the consumption binding. Before this, validating a root set and
then swapping a replay file changed the measured scene (cash and
all) while the rows kept the pre-swap roots_sha256.

One game, one content (X08): every root of a game must bind the
same replay_md5. The manifest check alone (basename == game id,
cache keyed by path) used to accept two different files named for
the same game and cluster their roots into a single "game";
identical bytes at different paths remain allowed.
"""

import argparse
import ast
import copy
import gzip
import hashlib
import io
import json
import math
import os
import sys
import tempfile
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


def _greedy_checked(pol, guard, feat, x, teacher, stats, where):
    """The gate's _guarded_greedy expanded inline, with the X07
    inference gate between act and apply: THIS step's policy logits
    and value must be finite, else the measurement is rejected on
    the spot. The expansion is call-for-call identical (act,
    reset_step, apply -- same objects, same order), so numerics are
    unchanged; only the check is added. The check is not redundant
    with the load gate: a single NaN head bias poisons the base
    slots' logits rows while the returned labels -- and, in the
    measured case, the final U itself -- stay plausible or even
    bit-identical to the healthy run, so no output-side check can
    see it. Only the inference surface can."""
    labels, logits, value = pol.act(x, feat.aux, teacher=teacher)
    if not (np.isfinite(logits).all() and math.isfinite(value)):
        bad = int(logits.size - np.isfinite(logits).sum())
        raise SystemExit(
            f"k0_duel: {where}: non-finite policy output at inference "
            f"(logits non-finite elements: {bad}, value: {value!r}): "
            f"corrupt inference must not be measured")
    guard.reset_step()
    return guard.apply(x, feat.aux, np.asarray(labels, dtype=np.int64),
                       logits, pol, feat.own_uid, teacher, stats)


def continue_game_2pol(pol_me, pol_opp, scene, cont_seed, teacher_style=2,
                       stats=None, where="continuation"):
    """Bare-net continuation of `scene` to terminal; U from the root
    seat. Line-mapped adaptation of the gate's continue_game bare path
    (see module docstring): the only change is per-seat policies --
    pol_me plays the root seat, pol_opp the opponent seat. The guided
    branch does not exist here (the shim carries no evaluate_root) and
    first_labels is always None, as in the gate's greedy-arm call.
    cont_seed is threaded for caliber parity with the gate du call
    shape; on the bare path it is numerically inert, exactly as in the
    gate's greedy arm. The greedy step itself is the gate's
    _guarded_greedy expanded inline (see _greedy_checked) so the X07
    inference gate can sit between act and apply; the terminal-U
    equivalence fixture proves the expansion is value-identical."""
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
            acts[seat] = _greedy_checked(
                pols[seat], guards[seat], feats[seat], xs[seat],
                teacher_style, stats,
                f"{where} seat {seat} step {step_no}")
        pre = sim.get_state()
        sim.step(feats[0].to_actions(np.asarray(acts[0], dtype=np.int64)),
                 feats[1].to_actions(np.asarray(acts[1], dtype=np.int64)))
        step_no += 1
    return _T._terminal_u(sim, me, pre,
                          episode_steps=int(CFG.get("episodeSteps", 300)))


def _read_weights_bytes(path):
    """The single read of a weights file (X04): every later use --
    md5 identity, finite gate, worker consumption -- works from
    these captured bytes, never from a second by-path read."""
    if not os.path.isfile(path):
        raise SystemExit(f"k0_duel: weights file not found: {path}")
    with open(path, "rb") as fh:
        return fh.read()


def _policy_from_bytes(path, blob):
    """Parse captured weights bytes behind the X07 finite gate: the
    bytes must parse as an npz of tensors and EVERY floating tensor
    must be entirely finite -- a single NaN/Inf element dies here,
    naming the file and the tensor, before the corrupt checkpoint
    can be measured as if healthy. Non-numeric entries (e.g. the
    string `meta`) are metadata, not weights, and pass. Checks only:
    the CachePolicy receives exactly the tensors np.load produced.
    `path` is carried for die messages; the bytes are the input."""
    try:
        tensors = dict(np.load(io.BytesIO(blob)))
    except Exception as exc:
        raise SystemExit(f"k0_duel: weights file {path} unreadable: "
                         f"{type(exc).__name__}: {exc}")
    if not tensors:
        raise SystemExit(f"k0_duel: weights file {path} holds no tensors")
    for name in sorted(tensors):
        arr = tensors[name]
        if arr.dtype.kind in "fc":
            bad = int(arr.size - np.isfinite(arr).sum())
            if bad:
                raise SystemExit(
                    f"k0_duel: weights file {path} tensor {name!r} has "
                    f"{bad} non-finite element(s) (NaN/Inf): corrupt "
                    f"weights must not be measured")
    return CachePolicy(tensors)


def _load_policy(path):
    return _policy_from_bytes(path, _read_weights_bytes(path))


def load_roots(manifest_path, step_min, step_max):
    """Load and hard-validate the root manifest. Returns (specs,
    roots_sha256, replay_blobs). Every spec gains ft / n_steps /
    replay_md5; replay_blobs maps each unique replay path to the
    exact bytes that were validated (X04: scene extraction later
    consumes these bytes, never a second by-path read). One game id
    must bind ONE replay content (X08): two paths carrying the same
    game with different bytes die here; identical bytes at different
    paths are allowed."""
    with open(manifest_path) as fh:
        raw = json.load(fh)
    if not isinstance(raw, list) or not raw:
        raise SystemExit("k0_duel: roots manifest must be a non-empty list")
    specs, seen = [], set()
    game_info = {}
    replay_blobs = {}
    game_binding = {}
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
            with open(path, "rb") as fh:
                blob = fh.read()
            replay_md5 = hashlib.md5(blob).hexdigest()
            d = json.loads(gzip.decompress(blob))
            ft, _cause = first_terminal_step(d["steps"])
            game_info[path] = (ft, len(d["steps"]), replay_md5)
            replay_blobs[path] = blob
        ft, n_steps, replay_md5 = game_info[path]
        bound = game_binding.setdefault(game, (replay_md5, path))
        if bound[0] != replay_md5:
            raise SystemExit(
                f"k0_duel: game {game!r} is bound to two different "
                f"replay contents: {bound[1]} (md5 {bound[0]}) vs "
                f"{path} (md5 {replay_md5}): one game id must bind "
                f"one replay content (X08)")
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
    return specs, roots_sha, replay_blobs


_W = {}


def _init_worker(student, parent, replay_blobs=None):
    # Each arm arrives as (path, bytes): the worker parses the exact
    # bytes run_duel hashed (X04), never a fresh by-path read.
    _W["pol_student"] = _policy_from_bytes(*student)
    _W["pol_parent"] = _policy_from_bytes(*parent)
    # Replay binding (X04): materialize the validated replay bytes
    # under their original basenames in a per-process scratch dir;
    # extract_scene (which reads only spec["path"]) then consumes
    # exactly the content load_roots hashed and validated. Scratch
    # volume is one copy per unique replay per process -- set
    # TMPDIR to disk for very large root sets.
    scratch = {}
    if replay_blobs:
        tmp = tempfile.mkdtemp(prefix="k0_duel_replays_")
        for path, blob in replay_blobs.items():
            sp = os.path.join(tmp, os.path.basename(path))
            with open(sp, "wb") as fh:
                fh.write(blob)
            scratch[path] = sp
    _W["replay_scratch"] = scratch


def _check_scene_finite(spec, scene):
    """X07 backstop for the feature surface: the root seat and the
    opponent seat step-0 feature tensors must be finite. (Later-step
    features are produced by the driver's own featurizer from sim
    state; the per-arm U gate below backstops the whole path.)"""
    for side in ("x_me", "x_opp"):
        x = scene[side]
        if not isinstance(x, dict):
            continue
        for k in sorted(x):
            v = x[k]
            if (isinstance(v, np.ndarray) and v.dtype.kind in "fc"
                    and not np.isfinite(v).all()):
                raise SystemExit(
                    f"k0_duel: root {spec['game']}@{spec['step']} "
                    f"feature {side}.{k} has non-finite element(s): "
                    f"corrupt scene features must not be measured")


def _duel_task(task):
    spec, cfg = task
    scratch = _W.get("replay_scratch") or {}
    if spec["path"] not in scratch:
        raise SystemExit(
            f"k0_duel: root {spec['game']}@{spec['step']} replay "
            f"{spec['path']} is not content-bound (no validated bytes "
            f"in this worker): refusing to extract by path (X04)")
    scene = extract_scene(dict(spec, path=scratch[spec["path"]]))
    _check_scene_finite(spec, scene)
    seed = cfg["seed_cont"] + spec["step"]
    where = f"root {spec['game']}@{spec['step']}"
    u_parent = continue_game_2pol(_W["pol_parent"], _W["pol_parent"],
                                  scene, seed,
                                  teacher_style=cfg["teacher_style"],
                                  where=f"{where} parent-arm")
    u_student = continue_game_2pol(_W["pol_student"], _W["pol_parent"],
                                   scene, seed,
                                   teacher_style=cfg["teacher_style"],
                                   where=f"{where} student-arm")
    for arm, u in (("parent", u_parent), ("student", u_student)):
        if not math.isfinite(u):
            raise SystemExit(
                f"k0_duel: root {spec['game']}@{spec['step']} produced "
                f"non-finite U on the {arm} arm: {u!r}")
    return {"game": spec["game"], "step": spec["step"], "me": spec["me"],
            "seat": spec["me"], "ft": spec["ft"],
            "u_student": u_student, "u_parent": u_parent,
            "du": u_student - u_parent, "seed": seed,
            "student_md5": cfg["student_md5"],
            "parent_md5": cfg["parent_md5"],
            "roots_sha256": cfg["roots_sha256"]}


def run_duel(student_path, parent_path, specs, roots_sha256,
             seed_cont=995400, teacher_style=2, workers=1,
             replay_blobs=None):
    if replay_blobs is None:
        raise SystemExit(
            "k0_duel: run_duel requires the replay_blobs returned by "
            "load_roots: scene extraction must consume the validated "
            "replay bytes, not a by-path re-read (X04)")
    for s in specs:
        if s["path"] not in replay_blobs:
            raise SystemExit(
                f"k0_duel: root {s['game']}@{s['step']} replay "
                f"{s['path']} has no validated bytes in replay_blobs "
                f"(X04)")
    student_blob = _read_weights_bytes(student_path)
    parent_blob = _read_weights_bytes(parent_path)
    cfg = {"seed_cont": seed_cont, "teacher_style": teacher_style,
           "student_md5": hashlib.md5(student_blob).hexdigest(),
           "parent_md5": hashlib.md5(parent_blob).hexdigest(),
           "roots_sha256": roots_sha256}
    tasks = [(s, cfg) for s in specs]
    arms = ((student_path, student_blob), (parent_path, parent_blob))
    if workers and workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers,
                                 initializer=_init_worker,
                                 initargs=(*arms, replay_blobs)) as ex:
            rows = list(ex.map(_duel_task, tasks))
    else:
        _init_worker(*arms, replay_blobs)
        rows = [_duel_task(t) for t in tasks]
    for r in rows:
        for k in ("u_student", "u_parent", "du"):
            if not math.isfinite(r[k]):
                raise SystemExit(
                    f"k0_duel: row {r['game']}@{r['step']} has "
                    f"non-finite {k}: {r[k]!r}; refusing to aggregate "
                    f"or write corrupt measurements")
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
    specs, roots_sha, replay_blobs = load_roots(a.roots, a.step_min,
                                                a.step_max)
    result = run_duel(a.student, a.parent, specs, roots_sha,
                      seed_cont=a.seed_cont,
                      teacher_style=a.teacher_style, workers=a.workers,
                      replay_blobs=replay_blobs)
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

#!/usr/bin/env python3
"""selfplay_loop_adapter -- interface layer between a selfplay_gen corpus
and k0_distill, for the KataGo full loop (reports/PLAN_KATAGO_LOOP_20261118.md
SS2/SS3; interface check reports/eng_verify/KATAGO_LOOP_INTERFACE_CHECK_20261118.md).

Why this file exists: the producer (code/producer/selfplay_gen.py,
bd62ead9) writes a cnet2_build replay corpus (index.json + replays/*.json.gz)
whose per-step records are {observation, action, reward, status} plus a
per-game terminal block {scores, margin, u}. It records NO search output
(its pack CONF has SEARCH=False): there are no per-root candidate arms,
priors P_A, or per-arm Q anywhere in the product. The consumer
(code/k0_distill.py, 2728fee6) reads ONLY a sealed
"cnet2-k0-annotation/1" generation (current.json pointer): per-root
feature rows (cnet2_build FIELDS) + policy_labels.npz {arms, n_arms,
priors, q, pi0, pi1, pi} + value_labels.npz {Q}, and constructs its
policy target from (Q, P_A) through the SS13-C formula. The two formats
are therefore NOT isomorphic, at either the container level or the
content level. This adapter is the interface layer between them; it
changes no existing file and calls the pinned seam tools as-is:

  1. terminal-U sidecar: the corpus index's per-game {margin, u,
     end_reason, terminal, steps} are copied verbatim into
     <out>/terminal_u.json. This preserves the loop's terminal-U labels
     at the interface layer. k0_distill 2728fee6 has NO terminal-U
     input slot (its value-side input is per-root per-arm Q, and its
     utility/wdl/margin heads are frozen), so the sidecar is a record,
     not a training input -- see the interface-check receipt SS4.
  2. features: cnet2_build via build_top.py (teacher row from the
     corpus's own strategy name) -> sealed build generation.
  3. search labels: k0_annotate (the pinned stage0_teacher.evaluate_root
     path) with the PRODUCING weights as teacher -- in the loop the
     parent annotates its own games -- over the caller's registered
     window [--window-lo, --window-hi), pi0 = pi1 = 1.0 -> sealed
     annotation generation carrying (arms, priors, Q, q) per root.
     This is where pi_search enters the chain: as the annotator's
     stored q, cross-checked by the consumer against its construction
     from (Q, P_A).
  4. seam: k0_join -> sealed "cnet2-k0-annotation/1" generation, which
     join itself preflights through the real consumer loader.
  5. bad-root prescan (plan SS3 hard gate): every joined root is run
     through k0_distill's own candidate-expansion rejection path
     (expand_candidate_batch); roots it refuses are reported by
     (game, step, seat) and their games named as problem games
     (whole-game exclusion is the caller's registered disposition).

Usage:
  python3 selfplay_loop_adapter.py --corpus CORPUS_DIR --weights PARENT.npz \
      --code-dir CODE_DIR --out OUT_DIR [--window-lo 200] [--window-hi 299] \
      [--workers 3] [--teacher-row 2] [--seed-teacher 7000]

Outputs under OUT_DIR: build/ annot/ joined/ (sealed generations),
terminal_u.json, adapter_summary.json. Refuses to reuse non-empty
build/ or joined/ dirs (the annotator has its own identity-checked
resume; the other two publishers are content-addressed but this driver
keeps one clean product per invocation).
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time


def _md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def _die(msg):
    raise SystemExit(f"selfplay_loop_adapter: {msg}")


def _run(cmd, cwd, env=None, label=""):
    t0 = time.time()
    print(f"[adapter] {label}: {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
    wall = time.time() - t0
    tail = (r.stdout.strip().splitlines() or [""])[-1][:400]
    print(f"[adapter] {label}: rc={r.returncode} wall={wall:.1f}s last: {tail}",
          flush=True)
    if r.returncode != 0:
        sys.stderr.write(r.stdout[-3000:] + "\n" + r.stderr[-3000:] + "\n")
        _die(f"{label} failed with rc={r.returncode}")
    return {"rc": r.returncode, "wall_s": round(wall, 1), "last_line": tail}


def _game_id(entry):
    f = entry["file"]
    name = os.path.basename(f)
    return name[:-len(".json.gz")] if name.endswith(".json.gz") else name


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", required=True,
                    help="selfplay_gen output dir (index.json + replays/)")
    ap.add_argument("--weights", required=True,
                    help="producing/parent weights .npz (teacher = producer)")
    ap.add_argument("--code-dir", required=True,
                    help="dir holding the pinned code stack (build_top.py, "
                         "k0_annotate.py, k0_join.py, k0_distill.py, ...)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--window-lo", type=int, default=200)
    ap.add_argument("--window-hi", type=int, default=299)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--teacher-row", type=int, default=2)
    ap.add_argument("--seed-teacher", type=int, default=7000)
    args = ap.parse_args()

    py = sys.executable
    code = os.path.abspath(args.code_dir)
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    summary = {"corpus": os.path.abspath(args.corpus),
               "weights": os.path.abspath(args.weights),
               "weights_md5": _md5_file(args.weights),
               "window": [args.window_lo, args.window_hi],
               "stages": {}}
    t_all = time.time()

    # ---- corpus index: game list + terminal-U sidecar -----------------
    idx_path = os.path.join(args.corpus, "index.json")
    with open(idx_path) as fh:
        index = json.load(fh)
    if not index:
        _die("corpus index.json is empty")
    games = [_game_id(e) for e in index]
    teacher_name = (index[0].get("strategies", {}).get("a", {}) or {}).get("name")
    if not teacher_name:
        _die("corpus index carries no strategy name for the build teacher row")
    sidecar = {
        "kind": "terminal-u sidecar (interface layer record)",
        "source_index": idx_path,
        "source_index_md5": _md5_file(idx_path),
        "consumed_by": None,
        "note": ("Per-game terminal labels copied verbatim from the "
                 "selfplay_gen corpus index (U = 0.5*W + 0.5*sigmoid(M/1000), "
                 "producer _u_of_margin). k0_distill 2728fee6 has no "
                 "terminal-U input slot; this file preserves the labels "
                 "for the loop's future E[U] value training, it is NOT a "
                 "k0_distill input."),
        "games": [{"game": _game_id(e), "file": e["file"],
                   "steps": e["steps"], "terminal": e["terminal"],
                   "end_reason": e["end_reason"], "winner": e["winner"],
                   "scores": e["scores"], "margin": e["margin"], "u": e["u"]}
                  for e in index],
    }
    with open(os.path.join(out, "terminal_u.json"), "w") as fh:
        json.dump(sidecar, fh, indent=1, ensure_ascii=False)
    summary["games"] = games
    summary["teacher_name"] = teacher_name
    print(f"[adapter] corpus: {len(games)} game(s), teacher name "
          f"{teacher_name!r}; terminal_u.json written", flush=True)

    # ---- stage 1: build (features) ------------------------------------
    build_out = os.path.join(out, "build")
    if os.path.isdir(build_out) and os.listdir(build_out):
        _die(f"build out dir non-empty: {build_out}")
    env = dict(os.environ)
    env["CNET2_TOP_TEACHERS"] = f"{teacher_name}@{args.teacher_row}"
    summary["stages"]["build"] = _run(
        [py, os.path.join(code, "build_top.py"), build_out,
         os.path.abspath(args.corpus)],
        cwd=code, env=env, label="build")

    # ---- stage 2: annotate (search labels; teacher = producer) --------
    games_json = os.path.join(out, "games.json")
    with open(games_json, "w") as fh:
        json.dump(games, fh)
    annot_out = os.path.join(out, "annot")
    summary["stages"]["annotate"] = _run(
        [py, os.path.join(code, "k0_annotate.py"),
         "--weights", os.path.abspath(args.weights),
         "--replays", os.path.join(os.path.abspath(args.corpus), "replays"),
         "--games", games_json, "--out", annot_out,
         "--window-lo", str(args.window_lo), "--window-hi", str(args.window_hi),
         "--pi0", "1.0", "--pi1", "1.0", "--seed-teacher", str(args.seed_teacher),
         "--workers", str(args.workers)],
        cwd=code, label="annotate")

    # ---- stage 3: join (consumer format) -------------------------------
    joined_out = os.path.join(out, "joined")
    if os.path.isdir(joined_out) and os.listdir(joined_out):
        _die(f"joined out dir non-empty: {joined_out}")
    summary["stages"]["join"] = _run(
        [py, os.path.join(code, "k0_join.py"),
         "--annot", annot_out, "--build", build_out, "--out", joined_out],
        cwd=code, label="join")
    with open(os.path.join(joined_out, "current.json")) as fh:
        summary["joined_generation"] = json.load(fh)["generation"]

    # ---- stage 4: prescan (trainer's own rejection path, all roots) ---
    sys.path.insert(0, code)
    import numpy as np
    import k0_distill as KD
    ds = KD.load_annotation(joined_out)
    info = ds["info"]
    feats, arms = ds["feats"], ds["arms"]
    R, K = int(info["roots"]), int(info["max_candidates"])
    keys = ds["keys"]
    # gid -> game name from the joined generation meta.
    gen_dir = ds["gen_dir"]
    with open(os.path.join(gen_dir, "meta.json")) as fh:
        meta = json.load(fh)
    gid2name = {int(g["gid"]): g["game"] for g in meta["games"]}

    def _refused(idx):
        """True iff the trainer's expansion refuses this root set;
        on refusal of a multi-root set, bisect to single roots."""
        try:
            KD.expand_candidate_batch(feats, arms, np.asarray(idx), K)
            return []
        except SystemExit:
            if len(idx) == 1:
                return list(idx)
            mid = len(idx) // 2
            return _refused(idx[:mid]) + _refused(idx[mid:])

    t0 = time.time()
    problem = []
    CH = 32
    for s in range(0, R, CH):
        problem += _refused(list(range(s, min(s + CH, R))))
    problem_keys = [{"game": gid2name.get(int(keys["game"][r]), str(keys["game"][r])),
                     "step": int(keys["step"][r]), "seat": int(keys["seat"][r])}
                    for r in sorted(problem)]
    summary["prescan"] = {
        "criterion": "k0_distill.expand_candidate_batch refusal (trainer "
                     "candidate-prefix cash-legality path, k0_distill.py "
                     "expand_candidate_batch / _root_cash_set)",
        "roots_total": R, "candidates_total": int(info["candidates"]),
        "problem_roots": len(problem_keys),
        "problem_games": sorted({p["game"] for p in problem_keys}),
        "problem_root_keys": problem_keys,
        "wall_s": round(time.time() - t0, 1),
    }
    print(f"[adapter] prescan: {R} roots, problem {len(problem_keys)} "
          f"(games {summary['prescan']['problem_games']})", flush=True)

    summary["wall_s_total"] = round(time.time() - t_all, 1)
    with open(os.path.join(out, "adapter_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, ensure_ascii=False)
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    main()

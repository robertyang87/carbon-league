#!/usr/bin/env python3
"""lb_relabel -- Line B expert iteration: search-relabelled endgame samples.

  python3 lb_relabel.py OUT_DIR --src corpus  CORPUS_DIR TEACHER_NAME TEACHER_ROW CLONE_W.npz [N_GAMES]
  python3 lb_relabel.py OUT_DIR --src selfplay CLONE_W.npz TEACHER_ROW SEED0 N [WORKERS]
  python3 lb_relabel.py --dry-run CLONE_W.npz TEACHER_ROW SEED     (one selfplay game, timing only, no shards)

WHY: m3s2r1's measured gap is endgame residual trees (6.8 vs real 2.1):
liquidation timing is a valuation/search decision BC cannot learn
(ep10->ep19 residual got WORSE). This tool produces the training rows
for m3s2r2: endgame states whose labels are the clone+search agent's
actions instead of the recorded teacher's (corpus) or the clone's own
(selfplay). Design frozen in reports/live/lb_expert_iteration_prep.md.

MECHANICS (both sources): one Feat per game is fed the genuine obs
stream from step 0 (identical to cnet2_build.build_game), so rows are
schema-identical to ds rows. The search agent (cnet2_search
make_search_agent, fleet DEFAULTS, weights=CLONE) is ALSO fed every
obs from step 0 with SEARCH=False -- its internal Feat/belief warms
exactly as in a live game at pure-net cost -- and flipped to
SEARCH=True at RELABEL_FROM (agent.cfg is exposed) so only endgame
steps pay search cost and only endgame rows are emitted.

ROW RULES (v1, single variable vs ds_m3s2):
  y        = Feat.labels(search action) for steps >= RELABEL_FROM
  uw       = 3.0 on own slots whose label differs from the clone's
             default action, 1.0 on agreeing own slots, 0 on empty
             (dagger_gen DG_UW_CHANGED convention)
  teacher  = TEACHER_ROW (2 for the Master3s family; the row the
             stand-in wrapper plays at inference)
  value    = tanh(final margin / 3000) of the recorded/driven game
             (selfplay: margin is the SCORE margin under the canonical
             terminal contract -- extinction ends the game at once, the
             extinct player scores steps-EPISODE_STEPS-1 and the survivor
             keeps its pre-terminal-step cash, mirroring eval_v53; the
             corpus branch keeps its recorded index scores, unchanged)
Output: OUT_DIR/shards/lb_<tag>.npz (cnet2_build.FIELDS + uw) +
games.jsonl, mergeable into a ds clone via merge_dagger.py
(DAGGER_W = sample weight of relabel rows, default 1.0).
Corpus shards also carry orig_id (the original replay file name): R06 --
merge_dagger assigns the shard its ORIGINAL game's train/val split, never a
fresh CRC roll on the derived shard name, so relabelled validation games can
never leak into the training split.
"""
import gzip
import json
import os
import random
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, "1")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "env_src", "carbon_challenge-master"))

RELABEL_FROM = int(os.environ.get("LB_FROM", "270"))
UW_CHANGED = float(os.environ.get("LB_UW_CHANGED", "3.0"))
# selfplay opponents: training-pool nets only, never exam opponents (dagger_gen rule)
POOL = ["exp/cnet2_v6t5g.py", "exp/cnet2_v5t5g.py", "exp/cnet2_v7s57_240.py",
        "exp/cnet2_v9s57_240.py", "exp/cnet2_style_t0.py"]
if os.environ.get("LB_POOL"):
    POOL = os.environ["LB_POOL"].split(",")


def _prefix_rows(F, x, feat, y):
    """uf/mf/legal under teacher forcing on label y (mirror cnet2_build)."""
    import numpy as np
    pf = F.Prefix(x, feat.aux)
    uf = np.zeros((F.OWN, F.UF), np.float32)
    mf = np.zeros((F.OWN, 5, F.MF), np.float32)
    legal = np.zeros((F.OWN, 5), bool)
    yy = y.copy()
    for s in range(F.OWN):
        if not x["tok_mask"][s]:
            continue
        uf[s], mf[s], legal[s] = pf.feats(s)
        if yy[s] >= 0 and not legal[s, yy[s]]:
            yy[s] = 0
        pf.advance(s, int(yy[s]) if yy[s] >= 0 else 0)
    return uf, mf, legal, yy


def _emit(F, rows, x, feat, y, a_default, carbon_full, side, value, teacher, step):
    import numpy as np
    uf, mf, legal, yy = _prefix_rows(F, x, feat, y)
    uw = np.zeros(F.OWN, np.float32)
    yd = feat.labels(a_default)
    for s in range(F.OWN):
        if x["tok_mask"][s] and yy[s] >= 0:
            uw[s] = UW_CHANGED if yy[s] != yd[s] else 1.0
    for k in ("maps", "scal", "tok_pos", "tok_kind", "tok_mask", "tok_cargo",
              "hist_cell", "hist_cargo", "hist_move"):
        rows[k].append(x[k])
    rows["uf"].append(uf); rows["mf"].append(mf); rows["legal"].append(legal)
    rows["y"].append(yy)
    order = F.CANON if side else F.GRID
    rows["carbon"].append(F.q8(np.asarray(carbon_full, np.float64)[order] / 100.0))
    rows["value"].append(np.tanh(value / 3000.0)); rows["teacher"].append(teacher)
    rows["gid"].append(0); rows["step"].append(step); rows["uw"].append(uw)
    return int(((yy != yd) & (uw > 0)).sum()), int((uw > 0).sum())


def _game_result(sim, side, steps_played, pre_cash):
    """Canonical terminal contract (F02) -- mirrors eval_v53's scoring block.
    Returns (margin, terminal, scores) from `side`'s perspective: extinction
    scores per eval_v53 (extinct = steps_played - EPISODE_STEPS - 1, survivor
    keeps pre_cash, the cash carried INTO the terminal step); a time-out
    finish scores final cash. dagger_gen and awr_gen carry the same helper;
    keep the three in sync."""
    import eval_v53
    post = sim.get_state()
    extinct = [eval_v53._extinct(post, p) for p in (0, 1)]
    if any(extinct):
        scores = [float(steps_played - eval_v53.EPISODE_STEPS - 1) if extinct[p]
                  else float(pre_cash[p]) for p in (0, 1)]
        return float(scores[side] - scores[1 - side]), "extinction", scores
    scores = [float(post["cash"][p]) for p in (0, 1)]
    return float(scores[side] - scores[1 - side]), "timeout", scores


def play_selfplay(task):
    """One clone-driven game; returns per-game diagnostics (+shard path unless dry)."""
    out, clone_w, row, seed, dry = task
    import numpy as np
    import eval_v53
    import cnet2_feat as F
    import cnet2_search as S
    import cnet2_agent as CA
    from cnet2_build import FIELDS
    from fastsim import FastCarbon
    opp_path = POOL[seed % len(POOL)]
    side = (seed // len(POOL)) % 2
    random.seed(eval_v53.rng_seed(seed, side))
    sim = FastCarbon(eval_v53.CFG).reset(seed=seed)
    clone = CA.make_agent(clone_w, teacher=row)
    sag = S.make_search_agent({"SEARCH": False, "TEACHER": row}, weights=clone_w)
    opp = eval_v53.load_agent_fn(os.path.join(HERE, opp_path))
    feat = F.Feat(side)
    rows = {k: [] for k in FIELDS}; rows["uw"] = []
    nchg = nsup = 0; times = []
    steps_played = 0; pre_cash = [float(sim.get_state()["cash"][p]) for p in (0, 1)]
    for step in range(299):
        o_me = sim.obs(side); o_op = sim.obs(1 - side)
        x = feat.observe(o_me)
        a_me = clone(o_me, eval_v53.CFG)
        if step == RELABEL_FROM:
            sag.cfg["SEARCH"] = True
        t0 = time.time()
        a_lab = sag(o_me, eval_v53.CFG)
        dt = time.time() - t0
        a_op = opp(o_op, eval_v53.CFG)
        if step >= RELABEL_FROM:
            times.append(dt)
            st = sim.get_state()
            full = np.asarray(sim.carbon, np.float64)
            mg_now = float(st["cash"][side] - st["cash"][1 - side])
            c, s_ = _emit(F, rows, x, feat, feat.labels(a_lab), a_me,
                          full, side, mg_now, row, step)
            nchg += c; nsup += s_
        pre_cash = [float(sim.get_state()["cash"][p]) for p in (0, 1)]
        sim.step(*((a_me, a_op) if side == 0 else (a_op, a_me)))
        steps_played = step + 1
        if any(eval_v53._extinct(sim.get_state(), p) for p in (0, 1)):
            break       # F02: extinction ends the game immediately -- no post-terminal rows
    mg, terminal, scores = _game_result(sim, side, steps_played, pre_cash)
    res = {"seed": seed, "opp": opp_path, "side": side, "margin": mg,
           "terminal": terminal, "end_step": steps_played, "scores": scores,
           "n": len(rows["y"]), "supervised": nsup, "changed": nchg,
           "ms_mean": round(1000 * sum(times) / max(1, len(times)), 1),
           "ms_max": round(1000 * max(times), 1) if times else 0.0,
           "stats": {k: sag.stats[k] for k in ("searched", "changed", "errors")}}
    if not dry:
        arr = {k: np.asarray(v).astype(FIELDS[k][0]) for k, v in rows.items() if k in FIELDS}
        arr["uw"] = np.asarray(rows["uw"], np.float16)
        # value must be the FINAL margin, not per-step running margin: fix in one pass
        arr["value"] = np.full(len(rows["y"]), np.tanh(mg / 3000.0), np.float16)
        fp = os.path.join(out, "shards", f"lb_sp_{seed}.npz")
        np.savez(fp, **arr); res["shard"] = fp
    return res


def play_corpus(task):
    """One recorded teacher game from a corpus trace; endgame rows relabelled.
    Margin comes from the corpus index entry (traces carry no names/scores)."""
    out, path, side, teacher_row, clone_w, key, mg, orig = task
    import numpy as np
    import eval_v53
    import cnet2_feat as F
    import cnet2_search as S
    from cnet2_build import FIELDS
    d = json.load(gzip.open(path))
    st = d["steps"]
    mg = float(mg)
    feat = F.Feat(side)
    import cnet2_agent as CA
    clone = CA.make_agent(clone_w, teacher=teacher_row)
    sag = S.make_search_agent({"SEARCH": False, "TEACHER": teacher_row}, weights=clone_w)
    rows = {k: [] for k in FIELDS}; rows["uw"] = []
    nchg = nsup = nchg_t = 0; times = []
    for t in range(len(st) - 1):
        o0 = st[t][0]["observation"]
        obs = {"step": o0["step"], "players": o0["players"],
               "carbon": st[t][side]["observation"]["carbon"], "player": side}
        if st[t][side].get("status") not in (None, "ACTIVE"):
            break
        x = feat.observe(obs)
        a_teacher = st[t + 1][side].get("action") or {}
        a_clone = clone(obs, eval_v53.CFG)
        if t == RELABEL_FROM:
            sag.cfg["SEARCH"] = True
        t0 = time.time()
        a_lab = sag(obs, eval_v53.CFG)
        dt = time.time() - t0
        if t >= RELABEL_FROM:
            times.append(dt)
            full = np.asarray(o0.get("full_carbon", obs["carbon"]), np.float64)
            c, s_ = _emit(F, rows, x, feat, feat.labels(a_lab),
                          a_clone, full, side, mg, teacher_row, t)
            nchg += c; nsup += s_
            yt = feat.labels(a_teacher)
            nchg_t += int(((rows["y"][-1] != yt) & (rows["uw"][-1] > 0)).sum())
    arr = {k: np.asarray(v).astype(FIELDS[k][0]) for k, v in rows.items() if k in FIELDS}
    arr["uw"] = np.asarray(rows["uw"], np.float16)
    arr["orig_id"] = np.array(orig)     # R06: original replay name -- merge inherits ITS split
    fp = os.path.join(out, "shards", f"lb_cp_{key}.npz")
    np.savez(fp, **arr)
    return {"key": key, "orig_id": orig,
            "split": "val" if zlib.crc32(orig.encode()) % 10 == 0 else "train",
            "side": side, "margin": mg, "n": len(rows["y"]),
            "supervised": nsup, "changed": nchg, "changed_vs_teacher": nchg_t,
            "ms_mean": round(1000 * sum(times) / max(1, len(times)), 1),
            "ms_max": round(1000 * max(times), 1) if times else 0.0, "shard": fp}


def main():
    if sys.argv[1] == "--dry-run":
        clone_w, row, seed = sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
        r = play_selfplay((None, clone_w, row, seed, True))
        print(json.dumps({"dry_run": r}, indent=1))
        return 0
    out = sys.argv[1]
    os.makedirs(os.path.join(out, "shards"), exist_ok=True)
    log = open(os.path.join(out, "games.jsonl"), "a")
    if sys.argv[2] == "--src" and sys.argv[3] == "selfplay":
        clone_w, row, s0, n = sys.argv[4], int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7])
        workers = int(sys.argv[8]) if len(sys.argv) > 8 else 4
        tasks = [(out, clone_w, row, s, False) for s in range(s0, s0 + n)]
        with ProcessPoolExecutor(workers) as ex:
            for r in ex.map(play_selfplay, tasks):
                log.write(json.dumps(r) + "\n"); log.flush(); print(json.dumps(r), flush=True)
    elif sys.argv[2] == "--src" and sys.argv[3] == "corpus":
        corpus, tname, row, clone_w = sys.argv[4], sys.argv[5], int(sys.argv[6]), sys.argv[7]
        ipath = os.path.join(corpus, "index.json")
        if not os.path.exists(ipath):
            print("ERROR: corpus index.json missing -- it is generated at build time "
                  "(ladder_corpus.py) and lives in the machine-local corpus dir "
                  "(e.g. DGX data/m3s2_corpus), not in the repo. Run there.", flush=True)
            return 2
        idx = json.load(open(ipath))
        tasks = []
        for g in idx:
            order = g["order"]
            names = [g["strategies"][order[s]]["name"] for s in (0, 1)]
            scores = [g["scores"][order[s]] for s in (0, 1)]
            for s in (0, 1):
                if names[s] == tname:
                    key = os.path.basename(g["file"]).replace(".json.gz", "")
                    tasks.append((out, os.path.join(corpus, g["file"]), s, row,
                                  clone_w, key, scores[s] - scores[1 - s],
                                  os.path.basename(g["file"])))
        print(json.dumps({"corpus_games": len(tasks)}), flush=True)
        with ProcessPoolExecutor(4) as ex:
            for r in ex.map(play_corpus, tasks):
                log.write(json.dumps(r) + "\n"); log.flush(); print(json.dumps(r), flush=True)
    else:
        print(__doc__); return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

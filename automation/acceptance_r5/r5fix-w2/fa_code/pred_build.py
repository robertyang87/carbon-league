"""pred_build -- build opponent next-action prediction samples per archetype.

Sample = one worker at one step of the ARCHETYPE player (the side whose
behaviour we want to predict). Features use ONLY the shared public blocks
(positions / cargo / type of both sides, trees, bases, cash, step) -- the
same quantities our live agent sees about a real opponent. The actor's
private carbon visibility is never touched.

Sources:
  alpha    : bc_data/claude_family_v2.npz   (val = train_bc.py VAL_FILES)
  deepseek : bc_data/deepseek_family.npz    (val = deepseek_val_files.json)
  v53      : traces/base_v53_vs_base_v53_seed*_side*.json.gz (mirror games,
             both streams are base_v53; split BY SEED: train <=2004)

Labels: recorded worker_actions for npz families (fixed pairing
obs_t -> act_{t+1}); position-delta-implied actions for traces.
prev_act: chained from the previous step of the same stream when the slot
is consistent (prev action explains the move into the current cell).
v53pred: faithful offline reproduction of base_v53's periodic-mover
detector (history window 26, periods (1,8,4,2,6,12), two-point confirm,
collectors only; the live blockade gate is NOT reproduced -- coverage in
the report is the detector's own, gate aside).

Attack labels (collectors only), defined on time-t positions:
  E1 = the worker's next cell holds >=1 opponent worker at t
  D  = the worker's next cell holds an opponent COLLECTOR whose cargo is
       strictly greater (the mover is lighter -> would win the collision)
"""

import gzip
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from bc_features import DIST, _xs, _ys  # noqa: E402

SIZE = 15
DELTA = {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1), 4: (-1, 0)}
ACT_BY_DELTA = {(0, 0): 0, (0, 1): 1, (1, 0): 2, (0, -1): 3, (-1, 0): 4}
PERIODS = (1, 8, 4, 2, 6, 12)
HIST_WIN = 26


def _nearest_field(targets):
    """(225,) distance from every cell to nearest target; 14 if none."""
    if len(targets) == 0:
        return np.full(SIZE * SIZE, 14.0)
    return DIST[:, np.asarray(targets, dtype=np.int64)].min(axis=1)


def _move_cell(pos, act):
    dx, dy = DELTA[int(act)]
    x = (int(_xs[pos]) + dx) % SIZE
    y = (int(_ys[pos]) + dy) % SIZE
    return (SIZE - 1 - y) * SIZE + x


class SampleSink:
    def __init__(self):
        self.rows = []

    def add(self, **kw):
        self.rows.append(kw)

    def arrays(self):
        keys = ["split", "strat", "wtype", "cargo", "d_base", "d_ebase",
                "d_tree", "d_foe", "step", "on_base", "on_ebase", "on_tree",
                "label", "prev_act", "e1", "dang", "rel_near", "d_foe_coll",
                "pred_cell", "next_cell", "pos"]
        out = {}
        for k in keys:
            out[k] = np.array([r[k] for r in self.rows])
        return out


def worker_row(sink, split, wtype, cargo, pos, step,
               f_base, f_ebase, f_tree, f_foe, own_bases, foe_bases,
               tree_cells, foe_workers, label, prev_act, pred_cell,
               strat=0):
    """foe_workers: list of (pos, cargo, type) of the OTHER side at t."""
    nxt = _move_cell(pos, label)
    e1 = dang = -1
    rel_near = np.nan
    d_foe_coll = 14
    if wtype == 0:
        foe_pos = {p for p, _, _ in foe_workers}
        e1 = 1 if nxt in foe_pos else 0
        dang = 0
        for p, c, t in foe_workers:
            if t == 0:
                dd = DIST[pos, p]
                if dd < d_foe_coll:
                    d_foe_coll = int(dd)
                    rel_near = c - cargo
                if p == nxt and c > cargo:
                    dang = 1
    sink.add(split=split, strat=strat, wtype=wtype, cargo=cargo,
             d_base=float(f_base[pos]),
             d_ebase=float(f_ebase[pos]), d_tree=float(f_tree[pos]),
             d_foe=float(f_foe[pos]), step=step,
             on_base=1 if pos in own_bases else 0,
             on_ebase=1 if pos in foe_bases else 0,
             on_tree=1 if pos in tree_cells else 0,
             label=label, prev_act=prev_act, e1=e1, dang=dang,
             rel_near=rel_near, d_foe_coll=d_foe_coll,
             pred_cell=pred_cell, next_cell=nxt, pos=pos)


def build_npz_family(npz_path, val_files, out_path):
    d = np.load(npz_path, allow_pickle=True)
    n = len(d["step"])
    step = d["step"]; player = d["player"]; gf = d["game_file"]
    workers = d["workers"]; wmask = d["workers_mask"]
    trees = d["trees"]; tmask = d["trees_mask"]
    bases = d["bases"]; cash = d["cash"]
    wact = d["worker_actions"]; bact = d["base_actions"]
    is_val = np.array([f in val_files for f in gf])

    # streams: (game_file, player) -> ordered point indices
    streams = {}
    for i in range(n):
        streams.setdefault((str(gf[i]), int(player[i])), []).append(i)
    for k in streams:
        streams[k].sort(key=lambda i: int(step[i]))

    sink = SampleSink()
    base_rows = []
    strat_names = []
    strat_ids = {}
    for (gfile, p), idxs in streams.items():
        sname = str(d["strategy"][idxs[0]])
        if sname not in strat_ids:
            strat_ids[sname] = len(strat_names)
            strat_names.append(sname)
        sid = strat_ids[sname]
        split = 1 if is_val[idxs[0]] else 0
        foe = 1 - p
        hist = {}          # slot -> position history (archetype collectors)
        prev = None        # (workers, wmask, wact) of previous point
        prev_step = -10
        for i in idxs:
            st = int(step[i])
            if st != prev_step + 1:
                hist = {}
                prev = None
            W = workers[i]; M = wmask[i]
            own_bases = [int(b) for b in bases[i][p] if int(b) >= 0]
            foe_bases = [int(b) for b in bases[i][foe] if int(b) >= 0]
            tpos = [int(trees[i][k][0]) for k in range(trees.shape[1])
                    if tmask[i][k]]
            foe_pos = [int(W[foe][j][0]) for j in range(10) if M[foe][j]]
            foe_list = [(int(W[foe][j][0]), float(W[foe][j][1]),
                         int(W[foe][j][2])) for j in range(10) if M[foe][j]]
            f_base = _nearest_field(own_bases)
            f_ebase = _nearest_field(foe_bases)
            f_tree = _nearest_field(tpos)
            f_foe = _nearest_field(foe_pos)
            tree_cells = set(tpos)
            # v53 periodic detector over the archetype's collectors
            pred_by_slot = {}
            alive = set()
            for j in range(10):
                if not (M[p][j] and int(W[p][j][2]) == 0):
                    continue
                alive.add(j)
                pos = int(W[p][j][0])
                h = hist.setdefault(j, [])
                # identity break: position jumped >1 from last record
                if h and DIST[h[-1], pos] > 1:
                    h = []
                    hist[j] = h
                h.append(pos)
                if len(h) > HIST_WIN:
                    del h[0]
                pred = -1
                for per in PERIODS:
                    if len(h) >= per + 2 and h[-1] == h[-1 - per] \
                            and h[-2] == h[-2 - per]:
                        pred = h[-per] if per > 1 else h[-1]
                        break
                pred_by_slot[j] = pred
            for j in list(hist):
                if j not in alive:
                    del hist[j]
            for j in range(10):
                if not M[p][j]:
                    continue
                wtype = int(W[p][j][2])
                pos = int(W[p][j][0])
                prev_act = -1
                if prev is not None and prev[1][p][j] and \
                        int(prev[0][p][j][2]) == wtype:
                    pa = int(prev[2][j])
                    if _move_cell(int(prev[0][p][j][0]), pa) == pos:
                        prev_act = pa
                worker_row(sink, split, wtype, float(W[p][j][1]), pos, st,
                           f_base, f_ebase, f_tree, f_foe, own_bases,
                           foe_bases, tree_cells, foe_list,
                           int(wact[i][j]), prev_act,
                           pred_by_slot.get(j, -1), strat=sid)
            for bi in range(2):
                if int(bases[i][p][bi]) < 0:
                    continue
                base_rows.append(dict(
                    split=split, bi=bi, step=st, cash=float(cash[i][p]),
                    n_coll=int(np.sum(M[p] & (W[p, :, 2] == 0))),
                    n_plan=int(np.sum(M[p] & (W[p, :, 2] == 1))),
                    label=int(bact[i][bi])))
            prev = (W, M, wact[i])
            prev_step = st
    arr = sink.arrays()
    barr = {k: np.array([r[k] for r in base_rows]) for k in
            ("split", "bi", "step", "cash", "n_coll", "n_plan", "label")}
    np.savez_compressed(out_path, strat_names=np.array(strat_names),
                        **arr, **{f"base_{k}": v for k, v in barr.items()})
    print(out_path, "workers", len(arr["label"]), "bases", len(barr["label"]),
          "val worker share", float(np.mean(arr["split"])))


def build_v53_family(out_path):
    import glob
    import re
    # side0 files only: side1 files are second recordings of the SAME games
    # (identical frames/final cash); each file already carries both players.
    files = sorted(glob.glob(os.path.join(
        HERE, "traces", "base_v53_vs_base_v53_*_side0.json.gz")))
    sink = SampleSink()
    base_rows = []
    games = []
    for path in files:
        m = re.search(r"seed(\d+)", path)
        seed = int(m.group(1))
        split = 0 if seed <= 2014 else 1
        games.append(seed)
        d = json.load(gzip.open(path))
        frames = d["frames"]
        bases_meta = {int(k): [c for _, c in v]
                      for k, v in d["meta"]["bases"].items()}
        for p in (0, 1):
            foe = 1 - p
            own_bases = bases_meta[p]
            foe_bases = bases_meta[foe]
            hist = {}  # wid -> positions
            for t in range(len(frames) - 1):
                fr, nx = frames[t], frames[t + 1]
                st = int(fr["step"])
                wk = fr["workers"]; wk_n = nx["workers"]
                my_w = {w: v for w, v in wk.items()
                        if w.startswith(f"player-{p}-worker")}
                foe_w = [(v[0], float(v[1]), 0 if v[2] == "C" else 1)
                         for w, v in wk.items()
                         if w.startswith(f"player-{foe}-worker")]
                tpos = [v[0] for v in fr["trees"].values()]
                f_base = _nearest_field(own_bases)
                f_ebase = _nearest_field(foe_bases)
                f_tree = _nearest_field(tpos)
                f_foe = _nearest_field([v[0] for v in foe_w])
                tree_cells = set(tpos)
                # v53 detector (wid-exact histories), collectors only
                pred_by_wid = {}
                for wid, v in my_w.items():
                    if v[2] != "C":
                        continue
                    h = hist.setdefault(wid, [])
                    h.append(int(v[0]))
                    if len(h) > HIST_WIN:
                        del h[0]
                    pred = -1
                    for per in PERIODS:
                        if len(h) >= per + 2 and h[-1] == h[-1 - per] \
                                and h[-2] == h[-2 - per]:
                            pred = h[-per] if per > 1 else h[-1]
                            break
                    pred_by_wid[wid] = pred
                for wid in list(hist):
                    if wid not in my_w:
                        del hist[wid]
                prevpos = {}
                if t > 0:
                    for wid, v in frames[t - 1]["workers"].items():
                        if wid in my_w:
                            prevpos[wid] = v[0]
                for wid, v in my_w.items():
                    if wid not in wk_n:
                        continue  # died: no realised move to label
                    pos, pos_n = int(v[0]), int(wk_n[wid][0])
                    dx = (int(_xs[pos_n]) - int(_xs[pos]) + 7) % SIZE - 7
                    dy = (int(_ys[pos_n]) - int(_ys[pos]) + 7) % SIZE - 7
                    if abs(dx) + abs(dy) > 1:
                        continue
                    label = ACT_BY_DELTA[(dx, dy)]
                    prev_act = -1
                    if wid in prevpos:
                        pdx = (int(_xs[pos]) - int(_xs[prevpos[wid]]) + 7) \
                            % SIZE - 7
                        pdy = (int(_ys[pos]) - int(_ys[prevpos[wid]]) + 7) \
                            % SIZE - 7
                        if abs(pdx) + abs(pdy) <= 1:
                            prev_act = ACT_BY_DELTA[(pdx, pdy)]
                    worker_row(sink, split, 0 if v[2] == "C" else 1,
                               float(v[1]), pos, st, f_base, f_ebase, f_tree,
                               f_foe, own_bases, foe_bases, tree_cells, foe_w,
                               label, prev_act, pred_by_wid.get(wid, -1))
                # base recruitment labels: new worker appearing at a base
                new_at = {}
                for wid, v in wk_n.items():
                    if wid.startswith(f"player-{p}-worker") \
                            and wid not in wk:
                        new_at.setdefault(int(v[0]), v[2])
                for bi, bc in enumerate(own_bases):
                    lab = 0
                    if bc in new_at:
                        lab = 1 if new_at[bc] == "C" else 2
                    base_rows.append(dict(
                        split=split, bi=bi, step=st,
                        cash=float(fr["cash"][p]),
                        n_coll=sum(1 for v in my_w.values() if v[2] == "C"),
                        n_plan=sum(1 for v in my_w.values() if v[2] == "P"),
                        label=lab))
    arr = sink.arrays()
    barr = {k: np.array([r[k] for r in base_rows]) for k in
            ("split", "bi", "step", "cash", "n_coll", "n_plan", "label")}
    np.savez_compressed(out_path, strat_names=np.array(["base_v53"]),
                        **arr, **{f"base_{k}": v for k, v in barr.items()})
    print(out_path, "games(seeds)", sorted(set(games)), "workers",
          len(arr["label"]), "bases", len(barr["label"]))


if __name__ == "__main__":
    ALPHA_VAL = {
        "replays/67743bde32a54e9cd9a6ac31a84871d3-1.json.gz",
        "replays/98e6ce19d9a3a347a2b3838b9c05e9f5-1.json.gz",
        "replays/1e93688a54a097262981c00e3e103699-1.json.gz",
        "replays/cf1ea7c4863bfb92892aee87b46479a6-1.json.gz",
        "replays/56e647d23f75cedb7c6d164e9025df4a-1.json.gz",
    }
    DS_VAL = set(json.load(open(os.path.join(
        HERE, "bc_data", "deepseek_val_files.json"))))
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "alpha"):
        build_npz_family(os.path.join(HERE, "bc_data", "claude_family_v2.npz"),
                         ALPHA_VAL, os.path.join(HERE, "pred_data_alpha.npz"))
    if which in ("all", "deepseek"):
        build_npz_family(os.path.join(HERE, "bc_data", "deepseek_family.npz"),
                         DS_VAL, os.path.join(HERE, "pred_data_deepseek.npz"))
    if which in ("all", "v53"):
        build_v53_family(os.path.join(HERE, "pred_data_v53.npz"))

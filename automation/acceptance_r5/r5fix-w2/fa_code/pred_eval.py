"""pred_eval -- offline evaluation of opponent next-action predictors.

Models (all fit on train split only, scored on val split):
  stay      : always predict STAY
  majority  : per-type modal action of the train set
  inertia   : predict the worker's previous action (chained samples only)
  v53pred   : base_v53 periodic-mover detector (built into pred_build):
              coverage + next-cell accuracy on collectors
  bucket    : context-bucket action distribution with hierarchical
              backoff + Laplace smoothing toward the parent level

Heads:
  worker action top-1 (all / collector / planter)
  base recruitment: top-1 + RECC/RECP recall (bucket vs always-none)
  attack events (collectors): bucket P(E1), P(D dangerous) with AUC,
              precision/recall @0.5, on all samples and on the
              opportunity subset (nearest foe worker within 2 cells)

Outputs pred_results.json (+ prints a summary).
"""

import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------- bucketisation ----------
def cargo_b(c):
    return np.select([c == 0, c < 75, c < 150, c < 300, c < 450],
                     [0, 1, 2, 3, 4], default=5).astype(np.int64)


def dist_b6(d):
    return np.select([d == 0, d == 1, d == 2, d == 3, d <= 6],
                     [0, 1, 2, 3, 4], default=5).astype(np.int64)


def dtree_b(d):
    return np.minimum(d, 3).astype(np.int64)


def dfoe_b(d):
    return np.minimum(d, 4).astype(np.int64)


def phase_b(s):
    return np.select([s < 30, s < 100, s < 200], [0, 1, 2],
                     default=3).astype(np.int64)


def cash_b(c):
    return np.select([c < 30, c < 60, c < 150], [0, 1, 2],
                     default=3).astype(np.int64)


def cnt_b(n):
    return np.minimum(n, 3).astype(np.int64)


def rel_b(rel_near, d_foe_coll):
    out = np.zeros(len(rel_near), dtype=np.int64)
    near = d_foe_coll <= 3
    out[near & (rel_near < 0)] = 1
    out[near & (rel_near == 0)] = 2
    out[near & (rel_near > 0) & (rel_near < 150)] = 3
    out[near & (rel_near >= 150)] = 4
    return out


# ---------- generic backoff bucket model ----------
class BackoffBuckets:
    """levels: list of key-column index tuples, finest first."""

    def __init__(self, levels, nclass, min_count=25, alpha=10.0):
        self.levels = levels
        self.nclass = nclass
        self.min_count = min_count
        self.alpha = alpha

    def fit(self, keys, y):
        self.tables = []
        for cols in self.levels:
            tab = {}
            if len(cols) == 0:
                tab[0] = np.bincount(y, minlength=self.nclass).astype(float)
            else:
                kk = keys[:, list(cols)]
                # encode multi-dim key as single int via mixed radix
                enc = np.zeros(len(y), dtype=np.int64)
                for j in range(kk.shape[1]):
                    enc = enc * 64 + kk[:, j]
                for e, yy in zip(enc, y):
                    row = tab.get(int(e))
                    if row is None:
                        row = np.zeros(self.nclass)
                        tab[int(e)] = row
                    row[yy] += 1
            self.tables.append(tab)
        return self

    def _enc(self, keys_row_block, cols):
        if len(cols) == 0:
            return np.zeros(len(keys_row_block), dtype=np.int64)
        kk = keys_row_block[:, list(cols)]
        enc = np.zeros(len(kk), dtype=np.int64)
        for j in range(kk.shape[1]):
            enc = enc * 64 + kk[:, j]
        return enc

    def predict_proba(self, keys):
        """keys: (n, ncols) int array -> (n, nclass) probabilities."""
        n = len(keys)
        # per-level encoded keys + counts/probs resolved level by level
        encs = [self._enc(keys, cols) for cols in self.levels]
        # bottom-up smoothed distributions per unique key per level
        level_probs = []
        for li, cols in enumerate(self.levels):
            enc = encs[li]
            # numpy 1.26: returns (unique, first_index, inverse) in
            # signature order -- do not swap.
            uniq, first, inv = np.unique(enc, return_index=True,
                                         return_inverse=True)
            probs = np.zeros((len(uniq), self.nclass))
            level_probs.append((uniq, inv, first, probs))
        # resolve from coarsest to finest
        resolved = [None] * len(self.levels)
        for li in reversed(range(len(self.levels))):
            uniq, inv, first, probs = level_probs[li]
            tab = self.tables[li]
            counts = np.zeros((len(uniq), self.nclass))
            for ui, e in enumerate(uniq):
                row = tab.get(int(e))
                if row is not None:
                    counts[ui] = row
            if li == len(self.levels) - 1:
                base = np.full(self.nclass, 1.0 / self.nclass)
                probs[:] = (counts + self.alpha * base) / \
                    (counts.sum(axis=1, keepdims=True) + self.alpha)
            else:
                # parent distribution: project rows one level coarser and
                # look up the resolved distribution; all rows sharing a
                # fine key share its projection, so take first occurrences.
                parent_enc = self._enc(keys, self.levels[li + 1])
                p_uniq = level_probs[li + 1][0]
                p_probs = resolved[li + 1]
                pos_in_parent = np.searchsorted(p_uniq, parent_enc)
                parent_of_key = p_probs[pos_in_parent][first]
                tot = counts.sum(axis=1)
                use = tot >= self.min_count
                smoothed = (counts + self.alpha * parent_of_key) / \
                    (tot[:, None] + self.alpha)
                probs[:] = np.where(use[:, None], smoothed, parent_of_key)
            resolved[li] = probs
        return resolved[0][level_probs[0][1]]


WORKER_LEVELS = [(0, 1, 2, 3, 4, 5, 6), (0, 1, 2, 3, 4), (0, 1, 3, 4),
                 (0, 1, 3), (0, 3), (0,), ()]
# key columns: 0 cargo_b, 1 dbase_b, 2 dtree_b, 3 dfoe_b, 4 phase,
#              5 on_base, 6 on_tree
ATTACK_LEVELS = [(0, 1, 2, 3), (0, 1, 2), (0, 2), (0,), ()]
# attack key columns: 0 cargo_b, 1 rel_b, 2 dfoecoll_b, 3 phase
BASE_LEVELS = [(0, 1, 2, 3, 4), (0, 1, 2, 3), (0, 1, 2), (0, 1), (0,), ()]
# base key columns: 0 phase, 1 cash_b, 2 ncoll_b, 3 nplan_b, 4 bi


def auc_score(y, s, rng, max_pairs=40000):
    pos = s[y == 1]
    neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    if len(pos) * len(neg) > max_pairs:
        pi = rng.choice(len(pos), size=int(np.sqrt(max_pairs)), replace=False)
        ni = rng.choice(len(neg), size=int(np.sqrt(max_pairs)), replace=False)
        pos, neg = pos[pi], neg[ni]
    wins = np.sum(pos[:, None] > neg[None, :]) + \
        0.5 * np.sum(pos[:, None] == neg[None, :])
    return float(wins / (len(pos) * len(neg)))


def eval_family(name, path, rng):
    d = np.load(path)
    tr = d["split"] == 0
    va = d["split"] == 1
    out = {"family": name,
           "n_train_workers": int(tr.sum()), "n_val_workers": int(va.sum())}
    wtype = d["wtype"]; label = d["label"]
    keys = np.stack([cargo_b(d["cargo"]), dist_b6(d["d_base"]),
                     dtree_b(d["d_tree"]), dfoe_b(d["d_foe"]),
                     phase_b(d["step"]), d["on_base"], d["on_tree"]],
                    axis=1).astype(np.int64)

    def acc(pred, mask):
        return float(np.mean(pred[mask] == label[mask])) if mask.sum() else \
            float("nan")

    res = {}
    # stay + majority baselines
    for tname, t in (("all", None), ("collector", 0), ("planter", 1)):
        m = va if t is None else (va & (wtype == t))
        res[f"stay_{tname}"] = acc(np.zeros(len(label), dtype=int), m)
        maj = {}
        for tt in (0, 1):
            mm = tr & (wtype == tt)
            maj[tt] = int(np.bincount(label[mm], minlength=5).argmax())
        res[f"majority_{tname}"] = acc(
            np.array([maj[int(x)] for x in wtype]), m)
    # inertia
    pa = d["prev_act"]
    chained = va & (pa >= 0)
    res["inertia_coverage"] = float(np.mean(pa[va] >= 0))
    res["inertia_acc"] = float(np.mean(pa[chained] == label[chained]))
    for tname, t in (("collector", 0), ("planter", 1)):
        m = chained & (wtype == t)
        res[f"inertia_{tname}"] = float(np.mean(pa[m] == label[m]))
    # v53 periodic detector (collectors)
    pred_cell = d["pred_cell"]; nxt = d["next_cell"]
    mc = va & (wtype == 0)
    fired = mc & (pred_cell >= 0)
    res["v53pred_coverage"] = float(np.mean(pred_cell[mc] >= 0))
    res["v53pred_cell_acc"] = float(np.mean(pred_cell[fired] == nxt[fired])) \
        if fired.sum() else float("nan")
    res["v53pred_n_fired"] = int(fired.sum())
    # bucket model per type
    bucket_pred = np.zeros(len(label), dtype=int)
    bucket_proba = np.zeros((len(label), 5))
    for t in (0, 1):
        mtr = tr & (wtype == t)
        mva = va & (wtype == t)
        model = BackoffBuckets(WORKER_LEVELS, 5).fit(keys[mtr], label[mtr])
        bucket_proba[mva] = model.predict_proba(keys[mva])
        bucket_pred[mva] = bucket_proba[mva].argmax(axis=1)
    for tname, t in (("all", None), ("collector", 0), ("planter", 1)):
        m = va if t is None else (va & (wtype == t))
        res[f"bucket_{tname}"] = acc(bucket_pred, m)
    # per-strategy val breakdown (only strategies with enough val samples)
    if "strat" in d and "strat_names" in d:
        strat = d["strat"]
        per = {}
        for si, sname in enumerate(d["strat_names"]):
            m = va & (strat == si)
            if m.sum() >= 300:
                per[str(sname)] = {"n": int(m.sum()),
                                   "bucket_top1": acc(bucket_pred, m),
                                   "stay": acc(np.zeros(len(label),
                                                        dtype=int), m)}
        if per:
            res["per_strategy_val"] = per
    # bucket timing on val (fit excluded from the timed window)
    mva = va
    mdl = BackoffBuckets(WORKER_LEVELS, 5).fit(keys[tr & (wtype == 0)],
                                               label[tr & (wtype == 0)])
    t0 = time.time()
    reps = 3
    for _ in range(reps):
        mdl.predict_proba(keys[mva & (wtype == 0)])
    dt = (time.time() - t0) / reps
    n_pred = int((mva & (wtype == 0)).sum())
    res["bucket_us_per_worker"] = dt / max(1, n_pred) * 1e6

    # ---- attack head (collectors) ----
    akeys = np.stack([cargo_b(d["cargo"]), rel_b(d["rel_near"],
                                                  d["d_foe_coll"]),
                      dfoe_b(d["d_foe_coll"]), phase_b(d["step"])],
                     axis=1).astype(np.int64)
    atk = {}
    for ev_name, ev in (("e1", d["e1"]), ("dang", d["dang"])):
        mtr = tr & (wtype == 0)
        mva = va & (wtype == 0)
        ytr = (ev[mtr] == 1).astype(int)
        yva = (ev[mva] == 1).astype(int)
        # binary backoff model: reuse BackoffBuckets with 2 classes
        model = BackoffBuckets(ATTACK_LEVELS, 2, min_count=40).fit(
            akeys[mtr], ytr)
        proba = model.predict_proba(akeys[mva])[:, 1]
        pred = (proba >= 0.5).astype(int)
        tp = int(np.sum((pred == 1) & (yva == 1)))
        fp = int(np.sum((pred == 1) & (yva == 0)))
        fn = int(np.sum((pred == 0) & (yva == 1)))
        atk[ev_name] = {
            "base_rate": float(np.mean(yva)),
            "auc": auc_score(yva, proba, rng),
            "prec@0.5": tp / max(1, tp + fp), "rec@0.5": tp / max(1, tp + fn),
            "n_pos": int(yva.sum())}
        # opportunity subset: nearest foe worker within 2
        opp = d["d_foe"][mva] <= 2
        if opp.sum() > 50 and yva[opp].sum() > 0:
            po = proba[opp]; yo = yva[opp]
            pro = (po >= 0.5).astype(int)
            tp = int(np.sum((pro == 1) & (yo == 1)))
            fp = int(np.sum((pro == 1) & (yo == 0)))
            fn = int(np.sum((pro == 0) & (yo == 1)))
            atk[ev_name + "_opp"] = {
                "base_rate": float(np.mean(yo)),
                "auc": auc_score(yo, po, rng),
                "prec@0.5": tp / max(1, tp + fp),
                "rec@0.5": tp / max(1, tp + fn), "n": int(opp.sum()),
                "n_pos": int(yo.sum())}
    res["attack"] = atk

    # ---- base recruitment head ----
    bkeys = np.stack([phase_b(d["base_step"]), cash_b(d["base_cash"]),
                      cnt_b(d["base_n_coll"]), cnt_b(d["base_n_plan"]),
                      d["base_bi"]], axis=1).astype(np.int64)
    blab = d["base_label"]; btr = d["base_split"] == 0
    bva = d["base_split"] == 1
    bmodel = BackoffBuckets(BASE_LEVELS, 3).fit(bkeys[btr], blab[btr])
    bpred = bmodel.predict_proba(bkeys[bva]).argmax(axis=1)
    bres = {"top1": float(np.mean(bpred == blab[bva])),
            "always_none_top1": float(np.mean(blab[bva] == 0))}
    for c, cn in ((1, "recc"), (2, "recp")):
        m = blab[bva] == c
        bres[f"{cn}_recall"] = float(np.mean(bpred[m] == c)) if m.sum() \
            else float("nan")
        bres[f"n_{cn}"] = int(m.sum())
    res["base"] = bres
    out["worker"] = res
    return out


def main():
    rng = np.random.default_rng(20261103)
    results = []
    for name, path in (("alpha", "pred_data_alpha.npz"),
                       ("deepseek", "pred_data_deepseek.npz"),
                       ("v53", "pred_data_v53.npz")):
        p = os.path.join(HERE, path)
        if not os.path.exists(p):
            print("skip", name)
            continue
        r = eval_family(name, p, rng)
        results.append(r)
        print(json.dumps(r, indent=1, ensure_ascii=False))
    json.dump(results, open(os.path.join(HERE, "pred_results.json"), "w"),
              indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()

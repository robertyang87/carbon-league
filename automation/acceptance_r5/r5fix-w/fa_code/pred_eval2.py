"""pred_eval2 -- addendum evaluation:
  * bucket+prev worker model (bucket tables conditioned on previous action)
  * attack-head threshold sweep on val (operating points for consumers)
  * single-call latency of the stateless bucket model (10-worker batch)
  * export of fitted tables to opponent_predictor_tables.npz
"""

import json
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np  # noqa: E402

from pred_eval import (BackoffBuckets, cargo_b, dist_b6, dtree_b, dfoe_b,
                       phase_b, cash_b, cnt_b, rel_b, ATTACK_LEVELS,
                       BASE_LEVELS, auc_score)

HERE = os.path.dirname(os.path.abspath(__file__))

# key cols for bucket+prev: 0 prev(0-4, 5=unknown), 1 cargo_b, 2 dbase_b,
# 3 dtree_b, 4 dfoe_b, 5 phase, 6 on_base, 7 on_tree
PREV_LEVELS = [(0, 1, 2, 3, 4, 5, 6, 7), (0, 1, 2, 4, 5), (0, 1, 4),
               (0, 1), (0,), (1, 4), ()]
WORKER_LEVELS = [(0, 1, 2, 3, 4, 5, 6), (0, 1, 2, 3, 4), (0, 1, 3, 4),
                 (0, 1, 3), (0, 3), (0,), ()]


def export_tables(store, prefix, model):
    for li, tab in enumerate(model.tables):
        if not tab:
            continue
        encs = np.array(sorted(tab.keys()), dtype=np.int64)
        cnts = np.stack([tab[int(e)] for e in encs])
        store[f"{prefix}_L{li}_enc"] = encs
        store[f"{prefix}_L{li}_cnt"] = cnts


def main():
    rng = np.random.default_rng(20261103)
    results = {}
    store = {}
    for name, path in (("alpha", "pred_data_alpha.npz"),
                       ("deepseek", "pred_data_deepseek.npz"),
                       ("v53", "pred_data_v53.npz")):
        d = np.load(os.path.join(HERE, path))
        tr = d["split"] == 0
        va = d["split"] == 1
        wtype = d["wtype"]; label = d["label"]
        prev = np.where(d["prev_act"] >= 0, d["prev_act"], 5)
        keysP = np.stack([prev, cargo_b(d["cargo"]), dist_b6(d["d_base"]),
                          dtree_b(d["d_tree"]), dfoe_b(d["d_foe"]),
                          phase_b(d["step"]), d["on_base"], d["on_tree"]],
                         axis=1).astype(np.int64)
        keysW = np.stack([cargo_b(d["cargo"]), dist_b6(d["d_base"]),
                          dtree_b(d["d_tree"]), dfoe_b(d["d_foe"]),
                          phase_b(d["step"]), d["on_base"], d["on_tree"]],
                         axis=1).astype(np.int64)
        r = {}
        predP = np.zeros(len(label), dtype=int)
        for t in (0, 1):
            mtr = tr & (wtype == t)
            mva = va & (wtype == t)
            mdl = BackoffBuckets(PREV_LEVELS, 5).fit(keysP[mtr], label[mtr])
            predP[mva] = mdl.predict_proba(keysP[mva]).argmax(axis=1)
            export_tables(store, f"{name}_wp{t}", mdl)
            mdlW = BackoffBuckets(WORKER_LEVELS, 5).fit(keysW[mtr],
                                                        label[mtr])
            export_tables(store, f"{name}_w{t}", mdlW)
        for tn, t in (("all", None), ("collector", 0), ("planter", 1)):
            m = va if t is None else (va & (wtype == t))
            r[f"bucketprev_{tn}"] = float(np.mean(predP[m] == label[m]))
        # attack threshold sweep
        akeys = np.stack([cargo_b(d["cargo"]),
                          rel_b(d["rel_near"], d["d_foe_coll"]),
                          dfoe_b(d["d_foe_coll"]), phase_b(d["step"])],
                         axis=1).astype(np.int64)
        sweep = {}
        for ev_name, ev in (("e1", d["e1"]), ("dang", d["dang"])):
            mtr = tr & (wtype == 0)
            mva = va & (wtype == 0)
            mdl = BackoffBuckets(ATTACK_LEVELS, 2, min_count=40).fit(
                akeys[mtr], (ev[mtr] == 1).astype(int))
            proba = mdl.predict_proba(akeys[mva])[:, 1]
            yva = (ev[mva] == 1).astype(int)
            rows = {}
            for th in (0.1, 0.2, 0.3, 0.5):
                pr = (proba >= th).astype(int)
                tp = int(np.sum((pr == 1) & (yva == 1)))
                fp = int(np.sum((pr == 1) & (yva == 0)))
                fn = int(np.sum((pr == 0) & (yva == 1)))
                rows[str(th)] = {"prec": tp / max(1, tp + fp),
                                 "rec": tp / max(1, tp + fn)}
            sweep[ev_name] = rows
            export_tables(store, f"{name}_atk_{ev_name}", mdl)
        r["attack_sweep"] = sweep
        # base tables export
        bkeys = np.stack([phase_b(d["base_step"]), cash_b(d["base_cash"]),
                          cnt_b(d["base_n_coll"]), cnt_b(d["base_n_plan"]),
                          d["base_bi"]], axis=1).astype(np.int64)
        bmdl = BackoffBuckets(BASE_LEVELS, 3).fit(
            bkeys[d["base_split"] == 0], d["base_label"][d["base_split"] == 0])
        export_tables(store, f"{name}_base", bmdl)
        # single-call latency: 10-worker batch through the stateless model
        mdlW = BackoffBuckets(WORKER_LEVELS, 5).fit(
            keysW[tr & (wtype == 0)], label[tr & (wtype == 0)])
        batch = keysW[va & (wtype == 0)][:10]
        for _ in range(20):
            mdlW.predict_proba(batch)
        t0 = time.time()
        reps = 200
        for _ in range(reps):
            mdlW.predict_proba(batch)
        r["bucket_call_us_10workers"] = (time.time() - t0) / reps * 1e6
        results[name] = r
        print(name, json.dumps(r, indent=1))
    np.savez_compressed(os.path.join(HERE, "opponent_predictor_tables.npz"),
                        **store)
    json.dump(results, open(os.path.join(HERE, "pred_results2.json"), "w"),
              indent=1)
    print("tables exported:", len(store), "arrays")


if __name__ == "__main__":
    main()

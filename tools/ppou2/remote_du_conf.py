#!/usr/bin/env python3
"""ppou2 it=20 EARLY CONFIRMATION exam paired Delta-U + bootstrap (B=20000).

Reserve blocks (one-shot, strategy-group ruling 2026-11-15 11:19):
  M2 seeds 984000-984499, E2 seeds 984500-984999 (500 clusters each).
Candidate (ckpt_20 pack ah f4fff163, used as-is, no repack):
  M2: ppou2_conf_it20_m2.jsonl ; E2: ppou2_conf_it20_e2.jsonl
Baselines (frozen start pack ah 11d80059, freshly produced on the blocks):
  M2 A: ppou2_conf_base_m2_a.jsonl ; M2 B: ppou2_conf_base_m2_b.jsonl
  E2:   ppou2_conf_base_e2.jsonl
Method identical to the checkpoint script: pairing by seed, cluster = seed
(value = mean over the seed's lines), M2 reading = mean of the A-paired
and B-paired per-seed deltas, combined = 1000 clusters (500 M2 batch-mean
+ 500 E2). Bootstrap resamples seed clusters within each examiner,
B=20000, rng seed 20261115. Criterion: combined CI lower bound > 0.
"""
import json

import numpy as np

BASE = "/root/carbon/probe_m2/data/bigeval_live/"
B = 20000
rng = np.random.default_rng(20261115)


def load_u(tag):
    d = {}
    n = 0
    errs = 0
    ahs = set()
    hhs = set()
    with open(BASE + tag + ".jsonl") as f:
        for line in f:
            r = json.loads(line)
            n += 1
            if r.get("err"):
                errs += 1
            ahs.add(r.get("ah"))
            hhs.add(r.get("hh"))
            d.setdefault(int(r["seed"]), []).append(float(r["u"]))
    return d, n, errs, ahs, hhs


def paired(cand, base):
    seeds = sorted(set(cand) & set(base))
    dv = np.array([np.mean(cand[s]) - np.mean(base[s]) for s in seeds])
    return seeds, dv


def boot(dv, n_boot=B):
    n = len(dv)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = dv[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def report(name, dv):
    lo, hi = boot(dv)
    out = {"field": name, "n_clusters": int(len(dv)),
           "point": float(dv.mean()), "ci_lo": lo, "ci_hi": hi}
    print(json.dumps(out))
    return out


cand_m2, n1, e1, ah1, hh1 = load_u("ppou2_conf_it20_m2")
cand_e2, n2, e2, ah2, hh2 = load_u("ppou2_conf_it20_e2")
base_m2a, nb1, eb1, ahb1, hhb1 = load_u("ppou2_conf_base_m2_a")
base_m2b, nb2, eb2, ahb2, hhb2 = load_u("ppou2_conf_base_m2_b")
base_e2, nb3, eb3, ahb3, hhb3 = load_u("ppou2_conf_base_e2")

meta = {"cand_m2": {"lines": n1, "errs": e1, "ah": sorted(ah1), "hh": sorted(hh1)},
        "cand_e2": {"lines": n2, "errs": e2, "ah": sorted(ah2), "hh": sorted(hh2)},
        "base_m2a": {"lines": nb1, "errs": eb1, "ah": sorted(ahb1)},
        "base_m2b": {"lines": nb2, "errs": eb2, "ah": sorted(ahb2)},
        "base_e2": {"lines": nb3, "errs": eb3, "ah": sorted(ahb3)}}
print(json.dumps({"meta": meta}))

s_a, dv_a = paired(cand_m2, base_m2a)
s_b, dv_b = paired(cand_m2, base_m2b)
assert s_a == s_b, "M2 A/B seed sets differ"
dv_m2 = (dv_a + dv_b) / 2.0
s_e, dv_e2 = paired(cand_e2, base_e2)

r_m2a = report("M2_vs_batchA", dv_a)
r_m2b = report("M2_vs_batchB", dv_b)
r_m2 = report("M2_batchmean", dv_m2)
r_e2 = report("E2", dv_e2)
dv_all = np.concatenate([dv_m2, dv_e2])
r_all = report("COMBINED", dv_all)

cand_u_m2 = float(np.mean([np.mean(v) for v in cand_m2.values()]))
cand_u_e2 = float(np.mean([np.mean(v) for v in cand_e2.values()]))
base_u_m2 = float(np.mean([np.mean(v) for v in base_m2a.values()] +
                          [np.mean(v) for v in base_m2b.values()]))
base_u_e2 = float(np.mean([np.mean(v) for v in base_e2.values()]))
print(json.dumps({"cand_meanU_m2": cand_u_m2, "cand_meanU_e2": cand_u_e2,
                  "base_meanU_m2_batches": base_u_m2, "base_meanU_e2": base_u_e2}))
print("DONE")

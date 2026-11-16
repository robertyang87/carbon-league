#!/usr/bin/env python3
"""Regression 1+2 for the route-B dedup pack:
  R1: embedded _SRC five-source md5 set identical to the original pack and
      to build2/build_record.json.
  R2: dequantised weights w1 AND w2 of the dedup pack are array-for-array
      identical to the original pack's (all tensors + meta).
Also checks: CONF identical, _B2 is _B1 alias identity in the dedup module.
Writes a JSON verdict to dedup/check_src_weights.json and prints it.
"""
import hashlib
import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SW = os.path.dirname(HERE)
sys.path.insert(0, os.path.expanduser("~/workspace/search-group/proto/vendor"))
from fastsim import FastCarbon  # noqa: E402

ORIG = os.path.join(SW, "echo4_search_int8.py")
DEDUP = os.path.join(HERE, "echo4_search_int8_dedup.py")
RECORD = os.path.join(SW, "build2", "build_record.json")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def md5s(s):
    return hashlib.md5(s.encode()).hexdigest()


def main():
    out = {"orig": ORIG, "dedup": DEDUP}
    mo = load("pack_orig_r12", ORIG)
    md = load("pack_dedup_r12", DEDUP)

    # ---- R1: _SRC md5 set ----
    rec = json.load(open(RECORD))["embedded_src_md5"]
    src_o = {k: md5s(v) for k, v in mo._SRC.items()}
    src_d = {k: md5s(v) for k, v in md._SRC.items()}
    out["src_md5_orig"] = src_o
    out["src_md5_dedup"] = src_d
    out["src_md5_record"] = rec
    out["R1_src_set_equal_orig_vs_dedup"] = src_o == src_d
    out["R1_src_set_equal_vs_record"] = src_d == rec
    out["R1_mods_equal"] = list(mo._MODS) == list(md._MODS)
    out["conf_equal"] = mo.CONF == md.CONF

    # ---- alias identity in the dedup module ----
    out["alias_B2_is_B1"] = md._B2 is md._B1
    out["alias_Q2_is_Q1"] = md._Q2 is md._Q1
    out["alias_M2_is_M1"] = md._M2 is md._M1

    # ---- R2: dequantised weights ----
    sim = FastCarbon()
    sim.reset(seed=986200)
    obs0 = sim.obs(0)
    mo.agent(obs0, {})
    md.agent(obs0, {})
    res = {}
    for slot, idx in (("w1", 0), ("w2", 1)):
        wo, wd = mo._W[idx], md._W[idx]
        keys_o, keys_d = set(wo), set(wd)
        mism = []
        n_arr = 0
        for k in sorted(keys_o & keys_d):
            a, b = wo[k], wd[k]
            if isinstance(a, np.ndarray):
                n_arr += 1
                if a.shape != b.shape or a.dtype != b.dtype or not np.array_equal(a, b):
                    mism.append(k)
            elif a != b:
                mism.append(k)
        res[slot] = {"keys_orig": len(keys_o), "keys_dedup": len(keys_d),
                     "key_sets_equal": keys_o == keys_d,
                     "arrays_compared": n_arr, "mismatches": mism}
    out["weights"] = res
    out["R2_w1_all_equal"] = (res["w1"]["key_sets_equal"] and not res["w1"]["mismatches"])
    out["R2_w2_all_equal"] = (res["w2"]["key_sets_equal"] and not res["w2"]["mismatches"])
    out["R2_w2_present_in_dedup"] = md._W[1] is not None

    out["verdict"] = ("PASS" if (out["R1_src_set_equal_orig_vs_dedup"] and out["R1_src_set_equal_vs_record"]
                                 and out["R1_mods_equal"] and out["conf_equal"]
                                 and out["alias_B2_is_B1"] and out["R2_w1_all_equal"]
                                 and out["R2_w2_all_equal"] and out["R2_w2_present_in_dedup"]) else "FAIL")
    json.dump(out, open(os.path.join(HERE, "check_src_weights.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))
    return 0 if out["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())

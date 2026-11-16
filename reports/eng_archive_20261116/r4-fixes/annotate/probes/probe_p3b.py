"""P3(b) probe: validate_records NaN bypass on q/priors (annotator),
and the join mirror _validate_annot_records. Run against a run dir whose
k0_annotate.py / k0_join.py are the version under test.

Usage: python3 probe_p3b.py RUN_DIR
Prints one line per variant: PASS (gate accepted) or DIE <msg>.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(sys.argv[1]))
import numpy as np  # noqa: E402
import k0_annotate as A  # noqa: E402


def clean_arrs():
    n, K = 2, 4
    return {"gid": np.array([0, 0], np.int32),
            "step": np.array([206, 207], np.int32),
            "seat": np.array([0, 1], np.int8),
            "arms": np.zeros((n, K, 12), np.int16),
            "n_arms": np.array([2, 2], np.int16),
            "priors": np.array([[.5, .5, 0, 0], [.5, .5, 0, 0]]),
            "q": np.array([[.5, .5, 0, 0], [.25, .75, 0, 0]]),
            "Q": np.array([[.5, .6, np.nan, np.nan], [.4, .7, np.nan, np.nan]]),
            "Q_std": np.zeros((n, K)),
            "pi0": np.array([1.0, 1.0]), "pi1": np.array([1.0, 1.0]),
            "pi": np.array([1.0, 1.0]),
            "belief": np.array([0, 0], np.int8),
            "n_unknown": np.array([3, 3], np.int32)}


VARIANTS = {}


def variant(name):
    def deco(fn):
        VARIANTS[name] = fn
        return fn
    return deco


@variant("clean")
def v_clean(a):
    return a


@variant("q_nan")
def v_qnan(a):
    a["q"][0, 0] = np.nan
    return a


@variant("priors_nan")
def v_pnan(a):
    a["priors"][0, 0] = np.nan
    return a


@variant("priors_inf")
def v_pinf(a):
    a["priors"][0, 0] = np.inf
    return a


@variant("priors_sum_half")
def v_psum(a):
    a["priors"][0] = [.25, .25, 0, 0]
    return a


def run(fn, label):
    for name, mut in VARIANTS.items():
        arrs = mut(clean_arrs())
        try:
            fn(arrs, (200, 299), 4, "probe")
            print(f"{label} {name}: PASS")
        except SystemExit as e:
            print(f"{label} {name}: DIE {e}")


run(A.validate_records, "annotator")
try:
    import k0_join as J
    run(J._validate_annot_records, "join     ")
except ImportError as e:
    print("join module not importable here:", e)

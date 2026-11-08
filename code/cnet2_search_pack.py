"""cnet2_search_pack -- single-file submission: cnet2 b128 policy (v12) + cnet2_search rollout layer.

python3 cnet2_search_pack.py OUT.py "CONF_DICT_LITERAL" [POLICY.npz] [ROLLOUT.npz]
Embeds cnet2_feat, cnet2_np, fastsim, cnet2_search sources (exec'd into private module objects, the
plain sys.modules names are restored afterwards) and both weight sets, int8 per-row quantised exactly
like cnet2_pack (dequantised == data/weights/*_q8.npz).  No __file__ use, no file access at play time.
"""
import base64
import json
import os
import sys
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cnet2_pack as P  # noqa: E402

MODS = ["cnet2_feat", "cnet2_np", "fastsim", "cnet2_search"]

TEMPLATE = '''"""Carbon League: cnet2 b128 policy (v12) + cnet2_search test-time rollout search (pure NumPy)."""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_k] = "1"   # force: a multi-game platform host must not run multithreaded BLAS
import base64, sys, types, zlib
import numpy as np

_SRC = %(src)r
_MODS = %(mods)r
_saved = {k: sys.modules.get(k) for k in _MODS}
_P = {}
try:
    for _k in _MODS:
        sys.modules.pop(_k, None)
    for _k in _MODS:
        _m = types.ModuleType(_k)
        sys.modules[_k] = _m
        exec(compile(_SRC[_k], _k + ".py", "exec"), _m.__dict__)
        _P[_k] = _m
finally:
    for _k, _v in _saved.items():
        if _v is None:
            sys.modules.pop(_k, None)
        else:
            sys.modules[_k] = _v
_S = _P["cnet2_search"]
CONF = %(conf)s
_Q1 = %(q1)s
_M1 = %(m1)r
_B1 = %(b1)r
_Q2 = %(q2)s
_M2 = %(m2)r
_B2 = %(b2)r


def _dequantize(raw, meta):
    out, off = {}, 0
    for k, shape, kind, nb in meta:
        cnt = int(np.prod(shape))
        if kind == "int8":
            q = np.frombuffer(raw, np.int8, cnt, off).astype(np.float32).reshape(shape[0], -1)
            sc = np.frombuffer(raw, "<f4", shape[0], off + cnt)
            out[k] = (q * sc[:, None]).reshape(shape)
        elif kind == "f16":
            out[k] = np.frombuffer(raw, "<f2", cnt, off).astype(np.float32).reshape(shape)
        else:
            out[k] = np.frombuffer(raw, "<f4", cnt, off).copy().reshape(shape)
        off += nb
    return out


_W = None
_AG = None


def agent(observation, configuration):
    global _W, _AG
    if _W is None:
        w1 = _dequantize(zlib.decompress(base64.b64decode(_B1)), _Q1); w1["meta"] = np.array(_M1)
        w2 = _dequantize(zlib.decompress(base64.b64decode(_B2)), _Q2); w2["meta"] = np.array(_M2)
        _W = (w1, w2)
    if _AG is None or observation["step"] == 0:
        _AG = _S.make_search_agent(CONF, weights=_W[0], rweights=_W[1])
    return _AG(observation, configuration)
'''


def main(out, conf, wp=None, wr=None):
    wd = os.path.join(HERE, "..", "data", "weights")
    wp = wp or os.path.join(wd, "cnet2_b128.npz")
    wr = wr or os.path.join(wd, "cnet2_s64.npz")
    parts = {}
    qmodes = {"1": os.environ.get("CNET2_QMODE", "int8"), "2": os.environ.get("CNET2_QMODE_ROLL", os.environ.get("CNET2_QMODE", "int8"))}
    for tag, path in (("1", wp), ("2", wr)):
        w = dict(np.load(path))
        P.QMODE = qmodes[tag]                    # policy and rollout net can be packed differently (int8 / f16)
        meta, raw, deq = P.quantize(w)
        qp = path.replace(".npz", "_q8.npz")
        if P.QMODE == "int8" and os.path.exists(qp):
            q = np.load(qp)
            assert all(np.array_equal(deq[k], q[k]) for k in deq), "dequantised != " + qp
        parts[tag] = (json.dumps(meta), str(w["meta"]), base64.b64encode(zlib.compress(raw, 9)).decode())
    eval(conf)  # must be a valid literal
    src = {m: open(os.path.join(HERE, m + ".py")).read() for m in MODS}
    code = TEMPLATE % dict(src=src, mods=MODS, conf=conf, q1=parts["1"][0], m1=parts["1"][1], b1=parts["1"][2],
                           q2=parts["2"][0], m2=parts["2"][1], b2=parts["2"][2])
    open(out, "w").write(code)
    print(json.dumps({"out": out, "bytes": len(code)}))


if __name__ == "__main__":
    main(*sys.argv[1:])

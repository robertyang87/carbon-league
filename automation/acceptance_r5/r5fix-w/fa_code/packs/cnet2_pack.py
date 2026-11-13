"""cnet2_pack -- single-file submission for a cnet2 policy.

Embeds cnet2_feat.py, cnet2_np.py, cnet2_agent.py sources (exec'd into module
objects) and the weights (int8 per-output-row for large tensors, fp32 for the
rest; zlib + base64). Also writes the dequantised weights as an .npz so an
unpacked wrapper can be checked game-for-game against the packed file.

Usage: python cnet2_pack.py WEIGHTS.npz OUT.py [teacher] [shield nocollide endgame as 0/1]
"""

import base64
import json
import os
import sys
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODS = ["cnet2_feat", "cnet2_np", "cnet2_agent"]


QMODE = os.environ.get("CNET2_QMODE", "int8")   # int8 | f16 (large tensors)


def quantize(w):
    meta, blobs, deq = [], [], {}
    for k in sorted(w):
        if k == "meta":
            continue
        v = np.asarray(w[k], np.float32)
        if v.ndim >= 2 and v.size >= 4096 and QMODE == "f16":
            b = v.astype("<f2")
            blobs.append(b.tobytes())
            meta.append([k, list(v.shape), "f16", b.nbytes])
            deq[k] = b.astype(np.float32)
        elif v.ndim >= 2 and v.size >= 4096:
            flat = v.reshape(v.shape[0], -1)
            sc = np.abs(flat).max(1) / 127.0
            sc[sc == 0] = 1.0
            q = np.clip(np.round(flat / sc[:, None]), -127, 127).astype(np.int8)
            sc = sc.astype("<f4")
            blobs += [q.tobytes(), sc.tobytes()]
            meta.append([k, list(v.shape), "int8", q.nbytes + sc.nbytes])
            deq[k] = (q.astype(np.float32) * sc[:, None]).reshape(v.shape)
        else:
            b = v.astype("<f4")
            blobs.append(b.tobytes())
            meta.append([k, list(v.shape), "f32", b.nbytes])
            deq[k] = b.copy()
    return meta, b"".join(blobs), deq


def dequantize(raw, meta):
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
    assert off == len(raw)
    return out


TEMPLATE = '''"""cnet2 packed submission (pure NumPy)."""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, "1")
import base64, json, sys, types, zlib
import numpy as np

_SRC = %(src)r
for _name in %(mods)r:
    if _name in sys.modules and getattr(sys.modules[_name], "_CNET2_PACKED", False):
        continue
    _m = types.ModuleType(_name)
    _m.__dict__["_CNET2_PACKED"] = True
    _m.__dict__["__file__"] = globals().get("__file__", _name + ".py")
    sys.modules[_name] = _m
    exec(compile(_SRC[_name], _name + ".py", "exec"), _m.__dict__)
_QMETA = %(qmeta)s
_WMETA = %(wmeta)r
_W64 = %(blob)r
_OPTS = %(opts)r


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


_AG = None
_WTS = None


def agent(observation, configuration):
    global _AG, _WTS
    if _WTS is None:
        _WTS = _dequantize(zlib.decompress(base64.b64decode(_W64)), _QMETA)
        _WTS["meta"] = np.array(_WMETA)
    if _AG is None or observation["step"] == 0:
        _AG = sys.modules["cnet2_agent"].make_agent(_WTS, **_OPTS)
    return _AG(observation, configuration)
'''


def main(wpath, out, teacher=1, shield=0, nocollide=0, endgame=0, pshield=0, rguard=0):
    w = dict(np.load(wpath))
    meta, raw, deq = quantize(w)
    deq_path = wpath.replace(".npz", "_q8.npz" if QMODE == "int8" else "_f16.npz")
    np.savez(deq_path, **deq, meta=w["meta"])
    src = {m: open(os.path.join(HERE, m + ".py")).read() for m in MODS}
    opts = dict(teacher=int(teacher), shield=bool(int(shield)), nocollide=bool(int(nocollide)),
                endgame=bool(int(endgame)), pshield=bool(int(pshield)), rguard=bool(int(rguard)))
    code = TEMPLATE % dict(src=src, mods=MODS, qmeta=json.dumps(meta), wmeta=str(w["meta"]),
                           blob=base64.b64encode(zlib.compress(raw, 9)).decode(), opts=opts)
    open(out, "w").write(code)
    print(json.dumps({"out": out, "bytes": len(code), "deq": deq_path, "opts": opts}))


if __name__ == "__main__":
    main(*sys.argv[1:])

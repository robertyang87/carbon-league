"""nn_pack_search2 -- single-file submission: nnfeat + nn_agent + fastsim + nn_search2 + weights.

python3 nn_pack_search2.py WEIGHTS.npz OUT.py "CONF_DICT_LITERAL" [VALUE_WEIGHTS.npz]
Module sources are embedded as strings and executed into private module objects registered in
sys.modules under unique names (no file access at play time). Needs only stdlib + NumPy.
"""
import base64
import io
import os
import sys
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def blob_of(path):
    d = np.load(path, allow_pickle=False)
    sd = {k: d[k].astype(np.float32) for k in d.files if k != "meta"}
    buf = io.BytesIO()
    np.savez(buf, **sd)
    return base64.b85encode(zlib.compress(buf.getvalue(), 9)).decode()


def main(weights, out, conf, vweights=None):
    srcs = {m: open(os.path.join(HERE, m + ".py")).read() for m in ("nnfeat", "nn_agent", "fastsim", "nn_search2")}
    lines = ['"""Carbon League: v9 policy (nn_master64 + rule layer) + nn_search2 tactical rollout search.\n"""',
             "import sys as _sys, types as _types, base64 as _b, io as _io, zlib as _z",
             "import numpy as np",
             "_SRC = %r" % srcs,
             "_BLOB = %r" % blob_of(weights),
             "_VBLOB = %r" % (blob_of(vweights) if vweights else None),
             "CONF = %s" % conf,
             "",
             "def _mod(name):",
             "    m = _types.ModuleType(name)",
             "    m.__file__ = __file__ if '__file__' in globals() else name",
             "    _sys.modules[name] = m",
             "    exec(compile(_SRC[name], name, 'exec'), m.__dict__)",
             "    return m",
             "",
             "# private copies: register under the plain names only while loading, then restore",
             "_saved = {k: _sys.modules.get(k) for k in _SRC}",
             "for _k in ('nnfeat', 'nn_agent', 'fastsim', 'nn_search2'):",
             "    _sys.modules.pop(_k, None)",
             "_S = None",
             "try:",
             "    for _k in ('nnfeat', 'nn_agent', 'fastsim', 'nn_search2'):",
             "        _m = _mod(_k)",
             "    _S = _sys.modules['nn_search2']",
             "finally:",
             "    for _k, _v in _saved.items():",
             "        if _v is None:",
             "            _sys.modules.pop(_k, None)",
             "        else:",
             "            _sys.modules[_k] = _v",
             "",
             "def _load(blob):",
             "    d = np.load(_io.BytesIO(_z.decompress(_b.b85decode(blob))))",
             "    return {k: d[k].astype(np.float32) for k in d.files}",
             "",
             "_W = _load(_BLOB)",
             "_VW = _load(_VBLOB) if _VBLOB else None",
             "_AG = None",
             "",
             "",
             "def agent(observation, configuration):",
             "    global _AG",
             "    if _AG is None or observation['step'] == 0:",
             "        _AG = _S.make_search_agent(CONF, weights=_W, vweights=_VW)",
             "    return _AG(observation, configuration)",
             ""]
    open(out, "w").write("\n".join(lines))
    print(out, os.path.getsize(out))


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], a[2], a[3] if len(a) > 3 else None)

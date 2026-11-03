"""nn_pack -- build a single-file submission from nnfeat.py + nn_agent.py + weights.

Usage: python3 nn_pack.py WEIGHTS.npz OUT.py ["header text"]
The output defines agent(observation, configuration) at top level, needs only
the standard library + NumPy, and embeds the weights (float16, zlib, base64).
"""

import base64
import io
import os
import re
import sys
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def strip_module(src, drop_imports):
    out = []
    for line in src.splitlines():
        if any(re.match(p, line) for p in drop_imports):
            continue
        out.append(line)
    return "\n".join(out)


def main(weights, out, header=""):
    d = np.load(weights, allow_pickle=False)
    sd = {k: d[k].astype(np.float16) for k in d.files if k != "meta"}
    buf = io.BytesIO()
    np.savez(buf, **sd)
    blob = base64.b85encode(zlib.compress(buf.getvalue(), 9)).decode()
    feat = open(os.path.join(HERE, "nnfeat.py")).read()
    ag = open(os.path.join(HERE, "nn_agent.py")).read()
    ag = strip_module(ag, [r"^import nnfeat as F", r"^sys\.path\.insert", r"^HERE = "])
    ag = ag.replace("F.", "")
    ag = re.sub(r'\n_AGENT = None\n\n\ndef agent\(observation, configuration\):.*?return _AGENT\(observation, configuration\)\n',
                "\n", ag, flags=re.S)
    src = ['"""' + (header or "Carbon League NN policy (egocentric CNN, NumPy inference).") + '\n"""',
           "# ---- nnfeat ----", feat.split('"""', 2)[2],
           "# ---- nn_agent ----", ag.split('"""', 2)[2],
           "# ---- embedded weights ----",
           "_BLOB = %r" % blob,
           "",
           "def _load_blob():",
           "    import base64 as _b, io as _io, zlib as _z",
           "    d = np.load(_io.BytesIO(_z.decompress(_b.b85decode(_BLOB))))",
           "    return {k: d[k].astype(np.float32) for k in d.files}",
           "",
           "_W = _load_blob()",
           "_AGENT = None",
           "",
           "",
           "def agent(observation, configuration):",
           "    global _AGENT",
           "    if _AGENT is None or observation['step'] == 0:",
           "        _AGENT = make_agent(_W)",
           "    return _AGENT(observation, configuration)",
           ""]
    text = "\n".join(src)
    with open(out, "w") as f:
        f.write(text)
    print(out, len(text.encode()), "bytes")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "")

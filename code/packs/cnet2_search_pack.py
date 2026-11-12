"""cnet2_search_pack -- single-file submission: cnet2 b128 policy (v12) + cnet2_search rollout layer.

python3 cnet2_search_pack.py OUT.py "CONF_DICT_LITERAL" [POLICY.npz] [ROLLOUT.npz]
CNET2_NO_ROLL=1 (conf with "SEARCH": False, "ROLL": "self") leaves the rollout net out -- the upload limit is 10 MB.
CNET2_ENS=a.npz,b.npz embeds extra policy nets; the pack then plays the mean of the members' logits (cnet2_np.Ensemble).
Embeds cnet2_feat, cnet2_np, fastsim, cnet2_search sources (exec'd into private module objects, the
plain sys.modules names are restored afterwards) and both weight sets, int8 per-row quantised exactly
like cnet2_pack (dequantised == data/weights/*_q8.npz).  No __file__ use, no file access at play time.
"""
import base64
import hashlib
import json
import os
import sys
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cnet2_pack as P  # noqa: E402

MODS = ["cnet2_feat", "cnet2_np", "fastsim", "cnet2_search"]

# --- N01 build-time source-consistency gate --------------------------------
# This packer embeds the module copies sitting next to it (HERE =
# code/packs/). Those copies must stay byte-identical to the authoritative
# sources in code/; drifting copies once shipped a stale pre-C01/C02 search
# (see reports/REVIEW_FIXES_20261113_R2.md, N01). ALLOW_DIFF is the explicit
# allowlist of module names permitted to differ from code/; it is empty by
# default, so ANY drift aborts the build whenever the authoritative tree is
# present. On a bare remote build machine without a code/ tree the gate
# degrades gracefully: the embedded hashes are still computed, embedded in
# the pack (_BUILD_HASHES) and printed in the build record, and the runbook
# requires one --check-source run in a repo checkout before release.
ALLOW_DIFF = ()

def _md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def module_hashes(base=None):
    base = base or HERE
    return {m: _md5_file(os.path.join(base, m + ".py")) for m in MODS}

def check_source(auth_dir=None):
    """Compare every embedded module against its authoritative code/ source.

    Returns (ok, record). The authoritative tree is auth_dir or HERE/..
    (this packer lives in code/packs/). When that tree does not contain all
    of MODS the check is skipped (record["checked"] is False) and ok is
    True: hash recording still happens, comparison is deferred to a repo
    checkout (packer --check-source, required by the runbook pre-release).
    """
    embedded = module_hashes()
    record = {"embedded_hashes": embedded, "authoritative_dir": None,
              "authoritative_hashes": {}, "checked": False,
              "mismatches": [], "missing_authoritative": [], "ok": True}
    auth = auth_dir or os.path.dirname(os.path.abspath(HERE))
    if not all(os.path.exists(os.path.join(auth, m + ".py")) for m in MODS):
        record["missing_authoritative"] = list(MODS)
        return True, record
    record["checked"] = True
    record["authoritative_dir"] = auth
    for m in MODS:
        ah = _md5_file(os.path.join(auth, m + ".py"))
        record["authoritative_hashes"][m] = ah
        if ah != embedded[m] and m not in ALLOW_DIFF:
            record["mismatches"].append(m)
    record["ok"] = not record["mismatches"]
    return record["ok"], record


TEMPLATE = '''"""Carbon League: cnet2 b128 policy (v12) + cnet2_search test-time rollout search (pure NumPy)."""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_k] = "1"   # force: a multi-game platform host must not run multithreaded BLAS
import base64, sys, types, zlib
import numpy as np

_SRC = %(src)r
_MODS = %(mods)r
_BUILD_HASHES = %(hashes)r  # N01 build record: md5 of each embedded module source, written by cnet2_search_pack at build time
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
_Q3 = %(q3)s
_M3 = %(m3)r
_B3 = %(b3)r
_ENS = %(ens)s


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
        w2 = None                              # packs built with CNET2_NO_ROLL=1 (pure net, no search) carry no rollout net
        if _B2:
            w2 = _dequantize(zlib.decompress(base64.b64decode(_B2)), _Q2); w2["meta"] = np.array(_M2)
        w3 = None
        if _B3:
            w3 = _dequantize(zlib.decompress(base64.b64decode(_B3)), _Q3); w3["meta"] = np.array(_M3)
        we = []
        for _q, _m, _b in _ENS:                 # extra ensemble members (conf key ENS is filled in here)
            _e = _dequantize(zlib.decompress(base64.b64decode(_b)), _q); _e["meta"] = np.array(_m); we.append(_e)
        _W = (w1, w2, w3, we)
    if _AG is None or observation["step"] == 0:
        _AG = _S.make_search_agent(dict(CONF, ENS=_W[3]) if _W[3] else CONF, weights=_W[0], rweights=_W[1], oweights=_W[2])
    a = _AG(observation, configuration)
    agent.last_stats = _AG.stats      # local eval only (per-game search stats); harmless on the platform
    return a
'''


def main(out, conf, wp=None, wr=None, wo=None, ens=()):
    # P3: this packer lives in code/packs/; the weights live in the repo-root
    # data/weights/ (cnet2_b128.npz / cnet2_s64.npz), two levels up. The old
    # HERE/../data/weights pointed at the nonexistent code/data/weights.
    wd = os.path.join(HERE, "..", "..", "data", "weights")
    wp = wp or os.path.join(wd, "cnet2_b128.npz")
    wr = wr or os.path.join(wd, "cnet2_s64.npz")
    parts = {}
    qmodes = {"1": os.environ.get("CNET2_QMODE", "int8"), "2": os.environ.get("CNET2_QMODE_ROLL", os.environ.get("CNET2_QMODE", "int8"))}
    qmodes["3"] = qmodes["1"]
    parts["3"] = ("None", "", "")                 # optional opening-phase policy (OPEN_NET_STEP)
    no_roll = os.environ.get("CNET2_NO_ROLL") == "1"
    if no_roll:      # the 10 MB upload limit: a pure-net pack does not need the rollout net (~1.1 MB int8, ~2.2 MB f16)
        c_ = eval(conf)
        assert c_.get("SEARCH") is False and c_.get("ROLL", "s64") != "s64", \
            'CNET2_NO_ROLL=1 needs "SEARCH": False and "ROLL": "self" in the conf (the policy net then stands in for the rollout net)'
        parts["2"] = ("None", "", "")
    for tag, path in (("1", wp),) + ((("2", wr),) if not no_roll else ()) + ((("3", wo),) if wo else ()):
        w = dict(np.load(path))
        P.QMODE = qmodes[tag]                    # policy and rollout net can be packed differently (int8 / f16)
        meta, raw, deq = P.quantize(w)
        qp = path.replace(".npz", "_q8.npz")
        if P.QMODE == "int8" and os.path.exists(qp):
            q = np.load(qp)
            assert all(np.array_equal(deq[k], q[k]) for k in deq), "dequantised != " + qp
        parts[tag] = (json.dumps(meta), str(w["meta"]), base64.b64encode(zlib.compress(raw, 9)).decode())
    ens_parts = []
    for path in ens:                             # CNET2_ENS=a.npz,b.npz : extra policy nets averaged with the main one
        w = dict(np.load(path)); P.QMODE = qmodes["1"]
        meta, raw, deq = P.quantize(w)
        ens_parts.append((meta, str(w["meta"]), base64.b64encode(zlib.compress(raw, 9)).decode()))
    assert "ENS" not in eval(conf), "pass ensemble members with CNET2_ENS, not in the conf (paths do not exist on the ladder)"
    eval(conf)  # must be a valid literal
    hashes, check = module_hashes(), check_source()[1]
    if check["checked"] and not check["ok"]:
        print(json.dumps({"error": "N01 source-consistency gate: embedded modules differ from authoritative code/ sources",
                          "mismatches": check["mismatches"]}), file=sys.stderr)
        raise SystemExit(2)
    if not check["checked"]:
        print("WARNING: authoritative code/ sources not found next to packs/; embedding module "
              "hashes without comparison -- run 'cnet2_search_pack.py --check-source' in a repo "
              "checkout before release (runbook requirement)", file=sys.stderr)
    src = {m: open(os.path.join(HERE, m + ".py")).read() for m in MODS}
    code = TEMPLATE % dict(src=src, mods=MODS, hashes=hashes, conf=conf, q1=parts["1"][0], m1=parts["1"][1], b1=parts["1"][2],
                           q2=parts["2"][0], m2=parts["2"][1], b2=parts["2"][2],
                           q3=parts["3"][0], m3=parts["3"][1], b3=parts["3"][2], ens=repr(ens_parts))
    open(out, "w").write(code)
    print(json.dumps({"out": out, "bytes": len(code), "module_hashes": hashes, "source_check": check}))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--check-source":
        # Repo-checkout consistency check (runbook pre-release requirement):
        #   python3 cnet2_search_pack.py --check-source [AUTHORITATIVE_CODE_DIR]
        # Exit 0 = every embedded module matches code/ (or is allowlisted);
        # exit 1 = drift found (or no authoritative tree to compare against).
        _ok, _rec = check_source(sys.argv[2] if len(sys.argv) > 2 else None)
        print(json.dumps(_rec, indent=1, sort_keys=True))
        sys.exit(0 if (_ok and _rec["checked"]) else 1)
    main(*sys.argv[1:], ens=[e for e in os.environ.get("CNET2_ENS", "").split(",") if e])

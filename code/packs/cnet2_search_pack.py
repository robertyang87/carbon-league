"""cnet2_search_pack -- single-file submission: cnet2 b128 policy (v12) + cnet2_search rollout layer.

python3 cnet2_search_pack.py OUT.py "CONF_DICT_LITERAL" [POLICY.npz] [ROLLOUT.npz]
CNET2_NO_ROLL=1 (conf with "SEARCH": False, "ROLL": "self") leaves the rollout net out -- the upload limit is 10 MB.
CNET2_ENS=a.npz,b.npz embeds extra policy nets; the pack then plays the mean of the members' logits (cnet2_np.Ensemble).
Embeds cnet2_feat, cnet2_np, fastsim, cnet2_search sources (exec'd into private module objects, the
plain sys.modules names are restored afterwards) and both weight sets, int8 per-row quantised exactly
like cnet2_pack (dequantised == data/weights/*_q8.npz).  No __file__ use, no file access at play time.

Post-build pack verification (F08):
  python3 cnet2_search_pack.py --verify-pack CANDIDATE.py [--expect-commit SHA] [--auth-dir DIR] [--record OUT.json]
  A standalone build (no authoritative code/ tree next to packs/) only records the embedded
  hashes with a warning; that does NOT verify the artifact. Before any pack built standalone
  is released, --verify-pack must pass against the authoritative sources: it extracts the
  candidate's embedded module sources, recomputes their md5s, cross-checks the pack's own
  _BUILD_HASHES record against them, compares every module against the authoritative tree
  (or the tree at --expect-commit, via git show) and reports the candidate file's own
  md5/sha256/bytes as its release identity. Exit 0 only if everything matches.
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


# --- F08 post-build pack verification ---------------------------------------
# --check-source compares the packer's neighbouring sources with code/; it
# never looks at a built artifact. A pack built standalone (old frozen
# search, only a warning) therefore sailed through a later clean-tree
# --check-source. --verify-pack closes that gap on the artifact itself.

def _extract_literal(text, name):
    """Pull a top-level `name = <literal>` assignment out of a pack file."""
    import ast as _ast
    for node in _ast.parse(text).body:
        if isinstance(node, _ast.Assign) and any(
                isinstance(t, _ast.Name) and t.id == name for t in node.targets):
            return _ast.literal_eval(node.value)
    return None


def _auth_hashes_for(auth_dir=None, expect_commit=None):
    """Authoritative module md5s + where they came from.

    Order: explicit --auth-dir; then `git show <expect_commit>:code/<m>.py`
    from the repo containing this packer; then the checkout tree next to
    packs/ (only without --expect-commit). Returns (hashes, source_desc);
    hashes is None when no authoritative source is available.
    """
    if auth_dir:
        if all(os.path.exists(os.path.join(auth_dir, m + ".py")) for m in MODS):
            return {m: _md5_file(os.path.join(auth_dir, m + ".py")) for m in MODS}, f"dir:{auth_dir}"
        return None, f"dir:{auth_dir} (incomplete)"
    if expect_commit:
        import subprocess
        repo = os.path.abspath(os.path.join(HERE, "..", ".."))
        out = {}
        try:
            for m in MODS:
                r = subprocess.run(["git", "-C", repo, "show", f"{expect_commit}:code/{m}.py"],
                                   capture_output=True, timeout=30)
                if r.returncode != 0:
                    return None, f"git:{expect_commit} (failed for {m})"
                out[m] = hashlib.md5(r.stdout).hexdigest()
            return out, f"git:{expect_commit}"
        except Exception as e:
            return None, f"git:{expect_commit} (error: {e})"
    auth = os.path.dirname(os.path.abspath(HERE))
    if all(os.path.exists(os.path.join(auth, m + ".py")) for m in MODS):
        return {m: _md5_file(os.path.join(auth, m + ".py")) for m in MODS}, f"dir:{auth}"
    return None, "unavailable"


def verify_pack(pack_path, auth_dir=None, expect_commit=None):
    """Verify a BUILT candidate pack against the authoritative sources.

    Returns (ok, record). ok is True only when: the pack embeds every MODS
    source; the md5 recomputed from the embedded sources matches the pack's
    own _BUILD_HASHES record (when present -- a missing record on a legacy
    pack is reported, not fatal, because the embedded sources themselves
    are what gets compared); and every embedded md5 equals the
    authoritative md5 (modules in ALLOW_DIFF exempted). The record also
    carries the pack file's own md5/sha256/bytes as its release identity.
    """
    raw = open(pack_path, "rb").read()
    record = {"pack": pack_path, "pack_bytes": len(raw),
              "pack_md5": hashlib.md5(raw).hexdigest(),
              "pack_sha256": hashlib.sha256(raw).hexdigest(),
              "expect_commit": expect_commit,
              "embedded_hashes": {}, "build_hashes_record": None,
              "record_consistent": None, "authoritative_source": None,
              "authoritative_hashes": {}, "mismatches": [],
              "missing_embedded": [], "ok": False}
    try:
        text = raw.decode("utf-8")
        srcs = _extract_literal(text, "_SRC")
        hashes_rec = _extract_literal(text, "_BUILD_HASHES")
    except Exception as e:
        record["error"] = f"cannot parse pack: {type(e).__name__}: {e}"
        return False, record
    if not isinstance(srcs, dict):
        record["error"] = "pack has no _SRC literal (not a cnet2_search_pack artifact)"
        return False, record
    for m in MODS:
        if m in srcs:
            record["embedded_hashes"][m] = hashlib.md5(srcs[m].encode("utf-8")).hexdigest()
        else:
            record["missing_embedded"].append(m)
    record["build_hashes_record"] = hashes_rec
    if isinstance(hashes_rec, dict):
        record["record_consistent"] = all(
            hashes_rec.get(m) == record["embedded_hashes"].get(m) for m in MODS)
    auth, desc = _auth_hashes_for(auth_dir, expect_commit)
    record["authoritative_source"] = desc
    if auth is None:
        record["error"] = "no authoritative source available (pass --auth-dir or --expect-commit in a git checkout)"
        return False, record
    record["authoritative_hashes"] = auth
    for m in MODS:
        if m in record["missing_embedded"]:
            record["mismatches"].append(m)
        elif record["embedded_hashes"].get(m) != auth[m] and m not in ALLOW_DIFF:
            record["mismatches"].append(m)
    record["ok"] = (not record["mismatches"]
                    and record["record_consistent"] is not False)
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
    if len(sys.argv) > 1 and sys.argv[1] in ("--help", "-h"):
        print(__doc__)
        print("Modes: build (default) | --check-source [CODE_DIR] | "
              "--verify-pack PACK [--expect-commit SHA] [--auth-dir DIR] [--record OUT.json]")
        sys.exit(0)
    if len(sys.argv) > 1 and sys.argv[1] == "--verify-pack":
        # Post-build artifact verification (F08). A pack built standalone
        # MUST pass this against the authoritative sources before release;
        # --check-source alone never inspects the artifact.
        _rest = sys.argv[2:]
        if not _rest:
            print("usage: cnet2_search_pack.py --verify-pack PACK [--expect-commit SHA] [--auth-dir DIR] [--record OUT.json]", file=sys.stderr)
            sys.exit(2)
        _pack = _rest[0]
        _kw = {}
        _record_out = None
        _i = 1
        while _i < len(_rest):
            if _rest[_i] == "--expect-commit":
                _kw["expect_commit"] = _rest[_i + 1]; _i += 2
            elif _rest[_i] == "--auth-dir":
                _kw["auth_dir"] = _rest[_i + 1]; _i += 2
            elif _rest[_i] == "--record":
                _record_out = _rest[_i + 1]; _i += 2
            else:
                print(f"unknown --verify-pack argument: {_rest[_i]}", file=sys.stderr)
                sys.exit(2)
        _ok, _rec = verify_pack(_pack, **_kw)
        print(json.dumps(_rec, indent=1, sort_keys=True))
        if _record_out:
            with open(_record_out, "w") as _f:
                json.dump(_rec, _f, indent=1, sort_keys=True)
        sys.exit(0 if _ok else 1)
    if len(sys.argv) > 1 and sys.argv[1] == "--check-source":
        # Repo-checkout consistency check (runbook pre-release requirement):
        #   python3 cnet2_search_pack.py --check-source [AUTHORITATIVE_CODE_DIR]
        # Exit 0 = every embedded module matches code/ (or is allowlisted);
        # exit 1 = drift found (or no authoritative tree to compare against).
        _ok, _rec = check_source(sys.argv[2] if len(sys.argv) > 2 else None)
        print(json.dumps(_rec, indent=1, sort_keys=True))
        sys.exit(0 if (_ok and _rec["checked"]) else 1)
    main(*sys.argv[1:], ens=[e for e in os.environ.get("CNET2_ENS", "").split(",") if e])

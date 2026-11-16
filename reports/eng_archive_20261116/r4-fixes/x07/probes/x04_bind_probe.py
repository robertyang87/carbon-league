"""X04 binding proof against the FIXED driver, three parts.
(a) Deterministic unit: capture the bytes of a weights path, then
    overwrite the path with corrupt bytes; loading from the captured
    bytes still yields the ORIGINAL tensors and the recorded md5 is
    the original's -- consumption follows the captured bytes, not
    whatever the path holds later.
(b) Structural: AST scan of k0_duel.py -- np.load is fed only from
    io.BytesIO (no by-path load remains in the module), and run_duel
    derives both recorded md5s from the captured blob variables.
(c) Live smoke: run_duel over a copy path in a thread while the
    path is overwritten with corrupt bytes mid-run; the run must
    complete with delta_u == 0.0 and student_md5 == the canonical
    md5 (the swap is inert), where the pre-binding driver would
    have consumed -- and died or measured -- the swapped file had
    the swap landed before its by-path load.
Run from the stage dir."""
import ast
import hashlib
import shutil
import sys
import threading
import time

import numpy as np

sys.path.insert(0, ".")
import k0_duel as D  # noqa: E402

CANON = "/home/hatch/workspace/heads-v2-diff/dgx.npz"
CANON_MD5 = "06a03d9cfeec239af3065a49fefec382"
SWAP = "../fixtures/swap_x04_live.npz"

# (a) unit ------------------------------------------------------------
shutil.copy(CANON, SWAP)
blob = D._read_weights_bytes(SWAP)
assert hashlib.md5(blob).hexdigest() == CANON_MD5
shutil.copy("../fixtures/corrupt_nan.npz", SWAP)   # path now corrupt
pol = D._policy_from_bytes(SWAP, blob)             # must NOT die
canon_v = dict(np.load(CANON))["whead.weight"].tobytes()
assert pol.w["whead.weight"].tobytes() == canon_v
print("(a) PASS: captured bytes load the original tensors after the "
      "path was overwritten with corrupt bytes; md5 identity is the "
      "captured bytes'")

# (b) structural ------------------------------------------------------
src = open("k0_duel.py").read()
tree = ast.parse(src)
np_load_args = []
for node in ast.walk(tree):
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "load"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "np"):
        np_load_args.append(ast.dump(node.args[0]) if node.args else "?")
assert len(np_load_args) == 1 and "BytesIO" in np_load_args[0], np_load_args
assert 'hashlib.md5(student_blob)' in src and 'hashlib.md5(parent_blob)' in src
print("(b) PASS: np.load has exactly one call site, fed by io.BytesIO; "
      "run_duel md5s derive from the captured blobs")

# (c) live smoke --------------------------------------------------------
shutil.copy(CANON, SWAP)
specs, sha, blobs = D.load_roots("../fixtures/roots_x07.json", 200, 299)
holder = {}


def run():
    try:
        holder["res"] = D.run_duel(SWAP, SWAP, specs, sha, replay_blobs=blobs)
    except BaseException as e:  # noqa: BLE001
        holder["err"] = e


t = threading.Thread(target=run)
t.start()
time.sleep(1.0)
shutil.copy("../fixtures/corrupt_nan.npz", SWAP)   # mid-run swap
t.join()
assert "err" not in holder, f"run died after swap: {holder.get('err')}"
res = holder["res"]
assert res["summary"]["delta_u"] == 0.0, res["summary"]
assert res["student"]["md5"] == CANON_MD5, res["student"]
print("(c) PASS: mid-run path swap inert -- run completed, "
      "delta_u == 0.0, recorded student_md5 == canonical (consumed "
      "bytes are the captured ones)")
print("X04 BIND PROBE: ALL PASS")

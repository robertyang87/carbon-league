"""Full regression battery for the W05 fix, run against exit/run (FIXED
exit_merge.py + current-main cnet2_exit_train.py). Sections:
  W = W05 items (corrupt meta / corrupt field / corrupt manifest on the
      reuse path; clean republish preserved),
  A = V01 (full chain loads; q corruption family; row/scal gates),
  B = V02 (reuse integrity, fault injection on rename and pointer flip),
  C = D1 segment-2 (pointer-side tamper refusals, legacy naming, direct
      sealed-dir refusal)."""
import io
import json
import os
import shutil
import sys
from contextlib import redirect_stdout

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(ROOT, "run")
sys.path.insert(0, RUN)
WORK = os.path.join(ROOT, "work")
SHARDS = os.path.join(WORK, "shards_w05")

import exit_merge  # noqa: E402
from cnet2_exit_train import _verify_exit_generation, load_exit, resolve_exit_arg  # noqa: E402

R = {}


def merge(out, shards=SHARDS, patch=None):
    real = exit_merge.os.replace
    if patch:
        exit_merge.os.replace = patch
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            exit_merge.main(out, [shards])
    finally:
        exit_merge.os.replace = real
    lines = [l for l in buf.getvalue().splitlines() if l.startswith("{")]
    return json.loads(lines[-1])


def merge_outcome(out, shards=SHARDS):
    """('OK', summary) or ('DIE', message) for a republish."""
    try:
        return "OK", merge(out, shards)
    except SystemExit as e:
        return "DIE", str(e)


def try_load(arg):
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            d = resolve_exit_arg(arg)
            D, is_val, meta = load_exit(d)
        return ("OK", int(is_val.shape[0])), buf.getvalue()
    except SystemExit as e:
        return ("DIE", str(e)), buf.getvalue()


def flat_copy(gen_dir, dst):
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(gen_dir, dst)
    os.remove(os.path.join(dst, "manifest.json"))
    return dst


def fresh_ds(name):
    out = os.path.join(WORK, name)
    if os.path.exists(out):
        shutil.rmtree(out)
    s = merge(out)
    return out, s, os.path.join(out, "generations", s["generation"])


# ================= W. W05 =================
outW, sumW, genW = fresh_ds("ds_w05reg")
manW = open(os.path.join(genW, "manifest.json"), "rb").read()
inoW = os.stat(genW).st_ino
stW, sW2 = merge_outcome(outW)
R["W3_clean_republish"] = {"status": stW, "reused": sW2.get("reused") if stW == "OK" else None,
                           "manifest_bytes_identical":
                               open(os.path.join(genW, "manifest.json"), "rb").read() == manW,
                           "gen_dir_same_inode": os.stat(genW).st_ino == inoW}
# W1: corrupt meta.json in place -> republish must die naming meta.json
meta_p = os.path.join(genW, "meta.json")
meta_good = open(meta_p, "rb").read()
meta = json.loads(meta_good)
meta["total"] = int(meta["total"]) + 7
with open(meta_p, "w") as fh:
    json.dump(meta, fh)
stW, msgW = merge_outcome(outW)
R["W1_corrupt_meta_republish"] = {"status": stW, "names_meta": "meta.json" in str(msgW)}
with open(meta_p, "wb") as fh:
    fh.write(meta_good)
stW, sW3 = merge_outcome(outW)
R["W1_repaired_meta_republish"] = {"status": stW,
                                   "reused": sW3.get("reused") if stW == "OK" else None}
# W2: corrupt a field .npy in place -> republish must die (pre-existing coverage)
val_p = os.path.join(genW, "value.npy")
val_good = open(val_p, "rb").read()
b = bytearray(val_good); b[len(b) // 2] ^= 0xFF
with open(val_p, "wb") as fh:
    fh.write(bytes(b))
stW, msgW = merge_outcome(outW)
R["W2_corrupt_field_republish"] = {"status": stW, "names_field": "value" in str(msgW)}
with open(val_p, "wb") as fh:
    fh.write(val_good)
# W2b: tamper manifest content without resealing -> republish must die
man_p = os.path.join(genW, "manifest.json")
man = json.loads(manW)
man["fields"]["value"]["rows"] += 1
with open(man_p, "w") as fh:
    json.dump(man, fh)
stW, msgW = merge_outcome(outW)
R["W2b_tampered_manifest_republish"] = {"status": stW}
with open(man_p, "wb") as fh:
    fh.write(manW)
stW, sW4 = merge_outcome(outW)
R["W2b_restored_republish"] = {"status": stW,
                               "reused": sW4.get("reused") if stW == "OK" else None}

# ================= A. V01 =================
outA, sumA, genA = fresh_ds("ds_fix")
st, _ = try_load(outA)
R["A1_full_chain_loads"] = st


def tamper(name, fn):
    d = flat_copy(genA, os.path.join(WORK, "flat_" + name))
    fn(d)
    st, _ = try_load(d)
    return st


def t_inf(d):
    a = np.load(os.path.join(d, "q.npy")).astype(np.float32); a[1, 0] = np.inf
    np.save(os.path.join(d, "q.npy"), a)


def t_slot_noq(d):
    a = np.load(os.path.join(d, "q.npy")).astype(np.float32); a[1, :] = np.nan
    np.save(os.path.join(d, "q.npy"), a)


def t_pi_on_nan(d):
    a = np.load(os.path.join(d, "q.npy")).astype(np.float32); a[1, 0] = np.nan
    np.save(os.path.join(d, "q.npy"), a)


def t_unsearched_finite(d):
    a = np.load(os.path.join(d, "q.npy")).astype(np.float32); a[0, 0] = 3.0
    np.save(os.path.join(d, "q.npy"), a)


def t_pi_no_slot(d):
    a = np.load(os.path.join(d, "pi.npy")).astype(np.float32); a[0, 0, 0] = 0.5
    np.save(os.path.join(d, "pi.npy"), a)


R["A4a_q_inf_evaluated"] = tamper("inf", t_inf)
R["A4b_slot_but_q_all_nan"] = tamper("slotnoq", t_slot_noq)
R["A4c_pi_positive_on_nan_q"] = tamper("pionnan", t_pi_on_nan)
R["A4d_unsearched_finite_q"] = tamper("unsfin", t_unsearched_finite)
R["A4e_pi_without_slot"] = tamper("pinoslot", t_pi_no_slot)


def t_rows(d):
    a = np.load(os.path.join(d, "y.npy")); np.save(os.path.join(d, "y.npy"), a[:9])


def t_scal(d):
    a = np.load(os.path.join(d, "scal.npy")).astype(np.float32); a[0] = np.nan
    np.save(os.path.join(d, "scal.npy"), a)


R["A5a_row_count_mismatch"] = tamper("rows", t_rows)
R["A5b_scal_nan"] = tamper("scal", t_scal)

# ================= B. V02 =================
outB, sumB, genB = fresh_ds("ds_v02fix")
genB_id = sumB["generation"]
ptr1 = json.load(open(os.path.join(outB, "current.json")))
man1 = open(os.path.join(genB, "manifest.json"), "rb").read()
ino1 = os.stat(genB).st_ino
s2 = merge(outB)
man2 = open(os.path.join(genB, "manifest.json"), "rb").read()
R["B1_republish"] = {"reused_flag": s2.get("reused"),
                     "manifest_bytes_identical": man1 == man2,
                     "gen_dir_same_inode": os.stat(genB).st_ino == ino1}
try:
    _verify_exit_generation(genB, ptr1)
    R["B1_old_pointer_verify"] = "PASS"
except SystemExit as e:
    R["B1_old_pointer_verify"] = "REFUSED: " + str(e)[:100]

sub = os.path.join(WORK, "shards_sub")
os.makedirs(sub, exist_ok=True)
shutil.copy(os.path.join(SHARDS, "990001_fxopp_s0.npz"), sub)


def boom_staging(src, dst):
    if ".staging-exit-" in os.path.basename(src):
        raise OSError("injected fault: rename staging->generation")
    return os.replace(src, dst)


try:
    merge(outB, shards=sub, patch=boom_staging)
    R["B2_fault_new_gen"] = {"publish": "no error (unexpected)"}
except OSError:
    R["B2_fault_new_gen"] = {"publish": "raised OSError"}
manB_after = open(os.path.join(genB, "manifest.json"), "rb").read()
ptrB = json.load(open(os.path.join(outB, "current.json")))
R["B2_old_gen_intact"] = {"manifest_unchanged": manB_after == man1,
                          "pointer_still_gen1": ptrB["generation"] == genB_id}
try:
    _verify_exit_generation(genB, ptrB)
    R["B2_pointer_verify"] = "PASS"
except SystemExit as e:
    R["B2_pointer_verify"] = "REFUSED: " + str(e)[:100]
s3 = merge(outB, shards=sub)
genB2 = os.path.join(outB, "generations", s3["generation"])
R["B2_recovery"] = {"new_gen": s3["generation"] != genB_id,
                    "old_gen_manifest_still_identical":
                        open(os.path.join(genB, "manifest.json"), "rb").read() == man1}
try:
    _verify_exit_generation(genB2, json.load(open(os.path.join(outB, "current.json"))))
    R["B2_new_pointer_verify"] = "PASS"
except SystemExit as e:
    R["B2_new_pointer_verify"] = "REFUSED: " + str(e)[:100]


def boom_ptr(src, dst):
    if os.path.basename(src).startswith(".current-"):
        raise OSError("injected fault: pointer flip")
    return os.replace(src, dst)


try:
    merge(outB, patch=boom_ptr)
    R["B3_fault_ptr_flip"] = {"publish": "no error (unexpected)"}
except OSError:
    R["B3_fault_ptr_flip"] = {"publish": "raised OSError"}
try:
    _verify_exit_generation(genB2, json.load(open(os.path.join(outB, "current.json"))))
    R["B3_pointer_still_valid"] = "PASS"
except SystemExit as e:
    R["B3_pointer_still_valid"] = "REFUSED: " + str(e)[:100]

# ================= C. D1 segment-2 =================
outC, sumC, genC = fresh_ds("ds_d1")
genC_id = sumC["generation"]
d1 = os.path.join(WORK, "ds_d1_flip")
if os.path.exists(d1):
    shutil.rmtree(d1)
shutil.copytree(outC, d1)
p = os.path.join(d1, "generations", genC_id, "value.npy")
b = bytearray(open(p, "rb").read()); b[len(b) // 2] ^= 0xFF
open(p, "wb").write(bytes(b))
st, _ = try_load(d1)
R["C1_field_byte_flip"] = st
d2 = os.path.join(WORK, "ds_d1_manrows")
if os.path.exists(d2):
    shutil.rmtree(d2)
shutil.copytree(outC, d2)
mp = os.path.join(d2, "generations", genC_id, "manifest.json")
man = json.load(open(mp)); man["fields"]["value"]["rows"] += 1
json.dump(man, open(mp, "w"))
st, _ = try_load(d2)
R["C2_manifest_rows_tamper"] = st
d3 = flat_copy(genC, os.path.join(WORK, "flat_legacy"))
st, out3 = try_load(d3)
R["C3_legacy_direct"] = {"status": st, "named_legacy": "exit_data_legacy_unsealed" in out3}
st, _ = try_load(genC)
R["C4_sealed_dir_direct"] = st

print(json.dumps(R, indent=1, default=str))

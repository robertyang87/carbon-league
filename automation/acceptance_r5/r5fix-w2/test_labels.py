"""Final acceptance: mixed exact-label chain (producer -> builder -> stage0 heads).
Classes assigned AFTER seeing builder val flags (need >=1 val game for heads evaluate):
normal exact / fault_one_side exact / unusable / legacy approx."""
import json, os, shutil, subprocess, sys, zlib
CODE = os.path.expanduser("~/workspace/r5fix-w2/fa_code")
WORK = os.path.expanduser("~/workspace/final-accept/work")
PY = os.path.expanduser("~/workspace/integ-accept/venv/bin/python")
sys.path.insert(0, CODE)
import numpy as np

PASS = []
def chk(name, cond, info=""):
    PASS.append((name, bool(cond)))
    print(("PASS " if cond else "FAIL ") + name, info)

corpus = os.path.join(WORK, "corpus_raw2")
if os.path.exists(corpus): shutil.rmtree(corpus)
env = dict(os.environ, SPGEN_STUB="1", PYTHONPATH=CODE + os.pathsep + os.path.join(CODE, "producer"))
r = subprocess.run([PY, os.path.join(CODE, "producer", "selfplay_gen.py"),
                    "--weights", "dummy.npz", "--seed0", "100", "--seed1", "108",
                    "--out", corpus, "--workers", "1", "--max-games", "8",
                    "--name", "claude-AlphaCarbon-Master"],
                   capture_output=True, text=True, env=env, timeout=900)
idx = json.load(open(os.path.join(corpus, "index.json")))
chk("producer made 8 games", len(idx) == 8, f"n={len(idx)}")
valflag = [zlib.crc32(os.path.basename(e["file"]).encode()) % 10 == 0 for e in idx]
if not any(valflag):
    # pad: duplicate game0's replay under a basename that lands in val
    import copy
    s = 0
    while zlib.crc32(f"valpad-{s}.json.gz".encode()) % 10 != 0: s += 1
    src = os.path.join(corpus, idx[0]["file"])
    dstname = f"valpad-{s}.json.gz"
    dst = os.path.join(os.path.dirname(src), dstname)
    shutil.copy(src, dst)
    e = copy.deepcopy(idx[0]); e["file"] = os.path.join(os.path.dirname(idx[0]["file"]), dstname)
    if "game" in e: e["game"] = 9990 + s
    idx.append(e); valflag.append(True)
    json.dump(idx, open(os.path.join(corpus, "index.json"), "w"))
    print(f"padded val game {dstname}")
train_g = [i for i, v in enumerate(valflag) if not v]
chk("at least 1 val + 3 train games", len(train_g) >= 3 and any(valflag), f"val={valflag}")
gi_fault, gi_unus, gi_approx = train_g[0], train_g[1], train_g[2]
gi_normal = next(i for i in train_g[3:] + [i for i, v in enumerate(valflag) if v][:1])

from cnet2_build import _u_of_margin, _wdl_value
def set_scores(e, sa, sb):
    e["scores"] = {"a": sa, "b": sb}
    e["margin"] = {"a": sa - sb, "b": sb - sa}
    e["u"] = {"a": _u_of_margin(sa - sb), "b": _u_of_margin(sb - sa)}
set_scores(idx[gi_fault], 100, 300); idx[gi_fault]["fault_sides"] = [0]
set_scores(idx[gi_unus], 400, 100); idx[gi_unus]["unusable"] = True
for k in ("margin", "u"): idx[gi_approx].pop(k, None)
json.dump(idx, open(os.path.join(corpus, "index.json"), "w"))
print(f"classes: fault={gi_fault} unusable={gi_unus} approx={gi_approx} normal={gi_normal}")

out = os.path.join(WORK, "ds_mixed2")
if os.path.exists(out): shutil.rmtree(out)
import cnet2_build as B
B.main(out, [corpus])
man = json.load(open(os.path.join(out, "current.json")))
gd = os.path.join(out, man["dir"])
gman = json.load(open(os.path.join(gd, "manifest.json")))
D = {k: np.load(os.path.join(gd, k + ".npy")) for k in ("gid", "margin_raw", "u_raw", "outcome_type", "label_status")}
gid = np.asarray(D["gid"])
from cnet2_build import OUTCOME_CODES, LABEL_EXACT, LABEL_APPROX, LABEL_UNUSABLE
def rows_of(gi): return gid == gi
for gi, oc, ls, nm in [(gi_normal, 0, LABEL_EXACT, "normal"), (gi_fault, 1, LABEL_EXACT, "fault"),
                       (gi_unus, 2, LABEL_UNUSABLE, "unusable"), (gi_approx, -1, LABEL_APPROX, "approx")]:
    rows = rows_of(gi)
    chk(f"{nm} rows+class", rows.sum() > 0 and (np.asarray(D['outcome_type'])[rows] == oc).all()
        and (np.asarray(D['label_status'])[rows] == ls).all(), f"rows={int(rows.sum())}")
mr = np.asarray(D["margin_raw"]); ur = np.asarray(D["u_raw"])
chk("normal u == U(margin)", np.allclose(ur[rows_of(gi_normal)], [_u_of_margin(x) for x in mr[rows_of(gi_normal)]], atol=1e-6))
chk("fault margin +-200", set(np.unique(mr[rows_of(gi_fault)]).tolist()) == {200.0, -200.0})
chk("fault u == WDL only", np.allclose(ur[rows_of(gi_fault)], [_wdl_value(x) for x in mr[rows_of(gi_fault)]], atol=1e-6),
    f"u={np.unique(ur[rows_of(gi_fault)])}")
chk("unusable NaN", np.isnan(mr[rows_of(gi_unus)]).all() and np.isnan(ur[rows_of(gi_unus)]).all())
chk("approx NaN raw", np.isnan(mr[rows_of(gi_approx)]).all())
lab = gman.get("labels", {}); print("manifest labels:", lab)
un_rows = int((np.asarray(D['label_status']) == LABEL_UNUSABLE).sum())
ap_rows = int((np.asarray(D['label_status']) == LABEL_APPROX).sum())
ex_rows = int((np.asarray(D['label_status']) == LABEL_EXACT).sum())
chk("manifest counts == dataset counts", lab.get("unusable_rows") == un_rows and lab.get("approx_rows") == ap_rows and lab.get("exact_rows") == ex_rows,
    f"man={lab} ds exact={ex_rows} approx={ap_rows} unusable={un_rows}")

import torch
from cnet2_model import CNet2
CFG = dict(C=32, NB=2, D=64, NH=4, L=1, E=16, G=8)
torch.manual_seed(3); net = CNet2(**CFG)
init = os.path.join(WORK, "init_heads.npz")
arrs = {k: v.numpy().astype(np.float32) for k, v in net.state_dict().items()}
arrs["meta"] = np.array(json.dumps({"cfg": CFG})); np.savez(init, **arrs)
hout = os.path.join(WORK, "heads_out2.npz")
env = dict(os.environ, BC_DEVICE="cpu", BC_RAM="1", BC_BS="64", PYTHONPATH=CODE)
r = subprocess.run([PY, os.path.join(CODE, "stage0_train_heads.py"), out, init, hout, "1"],
                   capture_output=True, text=True, env=env, timeout=900)
print(r.stdout[-900:]); print(r.stderr[-300:])
chk("heads exit0 + TRAIN-DONE", r.returncode == 0 and "TRAIN-DONE" in r.stdout, f"rc={r.returncode}")
mf = json.load(open(hout + ".manifest.json"))
print("heads manifest keys:", sorted(mf.keys()))
hl = mf.get("labels", {})
print("heads labels:", hl, "| approx_labels:", mf.get("approx_labels"), "| source:", mf.get("label_source"))
chk("heads manifest approx flagged", hl.get("approx_labels") is True and "mixed" in str(hl.get("label_source")))
chk("heads counts match", hl.get("unusable_rows") == un_rows and hl.get("approx_rows") == ap_rows,
    f"heads={hl} ds unusable={un_rows} approx={ap_rows}")
# unusable must not enter value-loss denominators: heads logs per-epoch label counts;
# check the printed epoch line reports train_label_unusable == unusable train rows
import re
mm = re.search(r'"train_label_unusable": (\d+)', r.stdout)
mv = re.search(r'"train_value_rows": (\d+)', r.stdout)
train_rows = int(re.search(r'"train": (\d+)', r.stdout).group(1))
chk("epoch log unusable count present", mm is not None and int(mm.group(1)) == un_rows, r.stdout[-300:])
chk("unusable excluded from value rows", mv is not None and int(mv.group(1)) == train_rows - un_rows,
    f"value_rows={mv.group(1) if mv else '?'} train={train_rows} unusable={un_rows}")
bad = [p for p in PASS if not p[1]]
print(f"\n== mixed labels v2: {len(PASS) - len(bad)}/{len(PASS)} passed ==")
sys.exit(1 if bad else 0)

"""Integrated acceptance on the MERGED cnet2_train.py (stage baseline).
F01 migration suite + F06 gate suite. Run with BC_DEVICE=cpu."""
import io, json, os, sys
from collections import OrderedDict
from contextlib import redirect_stdout

os.environ["BC_DEVICE"] = "cpu"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

import cnet2_feat as F
from cnet2_model import CNet2, migrate_state_dict, migrate_optimizer_state, NEW_HEAD_KEYS
import cnet2_train as T

PASS = []
def ok(name, cond, extra=""):
    assert cond, f"FAIL: {name} {extra}"
    PASS.append(name); print(f"PASS {name} {extra}")

CFG = dict(C=32, NB=2, D=64, NH=4, L=1, E=16, G=8)
torch.manual_seed(7)

def batch(B=2, Tn=14):
    g = torch.Generator().manual_seed(3)
    b = {
        "maps": torch.rand(B, F.P, 15, 15, generator=g),
        "tok_pos": torch.randint(0, 225, (B, Tn), generator=g),
        "tok_kind": torch.randint(0, 6, (B, Tn), generator=g),
        "tok_cargo": torch.rand(B, Tn, generator=g) * 10,
        "hist_cell": torch.randint(-1, 225, (B, Tn, F.H), generator=g),
        "hist_cargo": torch.rand(B, Tn, F.H, generator=g) * 10,
        "hist_move": torch.randint(0, 6, (B, Tn, F.H), generator=g),
        "tok_mask": torch.zeros(B, Tn, dtype=torch.bool),
        "scal": torch.rand(B, F.NS, generator=g),
        "teacher": torch.randint(0, F.N_TEACH, (B,), generator=g),
        "mf": torch.rand(B, F.OWN, 5, F.MF, generator=g),
        "uf": torch.rand(B, F.OWN, F.UF, generator=g),
        "legal": torch.rand(B, F.OWN, 5, generator=g) > 0.5,
        "y": torch.cat([torch.randint(0, 5, (B, 10), generator=g),
                        torch.randint(0, 3, (B, 2), generator=g)], 1),
        "value": torch.rand(B, generator=g) * 2 - 1,
        "carbon": torch.rand(B, F.N, generator=g),
    }
    b["tok_mask"][:, :12] = True
    b["legal"][:, :, 0] = True
    return b

# ---------- F01 ----------
full = CNet2(**CFG)
sd_full = full.state_dict()
old_sd = OrderedDict((k, v) for k, v in sd_full.items() if k not in NEW_HEAD_KEYS)

dmodel = CNet2()  # default cfg for the 59->65 count
_nn = [n for n, _ in dmodel.named_parameters()]
ok("F01-01 param count old->new is +6 (default cfg 96->102)", len(_nn) - len([n for n in _nn if n not in NEW_HEAD_KEYS]) == 6,
   f"{len([n for n in _nn if n not in NEW_HEAD_KEYS])}->{len(_nn)}")

m2 = CNet2(**CFG)
try:
    m2.load_state_dict(old_sd, strict=True); ok("F01-02 strict load of old sd fails", False)
except RuntimeError:
    ok("F01-02 strict load of old sd fails", True)

rep = migrate_state_dict(m2, old_sd, source="test-old")
ok("F01-03 migrate ok, missing == 6 head keys", sorted(rep["migrated_missing"]) == sorted(NEW_HEAD_KEYS))

b = batch()
with torch.no_grad():
    lg_f, v_f, _ = full(b); lg_m, v_m, _ = m2(b)
ok("F01-04 policy/value error 0 after migration",
   float((lg_f - lg_m).abs().max()) == 0.0 and float((v_f - v_m).abs().max()) == 0.0)

with torch.no_grad():
    for n, p in m2.named_parameters():
        if n in NEW_HEAD_KEYS: p.mul_(100.0)
    lg_m2, v_m2, _ = m2(b)
ok("F01-05 heads do not affect policy/value", float((lg_f - lg_m2).abs().max()) == 0.0
   and float((v_f - v_m2).abs().max()) == 0.0)

for tag, bad in [("missing stem", OrderedDict((k, v) for k, v in old_sd.items() if k != "stem.weight")),
                 ("extra key", OrderedDict(list(old_sd.items()) + [("bogus.weight", torch.zeros(1))]))]:
    try:
        migrate_state_dict(CNet2(**CFG), bad, source="test-bad"); ok(f"F01-06 migrate rejects {tag}", False)
    except RuntimeError:
        ok(f"F01-06 migrate rejects {tag}", True)

# old NPZ checkpoint (repo export format: float32 arrays in sd order + meta)
npz_path = "/tmp/integ_old.npz"
np.savez(npz_path, **{k: v.numpy() for k, v in old_sd.items()}, meta=np.array(json.dumps({"cfg": CFG})))
wz = np.load(npz_path)
old_sd_npz = OrderedDict((k, torch.from_numpy(wz[k])) for k in wz.files if k != "meta")
names_npz = T.old_param_names_checked(old_sd_npz, CNet2(**CFG), source="test-npz")
ok("F01-07 old NPZ: checked name derivation from file key order",
   names_npz == [n for n, _ in CNet2(**CFG).named_parameters() if n not in NEW_HEAD_KEYS])

def make_old_opt(model, lr, steps, seed):
    torch.manual_seed(seed)
    old_params = [p for n, p in model.named_parameters() if n not in NEW_HEAD_KEYS]
    opt = torch.optim.AdamW(old_params, lr=lr, weight_decay=0.01)
    for _ in range(steps):
        for i, p in enumerate(old_params):
            p.grad = torch.full_like(p, (i + 1) * 0.01)
        opt.step()
    return opt

def opt_migration_case(tag, lr, steps, seed):
    model = CNet2(**CFG)
    with torch.no_grad():  # align trunk weights with the reference model
        for n, p in model.named_parameters():
            if n in sd_full: p.copy_(sd_full[n])
    osd = OrderedDict((k, v) for k, v in model.state_dict().items() if k not in NEW_HEAD_KEYS)
    old_opt = make_old_opt(model, lr, steps, seed)
    old_state = {p: old_opt.state[p] for p in old_opt.param_groups[0]["params"]}
    new_opt = torch.optim.AdamW(model.parameters(), lr=9e-9)
    rep = T.migrate_optimizer_checked(new_opt, old_opt.state_dict(), osd, model, source=f"test-{tag}")
    ok(f"F01-{tag} carried/cold/steps/lr", len(rep["carried"]) == len(osd) and sorted(rep["cold"]) == sorted(NEW_HEAD_KEYS)
       and rep["steps_preserved"] == [float(steps)] and abs(new_opt.param_groups[0]["lr"] - lr) < 1e-12,
       f"steps={rep['steps_preserved']} lr={new_opt.param_groups[0]['lr']}")
    params = dict(model.named_parameters())
    old_params = [p for n, p in model.named_parameters() if n not in NEW_HEAD_KEYS]
    allmatch = True
    for p_old in old_params:
        nm = [n for n, q in model.named_parameters() if q is p_old][0]
        if not torch.equal(new_opt.state[params[nm]]["exp_avg"], old_state[p_old]["exp_avg"]):
            allmatch = False
    ok(f"F01-{tag} exp_avg bitwise on all old params (incl same-shape block biases)", allmatch)
    for p in model.parameters(): p.grad = torch.randn_like(p) * 0.01
    new_opt.step()
    ok(f"F01-{tag} post-migration step ok, heads cold-started then live",
       all("exp_avg" in new_opt.state[p] for n, p in model.named_parameters() if n in NEW_HEAD_KEYS))
    return model, new_opt

net_model, net_opt = opt_migration_case("08", 3e-3, 2, 11)
vnet_model, vnet_opt = opt_migration_case("09", 5e-3, 3, 23)

# raw function with a swapped same-shape name list silently misplaces moments (why the checked entry is mandatory)
swap_model = CNet2(**CFG)
osd2 = OrderedDict((k, v) for k, v in swap_model.state_dict().items() if k not in NEW_HEAD_KEYS)
old_opt2 = make_old_opt(swap_model, 3e-3, 1, 5)
names_good = [n for n, _ in swap_model.named_parameters() if n not in NEW_HEAD_KEYS]
names_bad = list(names_good)
i0, i1 = names_good.index("blocks.0.c1.bias"), names_good.index("blocks.1.c1.bias")
names_bad[i0], names_bad[i1] = names_bad[i1], names_bad[i0]
new_opt2 = torch.optim.AdamW(swap_model.parameters(), lr=9e-9)
migrate_optimizer_state(new_opt2, old_opt2.state_dict(), names_bad, [n for n, _ in swap_model.named_parameters()])
p1 = dict(swap_model.named_parameters())["blocks.1.c1.bias"]
oldp1 = [p for n, p in swap_model.named_parameters() if n == "blocks.1.c1.bias"][0]
misplaced = torch.equal(new_opt2.state[dict(swap_model.named_parameters())["blocks.0.c1.bias"]]["exp_avg"],
                        old_opt2.state[oldp1]["exp_avg"])
ok("F01-10 raw migrate with swapped same-shape names misplaces moments (checked entry required)", misplaced)

tampered = OrderedDict(old_sd_npz)
items = list(tampered.items()); items[3], items[4] = items[4], items[3]
try:
    T.old_param_names_checked(OrderedDict(items), CNet2(**CFG), source="test-tamper")
    ok("F01-11 tampered checkpoint key order rejected", False)
except RuntimeError:
    ok("F01-11 tampered checkpoint key order rejected", True)

bad_shape = OrderedDict(old_sd_npz); bad_shape["value.weight"] = torch.zeros(5, 5)
try:
    T.old_param_names_checked(bad_shape, CNet2(**CFG), source="test-shape")
    ok("F01-12 shape mismatch rejected", False)
except RuntimeError:
    ok("F01-12 shape mismatch rejected", True)

# round-trip: migrated model exports and reloads strictly; migrated optimizer reloads directly
rt_path = "/tmp/integ_rt.npz"
T.export(net_model, rt_path, {"cfg": CFG})
wz2 = np.load(rt_path)
fresh = CNet2(**CFG)
fresh.load_state_dict(OrderedDict((k, torch.from_numpy(wz2[k])) for k in wz2.files if k != "meta"), strict=True)
with torch.no_grad():
    lg1, v1, _ = net_model(b); lg2, v2, _ = fresh(b)
ok("F01-13 round-trip export->strict reload identical", float((lg1 - lg2).abs().max()) == 0.0
   and float((v1 - v2).abs().max()) == 0.0)
fresh_opt = torch.optim.AdamW(fresh.parameters(), lr=9e-9)
fresh_opt.load_state_dict(net_opt.state_dict())
pn = dict(fresh.named_parameters()); po = dict(net_model.named_parameters())
ok("F01-14 migrated optimizer state reloads directly",
   torch.equal(fresh_opt.state[pn["stem.weight"]]["exp_avg"], net_opt.state[po["stem.weight"]]["exp_avg"]))

# ---------- F06 gate ----------
def gate(D, dirn="testds"):
    return T._check_finite_weights(D, dirn)

s = gate({"w": np.array([0.5, 1.0, 2.0], np.float32), "uw": np.ones((3, 12), np.float32)})
ok("F06-01 gate normal summary", abs(s["prod_max"] - 2.0) < 1e-6 and s["prod_n"] == 36, str(s))

for tag, D, needle in [("uw inf", {"w": np.ones(2, np.float32), "uw": np.full((2, 12), np.inf, np.float32)}, "'uw'"),
                       ("w nan", {"w": np.array([1.0, np.nan], np.float32), "uw": np.ones((2, 12), np.float32)}, "'w'"),
                       ("prod overflow", {"w": np.full(2, 3e38, np.float32), "uw": np.full((2, 12), 3e38, np.float32)}, "w*uw")]:
    try:
        gate(D); ok(f"F06 gate rejects {tag}", False)
    except SystemExit as e:
        ok(f"F06 gate rejects {tag}", needle in str(e) and "testds" in str(e), str(e)[:80])

s = gate({"w": np.array([1.0, 1e-3], np.float32),
          "uw": np.array([[428293.7] + [1.0] * 11, [1.0] * 12], np.float32)})
ok("F06-02 gate extreme-but-finite passes, prod_max recorded", abs(s["prod_max"] - 428293.7) < 1.0, str(s))

# load() integration on a synthetic FIELDS dataset
from cnet2_build import FIELDS
dsdir = "/tmp/integ_ds"; os.makedirs(dsdir, exist_ok=True)
N = 6
for k, (dt, shp) in FIELDS.items():
    np.save(os.path.join(dsdir, k + ".npy"), np.zeros((N,) + tuple(shp), dt))
np.save(os.path.join(dsdir, "gid.npy"), np.arange(N) % 2)
np.save(os.path.join(dsdir, "w.npy"), np.array([1.0, 0.5, 1e-3, 2.0, 1.0, 1.0], np.float32))
np.save(os.path.join(dsdir, "uw.npy"), np.ones((N, F.OWN), np.float32))
json.dump({"games": [{"val": True}, {"val": False}]}, open(os.path.join(dsdir, "meta.json"), "w"))
buf = io.StringIO()
with redirect_stdout(buf):
    D, is_val, meta = T.load(dsdir)
ok("F06-03 load() attaches + prints product summary",
   "weight_gate" in meta and abs(meta["weight_gate"]["prod_max"] - 2.0) < 1e-6 and "weight_gate" in buf.getvalue(),
   str(meta.get("weight_gate")))

net = CNet2(**CFG)
bb = batch()
bbt = {k: v for k, v in bb.items()}
l0 = T.losses(net, bbt)[0]
bb2 = dict(bbt); bb2["w"] = torch.ones(2); bb2["uw"] = torch.ones(2, F.OWN)
l1 = T.losses(net, bb2)[0]
ok("F06-04 losses regression: unit weights == no weights", abs(float(l0) - float(l1)) < 1e-5,
   f"{float(l0):.6f} vs {float(l1):.6f}")
bb3 = dict(bbt); bb3["w"] = torch.tensor([1e-3, 1.0]); bb3["uw"] = torch.full((2, F.OWN), 1070.0)
l2 = T.losses(net, bb3)[0]
ok("F06-05 losses finite with extreme floored weights", bool(torch.isfinite(l2)), str(float(l2)))

print(f"\nALL {len(PASS)} CHECKS PASS")

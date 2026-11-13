"""S1 NumPy utility/cache interface checks; deterministic small network, no training.

Set CARBON_REPO or put this file under a repository containing code/fastsim.py.
JSON goes to stdout and, optionally, CARBON_REVIEW_OUT/network-results.json.
Canonical Studio heads_v2.npz is not needed and its precise numbers are not reproduced.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["BC_DEVICE"] = "cpu"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import numpy as np
import torch

repo = Path(os.environ["CARBON_REPO"]).resolve() if os.environ.get("CARBON_REPO") else next(
    p for p in Path(__file__).resolve().parents if (p / "code/fastsim.py").exists())
sys.path.insert(0, str(repo / "code"))
import cnet2_feat as F
import cnet2_np as NP
import cnet2_search as S
import stage0_teacher as T
import stage1_teacher_gate as GATE
from cnet2_model import CNet2
from eval_v53 import CFG
from fastsim import FastCarbon

torch.set_num_threads(1)
torch.manual_seed(20261113)
cfg = dict(C=16, NB=1, D=16, NH=4, L=1, E=8, G=4, TA=1, TNH=4)
net = CNet2(**cfg).eval()
with torch.no_grad():
    # CellAttn defaults to an identity residual; exercise nonzero attention branches.
    for layer in net.tattn:
        layer.out.weight.normal_(0, .04)
        layer.f2.weight.normal_(0, .04)
w = {k: v.detach().numpy().copy() for k, v in net.state_dict().items()}
w["meta"] = np.array(json.dumps({"cfg": net.cfg}))
pol = NP.Policy(w)
cache = S.CachePolicy(w)
baseline = os.environ.get("CARBON_BASELINE", "bc1a84d4")
old = types.ModuleType("baseline_cnet2_np")
old.__file__ = str(repo / "code/cnet2_np.py")
old_source = subprocess.check_output(["git", "-C", str(repo), "show", baseline + ":code/cnet2_np.py"], text=True)
exec(compile(old_source, "baseline_cnet2_np.py", "exec"), old.__dict__)
oldpol = old.Policy(w)

def same(a, b):
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b, equal_nan=True)
    return a == b

def err(a, b):
    return max(float(np.max(np.abs(x - y))) for x, y in zip(a, b))

keys = ("maps", "scal", "tok_pos", "tok_kind", "tok_mask", "tok_cargo", "hist_cell", "hist_cargo", "hist_move")
sim = FastCarbon(CFG).reset(seed=42)
features = [F.Feat(0), F.Feat(1)]
positions = []
for step in range(10):
    for seat in (0, 1):
        f = features[seat]
        x = f.observe(sim.obs(seat))
        positions.append((copy.deepcopy(x), copy.deepcopy(f.aux), seat, step))
    acts = []
    for seat in (0, 1):
        obs = sim.obs(seat)
        player = obs["players"][seat]
        a = {uid: "UP" for uid in player[2]}
        if step == 0:
            a[next(iter(player[1]))] = "RECCOLLECTOR"
        acts.append(a)
    sim.step(*acts)

results = {
    "head": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
    "baseline": baseline,
    "environment": {"python": sys.version.split()[0], "numpy": np.__version__, "torch": torch.__version__, "torch_threads": torch.get_num_threads()},
    "fixture": {"kind": "deterministic untrained small CNet2", "seed": 20261113, "cfg": net.cfg,
                "positions": len(positions), "sim_steps": 10, "seats": [0, 1], "canonical_heads_v2": False},
    "source_sha256": {name: hashlib.sha256((repo / "code" / name).read_bytes()).hexdigest()
                      for name in ("cnet2_np.py", "cnet2_model.py", "cnet2_search.py", "stage0_teacher.py", "stage1_teacher_gate.py")},
}
max_u = max_v = max_lg = max_encoder = 0.
default_same = sampled_same = rng_same = input_same = True
for i, (x, aux, seat, step) in enumerate(positions):
    teacher = i % F.N_TEACH
    before = copy.deepcopy((x, aux))
    r3 = pol.act(x, aux, teacher=teacher)
    r4 = pol.act(x, aux, teacher=teacher, want_u=True)
    rc = cache.act(x, aux, teacher=teacher, want_u=True)
    default_same &= same(r3, oldpol.act(x, aux, teacher=teacher)) and same(r3, r4[:3]) and same(r4, rc)
    max_encoder = max(max_encoder, err(pol.encode(x, teacher), cache.enc_for(x, teacher)))
    for sample_kinds in (None, {"base", "planter"}):
        generators = [np.random.default_rng(700 + i) for _ in range(3)]
        options = dict(teacher=teacher, temp=(.6, .8, 1.1), sample_kinds=sample_kinds)
        rs = [oldpol.act(x, aux, sample_rng=generators[0], **options),
              pol.act(x, aux, sample_rng=generators[1], **options),
              pol.act(x, aux, sample_rng=generators[2], want_u=True, **options)]
        sampled_same &= same(rs[0], rs[1]) and same(rs[1], rs[2][:3])
        rng_same &= same(generators[0].bit_generator.state, generators[1].bit_generator.state) and same(generators[1].bit_generator.state, generators[2].bit_generator.state)
    # Real autoregressive prefix uses the already-selected complete action sequence.
    y = r3[0]
    prefix = F.Prefix(x, aux)
    uf = np.zeros((F.OWN, F.UF), np.float32)
    mf = np.zeros((F.OWN, 5, F.MF), np.float32)
    legal = np.zeros((F.OWN, 5), bool)
    for slot in range(F.OWN):
        if x["tok_mask"][slot]:
            uf[slot], mf[slot], legal[slot] = prefix.feats(slot)
            prefix.advance(slot, int(y[slot]))
    b = {key: torch.as_tensor(np.array(x[key])[None]) for key in keys}
    b["maps"] = b["maps"].float() / 255
    for key in ("tok_pos", "tok_kind", "hist_cell", "hist_move"):
        b[key] = b[key].long()
    for key in ("scal", "tok_cargo", "hist_cargo"):
        b[key] = b[key].float()
    b.update(uf=torch.from_numpy(uf[None]), mf=torch.from_numpy(mf[None]), legal=torch.from_numpy(legal[None]),
             y=torch.from_numpy(y[None]), teacher=torch.tensor([teacher]))
    with torch.no_grad():
        lg, value, _, utility, _, _ = net.forward_heads(b)
    max_u = max(max_u, abs(r4[3] - float(utility[0])))
    max_v = max(max_v, abs(r4[2] - float(value[0])))
    if legal.any():
        max_lg = max(max_lg, float(np.max(np.abs(r4[1][legal] - lg.numpy()[0][legal]))))
    input_same &= same(before, (x, aux))
results["direct_parity"] = dict(default_triple_bit_equal_to_baseline=default_same, sampled_outputs_bit_equal=sampled_same,
                                sample_rng_state_equal=rng_same, input_features_aux_unchanged=input_same,
                                numpy_cache_encoder_max_error=max_encoder, torch_utility_max_error=max_u,
                                torch_legacy_value_max_error=max_v, torch_legal_logit_max_error=max_lg)
assert default_same and sampled_same and rng_same and input_same and max_encoder == 0
assert max_u < 1e-5 and max_v < 1e-5 and max_lg < 1e-5

x, aux, _, _ = positions[-2]
reference = pol.act(x, aux, teacher=3, want_u=True)
results["load_paths"] = {}
with tempfile.TemporaryDirectory(prefix="carbon-s1-network-") as td:
    path = Path(td) / "small.npz"
    np.savez(path, **w)
    with np.load(path, allow_pickle=False) as npz:
        for name, weights in (("npz_mapping", npz), ("npz_dict", dict(npz)), ("raw_dict", w),
                              ("prepared_with_meta", dict(pol.w, meta=w["meta"])),
                              ("dict_meta", dict(w, meta={"cfg": dict(net.cfg)}))):
            p = NP.Policy(weights)
            result = p.act(x, aux, teacher=3, want_u=True)
            results["load_paths"][name] = dict(exact=same(result, reference), G=p.G,
                transpose_exact=np.array_equal(p.w["utility.weightT"], w["utility.weight"].T))
            assert same(result, reference)
    # The worker loader must replace the policy/cache when a different weight file is loaded.
    GATE._init_worker(str(path))
    first = GATE._W["pol"]
    first.encode(x, 3)
    changed = {k: np.copy(v) for k, v in w.items()}
    changed["utility.bias"] += 1.
    changed["ctx.bias"] += .25
    path2 = Path(td) / "changed.npz"
    np.savez(path2, **changed)
    GATE._init_worker(str(path2))
    second = GATE._W["pol"]
    results["worker_weight_reload"] = dict(new_object=first is not second,
        empty_initial_cache=not hasattr(second, "last_sig"), preserved_G=second.G,
        fresh_model_exact=same(second.act(x, aux, teacher=3, want_u=True), NP.Policy(changed).act(x, aux, teacher=3, want_u=True)))
    assert results["worker_weight_reload"]["new_object"] and results["worker_weight_reload"]["empty_initial_cache"] and results["worker_weight_reload"]["fresh_model_exact"]

results["missing_keys"] = []
for missing in (("utility.weight",), ("utility.bias",), ("utility.weight", "utility.bias")):
    legacy = NP.Policy({k: v for k, v in w.items() if k not in missing})
    assert same(legacy.act(x, aux, teacher=3), reference[:3])
    rng = np.random.default_rng(9)
    before_rng = copy.deepcopy(rng.bit_generator.state)
    try:
        legacy.act(x, aux, teacher=3, sample_rng=rng, want_u=True)
    except KeyError as exc:
        results["missing_keys"].append(dict(removed=missing, raised=type(exc).__name__, key=exc.args[0],
            default_triple_works=True, sampling_rng_unchanged=same(before_rng, rng.bit_generator.state)))
    else:
        raise AssertionError("missing utility silently accepted")
results["stable_sigmoid"] = {str(z): NP._sigmoid(z) for z in (-1000, -100, 0, 100, 1000)}

results["cache_invalidation"] = []
for key in keys:
    xp = copy.deepcopy(x)
    if key == "maps": xp[key].flat[0] = (int(xp[key].flat[0]) + 37) % 256
    elif key == "scal": xp[key].flat[0] += .125
    elif key == "tok_mask": xp[key][0] = not xp[key][0]
    elif key == "tok_pos": xp[key][0] = (int(xp[key][0]) + 1) % F.N
    elif key == "tok_kind": xp[key][0] = (int(xp[key][0]) + 1) % 2
    elif key == "tok_cargo": xp[key][0] += 73.
    elif key == "hist_cell": xp[key][0, -1] = (int(xp[key][0, -1]) + 1) % F.N
    elif key == "hist_cargo": xp[key][0, -1] += 73.
    elif key == "hist_move": xp[key][0, -1] = (int(xp[key][0, -1]) + 1) % 5
    initial = cache.encode(x, 3)
    next_enc = cache.enc_for(xp, 3)
    delta = err(next_enc, pol.encode(xp, 3))
    results["cache_invalidation"].append(dict(field=key, different_signature=S.CachePolicy._sig(x, 3) != S.CachePolicy._sig(xp, 3), fresh_encode_max_error=delta,
                                              fixture_encoding_difference=err(initial, next_enc)))
    assert delta == 0 and results["cache_invalidation"][-1]["different_signature"]
cache.encode(x, 3)
teacher_error = err(cache.enc_for(x, 4), pol.encode(x, 4))
NP.act_tta(cache, x, aux, teacher=3, k=2)
tta_error = err(cache.enc_for(x, 3), pol.encode(x, 3))
results["cache_extra"] = dict(teacher_switch_error=teacher_error, tta_restore_error=tta_error)
assert teacher_error == 0 and tta_error == 0

class FreshPolicy(NP.Policy):
    def enc_for(self, x, teacher):
        return self.encode(x, teacher)

# Synthetic decision at an actual initial state triggers the existing recruit guard.
# This is a guarded decode equivalence test, not a claim about sampled policy frequency.
gx, ga, _, _ = positions[0]
gf = F.Feat(0)
gf.observe(FastCarbon(CFG).reset(seed=42).obs(0))
gy, gl, _ = pol.act(gx, ga, teacher=3)
gy = gy.copy()
gy[10:] = 0
guard_cfg = dict(T.TEACHER_GUARD_CFG)
stats_fresh, stats_cache = {}, {}
fresh_guard = S.BankruptcyGuard(guard_cfg)
cached_guard = S.BankruptcyGuard(guard_cfg)
fresh_labels = fresh_guard.apply(gx, ga, gy.copy(), gl, FreshPolicy(w), gf.own_uid, 3, stats_fresh)
cache.encode(x, 7)  # stale opposite teacher/state must be repaired in guard re-decode
cached_labels = cached_guard.apply(gx, ga, gy.copy(), gl, cache, gf.own_uid, 3, stats_cache)
results["guard_direct_fresh_vs_cache"] = dict(labels_exact=np.array_equal(fresh_labels, cached_labels), stats_fresh=stats_fresh, stats_cache=stats_cache)
assert np.array_equal(fresh_labels, cached_labels) and stats_cache.get("guard_recruit", 0) > 0

# Capture the real wrapper rather than treating two evaluate_root calls as independent NP/cache implementations.
original_class = T.CachePolicy
class SpyCache(S.CachePolicy):
    created = []
    def __init__(self, weights):
        super().__init__(weights)
        self.calls = []
        self.created.append(self)
    def act(self, *args, **kwargs):
        self.calls.append(dict(teacher=kwargs.get("teacher", 0), want_u=kwargs.get("want_u", False)))
        return super().act(*args, **kwargs)

fm, fo = F.Feat(0), F.Feat(1)
xm, xo = fm.observe(sim.obs(0)), fo.observe(sim.obs(1))
before_sim = copy.deepcopy(sim.get_state())
before_input = copy.deepcopy((xm, fm.aux, xo, fo.aux))
source_weights = {k: v.copy() for k, v in pol.w.items()}
np_random = np.random.get_state()
T.CachePolicy = SpyCache
try:
    teacher_result = T.evaluate_root(sim, 0, pol, xm, fm.aux, xo, fo.aux, fm, fo,
                                    teacher=3, K=2, M=1, H=1, seed_base=7000)
    wrapped = SpyCache.created[-1]
    u_calls = [c for c in wrapped.calls if c["want_u"]]
    results["teacher_wrapper"] = dict(G=wrapped.G, cfg_equal=wrapped.cfg == pol.cfg,
        utility_keys_present=all(k in wrapped.w for k in ("utility.weightT", "utility.bias")),
        source_weights_unchanged=all(np.array_equal(v, source_weights[k]) for k, v in pol.w.items()),
        direct_outputs_exact=same(wrapped.act(x, aux, teacher=3, want_u=True), reference),
        utility_leaf_calls=len(u_calls), all_calls_requested_teacher=all(c["teacher"] == 3 for c in wrapped.calls),
        counter_matches_real_calls=teacher_result["n_act_calls"] == len(wrapped.calls) - 1,
        root_sim_unchanged=same(before_sim, sim.get_state()), root_input_unchanged=same(before_input, (xm, fm.aux, xo, fo.aux)),
        numpy_global_rng_unchanged=same(np_random, np.random.get_state()),
        Q_mean=teacher_result["Q_mean"].tolist())
    assert wrapped.G == 4 and results["teacher_wrapper"]["direct_outputs_exact"] and len(u_calls) > 0
    assert results["teacher_wrapper"]["root_sim_unchanged"] and results["teacher_wrapper"]["root_input_unchanged"]
    assert results["teacher_wrapper"]["source_weights_unchanged"] and results["teacher_wrapper"]["numpy_global_rng_unchanged"]
    T.gen_candidates(pol, xm, fm.aux, teacher=3, K=2, counter=[0], guard=S.BankruptcyGuard(guard_cfg), own_uid=fm.own_uid)
    gw = SpyCache.created[-1]
    results["candidate_wrapper"] = dict(G=gw.G, cfg_equal=gw.cfg == pol.cfg,
        direct_outputs_exact=same(gw.act(x, aux, teacher=3, want_u=True), reference))
    assert gw.G == 4 and results["candidate_wrapper"]["direct_outputs_exact"]
finally:
    T.CachePolicy = original_class

results["limits"] = ["Untrained small deterministic fixture; does not reproduce canonical Studio heads_v2 figures or playing strength.",
                     "Nine cache-key perturbations are interface fixtures, not claims of complete-game reachability.",
                     "Stage1 creates a new policy on weight load; arbitrary in-place weight mutation/hot-swap is not an implemented cache contract.",
                     "No optimizer/training, no full match, no remote operation."]
results["passed"] = True
payload = json.dumps(results, ensure_ascii=False, indent=2) + "\n"
print(payload, end="")
if os.environ.get("CARBON_REVIEW_OUT"):
    out = Path(os.environ["CARBON_REVIEW_OUT"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "network-results.json").write_text(payload)

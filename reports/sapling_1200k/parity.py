"""Read-only Sapling export acceptance on real replay prefixes (CPU, one thread).

python reports/sapling_1200k/parity.py \
  --weight initial=/path/to/initial.npz --weight trained=/path/to/trained.npz \
  --weight legacy=/path/to/551k.npz --out reports/sapling_1200k/parity.json

Builds actual int8 packs in a TemporaryDirectory. Never writes beside input
weights. FP32->int8 differences are reported, not confused with pack/deq parity.
"""
import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_key] = "1"
import argparse
import contextlib
import copy
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
import cnet2_feat as F
import cnet2_np as NP
import cnet2_search as SEARCH
import cnet2_pack as PACK
from cnet2_agent import load_weights, make_agent
from rl2_ppo_cnet2 import Cnet2Adapter

torch.set_num_threads(1)
REPLAY = "data/master6s_corpus/21f9889b21072db820926c967d5bdb85-1.json.gz"
XKEYS = ("maps", "scal", "tok_pos", "tok_kind", "tok_mask", "tok_cargo", "hist_cell", "hist_cargo", "hist_move")
TOL = 3e-5
TEACHER = 2


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load_replay():
    path = ROOT / REPLAY
    raw = path.read_bytes() if path.exists() else subprocess.check_output(
        ["git", "show", "HEAD:" + REPLAY], cwd=ROOT)
    return json.loads(gzip.decompress(raw)), sha(raw)


def observation(replay, step, seat):
    row = replay["steps"][step]
    obs = dict(row[0]["observation"])
    obs.update(row[seat]["observation"])
    obs["player"] = seat
    obs.pop("full_carbon", None)
    return obs


def roots_from_replay(replay):
    roots, selected = [], set()
    for seat in (0, 1):
        feat = F.Feat(seat)
        special = set()
        for step in range(295):
            obs = observation(replay, step, seat)
            x = feat.observe(obs)
            choose = step in (0, 1, 12, 50, 150, 294)
            for tag, condition in (
                    ("one_free_slot", x["slots"] == 1 and x["cash"] >= 60),
                    ("one_recruit_budget", 30 <= x["cash"] < 60 and x["slots"] >= 2)):
                if condition and tag not in special:
                    special.add(tag)
                    choose = True
            if choose:
                roots.append(dict(seat=seat, step=step, x=copy.deepcopy(x), aux=copy.deepcopy(feat.aux)))
                selected.add((seat, step))
    assert len(roots) >= 12
    return roots, selected


def prefix(x, aux, labels):
    uf = np.zeros((F.OWN, F.UF), np.float32)
    mf = np.zeros((F.OWN, 5, F.MF), np.float32)
    legal = np.zeros((F.OWN, 5), bool)
    p = F.Prefix(x, aux)
    for s in range(F.OWN):
        if not x["tok_mask"][s]:
            continue
        uf[s], mf[s], legal[s] = p.feats(s)
        assert legal[s, labels[s]], (s, labels[s])
        p.advance(s, int(labels[s]))
    return uf, mf, legal


def tensor_batch(items):
    batch = {k: torch.from_numpy(np.ascontiguousarray(np.stack([z["x"][k] for z in items]))) for k in XKEYS}
    batch["maps"] = (batch["maps"].float() / 255).reshape(-1, F.P, F.S, F.S)
    for k in ("scal", "tok_cargo", "hist_cargo"):
        batch[k] = batch[k].float()
    for k in ("tok_pos", "tok_kind", "hist_cell", "hist_move"):
        batch[k] = batch[k].long()
    for k in ("uf", "mf", "legal", "y"):
        batch[k] = torch.from_numpy(np.ascontiguousarray(np.stack([z[k] for z in items])))
    batch["y"] = batch["y"].long()
    batch["teacher"] = torch.full((len(items),), TEACHER, dtype=torch.long)
    return batch


def compare(actual, expected, name, tolerance=TOL):
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert np.isfinite(actual).all() and np.isfinite(expected).all(), name + " nonfinite"
    error = float(np.max(np.abs(actual - expected)))
    assert error <= tolerance, (name, error, tolerance)
    return error


def numeric_checks(weights, roots):
    net = Cnet2Adapter().make_model(load_weights(weights)).cpu().eval()
    pol = NP.Policy(load_weights(weights))
    assert pol.G == int(net.cfg["G"])
    items, greedy, enc = [], [], []
    changed_features = changed_legal = 0
    for root in roots:
        x, aux = root["x"], root["aux"]
        base_y, base_lg, _, _ = pol.act(x, aux, teacher=TEACHER, want_u=True)
        greedy.append(base_y)
        enc.append(pol.encode(x, TEACHER))
        variants = [base_y]
        for pivot in (next((s for s in range(10) if x["tok_mask"][s]), None), 10):
            if pivot is None or not x["tok_mask"][pivot]:
                continue
            alternatives = [a for a in range(5) if base_lg[pivot, a] > -1e8 and a != base_y[pivot]]
            if not alternatives:
                continue
            target = alternatives[0]
            def hook(s, logits, pf, action, p=pivot, a=target):
                return a if s == p else None
            y = pol.act(x, aux, teacher=TEACHER, hook=hook)[0]
            variants.append(y)
        base_pf = prefix(x, aux, base_y)
        for y in variants:
            uf, mf, legal = prefix(x, aux, y)
            changed_features += int(not np.array_equal(uf, base_pf[0]) or not np.array_equal(mf, base_pf[1]))
            changed_legal += int(not np.array_equal(legal, base_pf[2]))
            _, lg, v, u = pol.act(x, aux, teacher=TEACHER, forced=y, want_u=True)
            items.append(dict(x=x, aux=aux, y=y, uf=uf, mf=mf, legal=legal, lg=lg, v=v, u=u))
    assert changed_features > 0 and changed_legal > 0, "fixture must exercise real prefix changes"
    b = tensor_batch(items)
    with torch.no_grad():
        t_lg, t_v, _ = net(b)
        tok, ctx, _ = net.encode(b)
        t_u, _, _ = net.heads(ctx)
        seq_lg, seq_v, _ = net.forward_seq(b)
    active = np.stack([z["x"]["tok_mask"][:F.OWN] for z in items])
    errors = {
        "torch_np_logits": compare(t_lg.numpy()[active], np.stack([z["lg"] for z in items])[active], "torch/NP logits"),
        "torch_np_value": compare(t_v.numpy(), [z["v"] for z in items], "torch/NP value"),
        "torch_np_utility": compare(t_u.numpy(), [z["u"] for z in items], "torch/NP utility"),
        "torch_sequence_logits": compare(seq_lg.numpy()[active], t_lg.numpy()[active], "sequence logits"),
        "torch_sequence_value": compare(seq_v.numpy(), t_v.numpy(), "sequence value"),
    }
    xs = [r["x"] for r in roots]
    auxs = [r["aux"] for r in roots]
    X = SEARCH.stack_x(xs)
    tok_b, ctx_b = SEARCH.encode_b(pol, X, np.full(len(xs), TEACHER, np.int64))
    errors["np_batch_tokens"] = compare(tok_b, np.stack([e[0] for e in enc]), "NP batch tokens")
    errors["np_batch_context"] = compare(ctx_b, np.stack([e[1] for e in enc]), "NP batch context")
    y_b = SEARCH.decode_b(pol, X, auxs, tok_b, ctx_b)
    np.testing.assert_array_equal(y_b, np.stack(greedy))
    # Explicitly force the independently constructed legal variants through the
    # batch decoder, whose dynamic prefix must match the scalar decoder.
    XI = SEARCH.stack_x([i["x"] for i in items])
    tb, cb = SEARCH.encode_b(pol, XI, np.full(len(items), TEACHER, np.int64))
    batch_logits = np.full((len(items), F.OWN, 5), NP.NEG, np.float32)
    forced = np.stack([i["y"] for i in items])
    labels = SEARCH.decode_b(pol, XI, [i["aux"] for i in items], tb, cb,
                             forced=forced, lg_out=batch_logits)
    np.testing.assert_array_equal(labels, forced)
    errors["np_batch_forced_logits"] = compare(batch_logits[active], np.stack([z["lg"] for z in items])[active], "NP batch forced logits")
    return dict(parameter_count=sum(p.numel() for p in net.parameters()), cfg=net.cfg,
                numpy_group_count=pol.G, torch_group_count=net.stem_n.num_groups,
                roots=len(roots), legal_joint_candidates=len(items),
                candidates_changing_prefix_features=changed_features,
                candidates_changing_prefix_legality=changed_legal,
                max_abs_errors=errors, batch_greedy_label_differences=0)


def pack_checks(weights, roots, selected, replay):
    with tempfile.TemporaryDirectory(prefix="sapling-parity-") as directory:
        src = Path(directory) / "weights.npz"
        dst = Path(directory) / "pack.py"
        shutil.copyfile(weights, src)
        # cnet2_pack writes its dequantized file beside src; src is a temporary
        # copy, so this test cannot mutate any training/input artifact.
        with contextlib.redirect_stdout(io.StringIO()):
            PACK.main(str(src), str(dst), teacher=TEACHER)
        deq = Path(directory) / "weights_q8.npz"
        numeric = numeric_checks(deq, roots)
        meta = json.loads(str(load_weights(deq)["meta"]))
        assert meta["cfg"]["G"] == numeric["cfg"]["G"]
        calls = 0
        for seat in (0, 1):
            spec = importlib.util.spec_from_file_location("sapling_parity_pack_%d_%d" % (seat, calls), dst)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            reference = make_agent(str(deq), teacher=TEACHER)
            for step in range(295):
                obs = observation(replay, step, seat)
                packed = module.agent(obs, replay["configuration"])
                direct = reference(obs, replay["configuration"])
                assert packed == direct, ("pack/deq actions", seat, step)
                calls += 1
            assert json.loads(module._WMETA)["cfg"]["G"] == numeric["cfg"]["G"]
        original = NP.Policy(load_weights(weights))
        quantized = NP.Policy(load_weights(deq))
        delta, changed, units = 0.0, 0, 0
        for r in roots:
            x, aux = r["x"], r["aux"]
            y, lg, _ = original.act(x, aux, teacher=TEACHER)
            qy, _, _ = quantized.act(x, aux, teacher=TEACHER)
            qlg = quantized.act(x, aux, teacher=TEACHER, forced=y)[1]
            active = x["tok_mask"][:F.OWN]
            delta = max(delta, float(np.max(np.abs(lg[active] - qlg[active]))))
            changed += int((y[active] != qy[active]).sum())
            units += int(active.sum())
        return dict(pack_bytes=dst.stat().st_size, pack_md5=hashlib.md5(dst.read_bytes()).hexdigest(),
                    pack_sha256=sha(dst.read_bytes()), dequantized_sha256=sha(deq.read_bytes()),
                    metadata_group_count=meta["cfg"]["G"], actual_pack_deq_calls=calls,
                    actual_pack_deq_action_differences=0, dequantized_numeric_checks=numeric,
                    quantization_diagnostic=dict(max_abs_same_prefix_logits_change=delta,
                        greedy_changed_slots=changed, active_slots=units,
                        note="FP32/int8 may differ; this is not a claim of preserved game strength."))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weight", action="append", required=True, help="label=/path/to/weights.npz")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert PACK.QMODE == "int8", "this acceptance requires actual int8 export"
    replay, digest = load_replay()
    roots, selected = roots_from_replay(replay)
    report = dict(schema_version="sapling-export-parity-v1", status="PASS", tolerance_abs=TOL,
        device="CPU", torch_threads=torch.get_num_threads(),
        python=sys.version.split()[0], numpy=np.__version__, torch=torch.__version__,
        replay_source=dict(path=REPLAY, compressed_sha256=digest,
            roots=[dict(seat=r["seat"], step=r["step"], cash=r["x"]["cash"], free_slots=r["x"]["slots"]) for r in roots],
            hidden_carbon_passed_to_policy=False, feature_history="Sequential public observations from step0."),
        weights={}, source_sha256={"code/"+p:sha((ROOT/"code"/p).read_bytes()) for p in (
            "cnet2_model.py", "cnet2_np.py", "cnet2_search.py", "cnet2_feat.py", "cnet2_agent.py", "cnet2_pack.py", "rl2_ppo_cnet2.py")},
        limitations=["Numerical/export acceptance only; no training, strength assessment or platform-resource certification.",
            "Preserve complete cfg.G metadata. The legacy no-config fallback is G=8 and is not suitable for C=93."])
    for entry in args.weight:
        label, path = entry.split("=", 1)
        path = Path(path)
        assert label not in report["weights"]
        assert path.is_file(), str(path)
        result = dict(input_filename=path.name, input_sha256=sha(path.read_bytes()))
        result["fp32_numeric_checks"] = numeric_checks(path, roots)
        result["int8_export"] = pack_checks(path, roots, selected, replay)
        report["weights"][label] = result
        print(label, "PASS", result["fp32_numeric_checks"]["cfg"], flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(args.out, flush=True)


if __name__ == "__main__":
    main()

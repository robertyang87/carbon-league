"""Two native CNet2 towers, with the original 8da4 public-state decoder and arbitration."""
import base64
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import zlib

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ACTOR = HERE.parents[1]
ROOT = ACTOR.parents[2]
SOURCE = ACTOR / "cargo_tree_action_learning_dev/candidate.py"
sys.path.insert(0, str(ROOT / "code"))
from cnet2_model import CNet2
import cnet2_pack

KEYS = (("_B1", "_Q1", "_M1"), ("_W64", "_QMETA", "_WMETA"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load():
    plan = json.loads((HERE / "protocol.json").read_text())
    if sha(SOURCE) != plan["parent_sha256"]:
        raise ValueError("8da4 source changed")
    spec = importlib.util.spec_from_file_location("frozen_8da4", SOURCE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    arrays, models = [], []
    for blob, layout, meta_name in KEYS:
        compressed = base64.b64decode(getattr(mod, blob))
        original = zlib.decompress(compressed)
        meta = getattr(mod, layout)
        weights = mod._dequantize(original, meta)
        restored_meta, restored, _ = cnet2_pack.quantize(weights)
        if restored_meta != meta or restored != original:
            raise ValueError("zero-update int8 export did not preserve the entire original payload")
        net = CNet2(**json.loads(getattr(mod, meta_name))["cfg"])
        net.load_state_dict({k: torch.from_numpy(v.copy()) for k, v in weights.items()}, strict=True)
        for k, tensor in net.state_dict().items():
            if not np.array_equal(tensor.numpy(), weights[k]):
                raise ValueError(k)
        models.append(net.eval())
        arrays.append(weights)
    mod._COUNTS = {r: dict(valid_slots=0, disagree=0, selected_student_in_disagreement=0,
                           selected_third_in_disagreement=0) for r in ("base", "collector", "planter")}
    return mod, arrays, models


def set_native(mod, arrays):
    native = sys.modules["cnet2_np"].Policy
    mod._NETS = [native({**w, "meta": getattr(mod, names[2])}) for w, names in zip(arrays, KEYS, strict=True)]


def record_native(mod):
    source = inspect.getsource(mod._return_decode)
    target = "action = int(np.argmax(np.where(legal, combined, -np.inf)))"
    if source.count(target) != 1:
        raise ValueError("frozen decoder changed")
    altered = source.replace(target, "capture(apply_residual, slot, legal, combined)\n        " + target)
    table = {}
    def capture(residual, slot, legal, logits):
        table[bool(residual)][slot] = (legal.copy(), logits.copy())
    # A new function object with the native module globals keeps its original counters and guard.
    mod.capture = capture
    exec(compile(altered, "<recording-only>", "exec"), vars(mod))
    def run(root):
        table.update({False: {}, True: {}})
        y = mod._mixed_actions(root["x"], root["aux"])
        branches = table.copy()
        return y, branches
    return run


def tensor_batch(x, style):
    values = {}
    for k in ("maps", "scal", "tok_pos", "tok_kind", "tok_cargo", "tok_mask",
              "hist_cell", "hist_cargo", "hist_move"):
        b = torch.from_numpy(np.ascontiguousarray(x[k]))[None]
        if k in ("tok_pos", "tok_kind", "hist_cell", "hist_move"):
            b = b.long()
        elif k == "tok_mask":
            b = b.bool()
        else:
            b = b.float()
        values[k] = b
    values["maps"] = values["maps"].reshape(1, -1, 15, 15) / 255.
    values["teacher"] = torch.tensor([style], dtype=torch.long)
    return values


def neuron_step(net, state, slot, previous, unit, moves):
    tok, ctx, hidden = state
    features = torch.cat((tok[0, slot], ctx[0], unit, moves.flatten()))[None]
    h = net.gru(net.din(features) + net.aemb.weight[previous], hidden)
    if slot < 10:
        z = net.whead(h)[0]
    else:
        z = torch.nn.functional.pad(net.bhead(h)[0], (0, 2))
    action_logits = z + net.V(moves) @ net.U(h)[0]
    return (tok, ctx, h), action_logits


def decode_pair(mod, nets, root):
    F = sys.modules["cnet2_feat"]
    x, aux = root["x"], root["aux"]
    b = tensor_batch(x, mod._HYBRID["style"])
    encoded = []
    for net in nets:
        tok, ctx, _ = net.encode(b)
        encoded.append((tok, ctx, net.h0(ctx)))
    tables, orders = {}, {}
    for residual in (False, True):
        states = list(encoded)
        prefix = F.Prefix(x, aux)
        y = np.full(F.OWN, -1, dtype=np.int64)
        previous = 8
        table = {}
        for slot in np.flatnonzero(x["tok_mask"][:F.OWN]):
            slot = int(slot)
            uf, mf, legal = prefix.feats(slot)
            unit = torch.from_numpy(uf).float()
            moves = torch.from_numpy(mf).float()
            scores = []
            for i, net in enumerate(nets):
                states[i], score = neuron_step(net, states[i], slot, previous, unit, moves)
                scores.append(score)
            z = mod._HYBRID["expert_weight"] * scores[0] + mod._HYBRID["student_weight"] * scores[1]
            if residual and slot < 10 and x["tok_kind"][slot] == 0:
                basis = mod._return_basis(x, aux, slot, mf) @ mod._RETURN_THETA
                z = z + torch.as_tensor(basis, dtype=z.dtype)
            mask = torch.from_numpy(legal)
            a = int(z.detach().masked_fill(~mask, -torch.inf).argmax())
            table[slot] = (legal, z)
            y[slot] = a
            prefix.advance(slot, a)
            previous = a + (5 if slot >= 10 else 0)
        tables[residual], orders[residual] = table, y
    baseline, alternative = orders[False], orders[True]
    protected = (np.arange(F.OWN) >= 10) | (x["tok_kind"][:F.OWN] == 1)
    chosen = not np.any(baseline[protected] != alternative[protected])
    if chosen and np.any(baseline != alternative):
        chosen = not bool(mod._return_conflicts(x, alternative) - mod._return_conflicts(x, baseline))
    return orders[chosen], tables, chosen

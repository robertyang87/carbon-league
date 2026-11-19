"""Public features and physical expert labels for the frozen native corpus."""
from pathlib import Path
import gzip
import hashlib
import json
import os
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ACTOR = EXP.parent
ROOT = ACTOR.parents[2]
PACK = EXP / "candidate.py"
BASE = EXP / "candidate.npz"
PARENT_SHA = "8da4b465f7fa6e601e51bd7c2e81f25825cb59b9b5ab285f7f56d0f6bbfe1cac"
sys.path[:0] = [str(ROOT / "code"), str(ACTOR / "blended_policy_return_dev")]
from fastsim import FastCarbon
import policy

XKEYS = ("maps", "scal", "tok_pos", "tok_kind", "tok_mask", "tok_cargo", "hist_cell", "hist_cargo", "hist_move")
ROLE = ("collector", "planter", "base")
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(path):
    return json.loads(gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes())

def atomic(path, raw):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(raw)
    os.replace(tmp, path)

def save(path, v):
    raw = (json.dumps(v, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    atomic(path, gzip.compress(raw, mtime=0) if path.suffix == ".gz" else raw)

def package(path=PACK):
    if sha(PACK) != PARENT_SHA:
        raise ValueError("different first-win parent")
    return policy.load(path)

def ticks(card, n, k):
    # One root drawn within each contiguous time stratum, with no outcome or action fields.
    rng = np.random.default_rng(int(card["sha256"][:16], 16) ^ 260119)
    return {int(rng.choice(v)) for v in np.array_split(np.arange(n), min(n, k))}

def actual_features(f, x, native_commands, model):
    F = sys.modules["cnet2_feat"]
    y = f.labels(native_commands)
    physical = {uid: cmd for uid,cmd in native_commands.items() if cmd and cmd != "NONE"}
    if f.to_actions(y) != physical:
        raise ValueError("physical teacher command did not round-trip")
    prefix = F.Prefix(x, f.aux)
    uf = np.zeros((F.OWN, F.UF), np.float32)
    mf = np.zeros((F.OWN, 5, F.MF), np.float32)
    legal = np.zeros((F.OWN, 5), bool)
    eligible = np.zeros(F.OWN, bool)
    residual = np.zeros((F.OWN, 5), np.float32)
    mask = x["tok_mask"][:F.OWN]
    if not np.array_equal(mask, y >= 0):
        raise ValueError("wrong active unit")
    cut = F.OWN
    for s in np.flatnonzero(mask):
        uf[s], mf[s], legal[s] = prefix.feats(s)
        if not legal[s, y[s]]:
            cut = int(s)
            break
        eligible[s] = legal[s].sum() > 1
        if s < 10 and x["tok_kind"][s] == 0:
            residual[s] = model._return_basis(x, f.aux, s, mf[s]) @ model._RETURN_THETA
        prefix.advance(s, int(y[s]))
    # No suffix after the first unrepresentable expert prefix is scored. A neutral
    # padded suffix keeps all Torch operations finite without propagating labels.
    legal[~legal.any(1), 0] = True
    return dict(**{k:x[k] for k in XKEYS}, y=y, uf=uf, mf=mf, legal=legal,
                eligible=eligible, residual=residual, cut=cut)

def tensorbatch(data, ids):
    import torch
    types = {"tok_pos":torch.long,"tok_kind":torch.long,"hist_cell":torch.long,"hist_move":torch.long,
             "y":torch.long,"tok_mask":torch.bool,"eligible":torch.bool,"legal":torch.bool}
    keys = (*XKEYS, "y", "uf", "mf", "legal", "eligible", "residual")
    b = {k:torch.as_tensor(data[k][ids]).to(types.get(k, torch.float32)) for k in keys}
    b["maps"] = b["maps"].reshape(len(b["y"]), -1, 15, 15) / 255.0
    b["teacher"] = torch.full((len(b["y"]),), 2, dtype=torch.long)
    return b

def torch_model(m, idx):
    import torch
    from cnet2_model import CNet2,migrate_state_dict
    net = CNet2(**json.loads(m._M1 if idx == 0 else m._WMETA)["cfg"])
    w = m._NETS[idx].w
    migrate_state_dict(net,{k:torch.from_numpy(w[k].copy()) for k in net.state_dict() if k in w})
    return net.eval()

def iter_public(c, k):
    tape = read(ROOT/c["path"])
    if sha(ROOT/c["path"]) != c["sha256"]:
        raise ValueError("replay digest changed")
    seat = 1-c["seat"]
    cfg = tape["game_configuration"]
    sim = FastCarbon(cfg).reset(seed=c["seed"],full_carbon=tape["initial_full_carbon"])
    times = ticks(c,len(tape["actions"]),k)
    for t, commands in enumerate(tape["actions"]):
        obs = sim.obs(seat)
        if set(obs) != {"carbon","players","player","step","trees","remainingOverageTime"} or obs["player"] != seat:
            raise ValueError("nonpublic observation or wrong acting seat")
        yield t,obs,cfg,commands[seat],t in times
        sim.step(*commands)

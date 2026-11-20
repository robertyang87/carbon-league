"""Fit a shared team-mode decoder and an equal-size independent-mode control."""
import argparse
import base64
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent
sys.path.insert(0, str(SOURCE))
import train as source
from train import native
common = source.common
PLAN = common.read(HERE/"protocol.json")


class Modes(nn.Module):
    def __init__(self, context):
        super().__init__()
        self.route = nn.Linear(context, 3)
        self.moves = nn.Parameter(torch.empty(3,3,15))
        self.bias = nn.Parameter(torch.empty(3,3,5))
        nn.init.normal_(self.moves, std=.035)
        nn.init.normal_(self.bias, std=.035)
        nn.init.normal_(self.route.weight, std=.01)
        nn.init.zeros_(self.route.bias)

    def forward(self, d, joint):
        ctx = d["ctx"]
        ctx = ctx / (ctx.square().mean(-1,keepdim=True) + 1e-6).sqrt()
        prior = self.route(ctx).log_softmax(-1)
        # Each mode changes legal action scores using the current unit role and the action's features.
        w = self.moves[d["role"]]
        bias = self.bias[d["role"]]
        adjustment = torch.einsum("bsaf,bszf->bsza", d["moves"], w) + bias
        raw = (d["base"].unsqueeze(2) + adjustment).masked_fill(~d["legal"].unsqueeze(2), -1e9)
        label = d["y"].clamp_min(0)[:,:,None,None].expand(-1,-1,3,1)
        observed = raw.log_softmax(-1).gather(-1,label).squeeze(-1) * d["usable"][:,:,None]
        if joint:
            nll = -torch.logsumexp(prior + observed.sum(1), dim=-1)
        else:
            nll = -(torch.logsumexp(prior[:,None,:] + observed, dim=-1) * d["usable"]).sum(1)
        return nll


def archive(split):
    data = source.read_split(split)
    model = common.package(SOURCE/"bc.py")
    partner, student = [common.torch_model(model,idx).eval() for idx in (0,1)]
    parts = {}
    with torch.inference_mode():
        for i in range(0,len(data["y"]),32):
            ids = np.arange(i,min(i+32,len(data["y"])))
            b = native.batch(data,ids)
            tok, ctx, _ = student.encode(b)
            ptok, pctx, _ = partner.encode(b)
            base = .25 * student.decode(b,tok,ctx) + .75 * partner.decode(b,ptok,pctx)
            # Train before protection. Deployment still uses the frozen protection on both models.
            value = dict(ctx=ctx,base=base,role=b["tok_kind"][:,:12].clamp(0,2),
                         moves=b["mf"],legal=b["legal"],y=b["y"],usable=b["eligible"])
            for key,tensor in value.items():
                parts.setdefault(key,[]).append(tensor)
    return {k:torch.cat(v) for k,v in parts.items()},data["terrain"],data["seat"]


def records(data, terrain, seat, nets):
    with torch.no_grad():
        base = -data["base"].masked_fill(~data["legal"],-1e9).log_softmax(-1).gather(-1,data["y"].clamp_min(0).unsqueeze(-1)).squeeze(-1)
        values = {"parent":(base*data["usable"]).sum(1).numpy()}
        for name,model in nets.items():
            values[name] = model(data,name=="joint").detach().numpy()
    return {str(t):{k:float(v[terrain==t].mean()) for k,v in values.items()} for t in sorted(set(terrain))}


def build(model, name):
    state = {k:v.detach().numpy().astype(np.float32) for k,v in model.state_dict().items()}
    path = HERE/(name+".npz")
    np.savez_compressed(path, **state)
    parent = (SOURCE/"bc.py").read_text()
    start = parent.index("def _return_decode(")
    end = parent.index("def _return_basis(",start)
    fragment = parent[start:end]
    first = "    previous = 8"
    second = "        action = int(np.argmax(np.where(legal, combined, -np.inf)))"
    third = "        labels[slot] = action"
    if any(fragment.count(key)!=1 for key in (first,second,third)):
        raise ValueError("changed production prefix")
    fragment=fragment.replace(first,first+"\n    prior = _mode_prior(encoded[1][2])")
    fragment=fragment.replace(second,"        mixed, components = _mode_score(prior, combined, int(x['tok_kind'][slot]), move, legal)\n        action = int(np.argmax(mixed))")
    fragment=fragment.replace(third,"        if _SHARE_MODE:\n            prior = _normalize_mode(prior + components[:, action])\n"+third)
    globals_code = (HERE/"runtime.py").read_text()
    embed = base64.b64encode(path.read_bytes()).decode()
    config = f"\nimport io as _mode_io\n_SHARE_MODE = {name=='joint'}\n_MODE_DATA = dict(np.load(_mode_io.BytesIO(base64.b64decode({embed!r})), allow_pickle=False))\n"
    output=HERE/(name+".py")
    output.write_text(parent[:start]+config+globals_code+"\n"+fragment+parent[end:])
    if output.stat().st_size > 10_000_000:
        raise ValueError("package too large")
    return dict(package_sha256=common.sha(output),weights_sha256=common.sha(path),bytes=output.stat().st_size)


def fit():
    if common.sha(SOURCE/"bc.py")!=PLAN["parent"] or common.sha(SOURCE/"data.json")!=PLAN["data"]:
        raise ValueError("changed production or data")
    d,terrain,seat = archive("train")
    torch.manual_seed(PLAN["seed"])
    start = Modes(d["ctx"].shape[1])
    nets = {}; history = {}; files = {}
    for name in PLAN["models"]:
        net = Modes(d["ctx"].shape[1]); net.load_state_dict(start.state_dict())
        optimizer = torch.optim.Adam(net.parameters(), lr=PLAN["lr"])
        rng = np.random.default_rng(PLAN["seed"]); log=[]
        for epoch in range(PLAN["epochs"]):
            before=[]
            for ids in np.array_split(rng.permutation(len(terrain)),len(terrain)//PLAN["batch"]):
                batch = {k:v[ids] for k,v in d.items()}
                nll=net(batch,name=="joint")
                reg=sum(p.square().mean() for p in net.parameters())
                loss=(nll/batch["usable"].sum(1).clamp_min(1)).mean()+PLAN["anchor"]*reg
                optimizer.zero_grad(set_to_none=True)
                loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),1,error_if_nonfinite=True)
                optimizer.step(); before.append(loss.item())
            log.append(float(np.mean(before)))
        nets[name]=net; history[name]=log; files[name]=build(net,name)
    common.save(HERE/"fit.json",dict(root_count=len(terrain),maps=len(set(terrain)), parameters=sum(v.numel() for v in start.parameters()),
            train_protocol=common.sha(HERE/"protocol.json"), serving=common.sha(HERE/"runtime.py"), source=common.sha(Path(__file__)),
            weight_changes={k:sum(int(torch.count_nonzero(w-start.state_dict()[p])) for p,w in n.state_dict().items()) for k,n in nets.items()},
            loss=history, releases=files, split="train"))
    print("trained",files,flush=True)


def assess():
    arms={}
    for name in PLAN["models"]:
        with np.load(HERE/(name+".npz"), allow_pickle=False) as f:
            model=Modes(f["route.weight"].shape[1])
            model.load_state_dict({k:torch.tensor(f[k]) for k in f.files})
        arms[name]=model
    output={}
    for split in ("val","dev"):
        d,t,s=archive(split)
        output[split]=records(d,t,s,arms)
    common.save(HERE/"likelihood.json",output)
    print("mean pre-guard dev team nll",{k:np.mean([v[k] for v in output["dev"].values()]) for k in ("parent","joint","independent")})


if __name__=="__main__":
    torch.set_num_threads(1)
    (fit if sys.argv[1]=="fit" else assess)()

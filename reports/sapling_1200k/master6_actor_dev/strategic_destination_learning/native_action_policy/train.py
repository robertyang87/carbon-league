"""Train the small conditional action readout on guarded physical Sapling decisions."""
import argparse
import ast
from collections import Counter
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(STUDY))
import free_entry as entry
from common import read, save, sha

PARENT = STUDY / "bc.py"
RULE = read(HERE / "protocol.json")
FEATURES = 15 + 9 + 20 + 5 + 2
WIDTH = 24


def check(flag, why):
    if not flag:
        raise ValueError(why)


def package(weights):
    text = PARENT.read_text()
    check(sha(PARENT) == RULE["parent"], "parent identity changed")
    tree = ast.parse(text)
    original, = [x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == "_return_decode"]
    function = ast.get_source_segment(text, original)
    function = function.replace("_return_decode(x, aux, encoded, apply_residual)",
                                "_native_decode(x, aux, encoded, apply_residual, forced=None, trace=None)")
    start = function.index("        action = int(np.argmax")
    end = function.index("        labels[slot] = action", start)
    select = """        if trace is not None:
            trace[slot] = (unit, move, legal, combined)
        if forced is not None and slot == forced[0]:
            action = int(forced[1])
            if not legal[action]:
                raise ValueError("Forced pivot became illegal")
        else:
            action = int(np.argmax(np.where(legal, combined, -np.inf)))
"""
    function = function[:start] + select + function[end:]
    params = []
    for key in ("W0", "B0", "W1"):
        values = weights[key].astype("<f4")
        packed = __import__("base64").b64encode(values.tobytes()).decode()
        params.append(f'_NATIVE_{key} = np.frombuffer(base64.b64decode("{packed}"), dtype="<f4").reshape({values.shape!r})')
    return text + "\n" + function + "\n" + "\n".join(params) + "\n" + (HERE / "runtime.py").read_text()


def initial():
    return dict(W0=np.zeros((FEATURES, WIDTH), np.float32), B0=np.zeros(WIDTH, np.float32),
                W1=np.zeros((WIDTH, 1), np.float32))


def load_source(path):
    spec = importlib.util.spec_from_file_location("native_pivot_public", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def collect(args):
    cache = args.work
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / "template.py"
    path.write_text(package(initial()))
    module = load_source(path)
    check(sha(STUDY/"sources.json") == RULE["sources"], "registered original sources changed")
    cards = read(STUDY/"sources.json")["cards"]
    map_sets = {key: {c["terrain"] for c in cards if c["split"] == key} for key in ("train","val","dev")}
    check([len(map_sets[k]) for k in ("train","val","dev")]==[24,4,4] and
          sum(map(len,map_sets.values())) == len(set.union(*map_sets.values())), "map split changed")

    for card in cards:
        if card["split"] not in ("train","val"):
            continue
        out = cache / (card["sha256"] + ".json.gz")
        if out.exists():
            found=read(out)
            check(found["template"]==sha(path) and found["source"]==card["sha256"], "wrong cached physical menus")
            continue
        rng = np.random.default_rng(int(card["sha256"][:16], 16)^RULE["seed"])
        times = {int(rng.choice(a)) for a in np.array_split(np.arange(6, min(295,card["transitions"]),6),12)}
        rows = []
        F = None
        f = None
        for t, obs, cfg, expert in entry.public_history(card):
            if not t:
                module.agent(obs, cfg)
                F = sys.modules["cnet2_feat"]
                f = F.Feat(card["expert_seat"])
            x = f.observe(obs)
            if t not in times:
                continue
            module._FEAT = f
            base, menu = module._native_menu(x,f.aux,t)
            if menu is None:
                rows.append(dict(t=t, possible=False))
                continue
            slot, actions, teams, features, logits = menu
            targets = f.labels(expert)
            prefix_match = np.array_equal(targets[:slot],base[:slot])
            target = int(targets[slot])
            check(np.array_equal(f.labels(f.to_actions(base)),base), "team roundtrip")
            check(all(np.array_equal(f.labels(f.to_actions(a)),a) for a in teams), "option roundtrip")
            rows.append(dict(t=t,possible=True,slot=int(slot),role=int(x["tok_kind"][slot]),
                        target=target,base=int(base[slot]),prefix_match=bool(prefix_match),
                        actions=actions.tolist(),outputs=teams[:,slot].tolist(),base_order=f.to_actions(base),
                        orders=[f.to_actions(a) for a in teams],legal_logits=logits[actions].tolist(),
                        features=features[actions].tolist(),expert=expert))
        check(len(rows)==12, "missing registered times")
        save(out,dict(template=sha(path),source=card["sha256"],rows=rows,terrain=card["terrain"],
                      split=card["split"],seed=card["seed"],seat=card["expert_seat"]))
        print("menus",card["split"],card["seed"],card["expert_seat"],flush=True)


class ActionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden=nn.Linear(FEATURES,WIDTH)
        self.out=nn.Linear(WIDTH,1,bias=False)
        nn.init.zeros_(self.out.weight)

    def forward(self,x):
        return self.out(torch.tanh(self.hidden(x))).squeeze(-1)


def batch(rows):
    n=len(rows)
    features=np.zeros((n,5,FEATURES),np.float32)
    logits=np.full((n,5),-1e8,np.float32)
    matches=np.zeros((n,5),np.float32)
    base=np.zeros(n,np.float32)
    for i,r in enumerate(rows):
        k=len(r["actions"])
        features[i,:k]=r["features"]
        logits[i,:k]=r["legal_logits"]
        matches[i,:k]=np.asarray(r["outputs"])==r["target"]
        base[i]=r["base"]==r["target"]
    return tuple(map(torch.from_numpy,(features,logits,matches,base)))


def terms(head,data):
    x,prior,match,base=data
    residual=head(x)
    probs=(prior+residual).softmax(-1)
    label_p=.5*base+.5*(probs*match).sum(-1)
    available=prior>-1e7
    centered=residual-(residual*available).sum(-1,keepdim=True)/available.sum(-1,keepdim=True)
    penalty=(centered.square()*available).sum(-1)/available.sum(-1)
    return -label_p.clamp_min(1e-12).log(),penalty


def usable(r):
    return r["possible"] and r["prefix_match"] and (r["target"] in r["outputs"] or r["target"]==r["base"])


def fit(args):
    torch.set_num_threads(1)
    torch.manual_seed(RULE["seed"])
    cards=[c for c in read(STUDY/"sources.json")["cards"] if c["split"] in ("train","val")]
    rows={"train":[],"val":[]}
    template=args.work/"template.py"
    check(template.read_text()==package(initial()), "cached features are from another runtime")
    for c in cards:
        d=read(args.work/(c["sha256"]+".json.gz"))
        check(d["source"]==c["sha256"] and d["template"]==sha(template), "stale sample")
        rows[c["split"]].extend(dict(r,terrain=c["terrain"],seed=c["seed"],seat=c["expert_seat"]) for r in d["rows"])
    eligible=[r for r in rows["train"] if usable(r)]
    check(len(eligible)>=96 and len({r["terrain"] for r in eligible})==24, "too few covered training maps")
    head=ActionHead()
    optimizer=torch.optim.AdamW(head.parameters(),lr=RULE["lr"],weight_decay=.001)
    rng=np.random.default_rng(RULE["seed"])
    terrain_count=Counter(r["terrain"] for r in eligible)
    weights=np.asarray([len(eligible)/(24*terrain_count[r["terrain"]]) for r in eligible],np.float32)
    data=batch(eligible)
    curve=[]
    for epoch in range(RULE["epochs"]):
        vals=[]
        perm=rng.permutation(len(eligible))
        for i in range(0,len(perm),RULE["batch"]):
            ids=perm[i:i+RULE["batch"]]
            loss,penalty=terms(head,[a[ids] for a in data])
            obj=((loss+RULE["residual_l2"]*penalty)*torch.from_numpy(weights[ids])).mean()
            check(torch.isfinite(obj).item(), "nonfinite head loss")
            optimizer.zero_grad()
            obj.backward()
            nn.utils.clip_grad_norm_(head.parameters(),2.,error_if_nonfinite=True)
            optimizer.step()
            vals.append(obj.detach().item())
        curve.append(float(np.mean(vals)))
    params=dict(W0=head.hidden.weight.detach().numpy().T,B0=head.hidden.bias.detach().numpy(),
                W1=head.out.weight.detach().numpy().T)
    np.savez(HERE/"head.npz",**params)
    candidate=HERE/"candidate.py"
    candidate.write_text(package(params) + "\n" + (HERE/"receipt.py").read_text())
    check(candidate.stat().st_size<=10_000_000,"package exceeds byte limit")
    zero=ActionHead()
    results={}
    held_rows=[]
    for terrain in sorted({r["terrain"] for r in rows["val"]}):
        rr=[r for r in rows["val"] if r["terrain"]==terrain]
        u=[r for r in rr if usable(r)]
        b=batch(u)
        with torch.no_grad():
            loss0=terms(zero,b)[0].numpy()
            loss1=terms(head,b)[0].numpy()
        broad=sum(len({tuple(sorted(cmd.items())) for cmd in [r["base_order"],*r["orders"]]})>=3 for r in rr if r["possible"])
        results[terrain]=dict(roots=len(rr),eligible=len(u),base_nll=float(loss0.mean()),trained_nll=float(loss1.mean()),
                              three_or_more_physical=broad)
        for r,a,b in zip(u,loss0,loss1):
            held_rows.append(dict(terrain=terrain,seed=r["seed"],seat=r["seat"],step=r["t"],role=r["role"],
                             slot=r["slot"],target=r["target"],nll_base=float(a),nll_trained=float(b),
                             parent=r["base_order"],orders=r["orders"],expert=r["expert"]))
    save(HERE/"held.json.gz",held_rows)
    save(HERE/"fit.json",dict(sources=sha(STUDY/"sources.json"),protocol=sha(HERE/"protocol.json"),
                  candidate=sha(candidate),head=sha(HERE/"head.npz"),parent=sha(PARENT),train_code=sha(Path(__file__)),
                  runtime=sha(HERE/"runtime.py"),receipt=sha(HERE/"receipt.py"),size=candidate.stat().st_size,train_roots=len(rows["train"]),
                  eligible=len(eligible),eligible_by_role=Counter(r["role"] for r in eligible),maps=24,
                  curve=curve,held_results=results,held_sha=sha(HERE/"held.json.gz"),steps=sum((len(eligible)+RULE["batch"]-1)//RULE["batch"] for _ in curve)))
    print(json.dumps(results),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=["collect","fit"])
    parser.add_argument("--work",type=Path,required=True)
    options=parser.parse_args()
    (collect if options.stage=="collect" else fit)(options)

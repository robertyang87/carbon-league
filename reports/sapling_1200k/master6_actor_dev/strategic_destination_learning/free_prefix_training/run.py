"""Learn the first legal disagreement of the current quantized Sapling on its own action prefix."""
import argparse
from collections import Counter
import copy
import importlib.util
from pathlib import Path
import statistics
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


# Reuse the tested tensor/publisher code without modifying any historical artifact.
common = load("prior_shadow_fit", STUDY / "learner_seat_corrections/model/fit.py")
native, c = common.native, common.c
entry = load("original_expert_history", STUDY / "free_entry.py")
RULE = c.read(HERE / "protocol.json")
PARENT = STUDY / "bc.py"
WEIGHTS = STUDY / "bc.npz"


def check(x, msg):
    if not x:
        raise ValueError(msg)


def roster(split):
    source = STUDY / "sources.json"
    check(c.sha(source) == RULE["source_sha"] and c.sha(PARENT) == RULE["parent"] and
          c.sha(WEIGHTS) == RULE["parent_weights"], "data or model identity changed")
    cards = c.read(source)["cards"]
    selected = [r for r in cards if r["seed"] in RULE[split + "_seeds"]]
    terrains = {s: {r["terrain"] for r in cards if r["seed"] in RULE[s + "_seeds"]} for s in ("train","held")}
    check(not terrains["train"] & terrains["held"] and all(len(terrains[s]) == len(RULE[s + "_seeds"]) for s in terrains),
          "terrain overlap or duplicate maps")
    check(all(r["split"] == ("train" if split == "train" else "val") and
              r["expert_seat"] == 1 - r["candidate_seat"] for r in selected), "role or split")
    check({(r["seed"], r["expert_seat"]) for r in selected} ==
          {(seed, seat) for seed in RULE[split + "_seeds"] for seat in (0,1)} and
          len(selected) == 2*len(terrains[split]), "incomplete seat pairs")
    return selected


def free(model, x, f):
    encoded = []
    for net in model._NETS:
        tokens, ctx = net.encode(x, 2)
        encoded.append((net, tokens, ctx, net.lin(ctx, "h0")))
    plain = model._return_decode(x, f.aux, encoded, False)
    alt = model._return_decode(x, f.aux, encoded, True)
    order, rejected = c.policy.guard(model, x, plain, alt)
    return order, rejected, encoded


def samples():
    F = sys.modules["cnet2_feat"]
    rows = []
    for card in roster("train"):
        f = F.Feat(card["expert_seat"])
        ticks = set(card["registered_times"][::2])
        check(len(ticks) == 16, "wrong registered train-root count")
        for t, obs, _, expert in entry.public_history(card):
            x = f.observe(obs)
            if t in ticks:
                state = copy.copy(f)
                state.own_uid = list(f.own_uid)
                state.aux = copy.deepcopy(f.aux)
                rows.append((x,state,expert,(card["seed"],card["expert_seat"],t)))
    check(len(rows) == 384, "incomplete training collection")
    return rows


def inspect_root(x, f, expert, sent):
    y = f.labels(expert)
    check(f.to_actions(y) == common.normalized(expert), "unrepresentable teacher command encoding")
    slots = np.flatnonzero(sent >= 0)
    check(np.array_equal(slots, np.flatnonzero(y >= 0)), "physical roster differs")
    pref = sys.modules["cnet2_feat"].Prefix(x,f.aux)
    for index,s in enumerate(slots):
        legal = pref.feats(s)[2]
        check(legal[sent[s]], "released action illegal on its own prefix")
        if sent[s] != y[s]:
            return int(s), int(y[s]), index, bool(legal[y[s]])
        pref.advance(s,int(sent[s]))
    return None, None, len(slots), False


def tensors(row, actor, parent, reader):
    x,f,expert,key = row
    sent, rejected, encoded = free(actor,x,f)
    slot, target, leading, supported = inspect_root(x,f,expert,sent)
    # All token/legality features (including covariates of the prefix) use the student's actual actions.
    data = c.actual_features(f,x,f.to_actions(sent),actor)
    full_cut = data.pop("cut")
    check(full_cut == len(sent), "student action unsupported under reconstruction")
    stop = slot + 1 if slot is not None else len(sent)
    fixed = np.zeros_like(data["residual"]); current = np.zeros_like(fixed)
    reader(x,f.aux,encoded,not rejected,sent,fixed,current,stop)
    previous = [(parent._NETS[0],*encoded[0][1:])]
    net = parent._NETS[1]
    tokens,ctx = net.encode(x,2)
    previous.append((net,tokens,ctx,net.lin(ctx,"h0")))
    prior = np.zeros_like(current)
    reader(x,f.aux,previous,not rejected,sent,np.zeros_like(fixed),prior,stop)
    mask = data["eligible"] & (np.arange(len(sent)) < (slot if slot is not None else stop))
    # Original actor's action dictionary is never used for later units, even when the teacher would be legal.
    data.update(fixed=fixed, anchor=mask, prior=prior, current=current)
    detail = dict(key=key,slot=slot,target=target,supported=supported,leading=leading,guard_rejected=bool(rejected),
                  sent=int(sent[slot]) if slot is not None else None)
    return data, detail


def fpweights(net, deq, original):
    return {k:original[k]+(v.detach().numpy()-deq[k]) for k,v in net.state_dict().items()}


def train():
    torch.manual_seed(RULE["seed"]);parent=c.package(PARENT);actor=c.package(PARENT)
    net=c.torch_model(parent,1)
    for k,p in net.named_parameters():
        p.requires_grad_(not k.startswith(native.VALUE))
    params=[p for p in net.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(params,lr=RULE["lr"],weight_decay=RULE["weight_decay"])
    reader=common.prepare.teacher_reader(actor)
    with np.load(WEIGHTS,allow_pickle=False) as z:
        original={k:z[k] for k in z}
        check(native.repack.quantize(WEIGHTS,z)==native.repack.literal(PARENT.read_text(),native.repack.STUDENT),
              "frozen publication failed to rebuild")
    dataset=samples()
    order=np.random.default_rng(RULE["seed"]).permutation(len(dataset)).reshape(-1,RULE["batch"])
    history=[]; roots=[]
    for step,batch in enumerate(order):
        model_weights=fpweights(net,parent._NETS[1].w,original)
        _,_,w=native.repack.cnet2_pack.quantize(model_weights)
        w["meta"]=original["meta"]
        actor._NETS[1]=sys.modules["cnet2_np"].Policy(w)
        collected=[tensors(dataset[i], actor, parent,reader) for i in batch]
        items=[a for a,_ in collected]; meta=[b for _,b in collected]; roots.extend(meta)
        data={k:np.stack([x[k] for x in items]) for k in items[0]}
        b=native.batch(data,np.arange(len(items)))
        logits=common.logits(net,b)
        prior=torch.from_numpy(data["prior"]).masked_fill(~b["legal"],-1e9).log_softmax(-1)
        kl=(prior.exp()*(prior-logits.log_softmax(-1))).sum(-1)
        anchor=torch.from_numpy(data["anchor"]);anchor_loss=(kl*anchor).sum()/anchor.sum().clamp_min(1)
        eligible=[i for i,r in enumerate(meta) if r["supported"]]
        if eligible:
            rows=torch.tensor(eligible);cols=torch.tensor([meta[i]["slot"] for i in eligible])
            gold=torch.tensor([meta[i]["target"] for i in eligible]);sent=torch.tensor([meta[i]["sent"] for i in eligible])
            gaps=logits[rows,cols,sent]-logits[rows,cols,gold]
            rank=torch.nn.functional.softplus(RULE["margin"]+gaps).sum()/len(meta)
        else:
            rank=logits[:,:,:1].clamp(-1,1).sum()*0
        if step==0:
            valid=anchor.clone()
            for i in eligible:valid[i,meta[i]["slot"]]=True
            mask=valid[...,None] & b["legal"];delta=logits.detach()-torch.from_numpy(data["current"])
            adj=delta-delta.masked_fill(~mask,0).sum(-1,keepdim=True)/mask.sum(-1,keepdim=True).clamp_min(1)
            parity=float(adj[mask].abs().max());check(parity<.003,"initial quantized-vs-gradient logits mismatch")
        loss=rank+RULE["kl"]*anchor_loss
        opt.zero_grad(set_to_none=True)
        check(torch.isfinite(loss),"nonfinite objective")
        if eligible or anchor.any():
            loss.backward(); grad=float(torch.nn.utils.clip_grad_norm_(params,RULE["clip"],error_if_nonfinite=True)); opt.step()
        else:grad=0.
        history.append(dict(step=step,supported=len(eligible),anchor_slots=int(anchor.sum()),loss=float(loss.detach()),grad=grad))
    common.HERE=HERE
    released=common.export(net,parent)
    c.save(HERE/"train_roots.json.gz",roots)
    c.save(HERE/"fit.json",dict(protocol=c.sha(HERE/"protocol.json"),script=c.sha(Path(__file__)),
                                rows=len(dataset),parity=parity,updates=history,roots=c.sha(HERE/"train_roots.json.gz"),**released))


def assess():
    data=c.read(HERE/"fit.json");path=HERE/"trained.py"
    check(c.sha(path)==data["package"] and data["protocol"]==c.sha(HERE/"protocol.json"),"new model identity")
    models={arm:load("entry_"+arm,path) for arm,path in (("parent",PARENT),("student",path))}
    rows=[]
    for card in roster("held"):
        ticks=set(card["registered_times"]);check(len(ticks)==24,"wrong held count")
        for t,obs,cfg,expert in entry.public_history(card):
            pubcfg={k:v for k,v in cfg.items() if k!="randomSeed"}
            orders={k:m.agent(obs,pubcfg) for k,m in models.items()}
            if t not in ticks:continue
            rec=dict(seed=card["seed"],terrain=card["terrain"],seat=card["expert_seat"],t=t,tape=card["sha256"],expert=expert)
            for k,m in models.items():
                # Keep the ordinary entry's exact historical observation and reconstruct only its selected root.
                # Feat.observe mutates history, so the step's backup features are obtained from a separate continuous reader below.
                rec[k]=orders[k]
            roles={uid:"base" for uid in obs["players"][card["expert_seat"]][1]}
            roles.update({u:v[2].lower() for u,v in obs["players"][card["expert_seat"]][2].items()})
            rec["roles"]=roles
            rows.append(rec)
        print("held",card["seed"],card["expert_seat"],len(rows),flush=True)
    # Independently rebuild the public prefix and count illegal targets/guard routing on the same roots.
    lookup={(r["seed"],r["seat"],r["t"]):r for r in rows}
    checked={k:c.package(p) for k,p in (("parent",PARENT),("student",path))}
    for card in roster("held"):
        F=sys.modules["cnet2_feat"];f=F.Feat(card["expert_seat"]);ticks=set(card["registered_times"])
        for t,obs,_,expert in entry.public_history(card):
            x=f.observe(obs)
            if t not in ticks:continue
            row=lookup[card["seed"],card["expert_seat"],t]
            row["slots"]=sum(uid is not None for uid in f.own_uid)
            for k,m in checked.items():
                sent,rejected,_=free(m,x,f); check(f.to_actions(sent)==row[k],"normal entry/reconstruction differs")
                slot,_,lead,legal=inspect_root(x,f,expert,sent)
                row[k+"_prefix"]=dict(slot=slot,leading=lead,supported=legal,guard_rejected=bool(rejected))
    def summarize(group):
        roles={r:Counter() for r in ("base","collector","planter")};stats=Counter()
        for row in group:
            stats["roots"]+=1;stats["changed"]+=row["parent"]!=row["student"]
            for k in models:
                p=row[k+"_prefix"];stats[k+"_exact"]+=p["slot"] is None
                stats[k+"_unsupported"]+=p["slot"] is not None and not p["supported"]
                stats[k+"_rejected"]+=p["guard_rejected"];stats[k+"_leading"]+=p["leading"]/row["slots"]
            for uid,role in row["roles"].items():
                target=row["expert"].get(uid,"NONE")
                a=row["parent"].get(uid,"NONE");b=row["student"].get(uid,"NONE")
                z=roles[role];z["total"]+=1;z["gained"]+=a!=target and b==target
                z["lost"]+=a==target and b!=target;z["wrong_to_other_wrong"]+=a!=b and a!=target and b!=target
                z["parent_correct"]+=a==target;z["student_correct"]+=b==target
        stats["leading_delta"]=(stats["student_leading"]-stats["parent_leading"])/len(group)
        return dict(stats=stats,roles=roles)
    by_map={str(v):summarize([r for r in rows if r["seed"]==v]) for v in RULE["held_seeds"]}
    by_seat={str(s):summarize([r for r in rows if r["seat"]==s]) for s in (0,1)};all_rows=summarize(rows)
    group=all_rows["stats"];workers=[all_rows["roles"][r] for r in ("collector","planter")]
    passed=(sum(v["stats"]["leading_delta"]>0 for v in by_map.values())>=3 and
            all(v["stats"]["leading_delta"]>0 and v["stats"]["changed"]>0 for v in by_seat.values()) and
            sum(v["gained"]-v["lost"] for v in workers)>0 and group["student_exact"]>group["parent_exact"])
    check(len(rows)==len(lookup)==192,"incomplete held assignment")
    c.save(HERE/"held_commands.json.gz",rows)
    c.save(HERE/"result.json",dict(package=data["package"],fit=c.sha(HERE/"fit.json"),held_evidence=c.sha(HERE/"held_commands.json.gz"),
          promotion=passed,pooled=all_rows,by_map=by_map,by_seat=by_seat,
          map_equal_leading_delta=statistics.mean(x["stats"]["leading_delta"] for x in by_map.values())))


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("operation",choices=("train","evaluate"));args=parser.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    (train if args.operation=="train" else assess)()

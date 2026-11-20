"""Fit a frozen Sapling policy against complete elite seat orders in archived Master3s2 games."""
import argparse
from collections import Counter
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
SOURCE = STUDY.parents[1] / "all_top5/master3s2_expert_orders"
PROTOCOL = "d373ccae0808919790786132824c893d762410747e95c45fea9fe4f333fd9aae"


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


# Load the recorded ladder convention before dynamically loading either self-contained policy.
inv = module("ladder_elite_archive", SOURCE / "inventory.py")
old = module("learner_correction_kernels", STUDY / "learner_seat_corrections/model/fit.py")
c, require = old.c, old.require
rule = c.read(HERE / "protocol.json")
ROLES = old.ROLES


def roster():
    require(c.sha(HERE / "protocol.json") == PROTOCOL, "Unregistered design")
    require(c.sha(SOURCE / "manifest.json") == rule["source_manifest_sha256"] and
            c.sha(SOURCE / "inventory.py") == rule["source_reader_sha256"], "Changed source release")
    require(c.sha(old.PARENT) == rule["starting_actor_package"] and
            c.sha(old.BASE) == rule["starting_weights_sha256"], "Changed starting actor")
    manifest = c.read(SOURCE / "manifest.json")["games"]
    require(len(manifest) == len({r["terrain"] for r in manifest}) == 30, "Manifest identity count")
    grouped = {}
    for split, status in (("fit","known_actor_train"), ("assessment","other_historical_train_only")):
        actual = {r["id"]:r for r in manifest if r["disposition"] == status}
        cards = rule["entries"][split]
        require(set(actual) == {r["id"] for r in cards}, "Registered source set differs")
        grouped[split] = []
        for entry in cards:
            card = actual[entry["id"]]
            require(all(entry[k] == card[k] for k in ("id","terrain","seat","seed","expert","replay_sha256")),
                    "Foreign replay/seat/role")
            grouped[split].append((card, set(entry["roots"])))
    require(sum(r["disposition"]=="protected" for r in manifest)==19, "Protected split differs")
    return grouped


def tape(card):
    source = inv.raw_bytes(card["archive"])
    require(inv.sha(source)==card["replay_sha256"], "Action archive mismatch")
    raw = json.loads(gzip.decompress(source))
    require(raw["id"]==card["uuid"] and raw["configuration"]["randomSeed"]==card["seed"] and
            inv.sha(bytes(raw["steps"][0][0]["observation"]["full_carbon"]))==card["terrain"] and
            raw["rewards"]==card["score_by_seat"] and raw["statuses"]==["DONE"]*2 and
            len(raw["steps"])==300, "Replay metadata/physical identity changed")
    return raw


class Actor:
    def __init__(self, path, seat):
        self.model = c.package(path)
        self.reader = old.prepare.teacher_reader(self.model)
        self.normal = module("elite_direct_public_actor", path)
        self.feat = sys.modules["cnet2_feat"].Feat(seat)

    def observe(self, obs, cfg):
        self.x = self.feat.observe(obs)
        self.free = old.normalized(self.normal.agent(obs, cfg))

    def root(self, label):
        m, f, x = self.model, self.feat, self.x
        enc = []
        for net in m._NETS:
            tok, ctx = net.encode(x, 2)
            enc.append((net, tok, ctx, net.lin(ctx, "h0")))
        plain = m._return_decode(x, f.aux, enc, False)
        residual = m._return_decode(x, f.aux, enc, True)
        sent, rejected = c.policy.guard(m, x, plain, residual)
        require(old.normalized(f.to_actions(sent))==self.free, "Full public policy differs from inspected route")
        a = c.actual_features(f, x, label["expert_order"], m)
        cut = int(a.pop("cut"))
        fixed = np.zeros_like(a["residual"])
        prior = np.zeros_like(fixed)
        self.reader(x, f.aux, enc, not rejected, a["y"], fixed, prior, cut)
        del a["residual"]
        complete = not np.any(a["tok_mask"][cut:len(a["y"])])
        require(complete and label["round_trip"] and label["all_slots_legal"], "Missing expert conditional action")
        prefix = []
        for slot in label["slots"]:
            s = slot["slot"]
            require(slot["prefix"]==prefix and f.own_uid[s]==slot["uid"] and
                    int(a["y"][s])==slot["canonical_action"] and bool(a["eligible"][s])==(sum(slot["legal"])>1) and
                    np.array_equal(a["legal"][s],slot["legal"]), "Conditional support or prior order differs")
            prefix.append([s,slot["canonical_action"]])
        require(int((a["y"]>=0).sum())==len(prefix), "Unsourced action slot")
        a.update(fixed=fixed, parent=prior, complete=np.bool_(complete))
        return a, bool(rejected)


def measured(a):
    score = np.where(a["legal"], a["parent"], -1e9)
    lp = score - score.max(-1,keepdims=True)
    lp -= np.log(np.exp(lp).sum(-1,keepdims=True))
    sel = a["eligible"]
    nll = -lp[np.arange(len(sel)), a["y"].clip(min=0)]
    kind = a["tok_kind"][:len(sel)]
    return {role: float(nll[sel & (kind==k)].sum(dtype=np.float64)) for k,role in enumerate(ROLES)}


def materialize(cases, paths, training):
    data, receipts = [], []
    for card, roots in cases:
        raw = tape(card)
        seat = card["seat"]
        actors = {name:Actor(path,seat) for name,path in paths.items()}
        pub, act = hashlib.sha256(), hashlib.sha256()
        count = 0
        for entry in inv.own_orders(card, raw):
            obs,t,label = entry["public"],entry["t"],entry["label"]
            pub.update(inv.canonical(obs)+b"\n")
            act.update(inv.canonical(label["expert_order"])+b"\n")
            for actor in actors.values():
                actor.observe(obs, raw["configuration"])
            if t not in roots:
                continue
            count += 1
            roles={s["uid"]:s["role"].lower() for s in label["slots"]}
            one = dict(id=card["id"],seed=card["seed"],terrain=card["terrain"],seat=seat,expert_name=card["expert"],
                       t=t,public_sha256=old.digest(obs),prior_own_orders_sha256=entry["prior_own_orders_sha256"],
                       target=label["expert_order"],roles=roles,policies={})
            for name,actor in actors.items():
                a,rejected = actor.root(label)
                require(not set(actor.free)-roles.keys(), "Foreign UID in released command")
                one["policies"][name] = dict(action=actor.free,nll=measured(a),guard_rejected=rejected)
                if training:
                    data.append(a)
            receipts.append(one)
        require(count==len(roots) and card["ledger"]["public_observations_sha256"]==pub.hexdigest() and
                card["ledger"]["observed_expert_orders_sha256"]==act.hexdigest(), "Source public/orders digest")
        print("verified",card["seed"],seat,count,tuple(paths),flush=True)
    return ({k:np.stack([r[k] for r in data]) for k in data[0]} if training else None),receipts


def train(cases):
    data, rows = materialize(cases, {"bc":old.PARENT}, True)
    params = rule["fit"]
    require(len(rows)==320 and data["complete"].all(),"Fit coverage changed")
    deployment = c.package(old.PARENT)
    torch.manual_seed(params["seed"])
    net = c.torch_model(deployment,1)
    for name,weight in net.named_parameters():
        weight.requires_grad_(not name.startswith(old.native.VALUE))
    variables=[v for v in net.parameters() if v.requires_grad]
    parity = old.parity(data,net)
    opt = torch.optim.AdamW(variables, lr=params["adamw_lr"], weight_decay=params["weight_decay"])
    order=np.random.default_rng(params["seed"]).permutation(len(rows)).reshape(-1,params["roots_per_update"])
    history=[]
    for ids in order:
        batch=old.native.batch(data,ids)
        opt.zero_grad(set_to_none=True)
        ce,kl=old.joint_loss(old.logits(net,batch),torch.from_numpy(data["parent"][ids]),batch,torch.from_numpy(data["complete"][ids]))
        loss=ce+params["kl_parent_to_student"]*kl
        require(bool(torch.isfinite(loss)), "Non-finite joint loss")
        loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(variables,params["max_grad_norm"],error_if_nonfinite=True)
        opt.step()
        history.append(dict(ce=float(ce.detach()),kl=float(kl.detach()),grad_before_clip=float(grad)))
    # The existing verified delta packer uses its module-local output directory.
    with patch.object(old,"HERE",HERE):
        identity=old.export(net,deployment)
    c.save(HERE/"fit.json",dict(protocol_sha256=PROTOCOL,source_manifest_sha256=rule["source_manifest_sha256"],
        trainer_sha256=c.sha(Path(__file__)),base=rule["starting_actor_package"],roots=len(rows),
        supervised_slots=int(data["eligible"].sum()),updates=len(history),parent_torch_error=parity,
        batches=history,**identity))
    print("frozen",identity["package"],flush=True)


def summarize(rows):
    roles={role:Counter() for role in (*ROLES,"all")}
    whole=Counter()
    for r in rows:
        p=r["policies"]; a=p["bc"]["action"]; b=p["trained"]["action"]; target=r["target"]
        whole["n"]+=1;whole["changed"]+=a!=b
        whole["old_exact"]+=a==target;whole["new_exact"]+=b==target
        for key in ("bc","trained"):
            whole[key+"_nll"]+=sum(p[key]["nll"].values())
            for role in ROLES:
                roles[role][key+"_nll"]+=p[key]["nll"][role]
        for uid,role in r["roles"].items():
            for dest in (roles[role],roles["all"]):
                y=target.get(uid,"NONE");x=a.get(uid,"NONE");z=b.get(uid,"NONE")
                dest["n"]+=1;dest["bc_correct"]+=x==y;dest["trained_correct"]+=z==y
                dest["changed"]+=x!=z;dest["unchanged"]+=x==z
                dest["gained"]+=x!=y and z==y;dest["lost"]+=x==y and z!=y
                dest["different_wrong"]+=x!=z and x!=y and z!=y
    for arm in ("bc","trained"):
        whole[arm+"_mean_joint_nll"]=whole[arm+"_nll"]/whole["n"]
        roles["all"][arm+"_nll"]=whole[arm+"_nll"]
        for stats in roles.values():
            stats[arm+"_entity_accuracy"]=stats[arm+"_correct"]/stats["n"] if stats["n"] else None
            stats[arm+"_mean_slot_nll"]=stats[arm+"_nll"]/stats["n"] if stats["n"] else None
    return {"whole":whole,"by_role":roles}


def assessment(cases):
    trained=HERE/"trained.py"
    frozen=c.read(HERE/"fit.json")
    require(c.sha(trained)==frozen["package"] and c.sha(HERE/"weights.npz")==frozen["weights"]
            and frozen["protocol_sha256"]==PROTOCOL, "Frozen export identity differs")
    arms={"bc":old.PARENT,"trained":trained}
    result={}
    for split in ("assessment","fit"):
        _,rows=materialize(cases[split],arms,False)
        c.save(HERE/(split+"_actions.json.gz"),rows)
        seeds=[card["seed"] for card,_ in cases[split]]
        summary={str(s):summarize([r for r in rows if r["seed"]==s]) for s in seeds}
        seats={str(s):summarize([r for r in rows if r["seat"]==s]) for s in (0,1)}
        equal={arm:{'joint_nll':float(np.mean([g["whole"][arm+"_mean_joint_nll"] for g in summary.values()])),
                    'entity_accuracy':float(np.mean([g["by_role"]["all"][arm+"_entity_accuracy"] for g in summary.values()]))}
               for arm in arms}
        result[split]=dict(map_equal=equal,maps=summary,physical_seats=seats,pooled=summarize(rows),
            maps_with_lower_nll=sum(g["whole"]["trained_mean_joint_nll"]<g["whole"]["bc_mean_joint_nll"] for g in summary.values()),
            sources=[{k:card[k] for k in ("id","seed","seat","expert","terrain","score_by_seat","replay_sha256")} for card,_ in cases[split]])
        print(split,equal,flush=True)
    h=result["assessment"];q=h["map_equal"]
    checks=dict(majority=h["maps_with_lower_nll"]>3,map_equal_nll=q["trained"]["joint_nll"]<q["bc"]["joint_nll"],
        map_equal_entity=q["trained"]["entity_accuracy"]>q["bc"]["entity_accuracy"],
        both_seats_changed=all(v["whole"]["changed"]>0 for v in h["physical_seats"].values()))
    c.save(HERE/"assessment.json",dict(fit_sha256=c.sha(HERE/"fit.json"),protocol_sha256=PROTOCOL,
        assessment_actions_sha256=c.sha(HERE/"assessment_actions.json.gz"),fit_actions_sha256=c.sha(HERE/"fit_actions.json.gz"),
        package_sha256=frozen["package"],checks=checks,eligible_for_development_consideration=all(checks.values()),
        original_matches=0,protected_tapes_opened=0,**result))
    print(checks,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=["precheck","train","assessment","all"])
    phase=parser.parse_args().phase
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    cases=roster()
    if phase in ("train","all"): train(cases["fit"])
    if phase in ("assessment","all"): assessment(cases)

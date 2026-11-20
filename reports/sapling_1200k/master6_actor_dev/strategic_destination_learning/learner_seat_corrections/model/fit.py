"""Learn whole expert commands on archived learner-seat trajectories; evaluate the ordinary release."""
import argparse
from collections import Counter
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
STUDY = HERE.parents[1]
INTAKE = HERE.parent / "intake"
NATIVE = STUDY.parent / "cargo_tree_action_learning_dev/native_actor_corpus"
sys.path.insert(0, str(NATIVE))
import fit as native
c = sys.modules["common"]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name,path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


prepare = module("public_expert_features",NATIVE/"prepare_data.py")
RULE = c.read(HERE/"protocol.json")
BASE = STUDY/"bc.npz"
PARENT = STUDY/"bc.py"
ROLES = ("collector","planter","base")
PUBLIC = {"step","player","carbon","players","trees","remainingOverageTime"}


def require(x, why):
    if not x:
        raise ValueError(why)


def normalized(cmd):
    return {uid:a for uid,a in cmd.items() if a and a!="NONE"}


def digest(x):
    import hashlib
    return hashlib.sha256(json.dumps(x,allow_nan=False,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()


def roster():
    require(c.sha(PARENT)==RULE["parent"] and c.sha(BASE)==RULE["parent_weights"],"Starting BC identity changed")
    manifest=c.read(INTAKE/"collection.json")
    return checked_roster(manifest)


def traces(split, info):
    result=[]
    for seed in RULE[split+"_seeds"]:
        for seat in (0,1):
            card=info[seed,seat]
            path=INTAKE/"traces"/f"{seed}_{seat}.json.gz"
            require(c.sha(path)==card["sha256"],"Recorded trace changed")
            tr=c.read(path)
            require((seed,seat)==(tr["seed"],tr["seat"]) and tr["teacher_sha256"]=="65f268b2919ce30fe19355115a928eada032a6beaa018d801662d434c083f2b5","Wrong teacher/physical seat")
            require([r["t"] for r in tr["rows"]]==list(range(32)), "Missing observed decision")
            require(len({r["public_sha256"] for r in tr["rows"]})==32,"Duplicate public input")
            for r in tr["rows"]:
                obs=r["observation"]
                require(digest(obs)==r["public_sha256"] and r["teacher_state"]["past_executed_destinations_observed"],
                        "Public digest or prior actual intent cache differed")
                require(set(obs)==PUBLIC and (obs["step"],obs["player"])==(r["t"],seat),"Feature visibility or seat")
            replay_input(tr,card)
            result.append(tr)
    require(len({tr["terrain"] for tr in result})==len(RULE[split+"_seeds"]), "Wrong terrain cardinality")
    return result


def materialize(split, info, path):
    model=c.package(path); reader=prepare.teacher_reader(model)
    free=module("learner_correction_released_actor",path)
    features=sys.modules["cnet2_feat"]
    data=[]; meta=[]
    for tr in traces(split,info):
        seat=tr["seat"]; f=features.Feat(seat)
        for row in tr["rows"]:
            obs=row["observation"]; y_cmd=normalized(row["teacher_order"])
            x=f.observe(obs)
            enc=[]
            for net in model._NETS:
                tokens,ctx=net.encode(x,2)
                enc.append((net,tokens,ctx,net.lin(ctx,"h0")))
            a=model._return_decode(x,f.aux,enc,False)
            b=model._return_decode(x,f.aux,enc,True)
            chosen,rejected=c.policy.guard(model,x,a,b)
            free_order=normalized(free.agent(obs,tr["game_configuration"]))
            require(free_order==normalized(f.to_actions(chosen)), "Instrumented action differs from actual release")
            arrays=c.actual_features(f,x,y_cmd,model)
            cut=int(arrays.pop("cut"));fixed=np.zeros_like(arrays["residual"]);prior=np.zeros_like(fixed)
            reader(x,f.aux,enc,not rejected,arrays["y"],fixed,prior,cut)
            del arrays["residual"]
            complete=not np.any(arrays["tok_mask"][cut:features.OWN])
            arrays.update(fixed=fixed,parent=prior,complete=np.bool_(complete))
            data.append(arrays)
            roles={uid:"base" for uid in obs["players"][seat][1]}
            roles.update({uid:w[2].lower() for uid,w in obs["players"][seat][2].items()})
            for cmd in (free_order,y_cmd):
                require(not set(cmd)-roles.keys(), "Action UID is not owned by acting player")
            meta.append(dict(seed=tr["seed"],seat=seat,terrain=tr["terrain"],source_sha256=tr["source_sha256"],
                behavior_package_sha256=tr["behavior_package_sha256"],t=row["t"],public_sha256=row["public_sha256"],
                free=free_order,actual_history_order=normalized(row["learner_order"]),expert=y_cmd,roles=roles,
                complete=complete,cut=cut,guard_rejected=bool(rejected)))
    return {k:np.stack([r[k] for r in data]) for k in data[0]},meta


def logits(model, b):
    return (b["residual"]+.25*model(b)[0]).masked_fill(~b["legal"],-1e9)


def joint_loss(scores, old, b, complete):
    lp=scores.log_softmax(-1); lp0=old.masked_fill(~b["legal"],-1e9).log_softmax(-1)
    action=-lp.gather(-1,b["y"].clamp_min(0).unsqueeze(-1)).squeeze(-1)
    kl=(lp0.exp()*(lp0-lp)).sum(-1)
    # Each team action is one joint event; do not reweight by the unit count on the board.
    weight=b["eligible"].float()*complete[:,None]
    n=complete.sum().clamp_min(1)
    return (action*weight).sum()/n,(kl*weight).sum()/n


def parity(data, model):
    high=0.
    with torch.no_grad():
        for begin in range(0,len(data["y"]),8):
            ids=np.arange(begin,min(len(data["y"]),begin+8));b=native.batch(data,ids)
            measured=logits(model,b);valid=b["legal"] & b["eligible"][:,:,None]
            gap=measured-torch.from_numpy(data["parent"][ids])
            gap-=gap.masked_fill(~valid,0).sum(-1,keepdim=True)/valid.sum(-1,keepdim=True).clamp_min(1)
            if valid.any():high=max(high,float(gap[valid].abs().max()))
    require(high<.003,"Numpy vs Torch logits differ: "+str(high))
    return high


def export(net, deployed):
    parent=deployed._NETS[1].w; state=net.state_dict()
    with np.load(BASE,allow_pickle=False) as ref:
        require(native.repack.quantize(BASE,ref) == native.repack.literal(PARENT.read_text(),native.repack.STUDENT),
                "Weights do not reproduce frozen parent")
        tensors={k:ref[k]+(v.detach().numpy()-parent[k]) for k,v in state.items()}
        w=HERE/"weights.npz";np.savez(w,**tensors,meta=ref["meta"])
        fields=native.repack.quantize(w,ref)
    source=PARENT.read_text();path=HERE/"trained.py";c.atomic(path,native.repack.substitute(source,fields).encode())
    require(native.repack.substitute(path.read_text(),native.repack.literal(source,native.repack.STUDENT))==source,
            "Package code or as-served partner changed")
    require(path.stat().st_size<=10000000,"Package exceeds ladder size")
    fresh=c.package(path);changed={k:int(np.count_nonzero(parent[k]!=fresh._NETS[1].w[k])) for k in state}
    require(all(changed[k] for k in ("stem.weight","gru.weight_ih","whead.weight")) and
            not any(n for k,n in changed.items() if k.startswith(native.VALUE)), "Quantized trainable/frozen tensors differ")
    return dict(package=c.sha(path),weights=c.sha(w),size=path.stat().st_size,changed_by_tensor=changed)


def train(info):
    data,meta=materialize("fit",info,PARENT)
    require(len(meta)==256 and all(any(r["complete"] and r["seed"]==s for r in meta) for s in RULE["fit_seeds"]),"Unusable fit corpus")
    model=c.package(PARENT);torch.manual_seed(RULE["seed"]);net=c.torch_model(model,1)
    for k,p in net.named_parameters():p.requires_grad_(not k.startswith(native.VALUE))
    params=[p for p in net.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(params,lr=RULE["learning_rate"],weight_decay=RULE["weight_decay"])
    drift=parity(data,net);history=[]
    order=np.random.default_rng(RULE["seed"]).permutation(len(meta)).reshape(-1,RULE["batch"])
    for ids in order:
        b=native.batch(data,ids);mask=torch.from_numpy(data["complete"][ids])
        opt.zero_grad(set_to_none=True)
        ce,kl=joint_loss(logits(net,b),torch.from_numpy(data["parent"][ids]),b,mask)
        loss=ce+RULE["kl"]*kl
        require(bool(torch.isfinite(loss)),"Nonfinite training loss")
        if (mask[:,None] & b["eligible"]).any():
            loss.backward()
            grad=float(torch.nn.utils.clip_grad_norm_(params,RULE["clip"],error_if_nonfinite=True))
            opt.step()
        else:grad=0.
        history.append(dict(n=len(ids),complete=int(mask.sum()),ce=float(ce.detach()),kl=float(kl.detach()),gradient_norm=grad))
    released=export(net,model)
    c.save(HERE/"fit.json",dict(protocol_sha256=c.sha(HERE/"protocol.json"),collection_sha256=c.sha(INTAKE/"collection.json"),
        trainer_sha256=c.sha(Path(__file__)),parent=RULE["parent"],fit_maps=RULE["fit_seeds"],roots=len(meta),
        all_supported=int(data["complete"].sum()),source_packages=sorted({r["behavior_package_sha256"] for r in meta}),
        updates=sum(s["gradient_norm"]>0 for s in history),batches=history,parity=drift,**released))


def assessment(info):
    frozen=c.read(HERE/"fit.json");new=HERE/"trained.py"
    require(c.sha(new)==frozen["package"] and c.sha(HERE/"weights.npz")==frozen["weights"] and
            c.sha(INTAKE/"collection.json")==frozen["collection_sha256"],"Evaluation came from a different training run")
    ds={};cards={}
    for name,path in (("bc",PARENT),("trained",new)):
        ds[name],cards[name]=materialize("held",info,path)
    rows=[]
    for i,(old,child) in enumerate(zip(cards["bc"],cards["trained"])):
        for key in ("seed","seat","t","public_sha256","source_sha256","expert","roles","complete","cut"):
            require(old[key]==child[key],"Historical test roots/target support mismatch")
        detail=dict(old);detail["trained"]=child["free"];detail["bc"]=detail.pop("free")
        detail["guard_route_changed"]=child["guard_rejected"]!=old["guard_rejected"]; detail["prefix_nll"]={}
        for arm,data in ds.items():
            scores=data["parent"][i].copy();scores[~data["legal"][i]]=-1e9
            lp=scores-scores.max(-1,keepdims=True);lp-=np.log(np.exp(lp).sum(-1,keepdims=True))
            labels=data["y"][i].clip(min=0);mask=data["eligible"][i];values=-lp[np.arange(len(labels)),labels]
            kinds=data["tok_kind"][i,:len(labels)]
            detail["prefix_nll"][arm]={role:[float(values[mask & (kinds==k)].sum()),int((mask & (kinds==k)).sum())] for k,role in enumerate(ROLES)}
        rows.append(detail)
    def summarize(group):
        roles={k:Counter() for k in (*ROLES,"all")};whole=Counter();nll={arm:Counter() for arm in ds}
        complete=sum(r["complete"] for r in group)
        for r in group:
            changed=(r["trained"]!=r["bc"]);whole["changed"]+=changed
            for arm in ds:whole[arm+"_hit"]+=r[arm]==r["expert"]
            for uid,role in r["roles"].items():
                y=r["expert"].get(uid,"NONE");a=r["bc"].get(uid,"NONE");b=r["trained"].get(uid,"NONE")
                for name in (role,"all"):
                    q=roles[name];q["total"]+=1;q["bc_correct"]+=a==y;q["trained_correct"]+=b==y
                    q["changed"]+=a!=b;q["gained"]+=b==y and a!=y;q["lost"]+=a==y and b!=y
                    q["wrong_to_different_wrong"]+=a!=b and a!=y and b!=y
            for arm in ds:
                for role,(value,n) in r["prefix_nll"][arm].items():
                    nll[arm][role+"_supported_slots"]+=n;nll[arm][role+"_prefix_nll"]+=value
                    if r["complete"]:nll[arm]["joint_nll_sum"]+=value
        for arm in ds:
            nll[arm]["joint_nll_per_complete_root"]=nll[arm]["joint_nll_sum"]/complete if complete else None
            for role in ROLES:
                n=nll[arm][role+"_supported_slots"]
                nll[arm][role+"_prefix_nll_per_slot"]=nll[arm][role+"_prefix_nll"]/n if n else None
        return dict(n=len(group),complete=complete,whole=whole,entity=roles,nll=nll)
    pooled=summarize(rows);by_map={str(s):summarize([r for r in rows if r["seed"]==s]) for s in RULE["held_seeds"]}
    by_seat={str(s):summarize([r for r in rows if r["seat"]==s]) for s in (0,1)}
    per_map_seat={f"{seed}_{s}":summarize([r for r in rows if r["seed"]==seed and r["seat"]==s]) for seed in RULE["held_seeds"] for s in (0,1)}
    eligible=all(v["complete"] and v["whole"]["changed"]>0 and v["nll"]["trained"]["joint_nll_per_complete_root"]<v["nll"]["bc"]["joint_nll_per_complete_root"] for v in by_map.values()) and pooled["entity"]["all"]["gained"]>pooled["entity"]["all"]["lost"]
    c.save(HERE/"held_actions.json.gz",rows)
    c.save(HERE/"assessment.json",dict(status="candidate_for_original_development" if eligible else "offline_only",
        package_sha256=frozen["package"],fit_sha256=c.sha(HERE/"fit.json"),held_actions_sha256=c.sha(HERE/"held_actions.json.gz"),
        pooled=pooled,by_map=by_map,by_physical_seat=by_seat,by_map_seat=per_map_seat,
        actual_strength_games=0,source_packages=sorted({r["behavior_package_sha256"] for r in rows})))


def checked_roster(manifest):
    plan=c.read(INTAKE/"protocol.json")
    expected={(seed,seat):split for split in ("fit","held") for seed in RULE[split+"_seeds"] for seat in (0,1)}
    originals={(r["seed"],r["seat"]):r for r in plan["tickets"]}
    require(len(expected)==len(originals)==len(plan["tickets"])==12 and set(originals)==set(expected),"Intake scope differs")
    require(plan["expected_roots"]==384 and plan["root_steps"]==list(range(32)),"Frozen root set differs")
    require(all(r["split"]==expected[k] for k,r in originals.items()),"Wrong registered map split")
    require(manifest.get("training_eligible") is True and not manifest.get("issues"),"Shadow intake not qualified for training")
    rows=manifest["tickets"]
    info={(r["seed"],r["seat"]): r for r in rows}
    require(set(info)==set(expected) and len(rows)==12,"Unregistered source or incomplete intake")
    for key,card in info.items():
        reg=originals[key]
        require(card["eligible"] and card["n_rows"]==32 and card["split"]==reg["split"]
                and card["terrain"]==reg["terrain"],"Incomplete or wrong training ticket")
        require((c.ROOT/card["trace_path"]).resolve()==(INTAKE/"traces"/f"{key[0]}_{key[1]}.json.gz").resolve(),"Trace path mismatch")
        # Source and its role derive from the pre-label registration even if they repeat between phases.
        card["source_registration"]=reg
    return info


def replay_input(tr, card):
    reg=card["source_registration"];path=c.ROOT/reg["source_path"]
    require(tr["source_sha256"]==reg["source_sha256"]==c.sha(path) and tr["terrain"]==reg["terrain"] and
            tr["protocol_sha256"]==c.sha(INTAKE/"protocol.json") and tr["split"]==reg["split"] and
            tr["behavior_package_sha256"]==reg["learner_pack_sha256"],"Changed raw trajectory or behavior policy")
    raw=c.read(path);commands=raw["actions"]
    require(raw["candidate_seat"]==tr["seat"] and digest(commands)==reg["body_actions_sha256"],"Changed historical side or full command log")
    sim=c.FastCarbon(raw["game_configuration"]).reset(seed=tr["seed"])
    import hashlib
    require(hashlib.sha256(bytes(map(int,sim.carbon))).hexdigest()==tr["terrain"],"Different physical opening")
    for r in tr["rows"]:
        t=r["t"];obs=sim.obs(tr["seat"])
        require(all(obs[k]==r["observation"][k] for k in ("player","step","players","trees","carbon")),
                "Shadow observation not reconstructed from the past on the learner's physical seat")
        require(normalized(commands[t][tr["seat"]])==normalized(r["learner_order"]) and
                normalized(commands[t][1-tr["seat"]])==normalized(r["opponent_order"]), "Shadow teacher was played or source actions drifted")
        sim.step(*commands[t])


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("phase",choices=("train","held","all","precheck"))
    args=ap.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    if args.phase=="precheck":
        p=c.read(INTAKE/"protocol.json")
        print("protocol maps",[(split, sorted({x["seed"] for x in p["tickets"] if x["split"]==split})) for split in ("fit","held")])
        require(all(sorted({x["seed"] for x in p["tickets"] if x["split"]==s})==sorted(RULE[s+"_seeds"]) for s in ("fit","held")) and p["expected_roots"]==384,"Different map contract")
        return
    info=roster()
    if args.phase in ("train","all"):train(info)
    if args.phase in ("held","all"):assessment(info)
if __name__=="__main__":main()

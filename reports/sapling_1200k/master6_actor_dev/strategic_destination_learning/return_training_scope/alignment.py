"""Full-network within-training-map sign test at the actual deployed 851c operating point."""
from collections import defaultdict
import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import zipfile

import numpy as np
import torch

from audit import INTAKE, HERE, learning, o

PROTOCOL = "6e583eb321d13267af7da68dce648f33f8be8bd3bbb401b64449fd0d226323df"


def recover(ep, base, teacher, schedule):
    """Replay every public transition, rebuild the actual distribution at every scheduled root."""
    seed,seat=ep["seed"],ep["seat"]
    path=o.ROOT/ep["archive_path"]
    o.require(o.sha(path)==ep["archive_sha256"],"training zip changed")
    by_step={d["step"]:d for d in ep["events"]}
    o.require(set(by_step)==schedule and len(by_step)==12, "unexpected schedule")
    check=[]; orders=hashlib.sha256(); roots=[]
    with tempfile.TemporaryDirectory() as td, zipfile.ZipFile(path) as zipped:
        reg=Path(td)/"registration.json"; reg.write_bytes(zipped.read(ep["registration_member"]))
        receipt=json.loads(zipped.read(ep["receipt_member"])); tape=zipped.read(ep["turns_member"])
        o.require(o.sha(reg)==ep["registration_sha256"] and
                  learning.sha_bytes(tape)==ep["tape_sha256"]==receipt["turns_sha256"],"missing binding")
        live=learning.menu.open_live(reg);env,cfg=live.official()
        o.require(learning.sha_bytes(bytes(env.states[0].observation.full_carbon))==ep["terrain"],"terrain mismatch")
        import sys
        feature=sys.modules["cnet2_feat"].Feat(seat)
        bank=[5.,5.]
        for t,line in enumerate(tape.splitlines()):
            f=json.loads(line);pub=live.observe(env,seat,bank[seat])
            public_hash=learning.sha_bytes(live.encoded(pub))
            o.require(not env.done and t==f["step"]==pub["step"] and pub["player"]==seat and
                      "full_carbon" not in pub and public_hash==f["public_observation_sha256"][seat]
                      and all(k["status"]=="ok" for k in f["calls"]), "corrupt public history")
            commands=[k["response"]["action"] for k in f["calls"]]
            orders.update(learning.compact(commands)+b"\n")
            x=feature.observe(pub)
            if t in schedule:
                e=by_step[t]
                o.require(pub==e["public_observation"] and public_hash==e["public_observation_sha256"]
                          and commands[seat]==e["issued"],"misaligned issued order")
                public=(x,feature.aux,list(feature.own_uid))
                site=dict(public=public,seat=seat,step=t)
                opts,p=learning.true_free(base,site)
                o.require(opts==e["options"] and np.max(abs(np.subtract(p,e["probabilities"])))<1e-12
                          and bool(len(opts)==2)==e["supported"],"fresh physical distribution mismatch")
                if e["supported"]:
                    row,meta=learning.build_features(base,teacher,feature,x,e,ep["advantage"],seed,seat)
                    meta.update(arm=ep["arm"], public_observation_sha256=public_hash)
                    roots.append((row,meta,copy.deepcopy(public)))
                check.append(t)
            bank=f["overage_remaining_seconds"];env.step(commands)
        o.require(env.done and [s.reward for s in env.states]==ep["official_scores_by_seat"] and
                  set(check)==schedule and len(check)==len(schedule), "sample incomplete or incorrect payoffs")
    return roots,orders.hexdigest()


def gradient(roots,models,base):
    data,rows=learning.a.network_data(roots)
    for m in models:m.zero_grad(set_to_none=True)
    gap_error=0.;prob_error=0.;lp_error=0.
    for ix in o.groups(len(roots)):
        b=o.tensorbatch(data,ix)
        ids=torch.arange(len(ix))
        slots=torch.as_tensor(data["slot"][ix]);acts=torch.as_tensor(data["moves"][ix])
        projected=[m(b)[0][ids,slots].gather(1,acts).double() for m in models]
        residual=[]
        for i in ix:
            row,meta,(x,aux,_)=roots[i];s=meta["slot"]
            add=np.zeros(5)
            if meta["route"] and x["tok_kind"][s]==0:
                add=base._return_basis(x,aux,s,row["mf"][s])@base._RETURN_THETA
            residual.append(add[meta["moves"]])
        z=(base._HYBRID["expert_weight"]*projected[0]+base._HYBRID["student_weight"]*projected[1]+
           torch.tensor(np.array(residual),dtype=torch.float64))
        gap=z[:,0]-z[:,1];native=torch.tensor([rows[i]["observed_gap"] for i in ix],dtype=torch.float64)
        p=torch.tensor([rows[i]["observed_p0"] for i in ix],dtype=torch.float64)
        sign=torch.tensor([1. if rows[i]["pick"]==0 else -1. for i in ix],dtype=torch.float64)
        a=torch.tensor([rows[i]["advantage"] for i in ix],dtype=torch.float64)
        recorded=torch.tensor([rows[i]["old_logp"] for i in ix],dtype=torch.float64)
        gap_error=max(gap_error,float((native-gap.detach()).abs().max()))
        prob_error=max(prob_error,float((p-gap.detach().sigmoid()).abs().max()))
        lp_error=max(lp_error,float((recorded-torch.nn.functional.logsigmoid(sign*gap.detach())).abs().max()))
        anchored=native+(gap-gap.detach())
        loss=(a*torch.nn.functional.logsigmoid(sign*anchored)).sum()/48
        o.require(torch.isfinite(loss).item(), "nonfinite score")
        loss.backward()
    o.require(prob_error<5e-5 and lp_error<1e-4,"dual tower or conditional prefix changed")
    flat=[];ident=[]
    for m in models:
        arrays=[];counts={}
        for name,p in m.named_parameters():
            if not p.requires_grad:continue
            grad=p.grad.detach().reshape(-1).numpy() if p.grad is not None else np.zeros(p.numel(),np.float32)
            o.require(np.isfinite(grad).all(),"nonfinite gradient")
            arrays.append(grad);counts[name]=p.numel()
        flat.append(np.concatenate(arrays));ident.append(counts)
    return flat,ident, dict(supported=len(rows),max_gap_error=gap_error,max_probability_error=prob_error,max_selected_logp_error=lp_error)


def map_comparison(vectors,seeds):
    matrix=np.stack(vectors).astype(np.float64)
    gram=matrix@matrix.T
    norm=np.sqrt(np.diag(gram));o.require(np.all(norm>0),"empty map direction")
    cos=gram/norm[:,None]/norm[None,:]
    off=np.triu_indices(len(seeds),1)
    folds=[]
    for i,seed in enumerate(seeds):
        ix=[j for j in range(len(seeds)) if j!=i]
        dot=float(gram[i,ix].mean()); other=float(gram[np.ix_(ix,ix)].mean())**.5
        folds.append(dict(seed=seed, norm=float(norm[i]), held_dot_five=dot, other_five_norm=other,
            cosine=dot/(other*norm[i]), dot_unit_normalized_five=float(cos[i,ix].mean())))
    pairs=[dict(seed_left=seeds[i],seed_right=seeds[j],dot=float(gram[i,j]),cosine=float(cos[i,j]))
           for i,j in zip(*off)]
    npositive=sum(r["held_dot_five"]>0 for r in folds);s=float(gram[off].sum())
    return dict(physical_seeds=seeds,parameters=int(matrix.shape[1]),folds=folds,raw_gram=gram.tolist(),
        cosine_matrix=cos.tolist(),pairs=pairs,positive_folds=npositive,positive_of_15=int((gram[off]>0).sum()),
        distinct_pair_dot_sum=s,pass_screen=npositive>=4 and s>0)


def main():
    o.require(o.sha(HERE/"alignment_protocol.json")==PROTOCOL,"registration changed")
    p=INTAKE/"hosted/training_manifest.json";plan=o.read(INTAKE/"learning.json")
    manifest,eps=learning.training_manifest(plan,p)
    base=o.package(learning.BEHAVIOR);teacher=o.teacher_reader(base)
    nets=[o.torch_model(base,i) for i in (0,1)]
    for net in nets:
        for k,v in net.named_parameters():v.requires_grad_(not k.startswith(o.VALUE))
    seeds=sorted(set(e["seed"] for e in eps));vecs=[[],[]];maps=[];cardinals=None;all_hashes=set()
    for seed in seeds:
        roots=[];hashes=set();rs=[]
        for ep in (e for e in eps if e["seed"]==seed):
            r,h=recover(ep,base,teacher,set(plan["scheduled_turns"]))
            o.require((seed,h) not in all_hashes and h not in hashes, "duplicate actual joint trajectory")
            hashes.add(h);all_hashes.add((seed,h));roots.extend(r)
            rs.append(dict(seat=ep["seat"],arm=ep["arm"],u=ep["u"],advantage=ep["advantage"],official_scores=ep["official_scores_by_seat"],
                           trajectory_orders_sha256=h,archive_sha256=ep["archive_sha256"],supported=len(r)))
        grad,ident,meta=gradient(roots,nets,base)
        o.require(cardinals is None or ident==cardinals,"gradient support dimension changed")
        cardinals=ident
        for i,g in enumerate(grad):vecs[i].append(g)
        maps.append(dict(seed=seed,terrain=next(e["terrain"] for e in eps if e["seed"]==seed),episodes=rs,**meta))
        print(seed,len(roots),round(meta["max_probability_error"],8),flush=True)
    out={k:map_comparison(v,seeds) for k,v in zip(("it50","sapling"),vecs)}
    reference=o.read(INTAKE/"fit.json")["leave_one_physical_map_out"]["folds"]
    par={r["held_terrain_seed"]:r["cosine"] for r in reference}
    diff=max(abs(r["cosine"]-par[r["seed"]]) for r in out["sapling"]["folds"])
    o.require(diff<.00005 and sum(r["supported"] for r in maps)==275,"historical Sapling gradients or cohort mismatch")
    decision=all(v["pass_screen"] for v in out.values())
    o.save(HERE/"alignment.json",dict(protocol_sha256=PROTOCOL,program_sha256=o.sha(Path(__file__)),data_sha256=o.sha(p),
        parent_sha256=o.sha(learning.BEHAVIOR),prior_sapling_fit_sha256=o.sha(INTAKE/"fit.json"),prior_sapling_max_cosine_error=diff,
        maps=maps,towers=out,parameter_counts_by_tensor=dict(it50=cardinals[0],sapling=cardinals[1]),eligible_for_one_step=decision,
        optimizer_steps=0,new_matches=0))
    print([(k,v["positive_folds"],v["positive_of_15"],v["distinct_pair_dot_sum"]) for k,v in out.items()],flush=True)


if __name__=="__main__":
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(621758460)
    main()

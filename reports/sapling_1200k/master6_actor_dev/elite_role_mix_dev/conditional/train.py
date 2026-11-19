"""Fit an action correction on mixed-policy disagreement slots using only running-policy observables."""
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("role_fit",HERE.parent/"fit.py")
role_fit=importlib.util.module_from_spec(spec);spec.loader.exec_module(role_fit)
ref=role_fit.ref
SRC=(HERE/"runtime.py").read_text()
feature_env={"np":np}
exec(compile(SRC,str(HERE/"runtime.py"),"exec"),feature_env)
feature=feature_env["_role_residual_features"]
SEED=29037813
NP=Path("/tmp/sapling-conditional-role-v1-shards")


@torch.inference_mode()
def collect(data,nets,ix):
    accum=[]
    for start in range(0,len(ix),24):
        rows=ix[start:start+24]
        b=ref.common.batch(data,rows); expert,student=[m(b)[0].numpy() for m in nets]
        valid=ref.common.actionable(b).numpy(); br,bs=np.where(valid)
        y=b["y"].numpy()[br,bs]
        kinds=b["tok_kind"].numpy()[br,bs]; land=data["gid"][rows][br]; steps=data["step"][rows][br]
        e=expert[br,bs]; s=student[br,bs]; legal=b["legal"].numpy()[br,bs]
        un=b["uf"].numpy()[br,bs]; mv=b["mf"].numpy()[br,bs]
        dis=(e.argmax(1)!=s.argmax(1))&(steps>0)
        feat=np.zeros((len(br),5,7+mv.shape[-1]*3+7),np.float32)
        for j in np.flatnonzero(dis):
            feat[j]=feature(kinds[j],steps[j],un[j],mv[j],legal[j],e[j],s[j])
        accum.append((land,kinds,y,0.75*e+0.25*s,feat,dis))
    return tuple(np.concatenate(vals) for vals in zip(*accum))


def scores(arr,theta):
    gid,role,y,z,x,dis=arr
    delta=2*np.tanh(np.einsum("jak,jk->ja",x,theta[role])/2)
    old=z.argmax(-1); new=(z+delta).argmax(-1)
    summary={}
    for key,idx in {"all":np.ones(len(y),bool),"disagreement":dis,
                    **{name:role==r for r,name in enumerate(role_fit.ROLE)}}.items():
        map_id=np.unique(gid[idx]);dif=[];before=[];after=[]
        for g in map_id:
            m=idx&(gid==g); a=(old[m]==y[m]).mean();b=(new[m]==y[m]).mean()
            dif.append(b-a);before.append(a);after.append(b)
        boot=np.random.default_rng(SEED).choice(dif,(5000,len(dif)),replace=True).mean(1)
        summary[key]=dict(maps=len(map_id),slots=int(idx.sum()),changed=int((new[idx]!=old[idx]).sum()),
                          parent=float(np.mean(before)),candidate=float(np.mean(after)),
                          paired_map_delta=float(np.mean(dif)),map_bootstrap_95=np.quantile(boot,[.025,.975]).tolist())
    return summary


def run():
    torch.manual_seed(SEED);torch.set_num_threads(1)
    plan=json.loads((role_fit.TRAIN/"panel.json").read_bytes())["games"]
    prior=json.loads((HERE.parent/"protocol.json").read_bytes())["external_games"]
    external=role_fit.new_maps(plan+prior)
    tr=[g for g in plan if g["split"]=="train"]
    role_fit.save(HERE/"protocol.json",dict(parent=ref.sha(role_fit.PARENT),external_games=external,
        train_maps=len(tr),split="all original actor maps and first role-mix external 24 maps excluded from this external panel",
        schedule="12 distinct roots per training map, equal maps, seed 29037813; learn on disagreement slots only",
        loss="map-equal cross entropy of actual expert, logit temperature 1.5 for fitting, l2 .01, 120 full-batch Adam steps .035, max adjustment 2 tanh; no test selection or per-map rule",
        gate="external teacher and actual policy map-mean action agreement both improve, actual joint changes >=5%; do not claim match-level strength"))
    nets=(ref.expert_model(),ref.common.quantized(ref.common.model(ref.PARENT,ref.PROTOCOL["parent_weights_sha256"])[0]).eval())
    d=ref.load(tr);rg=np.random.default_rng(SEED)
    ix=np.concatenate([rg.choice(np.where(d["gid"]==g["gid"])[0],min(g["rows"],12),False) for g in tr])
    train=collect(d,nets,ix)
    gid,role,y,z,x,keep=train
    _,inv,n=np.unique(gid[keep],return_inverse=True,return_counts=True)
    sample_w=torch.tensor(1./(len(n)*n[inv]),dtype=torch.float32)
    kinds=torch.from_numpy(role[keep].astype(np.int64))
    target=torch.from_numpy(y[keep].astype(np.int64))
    base=torch.from_numpy(z[keep].astype(np.float32))
    terms=torch.from_numpy(x[keep].astype(np.float32))
    theta=torch.zeros((3,x.shape[-1]),requires_grad=True)
    opt=torch.optim.Adam([theta],lr=.035)
    for it in range(120):
        delta=2*torch.tanh(torch.einsum("jak,jk->ja",terms,theta[kinds])/2)
        loss=(torch.nn.functional.cross_entropy((base+delta)/1.5,target,reduction="none")*sample_w).sum()+.01*theta.square().mean()
        opt.zero_grad();loss.backward();opt.step()
    coeff=theta.detach().numpy()
    old=role_fit.PARENT.read_text()
    expr='combined = _HYBRID["expert_weight"] * scores[0] + _HYBRID["student_weight"] * scores[1]'
    if old.count(expr)!=2:raise ValueError("unexpected actor implementation")
    pack=old.replace(expr,expr+"\n        combined = combined + _role_residual(x, slot, unit, move, legal, scores[0], scores[1])")
    pack+="\n_ROLE_THETA = np.asarray("+repr(coeff.tolist())+", dtype=np.float32)\n"+SRC
    candidate=HERE/"candidate.py";candidate.write_text(pack)
    NP.mkdir(exist_ok=True)
    fresh=[dict(g,shard_sha256=r["sha256"],rows=r["rows"]) for g in external for r in [role_fit.data_build.prep(g,NP)]]
    d=ref.load(fresh,NP)
    test=collect(d,nets,np.arange(len(d["gid"])))
    result=dict(parent_sha256=ref.sha(role_fit.PARENT),candidate_sha256=ref.sha(candidate),candidate_bytes=candidate.stat().st_size,
                fitted_parameters=int(coeff.size),training_root_count=len(ix),train=scores(train,coeff),external=scores(test,coeff))
    role_fit.save(HERE/"fit.json",result)
    print(result["external"]["all"],flush=True)


if __name__=="__main__":
    run()

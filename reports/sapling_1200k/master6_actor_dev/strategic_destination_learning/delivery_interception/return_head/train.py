"""Fit one preset pair of small deposit predictors on ten maps and score the other five."""
from collections import Counter
import gzip
import json
from pathlib import Path
import time

import numpy as np
import torch

import dataset
from features import FREIGHT_PATH,NAMES
from metrics import evaluate
from predict import ARMS,outputs,sha

HERE=Path(__file__).resolve().parent
LOCK="dfeb10b6d289a434620d8ae255f7bbacee7b3612402a646dd2aeb6119da3c670"


def save(path,value):
    tmp=path.with_suffix(path.suffix+".part")
    payload=(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+"\n").encode()
    tmp.write_bytes(gzip.compress(payload,mtime=0) if path.suffix==".gz" else payload)
    tmp.replace(path)


def balanced(terrain):
    names,idx,count=np.unique(terrain,return_inverse=True,return_counts=True)
    return 1/(len(names)*count[idx])


def main():
    start=time.perf_counter()
    if sha(HERE/"protocol.json")!=LOCK:
        raise ValueError("preregistered plan changed")
    plan=json.loads((HERE/"protocol.json").read_bytes())
    x,rows=dataset.build()
    t=np.array([r["terrain"] for r in rows])
    a=np.array([r["amount"] for r in rows]); y=a>0
    fit=np.isin(t,plan["training_terrains"]);held=np.isin(t,plan["held_terrains"])
    if np.any(fit==held) or len(np.unique(t[fit]))!=10 or len(np.unique(t[held]))!=5 or len(rows)!=885:
        raise ValueError("frozen partitions not disjoint and complete")
    wt=balanced(t[fit])
    mu=wt@x[fit]
    sd=np.sqrt(wt@np.square(x[fit]-mu));sd[sd<1e-8]=1.
    z=np.clip((x[fit]-mu)/sd,-6,6)
    positive=y[fit]
    wpos=balanced(t[fit][positive])
    log_a=np.log1p(a[fit][positive])
    av=float(wpos@log_a); asc=float(np.sqrt(wpos@np.square(log_a-av)))
    if asc<=1e-8:
        raise ValueError("unidentifiable positive-amount scale")
    weights=dict(names=np.asarray(NAMES),mean=mu,scale=sd,positive_mean=av,positive_scale=asc,
                 max_log_payment=float(max(log_a)))
    pred={}
    tensor=lambda v:torch.as_tensor(v,dtype=torch.float64)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    for k in ARMS:
        torch.manual_seed(621758490)
        network=torch.nn.Sequential(torch.nn.Linear(len(NAMES),12),torch.nn.Tanh(),torch.nn.Linear(12,2)).double()
        mask=np.ones(len(NAMES))
        if k=="geometry":
            mask[2:]=0
        inputs=tensor(z*mask)
        labels=tensor(positive)
        w=tensor(wt); wp=tensor(wpos); normalized_amount=tensor((log_a-av)/asc)
        opt=torch.optim.AdamW(network.parameters(),lr=.005,weight_decay=.002)
        for _ in range(100):
            opt.zero_grad(set_to_none=True)
            raw=network(inputs)
            binary=torch.nn.functional.binary_cross_entropy_with_logits(raw[:,0],labels,reduction="none")
            loss=binary@w+.15*(raw[positive,1]-normalized_amount).square()@wp
            loss.backward()
            opt.step()
        weights.update({k+"_mask":mask,k+"_hidden":network[0].weight.detach().numpy().copy(),
                        k+"_hidden_bias":network[0].bias.detach().numpy().copy(),k+"_output":network[2].weight.detach().numpy().copy(),
                        k+"_output_bias":network[2].bias.detach().numpy().copy()})
        pred[k]=outputs(weights,x,k)
        with torch.no_grad():
            p=network(inputs)[:,0].sigmoid().numpy()
        if np.max(abs(p-pred[k][0][fit]))>1e-12:
            raise ValueError("exported NumPy model disagrees with fitted PyTorch")
    tmp=HERE/"weights.part"
    with tmp.open("wb") as f:
        np.savez_compressed(f,**weights)
    tmp.replace(HERE/"weights.npz")
    result={}
    for k,take in (("fit",fit),("held",held)):
        result[k]=evaluate([r for r,v in zip(rows,take) if v],{arm:(p[take],at[take]) for arm,(p,at) in pred.items()})
    for j,r in enumerate(rows):
        r["split"]="fit" if fit[j] else "held"
        r["probability"]={k:float(pred[k][0][j]) for k in ARMS}
        r["cash_prediction"]={k:float(pred[k][1][j]) for k in ARMS}
    save(HERE/"predictions.json.gz",dict(rows=rows))
    result["source_sha256"]=plan["source_case_sha256"]
    result["trial_predictions_sha256"]=sha(HERE/"predictions.json.gz")
    result["per_binary_or_amount_head_total_joint_parameters"]=len(NAMES)*12+12+24+2
    result["seconds"]=time.perf_counter()-start
    result["observed_horizon_counts"]=dict(Counter(int(r["observed_joint_turns"]) for r in rows))
    save(HERE/"result.json",result)
    files={"weights":HERE/"weights.npz","features":HERE/"features.py","predict":HERE/"predict.py",
           "freight_features":FREIGHT_PATH,"protocol":HERE/"protocol.json","dataset":HERE/"dataset.py",
           "trainer":Path(__file__).resolve(),"metrics":HERE/"metrics.py","result":HERE/"result.json"}
    release=dict(sha256={k:sha(f) for k,f in files.items()},training_terrains=plan["training_terrains"],
                 held_terrains=plan["held_terrains"],training_rows=int(fit.sum()),within_study_held_rows=int(held.sum()))
    save(HERE/"release.json",release)
    print("release",sha(HERE/"release.json"),json.dumps({s:[result[s]["roots"],result[s]["positives"],
          result[s]["differences"],result[s]["threshold_switches"]] for s in ("fit","held")}),flush=True)


if __name__=="__main__":
    main()

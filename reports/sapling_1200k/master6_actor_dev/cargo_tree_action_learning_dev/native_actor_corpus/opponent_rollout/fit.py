"""Train one 4,080-parameter original-action model with clustered maps and physical seats."""
from collections import Counter
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from model import CONTRACT,HERE,ROLE,SOURCES,compact_fit_card,load,save,sha,ticks_for,validate

def main():
    start=time.process_time();wall=time.monotonic()
    plan=validate(); torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.manual_seed(plan["seed"])
    cards=load(SOURCES)["train"];ticks=ticks_for(cards)
    cache=Path(tempfile.gettempdir())/("sapling-opponent-frames-"+sha(CONTRACT)[:16]+"-"+sha(HERE/"model.py")[:12])
    cache.mkdir(exist_ok=True)
    arrays={k:[] for k in ("mats","legal","y","role")};counts=Counter();coverage=[]
    for i,c in enumerate(cards):
        target=cache/(c["raw_sha256"]+".npz")
        if not target.exists():
            converted,stats=compact_fit_card(c,ticks[c["raw_sha256"]])
            tmp=target.with_name(target.name+".tmp")
            with tmp.open("wb") as f:np.savez(f,**converted,**{"s_"+k:v for k,v in stats.items()})
            os.replace(tmp,target)
        with np.load(target)as a:
            for key,parts in arrays.items():parts.append(a[key])
            counts.update({k[2:]:int(a[k]) for k in a if k.startswith("s_")})
        coverage.append({"sha256":c["raw_sha256"],"seed":c["seed"],"expert_seat":c["expert_seat"],
                         "selected_t":ticks[c["raw_sha256"]],"learned_slots":len(arrays["y"][-1])})
        if (i+1)%20==0:print("verified train tapes",i+1,"CPU",round(time.process_time()-start,1),flush=True)
        if time.process_time()-start>1200:raise TimeoutError("frozen CPU ceiling")
    tic=time.process_time();data={k:np.concatenate(v)for k,v in arrays.items()};del arrays
    xx=torch.from_numpy(data["mats"]);legal=torch.from_numpy(data["legal"]);y=torch.from_numpy(data["y"].astype(np.int64))
    mlp=torch.nn.Sequential(torch.nn.Linear(83,48),torch.nn.Tanh(),torch.nn.Linear(48,1,bias=False))
    initial=[p.detach().clone() for p in mlp.parameters()]
    opt=torch.optim.Adam(mlp.parameters(),lr=.002);losses=[];updates=0
    for epoch in range(12):
        loss_sum=0.
        for ix in torch.randperm(len(y)).split(384):
            logits=mlp(xx[ix].float()).squeeze(-1).masked_fill(~legal[ix],-1e9)
            loss=torch.nn.functional.cross_entropy(logits,y[ix]);opt.zero_grad();loss.backward();opt.step()
            loss_sum+=float(loss.detach())*len(ix);updates+=1
        losses.append(loss_sum/len(y));print("epoch",epoch+1,round(losses[-1],5),flush=True)
        if time.process_time()-start>1200:raise TimeoutError("frozen CPU ceiling")
    weights=dict(hidden=mlp[0].weight.detach().numpy().T,bias=mlp[0].bias.detach().numpy(),head=mlp[2].weight.detach().numpy().T)
    target=HERE/"model.npz";np.savez_compressed(target,**weights)
    with torch.no_grad():
        sample=xx[:128].float();actual=np.tanh(sample.numpy()@weights["hidden"]+weights["bias"])@weights["head"]
        error=float(np.max(np.abs(actual-mlp(sample).numpy())))
    stats=dict(seed=plan["seed"],parameters=sum(v.size for v in weights.values()),raw_tapes=len(cards),
               action_roots=sum(len(v)for v in ticks.values()),role_slot_counts=dict(Counter(ROLE[int(r)]for r in data["role"])),
               fitted_slots=len(y),prefix_counts=dict(counts),update_count=updates,epoch_mean_nll=losses,
               changed_parameters=sum(int(torch.count_nonzero(a!=b)) for a,b in zip(initial,mlp.parameters())),
               weights_sha256=sha(target),input_protocol_sha256=sha(CONTRACT),source_sha256=sha(SOURCES),
               torch_version=torch.__version__,numpy_version=np.__version__,cpu_affinity=sorted(os.sched_getaffinity(0)),
               numerical_threads=torch.get_num_threads(),optimization_cpu_seconds=time.process_time()-tic,
               total_cpu_seconds=time.process_time()-start,elapsed_seconds=time.monotonic()-wall,numpy_torch_max_abs_error=error,
               code_sha256={k:sha(HERE/k)for k in ("model.py","fit.py")})
    assert stats["parameters"]==4080 and stats["action_roots"]==8064 and error<2e-5
    save(HERE/"training_roots.json.gz",coverage);stats["root_index_sha256"]=sha(HERE/"training_roots.json.gz")
    save(HERE/"fit.json",stats);print("fit complete",stats,flush=True)
if __name__=="__main__":main()

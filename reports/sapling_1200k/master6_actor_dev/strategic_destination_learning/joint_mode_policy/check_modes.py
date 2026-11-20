"""Verify the served posterior on a real legal physical prefix in both seats."""
import numpy as np
import torch
from train_modes import HERE,SOURCE,PLAN,Modes,common,native,source
from public_eval import module


def main():
    data=source.read_split("train")
    ids=[int(np.flatnonzero(data["seat"]==s)[0]) for s in (0,1)]
    b=native.batch(data,ids)
    pair=common.package(SOURCE/"bc.py")
    partner,student=[common.torch_model(pair,i).eval() for i in (0,1)]
    with torch.no_grad():
        tok,ctx,_=student.encode(b)
        pt,pc,_=partner.encode(b)
        base=.25*student.decode(b,tok,ctx)+.75*partner.decode(b,pt,pc)
    batch=dict(base=base,ctx=ctx,moves=b["mf"],role=b["tok_kind"][:,:12].clamp(0,2),legal=b["legal"],y=b["y"],usable=b["eligible"])
    result={}
    for name in PLAN["models"]:
        agent=module(name,HERE/(name+".py"))
        model=Modes(ctx.shape[1]); model.load_state_dict({k:torch.tensor(v) for k,v in agent._MODE_DATA.items()})
        with torch.no_grad(): nll=model(batch,name=="joint").numpy()
        trace=[]
        for i in range(2):
            prior=agent._mode_prior(ctx[i].numpy())
            likelihood=0.
            for s in np.flatnonzero(b["eligible"][i].numpy()):
                mixed,components=agent._mode_score(prior,base[i,s].numpy(),int(batch["role"][i,s]),batch["moves"][i,s].numpy(),b["legal"][i,s].numpy())
                y=int(b["y"][i,s])
                likelihood-=float(mixed[y])
                if agent._SHARE_MODE:
                    prior=agent._normalize_mode(prior+components[:,y])
            trace.append(dict(seat=int(data["seat"][ids[i]]),served=likelihood,torch=float(nll[i]),absolute_error=abs(likelihood-float(nll[i]))))
        if max(r["absolute_error"] for r in trace)>2e-5:raise ValueError("conditional probabilities changed")
        result[name]=trace
    common.save(HERE/"parity.json",result)


if __name__=="__main__":
    torch.set_num_threads(1)
    main()

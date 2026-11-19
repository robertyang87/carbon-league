"""Localize changes in the pure Sapling policy on the predeclared twenty held maps."""
from collections import defaultdict
import copy
import json
from statistics import mean

import numpy as np
import torch

import train as training

ENCODER={"stem","stem_n","blocks","tattn","tok1","tok2","fuse","attn","sc1","sc2","teach","ctx"}
DECODER={"h0","din","aemb","gru","whead","bhead","U","V"}
OTHER={"value","utility","wdl","margin","aux"}


def hybrids(old, new):
    old_state=old.state_dict()
    new_state=new.state_dict()
    assert old_state.keys()==new_state.keys()
    models={"parent":old, "final":new}
    for name, region in (("new_encoder",ENCODER), ("new_decoder",DECODER)):
        state={}
        for key, old_value in old_state.items():
            prefix=key.split(".")[0]
            if prefix not in ENCODER | DECODER | OTHER:
                raise ValueError(f"unclassified tensor {key}")
            if prefix in OTHER and not torch.equal(old_value,new_state[key]):
                raise ValueError(f"unregistered auxiliary head change {key}")
            state[key]=new_state[key] if prefix in region else old_value
        model=copy.deepcopy(old)
        model.load_state_dict(state,strict=True)
        models[name]=model
    return models


def main():
    torch.set_num_threads(1)
    recipe=training.check()
    fit=training.load(training.HERE/"fit.json")
    old,_=training.common.model(training.START,recipe["parent_npz_sha256"])
    new,_=training.common.model(training.HERE/"candidate.npz",fit["weight_sha256"])
    models=hybrids(old.eval(),new.eval())
    games=[g for g in training.assemble() if g["split"]=="holdout"]
    data=training.ref.load(games,training.SHARDS)
    counts=defaultdict(lambda:np.zeros(3,np.float64))
    with torch.inference_mode():
        for ix in range(0,len(data["gid"]),16):
            sl=slice(ix,ix+16)
            b=training.common.batch(data,sl)
            mask=training.common.actionable(b)
            y=b["y"].clamp_min(0)
            roles=b["tok_kind"][:,:y.shape[1]]
            for key,model in models.items():
                logits=model(b)[0]
                loss=-logits.log_softmax(-1).gather(-1,y.unsqueeze(-1)).squeeze(-1)
                matched=logits.argmax(-1)==y
                for j,gid in enumerate(data["gid"][sl]):
                    for k,role in enumerate(("collector","planter","base","all")):
                        valid=mask[j] if role=="all" else mask[j] & (roles[j]==k)
                        counts[(int(gid),role,key)]+=(int(valid.sum()),int(matched[j][valid].sum()),float(loss[j][valid].sum()))
    means=[]
    for role in ("collector","planter","base","all"):
        for name in models:
            values=[v for (_,r,k),v in counts.items() if (r,k)==(role,name) and v[0]]
            means.append(dict(role=role,network=name,maps=len(values),free_slots=int(sum(v[0] for v in values)),
                mean_map_accuracy=mean(v[1]/v[0] for v in values),mean_map_ce=mean(v[2]/v[0] for v in values)))
    result=dict(parent=recipe["parent_npz_sha256"],final=fit["weight_sha256"],split="fixed teacher prefixes; FP32 pure policy; held maps only",
                map_equal=means, per_map=[dict(gid=g,role=r,network=n,total=int(v[0]),correct=int(v[1]),sum_ce=v[2])
                    for (g,r,n),v in sorted(counts.items())])
    target=training.HERE/"encoder_decoder_audit.json"
    if target.is_file() and json.loads(target.read_bytes())!=result:
        raise ValueError("previous audit differs")
    target.write_text(json.dumps(result,indent=1,allow_nan=False)+"\n")
    print([r for r in means if r["role"]=="all"])


if __name__=="__main__":
    main()

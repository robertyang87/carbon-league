"""Audit every in-scope v1 base choice from the six tapes using exactly the shipped model and thresholds."""
from collections import Counter
import importlib.util
import sys
import numpy as np
from build import BASE,DEV,HERE,ROOT,read,write,sha,emit
sys.path.insert(0,str(ROOT/'code'))
from fastsim import FastCarbon


def one(game):
    path=ROOT/game["source"];raw=read(path);seat=game["seat"];cfg=raw["game_configuration"]
    spec=importlib.util.spec_from_file_location("inspect_workforce_only",HERE/"agent.py")
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    original=m._workforce
    detail=[]
    def inspect(x,aux,enc,parent,branch):
        F=sys.modules["cnet2_feat"];stage=299-int(x["left"])
        if 8<=stage<=170 and np.sum(x["tok_mask"][:10]&(x["tok_kind"][:10]==1))>=2 and 2 in parent[10:]:
            same,rows=m._wf_decode(x,aux,enc,branch,teacher=parent)
            assert np.array_equal(same,parent)
            for r in rows:
                s=r["slot"];v=r["info"]
                if parent[s]!=2:continue
                z=np.clip((r["features"]-m._WF_MU)/m._WF_SD,-4,4)
                lp=r["scores"]+np.r_[0.,np.clip(z@m._WF_COEF,-3,3)]
                p=np.exp(lp-lp.max());p/=p.sum();a=int(p.argmax())
                reason=None
                if not all(r["legal"]):reason="not_legal"
                elif v["risk"] or v["occupied"] or v["free"]<=0:reason="danger_or_reserved"
                elif a==2:reason="model_chooses_same_planter"
                elif p[a]<.5 or p[a]-p[2]<.1:reason="confidence_or_contrast"
                elif a==0 and v["ncollector"]<1:reason="protect_last_collector"
                else:
                    other=sum(parent[j]>0 for j in range(s+1,F.OWN) if x["tok_mask"][j])
                    if v["cash"]-v["spend"]-30*(other+int(a!=0))<30:reason="preserve_cash"
                    else:
                        new,_=m._wf_decode(x,aux,enc,branch,forced_slot=s,forced=a,record=False)
                        if new is None or np.flatnonzero(new!=parent).tolist()!=[s]:reason="other_prefix_commands_change"
                        elif m._return_conflicts(x,new)-m._return_conflicts(x,parent):reason="new_conflict"
                        else:reason="committed"
                detail.append(dict(t=stage,base=m._FEAT.own_uid[s],reason=reason,scores_prior=r["scores"].tolist(),
                    fitted_probabilities=p.tolist(),predicted=F.BASE_ACTS[a],cash=v["cash"],planned_plant_spend=v["spend"]))
        out=original(x,aux,enc,parent,branch)
        assert np.array_equal(parent,out),"archived zero-trigger decision no longer matches"
        return out
    m._workforce=inspect
    sim=FastCarbon(cfg).reset(seed=raw["seed"]);cnt=0
    for t,acts in enumerate(raw["actions"][:171]):
        obs=sim.obs(seat);me=obs["players"][seat]
        eligible=t>=8 and sum(w[2]=="PLANTER" for w in me[2].values())>=2 and any(acts[seat].get(b)=="RECPLANTER" for b in me[1])
        if t==0 or eligible:
            assert m.agent(obs,cfg)==acts[seat]
            if eligible:cnt+=1
        else:m._FEAT.observe(obs)
        sim.step(*acts)
    assert len({x["t"] for x in detail})==cnt==game["eligible_parent_turns_until_commit"]
    return dict(seed=game["seed"],seat=seat,turns=cnt,base_decisions=detail)


if __name__=="__main__":
    assert sha(emit())==read(HERE/"roots.json")["candidate_sha256"]
    r=read(HERE/"screen.json")
    games=[one(g) for g in r["games"]]
    c=Counter(u["reason"] for g in games for u in g["base_decisions"])
    a=Counter(u["predicted"] for g in games for u in g["base_decisions"])
    write(HERE/"funnel.json",dict(eligible_turns=sum(g["turns"] for g in games),base_slots=sum(c.values()),
              first_rejection_reasons=dict(c),predicted_before_safety=dict(a),games=games))
    print(dict(c),dict(a))

"""Audit change-of-measure against previously signed menus on held trajectories."""
import gzip
import importlib.util
import json
from pathlib import Path
import sys
from collections import Counter
import numpy as np
from build import HERE, SOURCE, PARENT, sha, OLD_PACKAGE
from kernel import physical

sys.path.insert(0, str(HERE.parent))
import free_entry

def load(path, alias):
    spec = importlib.util.spec_from_file_location(alias,path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def key(x):
    return tuple(sorted(x.items()))

def require(ok,why):
    if not ok: raise ValueError(why)

def main():
    source = SOURCE/"entry.json.gz"
    old_gate=json.loads((SOURCE/"gate.json").read_bytes())
    require(old_gate["candidate_sha256"]==OLD_PACKAGE and sha(source)==old_gate["entry_sha256"],"Source verification changed")
    reference={(r["seed"],r["seat"],r["t"]):r for r in json.loads(gzip.decompress(source.read_bytes()))}
    model=load(HERE/"sampler.py","onpolicy_sampler")
    base=load(PARENT,"onpolicy_incumbent")
    fixture=json.loads((HERE.parent/"sources.json").read_bytes())
    rng=np.random.default_rng(61001)
    zero=np.zeros(24,np.float32)
    hypothetical=rng.normal(0.,.04,size=24).astype(np.float32)
    eps=.0001
    stats=Counter()
    prob_gap=kernel_gap=deriv_gap=0.
    records=[]
    for card in fixture["cards"]:
        if card["split"]!="val":
            continue
        model._NATIVE_RNG=np.random.default_rng(int(card["sha256"][:12],16)^61001)
        for t,obs,cfg,_ in free_entry.public_history(card):
            ref=reference.get((card["seed"],card["expert_seat"],t))
            if ref is None and t not in (0,1):
                base._FEAT.observe(obs)
                model._FEAT.observe(obs)
                continue
            cfg={k:v for k,v in cfg.items() if k!="randomSeed"}
            action=model.agent(obs,cfg)
            true_base=base.agent(obs,cfg)
            log=model._LAST_POLICY
            require(log["issued"]==action and true_base==log["parent"],"Public issuer or frozen incumbent mismatch")
            options=log["orders"]
            keys=[key(a["action"]) for a in options]
            require(len(keys)==len(set(keys)) and key(action) in keys,"Ambiguous physical output")
            prob=np.array([a["probability"] for a in options])
            prior=np.array([a["prior_probability"] for a in options])
            actual=keys.index(key(action))
            require(np.max(abs(prob-prior))<1e-11 and abs(np.log(prob[actual])-log["logp"])<1e-12,"Behavior logp")
            menu=log["policy_features"]
            p,g=physical(menu,len(options),zero)
            kernel_gap=max(kernel_gap,float(np.max(abs(p-prob))))
            require(np.max(abs(p-prob))<5e-6 and np.max(abs(p@g))<1e-10, "Independent policy or score expectation changed")
            if ref is None:
                require(len(keys)==1 and action==true_base and menu is None,"Ordinary step changed")
                stats["ordinary"]+=1
            else:
                old={key(o["action"]):o["prior_probability"] for o in ref["options"]}
                require(set(keys)==set(old) and true_base==ref["parent"],"Unseen team order or new protected baseline")
                gap=max(abs(old[k]-p[i]) for i,k in enumerate(keys))
                prob_gap=max(prob_gap,gap)
                require(gap<5e-6,"Different initial exploration from archived untrained correction")
                p1,g1=physical(menu,len(options),hypothetical)
                def smooth(weight):
                    h=np.asarray(menu["hidden"],np.float64)
                    l=np.asarray(menu["logits"],np.float64)+h@weight
                    distribution=np.exp(l-l.max())
                    masses=np.bincount(menu["route_to_order"], 0.5*distribution/distribution.sum(), minlength=len(options))
                    masses[menu["base_index"]]+=.5
                    return masses
                up=smooth(hypothetical.astype(np.float64)+eps*zero_direction.astype(np.float64))
                down=smooth(hypothetical.astype(np.float64)-eps*zero_direction.astype(np.float64))
                delta=max(abs((np.log(up)-np.log(down))/(2*eps)-g1@zero_direction))
                deriv_gap=max(deriv_gap,float(delta))
                require(delta<5e-5,"Incorrect grouped score gradient")
                stats["pivot"]+=1
                stats["changed"]+=int(action!=true_base)
                stats["three_options"]+=int(len(keys)>=3)
            records.append(dict(seed=card["seed"],terrain=card["terrain"],seat=obs["player"],step=t,
                                issued=action,logp=log["logp"],parent=true_base,orders=options))
    require(stats["ordinary"]==16 and stats["pivot"]==len(reference)==96,"Historical denominator")
    payload=(json.dumps(records,sort_keys=True,separators=(",",":"))+"\n").encode()
    (HERE/"old_entry.json.gz").write_bytes(gzip.compress(payload,mtime=0))
    report=dict(sampler=sha(HERE/"sampler.py"),head=sha(HERE/"behavior.npz"),strong_parent=sha(PARENT),
                source_behavior=OLD_PACKAGE,source_archive=sha(source),records=sha(HERE/"old_entry.json.gz"),
                historical_maps=len({r["terrain"] for r in records}),counts=stats,
                zero_head_probability_max_difference_vs_recorded_prior=prob_gap,max_recomputed_prob_error=kernel_gap,
                grouped_score_directional_derivative_max_error=deriv_gap,eligible_for_training_collection=True,
                strength_evaluation_eligible=False,standalone_bytes=(HERE/"sampler.py").stat().st_size)
    (HERE/"gate.json").write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    print(report)

zero_direction=np.random.default_rng(16431).normal(size=24)
zero_direction=(zero_direction/np.linalg.norm(zero_direction)).astype(np.float32)
if __name__=="__main__":
    main()

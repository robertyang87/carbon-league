"""Check the publication's physical distributions on every prespecified validation root."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path

from train import HERE, STUDY, PARENT, RULE, entry, load_source, read, save, sha, check
import numpy as np
import sys


def key(order):
    return tuple(sorted(order.items()))


def expected(row, head):
    base=key(row["base_order"])
    grouped=defaultdict(lambda: [0.,0.])
    grouped[base]=[.5,.5]
    x=np.asarray(row["features"],np.float32)
    z=np.asarray(row["legal_logits"],np.float32)
    scores=z+(np.tanh(x@head["W0"]+head["B0"])@head["W1"]).ravel()
    def softmax(a):
        a=a.astype(np.float64)
        e=np.exp(a-a.max())
        return e/e.sum()
    for act, p, q in zip(row["orders"],softmax(scores),softmax(z)):
        dest=grouped[key(act)]
        dest[0]+=.5*p
        dest[1]+=.5*q
    return grouped


def verify(work):
    fit=read(HERE/"fit.json")
    candidate=HERE/"candidate.py"
    check(sha(candidate)==fit["candidate"] and sha(HERE/"head.npz")==fit["head"],"release drift")
    model, parent=load_source(candidate),load_source(PARENT)
    costs=[]
    errors=[]
    records=[]
    ordinary=0
    draws=0
    roles=Counter()
    import time
    with np.load(HERE/"head.npz",allow_pickle=False) as h:
        head={k:h[k] for k in h}
    for card in read(STUDY/"sources.json")["cards"]:
        if card["split"]!="val":
            continue
        batch=read(work/(card["sha256"]+".json.gz"))
        rows={r["t"]:r for r in batch["rows"]}
        model._NATIVE_RNG=np.random.default_rng(int(card["sha256"][:16],16)^RULE["seed"])
        for t,obs,config,expert in entry.public_history(card):
            if t not in rows and t not in (0,1):
                model._FEAT.observe(obs)
                parent._FEAT.observe(obs)
                continue
            check(t!=0 or card["expert_seat"]==obs["player"],"seat")
            start=time.perf_counter()
            actual=model.agent(obs,config)
            costs.append(time.perf_counter()-start)
            base=parent.agent(obs,config)
            receipt=model._LAST_POLICY
            dist={key(r["action"]):(r["probability"],r["prior_probability"]) for r in receipt["orders"]}
            check(len(dist)==len(receipt["orders"]) and receipt["issued"]==actual and receipt["parent"]==base,
                  "sampling/parent public-entry parity")
            check(key(actual) in dist and abs(np.log(dist[key(actual)][0])-receipt["logp"])<1e-12,"issued logp")
            check(all(np.isfinite(v) and v>0 for pair in dist.values() for v in pair)
                  and np.allclose(np.array(list(dist.values())).sum(axis=0),1.,atol=1e-10,rtol=0),"mass")
            if t not in rows:
                check(len(dist)==1 and actual==base,"ordinary step")
                ordinary+=1
                continue
            r=rows[t]
            want=expected(r,head) if r["possible"] else {key(base):[1.,1.]}
            check(set(want)==set(dist) and key(base) in dist,"different physical support")
            gap=max(abs(np.array(want[k])-dist[k]).max() for k in want)
            errors.append(float(gap))
            check(gap<5e-6,"runtime and archived feature probabilities disagree")
            roles[str(receipt["pivot_role"])]+=1
            draws+=int(actual!=base)
            records.append(dict(seed=card["seed"],terrain=card["terrain"],seat=obs["player"],t=t,
                                support=len(dist),issued=actual,teacher=expert,parent=base,options=receipt["orders"],
                                slot=receipt["pivot_slot"],role=receipt["pivot_role"],logp=receipt["logp"],gap=float(gap)))
    m=fit["held_results"]
    delta=np.array([r["trained_nll"]-r["base_nll"] for r in m.values()])
    broad=[r for r in records if r["support"]>=3]
    sufficient=len(broad)>=8 and len({r["terrain"] for r in broad})>=2
    check(len(records)==96 and ordinary==16 and len(m)==4,"coverage")
    save(HERE/"entry.json.gz",records)
    eligible=bool(np.mean(delta)<0 and sum(delta<0)>=3 and sufficient and draws>0)
    save(HERE/"gate.json",dict(
        schema="native-physical-proposal-gate/1",candidate_sha256=sha(candidate),parent_sha256=sha(PARENT),
        head_sha256=sha(HERE/"head.npz"),fit_sha256=sha(HERE/"fit.json"),entry_sha256=sha(HERE/"entry.json.gz"),
        physical_change_proven=draws>0,eligible_for_original_development=eligible,
        scope="Proposal-init eligibility only. Additional independent mechanism review required before live dispatch.",
        held_roots=len(records),matched_ordinary=ordinary,max_probability_error=max(errors),
        training_physical_maps=fit["maps"],held_physical_maps=len(m),improving_held_maps=int(sum(delta<0)),
        equal_map_pivot_nll_change=float(delta.mean()),three_or_more_orders=len(broad),
        broad_maps=len({r["terrain"] for r in broad}),seeded_draws_different_from_parent=draws,
        roles=roles,heat_local_s={"p95":float(np.percentile(costs,95)),"max_includes_cold":float(max(costs))},
        official_new_games=0))
    print(read(HERE/"gate.json"),flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--work",required=True,type=Path)
    verify(parser.parse_args().work)

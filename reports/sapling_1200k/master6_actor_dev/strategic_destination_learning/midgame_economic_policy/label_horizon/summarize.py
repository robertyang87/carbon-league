"""Compare the two frozen demonstration weights with physical-map clusters and disjoint outcome intervals."""
from collections import Counter,defaultdict
import json
from statistics import mean
import numpy as np
from extract import HERE, LANE, CACHE, read, signature, write

REPS=10000
SEED=2026112060


def metrics(r):
    result={}
    for who,p in zip(("own","rival"),r["sides"]):
        receipts=p["root_receipts"]
        x={"purchase":float(bool(receipts)),"plant":float(any(t["kind"]=="plant" for t in receipts)),
           "seize":float(any(t["kind"]=="seize" for t in receipts)),
           "root_acquired_rent20":sum(t["rent20"] for t in receipts),
           "root_acquired_rent60":sum(t["rent60"] for t in receipts) if r["complete_60"] else None,
           "root_acquired_rent_to_terminal":sum(t["rent_observed_terminal"] for t in receipts),
           "terminal_score":p["official_score"],"initial_cash":p["initial_cash"],
           "after20_to_terminal_official":p["after_20_to_terminal_official"],
           "after20_to_terminal_cash":p["after_20_to_terminal_bank"]}
        for h in ("root","early","late"):
            box=p[h] or {}
            for k in ("net_cash","tree","tree_spend","tree_net","bank","donation","recruit"):
                x[h+"_"+k]=box.get(k)
        for suffix in ("20","60","_to_terminal"):
            rent=x["root_acquired_rent"+suffix]
            x["new_tree_return"+suffix]=(rent-x["root_tree_spend"] if rent is not None and x["root_tree_spend"] is not None else None)
        result[who]=x
    result["edge"]={k:(result["own"][k]-result["rival"][k] if result["own"][k] is not None and result["rival"][k] is not None else None) for k in result["own"]}
    return result


def ci(values):
    a=np.array(values,dtype=float)
    rng=np.random.default_rng(SEED)
    ix=rng.integers(len(a),size=(REPS,len(a)))
    return np.quantile(a[ix].mean(axis=1),[.025,.975]).tolist()


def aggregate(rows):
    groups=defaultdict(list)
    for r in rows:
        groups[r["terrain"]].append(r)
    out={}
    for who in ("own","rival","edge"):
        out[who]={}
        for key in rows[0]["metrics"][who]:
            uniform=[];weighted=[];n=0
            for group in groups.values():
                good=[r for r in group if r["metrics"][who][key] is not None]
                if not good:continue
                vals=[r["metrics"][who][key] for r in good];ws=[r["weight"] for r in good]
                uniform.append(float(np.mean(vals)))
                weighted.append(float(np.average(vals,weights=ws)))
                n+=len(good)
            if uniform:
                changes=np.array(weighted)-uniform
                out[who][key]={"n":n,"maps":len(changes),"uniform":mean(uniform),"weighted":mean(weighted),
                               "delta":float(changes.mean()),"delta_cluster_95":ci(changes)}
    return out


def paid_vs_non(rows):
    result={}
    for who in ("own","rival"):
        groups=defaultdict(lambda:[[],[]])
        for r in rows:
            p=int(r["metrics"][who]["purchase"])
            groups[r["terrain"]][p].append(r["weight"])
        contrasts=[mean(b)-mean(a) for a,b in groups.values() if a and b]
        result[who]=dict(maps_with_both=len(contrasts),total_maps=len(groups),
                         within_map_weight_paid_minus_unpaid=mean(contrasts),cluster_95=ci(contrasts))
    return result


def partial_correlations(rows):
    """Map+clock effects are removed from predictors and responses with map-equal observation weights."""
    rows=[r for r in rows if r["complete_60"]]
    terrain=sorted({r["terrain"] for r in rows})
    times=sorted({r["t"] for r in rows})
    groups=Counter(r["terrain"] for r in rows)
    X=np.zeros((len(rows),len(terrain)+len(times)-1))
    for i,r in enumerate(rows):
        X[i,terrain.index(r["terrain"])]=1
        j=times.index(r["t"])
        if j>0:X[i,len(terrain)+j-1]=1
    w=np.array([1/groups[r["terrain"]] for r in rows])
    cols=["observed_early","late","remaining_to_end","weight","residual"]
    v=np.array([[r["target"],r["metrics"]["edge"]["late_net_cash"],r["metrics"]["edge"]["after20_to_terminal_official"],r["weight"],r["residual"]] for r in rows])
    sw=np.sqrt(w)[:,None]
    beta=np.linalg.lstsq(X*sw,v*sw,rcond=None)[0]
    centered=(v-X@beta)*sw
    covariance=centered.T@centered
    scale=np.sqrt(np.diag(covariance))
    cor=covariance/scale[:,None]/scale[None,:]
    return dict(maps=len(terrain),n=len(rows),clock_fixed=times,columns=cols,matrix=cor.tolist())


def contrasts(rows):
    complete=[r for r in rows if r["complete_60"] and r["verified_ledger"]]
    rlo,rhi=np.quantile([r["residual"] for r in complete],[.1,.9])
    cases={}
    for name,criterion,sort_key in (
        ("high_residual_future_reversal",lambda r:r["residual"]>=rhi and r["weight"]>1 and r["metrics"]["edge"]["late_net_cash"]<0,
         lambda r:r["metrics"]["edge"]["late_net_cash"]),
        ("low_residual_future_recovery",lambda r:r["residual"]<=rlo and r["weight"]<1 and r["metrics"]["edge"]["late_net_cash"]>0,
         lambda r:-r["metrics"]["edge"]["late_net_cash"]),
        ("upweighted_purchase",lambda r:r["weight"]>1 and r["metrics"]["own"]["purchase"]>0,lambda r:-r["residual"]),
        ("downweighted_profitable_purchase",lambda r:r["weight"]<1 and r["metrics"]["own"]["new_tree_return_to_terminal"]>0,
         lambda r:-r["metrics"]["own"]["new_tree_return_to_terminal"])):
        eligible=[r for r in complete if criterion(r)]
        chosen=sorted(eligible,key=lambda r:(sort_key(r),r["gid"],r["t"]))[:1]
        cases[name]=dict(eligible=len(eligible),examples=[{k:r[k] for k in ("gid","terrain","teacher","opponent","seat","t","weight","residual","metrics","sides")} for r in chosen])
    return dict(residual_decile_thresholds=[rlo,rhi],cases=cases)


def main():
    rows=read(HERE/"training_cash.json.gz")
    for r in rows:r["metrics"]=metrics(r)
    by_game={r["gid"]:r for r in rows}
    missed=[dict(gid=k,turns=v["turns"],missing=[t for t in (30,50,70,90,110) if t+20>v["turns"]]) for k,v in by_game.items() if v["turns"]<299]
    for q in missed:
        game=read(CACHE/(q["gid"]+".json.gz"))
        last=game["steps"][-1][0]["observation"]["players"]
        price=min(game["configuration"][k] for k in ("recPlanterCost","recCollectorCost"))
        q["extinct_physical_seats"]=[s for s,(cash,bases,workers,trees) in enumerate(last) if not workers and not trees and (not bases or cash<price)]
        q["official_physical_scores"]=game["rewards"]
        if not q["extinct_physical_seats"]:
            raise ValueError("Early game lacks a verified extinction")
    c=dict(registered_maps=234,registered_times_per_map=5,possible=1170,
           source_rows=len(rows),kept_fixed20=sum(r["t"]+20<=r["turns"] for r in rows),kept_fixed60=sum(r["complete_60"] for r in rows),
           extinct_or_early_games=missed,early_missing60=[{k:r[k] for k in ("gid","seat","t","turns")} for r in rows if not r["complete_60"]],
           exact_ledger_rows=sum(r["verified_ledger"] for r in rows),ledger_gaps=[{k:r[k] for k in ("gid","t","missing_ledger_reason")} for r in rows if not r["verified_ledger"]])
    out=dict(schema="fixed-training-weight-diagnostic/1",train_labels_sha256=signature(LANE/"advantage.json.gz"),
             audit_roots_sha256=signature(HERE/"training_cash.json.gz"),scope=c,weights=aggregate(rows),paid_vs_non=paid_vs_non(rows),
             residual_correlations=partial_correlations(rows),counterexamples=contrasts(rows),root_purchase=Counter(
                     str((r["metrics"]["own"]["purchase"],r["metrics"]["rival"]["purchase"])) for r in rows))
    write(HERE/"summary.json",out)


if __name__=="__main__":
    main()
